import logging
from ast import literal_eval as make_tuple
import torch
import numpy
import neusum

try:
    import ipdb
except ImportError:
    pass                                                                        #initial required import

lower = True
seq_length = 100
max_doc_len = 80
report_every = 100000
shuffle = 1
norm_lambda = 5                           #fix values are assigned altough they are also pass as parameter but in case when not passed it is default value

logger = logging.getLogger(__name__)     #for log file get log file


def makeVocabulary(filenames, size):                                                         #for the purpose of making vocabulary UNK_WORD means unknown words
    vocab = neusum.Dict([neusum.Constants.PAD_WORD, neusum.Constants.UNK_WORD,
                         neusum.Constants.BOS_WORD, neusum.Constants.EOS_WORD], lower=lower)   #convert all words to lower alphabet form
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:                                            #strip words where ever blank space found and add them to vocabulary
            for sent in f.readlines():
                for word in sent.strip().split(' '):
                    vocab.add(word)

    originalSize = vocab.size()                                                               #size of original vocabulary we get 732,204 words
    vocab = vocab.prune(size)                                               #we reduce the vocabulary to top 100,000 words as the later are very few in occurance
    logger.info('Created dictionary of size %d (pruned from %d)' %                             
                (vocab.size(), originalSize))                               #feed into log file the data about vocabulary

    return vocab                                                           


def initVocabulary(name, dataFiles, vocabFile, vocabSize):                 #this will initialize the process of vocabulary
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        logger.info('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = neusum.Dict(lower=lower)
        vocab.loadFile(vocabFile)
        logger.info('Loaded ' + str(vocab.size()) + ' ' + name + ' words')                         #if any vocabulary is present read it and update log file

    if vocab is None:
        # If a dictionary is still missing, generate it.
        logger.info('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFiles, vocabSize)

        vocab = genWordVocab                                                     #if vocab is not present generate it            

    return vocab


def saveVocabulary(name, vocab, file):
    logger.info('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)                                                           #save vocab into vocab.txt file


def np_softmax(x, a=1):
    """Compute softmax values for each sets of scores in x."""
    return numpy.exp(a * x) / numpy.sum(numpy.exp(a * x), axis=0)                   #implementation of softmax function using numpy


def makeData(srcFile, tgtFile, train_oracle_file, train_src_rouge_file, srcDicts, tgtDicts):    #this block will generate the data files
    src, tgt = [], []
    src_raw = []
    src_rouge = []
    oracle = []
    sizes = []
    count, ignored = 0, 0                                           #initializing required data structure

    logger.info('Processing %s & %s ...' % (srcFile, tgtFile))     #update log file
    srcF = open(srcFile, encoding='utf-8')                         #open required file using utf-8 encoading
    tgtF = open(tgtFile, encoding='utf-8')
    oracleF = open(train_oracle_file, encoding='utf-8')           #oraclef file has rouge score for the selected sentence and there index those are selected when extract tgt are generated from abstract target using greedy approch
    src_rougeF = open(train_src_rouge_file, encoding='utf-8')     #this file contains rouge score for all the lines in an article with respect to there corresponding highlights using final greedy approch

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()
        oline = oracleF.readline()
        src_rouge_line = src_rougeF.readline()                    #code to read lines

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "" or src_rouge_line == "":
            logger.info('WARNING: source and target do not have the same number of sentences')
            break                                                       #check whether source and target has equL NUMBER OR ARTICLE OR not if not give a warning and terminate

        sline = sline.strip()
        tline = tline.strip()
        oline = oline.strip()
        src_rouge_line = src_rouge_line.strip()                 #strip initialize for document

        # source and/or target are empty
        if sline == "" or tline == "" or ('None' in oline) or ('nan' in src_rouge_line):
            logger.info('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue                                                        #user warning if an empty line is present we face some empty line in dataset

        srcSents = sline.split('##SENT##')[:max_doc_len]  #strip line whereever ##SENT## delimiter are present till the reach of maximum doc length which is 80 in this case
        tgtSents = tline.split('##SENT##')                #slpit all other
        rouge_gains = src_rouge_line.split('\t')[1:]
        srcWords = [x.split(' ')[:seq_length] for x in srcSents]
        tgtWords = ' '.join(tgtSents)
        oracle_combination = make_tuple(oline.split('\t')[0])
        # oracle_combination = [(x + 1) for x in oracle_combination] + [0]
        oracle_combination = [x for x in oracle_combination]  # no sentinel

        index_out_of_range = [x >= max_doc_len for x in oracle_combination]
        if any(index_out_of_range):
            logger.info('WARNING: oracle exceeds max_doc_len, ignoring (' + str(count + 1) + ')')
            continue                                                        #compare scores with the number of sentences and if greater give a user warning
                                                                            #and update logger accordingly
        src_raw.append(srcSents)                                            #append all files

        src.append([srcDicts.convertToIdx(word,
                                          neusum.Constants.UNK_WORD) for word in srcWords])
        tgt.append(tgtWords)

        oracle.append(torch.LongTensor(oracle_combination))
        rouge_gains = [[float(gain) for gain in x.split(' ')] for x in rouge_gains]
        # rouge_gains = [torch.FloatTensor(x) for x in rouge_gains]
        # rouge_gains = [(x - torch.min(x)) / (torch.max(x) - torch.min(x)) for x in rouge_gains][:1]
        rouge_gains = [numpy.array(x) for x in rouge_gains]
        rouge_gains = [(x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)) for x in rouge_gains]
        rouge_gains = [torch.from_numpy(np_softmax(x, norm_lambda)).float() for x in rouge_gains]
        src_rouge.append(rouge_gains)

        sizes += [len(srcWords)]

        count += 1

        if count % report_every == 0:
            logger.info('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()
    oracleF.close()
    src_rougeF.close()                                                         #close all other file upon updating log file

    if shuffle == 1:                                                            #shuffle sentences for next epoch
        logger.info('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        src_raw = [src_raw[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        oracle = [oracle[idx] for idx in perm]
        src_rouge = [src_rouge[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    logger.info('... sorting sentences by size')                 #sort sentence by size as the short sentence are useful for summary and it contains more info/word
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    src_raw = [src_raw[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]
    oracle = [oracle[idx] for idx in perm]
    src_rouge = [src_rouge[idx] for idx in perm]

    logger.info('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
                (len(src), ignored, seq_length))
    return src, src_raw, tgt, oracle, src_rouge                  #a log update for the data is prepared and then return the prepared data


def prepare_data_online(train_src, src_vocab, train_tgt, tgt_vocab, train_oracle, train_src_rouge):
    dicts = {}
    dicts['src'] = initVocabulary('source', [train_src], src_vocab, 0)    #innitialize vocab for src and target
    dicts['tgt'] = initVocabulary('target', [train_tgt], tgt_vocab, 0)

    logger.info('Preparing training ...')
    train = {}
    train['src'], train['src_raw'], train['tgt'], \
    train['oracle'], train['src_rouge'] = makeData(train_src, train_tgt,
                                                   train_oracle, train_src_rouge,
                                                   dicts['src'], dicts['tgt'])         #update log file and return the data set

    dataset = {'dicts': dicts,
               'train': train,
               # 'valid': valid
               }
    return dataset
