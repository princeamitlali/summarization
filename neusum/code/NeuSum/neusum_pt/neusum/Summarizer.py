import neusum
import torch.nn as nn
import torch
from torch.autograd import Variable

try:
    import ipdb
except ImportError:
    pass


class Summarizer(object):
    def __init__(self, opt, model=None, dataset=None, logger=None):   #this is the initialization for the summarizer it will eventually generate the summary
        self.opt = opt

        if model is None:  
            print('+____________________________________not available________________________________________________________________________-')                                  #check if any model is available of not if not then
            checkpoint = torch.load(opt.model)

            model_opt = checkpoint['opt']                  #find the last saved checkpoint
            if logger is not None:                          #if loger file is present it will update the log file
                logger.info('Loading model from {0}'.format(opt.model))
                logger.info('model_opt')
                logger.info(model_opt)
            self.src_dict = checkpoint['dicts']['src']
            self.tgt_dict = checkpoint['dicts']['tgt']

            self.enc_rnn_size = model_opt.doc_enc_size
            self.dec_rnn_size = model_opt.dec_rnn_size          #initialize the details into it
            sent_encoder = neusum.Models.Encoder(model_opt, self.src_dict)    #start the sentence level encoding
            doc_encoder = neusum.Models.DocumentEncoder(model_opt)            #start the document level encoding
            pointer = neusum.Models.Pointer(model_opt, self.tgt_dict)
            if hasattr(model_opt, 'dec_init'):                                  #there are two type of decoder are permitted simple means without attention and other
                if model_opt.dec_init == "simple":                              #with the attention and if neither of two select it will genrate error
                    decIniter = neusum.Models.DecInit(model_opt)
                elif model_opt.dec_init == "att":
                    decIniter = neusum.Models.DecInitAtt(model_opt)
                else:
                    raise ValueError('Unknown decoder init method: {0}'.format(model_opt.dec_init))    #the default decoder is set to simple
            else:
                # TODO: some old model do not have this attribute in it
                decIniter = neusum.Models.DecInit(model_opt)

            model = neusum.Models.NMTModel(sent_encoder, doc_encoder, pointer, decIniter, None)        #loading the model this is the archietecture which is used in the papaer

            #model.load_state_dict(checkpoint['model']) 
            model.load_state_dict(torch.load('/home/prince/workspace/neusum/data/cnndm/models/neusum/model_devRouge_0.1891_e6.pt'), strict=False)                     #load the last checkpoint checkpoint means the value for the learning paramemter

            if opt.cuda:
                model.cuda()
            else:
                model.cpu()                                     #run model on either cpu or gpu

        else:                                                   #if model is present
            self.src_dict = dataset['dicts']['src']
            self.tgt_dict = dataset['dicts']['tgt']

            self.enc_rnn_size = opt.doc_enc_size
            self.dec_rnn_size = opt.dec_rnn_size
            self.opt.cuda = True if len(opt.gpus) >= 1 else False
            self.opt.n_best = 1
            self.opt.replace_unk = False

        self.tt = torch.cuda if opt.cuda else torch
        self.model = model
        self.model.eval()

        self.copyCount = 0                                              #initialize the values

    def buildData(self, srcBatch, srcRaw, tgtRaw, oracleBatch, srcRougeBatch):      #this will generate the source dataset
        srcData = [[self.src_dict.convertToIdx(b,
                                               neusum.Constants.UNK_WORD) for b in doc] for doc in srcBatch]

        return neusum.Dataset(srcData, srcRaw, tgtRaw, oracleBatch, srcRougeBatch, self.opt.batch_size,
                           self.opt.max_doc_len, self.opt.cuda, volatile=True)

    def buildTargetTokens(self, pred, src, attn):                               #this will generate the token as for calculating rouge f1 score we need tokens
        tokens = self.tgt_dict.convertToLabels(pred, neusum.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == neusum.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex[0]]
        return tokens

    def translateBatch(self, batch):     #this block is for a batch which has 64 article
        """
        input: (wrap(srcBatch), lengths, doc_lengths), (src_raw,), \
               (tgtBatch,), (oracleBatch, oracleLength), \
               Variable(torch.LongTensor(list(indices)).view(-1).cuda(), volatile=self.volatile)
        """

        batchSize = batch[0][2].size(1)
        beamSize = self.opt.beam_size      #initialize the batch size and beam size

        #  (1) Encode the document
        srcBatch = batch[0]
        indices = batch[4]
        doc_hidden, doc_context, doc_sent_mask = self.model.encode_document(srcBatch, indices)     #encoding from models.py

        if isinstance(self.model.decIniter, neusum.Models.DecInitAtt):     #if decoder with attention is used
            enc_hidden = self.model.decIniter(doc_context, doc_sent_mask)
        elif isinstance(self.model.decIniter, neusum.Models.DecInit):      #decoder without attn direction denote the direction of bi-directional gru 2 means backward gru
            if self.model.decIniter.num_directions == 2:
                enc_hidden = self.model.decIniter(doc_hidden[1])  # [1] is the last backward hiden
            else:
                enc_hidden = self.model.decIniter(doc_hidden[0])          #forward gru
        else:
            raise ValueError("Unknown decIniter type")

        pointer_precompute, reg_precompute = None, None

        decStates = enc_hidden  # batch, dec_hidden

        # Expand tensors for each beam.
        context = Variable(doc_context.data.repeat(1, beamSize, 1))
        decStates = Variable(decStates.unsqueeze(0).data.repeat(1, beamSize, 1))
        # pointer_att_vec = self.model.make_init_att(context)
        reg_att_vec = self.model.make_init_att(context)              # apply attention and initialize mask, mask is used to favor it is just like attaching weight
        with torch.no_grad():
            padMask = Variable(doc_sent_mask.data.repeat(beamSize, 1))
        baseMask = doc_sent_mask.data.clone()

        beam = [neusum.Beam(beamSize, self.opt.cuda, self.opt.force_max_len) for k in range(batchSize)] #beam is initialized it is used to find novel sentences
        batchIdx = list(range(batchSize))
        remainingSents = batchSize                    #this act as sentence scoring and selection together process

        for i in range(self.opt.max_decode_step):
            # Prepare decoder input.
            input = torch.stack([b.getCurrentState() for b in beam
                                 if not b.done]).transpose(0, 1).contiguous().view(1, -1)
            if i > 0:
                all_masks = torch.stack(
                    [b.get_doc_sent_mask(baseMask[idx]) for idx, b in enumerate(beam)
                     if not b.done]).transpose(0, 1).contiguous()
                with torch.no_grad():
                    all_masks = Variable(all_masks.view(-1, all_masks.size(2)))
            else:
                all_masks = padMask
            decStates, attn, reg_att_vec, reg_precompute = self.model.pointer(
                decStates, context, padMask, [all_masks],
                reg_att_vec, reg_precompute,
                1, reg_att_vec)                                                                      #applying mask

            # batch x beam x numWords
            attn = attn.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()      #initiallizing attn

            active = []
            father_idx = []
            for b in range(batchSize):
                if beam[b].done:
                    continue

                idx = batchIdx[b]
                if not beam[b].advance(attn.data[idx]):
                    active += [b]
                    father_idx.append(beam[b].prevKs[-1])  # this is very annoying, getting the index of the selected sentences in respectively

            if not active:
                break

            # to get the real father index
            real_father_idx = []
            for kk, idx in enumerate(father_idx):
                real_father_idx.append(idx * len(father_idx) + kk)              #finf=d the index of the sentence from the actual article

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}   

            def updateActive(t, rnnSize):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                with torch.no_grad():
                    return Variable(view.index_select(1, activeIdx) \
                                .view(*newSize))

            decStates = updateActive(decStates, self.dec_rnn_size)
            context = updateActive(context, self.enc_rnn_size)
            reg_att_vec = updateActive(reg_att_vec, self.enc_rnn_size)
            reg_precompute = None
            padMask = updateActive(padMask, padMask.size(1))                                             #update param

            # set correct state for beam search
            previous_index = torch.stack(real_father_idx).transpose(0, 1).contiguous()
            with torch.no_grad():
                previous_index = Variable(previous_index)
            decStates = decStates.view(-1, decStates.size(2)).index_select(0, previous_index.view(-1)).view(
                *decStates.size())
            reg_att_vec = reg_att_vec.view(-1, reg_att_vec.size(1)).index_select(0, previous_index.view(
                -1)).view(*reg_att_vec.size())

            remainingSents = len(active)

        # (4) package everything up
        allHyp, allScores, allAttn = [], [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            allHyp += [hyps]

        return allHyp, allScores, allAttn, None

    def translate(self, src_batch, src_raw, tgt_raw):
        #  (1) convert words to indexes
        dataset = self.buildData(src_batch, src_raw, tgt_raw, None, None)
        # (wrap(srcBatch),  lengths), (wrap(tgtBatch), ), indices
        batch = dataset[0]

        #  (2) translate
        pred, predScore, attn, _ = self.translateBatch(batch)

        #  (3) convert indexes to words
        predBatch = []
        predict_id = []
        src_raw = batch[1][0]
        for b in range(len(src_raw)):
            predBatch_nbest = []
            predict_id_nbest = []
            for n in range(self.opt.n_best):
                n = 0
                selected_sents = []
                selected_id = []
                for idx in pred[b][n]:
                    if idx >= len(src_raw[b]):
                        break
                    selected_sents.append(src_raw[b][idx])
                    selected_id.append(idx)
                predBatch_nbest.append(' '.join(selected_sents))
                predict_id_nbest.append(tuple(selected_id))
            predBatch.append(predBatch_nbest)
            predict_id.append(predict_id_nbest)

        return predBatch, predict_id, predScore, None
