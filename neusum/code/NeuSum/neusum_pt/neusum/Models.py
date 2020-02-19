import torch
import torch as T
import torch.nn as nn
from torch.autograd import Variable
import neusum.modules
#import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

try:
    import ipdb
except ImportError:
    pass
                                                                                                        #import the required packages

class Encoder(nn.Module):                                  #simple encoder without attention
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.dict = dicts
        self.num_directions = 2 if opt.sent_brnn else 1
        self.sent_enc_size = opt.sent_enc_size
        assert opt.sent_enc_size % self.num_directions == 0
        self.hidden_size = self.sent_enc_size // self.num_directions
        self.word_emb_size = opt.word_vec_size
        self.freeze_word_vecs_enc = opt.freeze_word_vecs_enc       #initializee

        input_size = opt.word_vec_size

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=neusum.Constants.PAD)      #word embedding initialization
        self.rnn = nn.GRU(input_size, self.hidden_size,
                          num_layers=opt.layers,
                          dropout=opt.sent_dropout,
                          bidirectional=opt.sent_brnn)                    #bi-directional gru
        if opt.sent_dropout > 0:
            self.dropout = nn.Dropout(opt.sent_dropout)           #if dropout is set drop values
        else:
            self.dropout = None

    def load_pretrained_vectors(self, opt, logger):        #load pre-trained vector in our case it is glove vector of 50d
        if opt.pre_word_vecs_enc is not None:
            from neusum.xutils import load_pretrain_embedding
            pretrained = load_pretrain_embedding(logger, opt.pre_word_vecs_enc, self.dict, self.word_emb_size, None)
            self.word_lut.weight.data.copy_(pretrained)
            # self.word_lut.weight.requires_grad = False

    def forward(self, input, hidden=None):
        """
        input: (wrap(srcBatch), lengths)
        """
        lengths = input[1].data.view(-1).tolist()  # lengths data is wrapped inside a Variable
        if self.freeze_word_vecs_enc:
            wordEmb = self.word_lut(input[0]).detach()
        else:
            wordEmb = self.word_lut(input[0])                   #word embedding
        emb = pack(wordEmb, lengths)
        outputs, hidden_t = self.rnn(emb, hidden)               #gru line 33 this gru is for sentence level encoding and selection and scoring
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        if self.dropout is not None:
            outputs = self.dropout(outputs)
            hidden_t = self.dropout(hidden_t)
        return hidden_t, outputs


class DocumentEncoder(nn.Module):                       #document level encoading
    def __init__(self, opt):
        self.layers = opt.layers
        self.doc_len = opt.max_doc_len
        self.num_directions = 2 if opt.doc_brnn else 1
        self.doc_enc_size = opt.doc_enc_size
        assert opt.doc_enc_size % self.num_directions == 0                      #if correct only then proceed 
        self.hidden_size = self.doc_enc_size // self.num_directions
        input_size = opt.sent_enc_size

        super(DocumentEncoder, self).__init__()
        self.rnn = nn.GRU(input_size, self.hidden_size,
                          num_layers=opt.layers,
                          dropout=opt.doc_dropout,
                          bidirectional=opt.doc_brnn)                       #bi-directional gru for document level encoading
        if opt.doc_dropout > 0:
            self.dropout = nn.Dropout(opt.doc_dropout)
        else:
            self.dropout = None                                             #the dropout is set in this condition is 0.2 if not no dropout take place
        if opt.use_self_att:
            self.self_att = neusum.modules.ConcatAttention(self.doc_enc_size, self.doc_enc_size, opt.self_att_size)
            self.self_att_rnn = nn.GRU(self.doc_enc_size * 2, self.hidden_size,
                                       num_layers=1, dropout=0,
                                       bidirectional=opt.doc_brnn)
        else:
            self.self_att = None                            #if self attention is used it will be infused with the bi-directional gru

    def gen_mask_with_length(self, doc_len, batch_size, lengths):
        mask = torch.ByteTensor(batch_size, doc_len).cuda().zero_()
        ll = lengths.data.view(-1).tolist()
        for i in range(batch_size):
            for j in range(doc_len):
                if j >= ll[i]:
                    mask[i][j] = 1
        mask = mask.float()
        return mask                                                             #this will generate the mask

    def self_attention(self, contexts, mask):                                   #self attention first split apply attention stack them together
        precompute = None
        contexts_t = contexts.transpose(0, 1).contiguous()
        self.self_att.applyMask(mask)
        all_h = []
        for h in contexts.split(1, dim=0):
            new_h, attn, precompute = self.self_att(h.squeeze(0), contexts_t, precompute)
            all_h.append(new_h)
        res = torch.stack(all_h)
        return res

    def forward(self, input, hidden=None):
        """
        input: (sentence_vectors, src[2])
        """

        sent_vec = input[0].view(-1, self.doc_len, input[0].size(1))  # (batch, doc_len, dim)
        sent_vec = sent_vec.transpose(0, 1).contiguous()  # (doc_len, batch, dim)
        lengths = input[1].cuda()                                                               #get the sentence vector and then apply embedding
        sorted_lengths, orig_index = torch.sort(lengths, descending=True)
        _, restore_idex = torch.sort(orig_index)
        new_input = sent_vec.index_select(1, orig_index.view(-1))

        emb = pack(new_input, sorted_lengths.data.view(-1).tolist())  # TODO need to be sorted by length
        outputs, hidden_t = self.rnn(emb, hidden)                       #output is get from gru
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        hidden_t = hidden_t.index_select(1, restore_idex.view(-1))
        outputs = outputs.index_select(1, restore_idex.view(-1))      #the hidden layer concept is used to calculate the history then apply mask on the output

        doc_sent_mask = self.gen_mask_with_length(outputs.size(0), outputs.size(1), input[1])
        with torch.no_grad():
            doc_sent_mask = Variable(doc_sent_mask, requires_grad=False)

        if self.dropout is not None:
            hidden_t = self.dropout(hidden_t)
            outputs = self.dropout(outputs)                             #if dropout applicable 
        if self.self_att is not None:                                   #if self attention is applied
            self_att_outputs = self.self_attention(outputs, doc_sent_mask) #apply self attention
            outputs = torch.cat((outputs, self_att_outputs), dim=2)     #concatnation is applied
            new_sent_vec = outputs.index_select(1, orig_index.view(-1))
            new_sent_vec_pack = pack(new_sent_vec, sorted_lengths.data.view(-1).tolist())
            new_outputs, _ = self.self_att_rnn(new_sent_vec_pack, hidden)
            new_outputs = unpack(new_outputs)[0]
            new_outputs = new_outputs.index_select(1, restore_idex.view(-1))
            outputs = new_outputs
            if self.dropout is not None:
                outputs = self.dropout(outputs)                                   #new output will be calculated accordingly and if droup-out applicable apply that

        return hidden_t, outputs, doc_sent_mask


class StackedGRU(nn.Module):                                            #stacking the bi-directional gru
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1


def get_hard_attention_index(seq_len, batch_size, indices):     #hard attention is treated as focused so it is use to focus 
    if isinstance(indices, Variable):
        index_data = indices.data
    else:
        index_data = indices
    buf = []
    for batch_id, seq_idx in enumerate(index_data):
        idx = seq_idx * batch_size + batch_id
        buf.append(idx)
    return torch.LongTensor(buf)


class Pointer(nn.Module):                   #pointer nework they use to suppport the decoder altough in original papger they use gru architecture
    def __init__(self, opt, dicts):
        self.opt = opt
        self.layers = opt.layers
        input_size = opt.doc_enc_size

        super(Pointer, self).__init__()
        self.rnn = StackedGRU(opt.layers, input_size, opt.dec_rnn_size, opt.dec_dropout)
        self.scorer = neusum.modules.ScoreAttention(opt.doc_enc_size, opt.dec_rnn_size, opt.att_vec_size)
        self.hidden_size = opt.dec_rnn_size

    def forward(self, hidden, context, doc_sent_mask, src_pad_masks, pre_att_hard,
                att_precompute_hard, max_step, prev_targets):
        """

        :param hidden: pointer network RNN hidden
        :param context: the document sentence vectors (doc_len, batch, dim)
        :param doc_sent_mask: doc_sent_mask for data pad masking (batch, doc_len)
        :param src_pad_masks: [src_pad_mask for t in times] for rule masking
        :param pre_att_hard: previous hard attention
        :param att_precompute_hard: hard attention precompute
        :param max_step:
        :param prev_targets: (step, batch)
        :return:
        """
        cur_context_hard = pre_att_hard

        all_scores = []
        self.scorer.applyMask(doc_sent_mask)
        hard_context_buf = context.view(-1, context.size(2))
        batch_first_context = context.transpose(0, 1).contiguous()

        for i in range(max_step):
            input_vector = cur_context_hard
            output, hidden = self.rnn(input_vector, hidden)
            reg_score, att_precompute_hard = self.scorer(output, batch_first_context, att_precompute_hard)
            all_scores.append(reg_score)
            if self.training and max_step > 1:
                max_idx = prev_targets[i]
                hard_max_idx = get_hard_attention_index(context.size(0), context.size(1), max_idx).cuda()
                hard_max_idx = Variable(hard_max_idx, requires_grad=False, volatile=(not self.training))
                cur_context_hard = hard_context_buf.index_select(dim=0, index=hard_max_idx)
            elif not self.training:
                max_score, max_idx = reg_score.max(dim=1)
                hard_max_idx = get_hard_attention_index(context.size(0), context.size(1), max_idx).cuda()
                with torch.no_grad():
                    hard_max_idx = Variable(hard_max_idx, requires_grad=False)
                cur_context_hard = hard_context_buf.index_select(dim=0, index=hard_max_idx)
        scores = torch.stack(all_scores)

        return hidden, scores, cur_context_hard, att_precompute_hard


class DecInit(nn.Module):                                                   #simple decoder unit
    def __init__(self, opt):
        super(DecInit, self).__init__()
        self.num_directions = 2 if opt.doc_brnn else 1
        assert opt.doc_enc_size % self.num_directions == 0
        self.enc_rnn_size = opt.doc_enc_size
        self.dec_rnn_size = opt.dec_rnn_size
        self.initer = nn.Linear(self.enc_rnn_size // self.num_directions, self.dec_rnn_size)

    def forward(self, last_enc_h):
        # batchSize = last_enc_h.size(0)
        # dim = last_enc_h.size(1)
        return T.tanh(self.initer(last_enc_h))


class DecInitAtt(nn.Module):                                               #decoder with attention
    def __init__(self, opt):
        super(DecInitAtt, self).__init__()
        self.num_directions = 2 if opt.doc_brnn else 1
        assert opt.doc_enc_size % self.num_directions == 0
        self.enc_rnn_size = opt.doc_enc_size
        self.dec_rnn_size = opt.dec_rnn_size
        self.init_query = nn.Parameter(torch.zeros(1, self.enc_rnn_size))
        self.init_att = neusum.modules.ConcatAttention(self.enc_rnn_size, self.enc_rnn_size, opt.att_vec_size)
        self.transformer = nn.Linear(self.enc_rnn_size, self.dec_rnn_size)

    def forward(self, context, mask):
        batchSize = context.size(1)
        self.init_att.applyMask(mask)
        res, _, _ = self.init_att(self.init_query.expand(batchSize, self.init_query.size(1)),
                                  context.transpose(0, 1),
                                  None)
        res = self.transformer(res)
        return res


class NMTModel(nn.Module):                                                  #our model architecture
    def __init__(self, sent_encoder, doc_encoder, pointer, decIniter, reward_cal):
        super(NMTModel, self).__init__()
        self.sent_encoder = sent_encoder
        self.doc_encoder = doc_encoder
        self.pointer = pointer
        self.decIniter = decIniter
        self.reward_cal = reward_cal

    def make_init_att(self, context):                                          #initialize attention
        """

        :param context: (seq_len, batch, dim)
        :return:
        """
        batch_size = context.size(1)
        h_size = (batch_size, self.doc_encoder.hidden_size * self.doc_encoder.num_directions)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def encode_document(self, src, indices):                                               #encode the document at sentece level as well as document level encoding
        """
        Encode the document.                                                               #also after encoding words and sentences are sorted

        :param src: (wrap(srcBatch), lengths, doc_lengths)
        :param indices: Variable(torch.LongTensor(list(indices)).view(-1).cuda(), volatile=self.volatile)
        :return: doc_hidden, doc_context, doc_sent_mask
        """
        enc_hidden, context = self.sent_encoder(src)
        sentence_vectors = enc_hidden.transpose(0, 1).contiguous().view(enc_hidden.size(1), -1)
        _, restore_index = torch.sort(indices, dim=0)
        sentence_vectors = sentence_vectors.index_select(0, restore_index)
        doc_hidden, doc_context, doc_sent_mask = self.doc_encoder((sentence_vectors, src[2]))

        return doc_hidden, doc_context, doc_sent_mask

    def gen_all_masks(self, base_mask, targets):                         #apply mask
        batch_size = targets.size(0)
        res = []
        res.append(base_mask)
        for i in range(targets.size(1)):
            next_mask = res[-1].data.clone()
            for j in range(batch_size):
                if targets.data[j][i] < next_mask.size(1):
                    next_mask[j][targets.data[j][i]] = 1
            next_mask = Variable(next_mask, requires_grad=False, volatile=(not self.training))
            res.append(next_mask)
        return res

    def forward(self, input):
        """
        input: (wrap(srcBatch), lengths, doc_lengths), (src_raw,), \
               (tgtBatch,), (simple_wrap(oracleBatch), oracleLength), \
               simple_wrap(torch.LongTensor(list(indices)).view(-1)), (simple_wrap(src_rouge_batch),)
        """

        doc_hidden, doc_context, doc_sent_mask = self.encode_document(input[0], input[4])                 #document level encoading

        init_att = self.make_init_att(doc_context)                                                        #applying attention then decoading
        if isinstance(self.decIniter, DecInitAtt):
            dec_hidden = self.decIniter(doc_context, doc_sent_mask).unsqueeze(0)
        elif isinstance(self.decIniter, DecInit):
            if self.decIniter.num_directions == 2:
                dec_hidden = self.decIniter(doc_hidden[1]).unsqueeze(0)  # [1] is the last backward hiden
            else:
                dec_hidden = self.decIniter(doc_hidden[0]).unsqueeze(0)
        else:
            raise ValueError("Unknown decIniter type")

        max_point_step = input[3][1].max().item()
        prev_att = init_att
        pointer_precompute_hard = None                                                                      

        all_masks = self.gen_all_masks(doc_sent_mask, input[3][0])

        oracle_targets = input[3][0]

        dec_hidden, scores, att_vec_hard, pointer_precompute_hard = self.pointer(
            dec_hidden, doc_context,
            doc_sent_mask, all_masks,
            prev_att, pointer_precompute_hard,
            max_point_step,
            oracle_targets.transpose(0, 1).contiguous())                                            #history is generated and pointer network are used for decoding
        return scores, doc_sent_mask
