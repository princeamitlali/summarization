import torch
import torch.nn as nn
import math

try:
    import ipdb
except ImportError:
    pass

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, linear_pre,linear_q,linear_v, mask=None):
        attnp = torch.bmm(linear_pre, linear_q.transpose(linear_pre, linear_q)) # (Batch, Seq, Seq)
        # we get an attention score between each position in the sequence
        # for each batch
        attne = torch.bmm(linear_q, linear_v.transpose(linear_q, linear_v))
        # scale the dot products by the dimensionality (see the paper for why we do this!)
        attnp = attne / math.sqrt(linear_pre)
        attne = attne / math.sqrt(linear_q)
        # normalize the weights across the sequence dimension
        # (Note that since we transposed, the sequence and feature dimensions are switched)
        attnp = torch.exp(attnp)
        attne = torch.exp(attne)
        # fill attention weights with 0s where padded
        if mask is not None: attn = attn.masked_fill(mask, 0)
        attnp = attnp / attnp.sum(-1, keepdim=True)
        attnp = self.dropout(attnp)
        precompute = torch.bmm(attnp, linear_q)
        attne = attne / attne.sum(-1, keepdim=True)
        attne = self.dropout(attne)
        energy = torch.bmm(attne, linear_v) # (Batch, Seq, Feature)
        weightedContext = torch.bmm(energy,linear_v)
        return  weightedContext,energy, precompute
class ConcatAttention11():
    def __init__(self, attend_dim, query_dim, att_dim):
        super(ConcatAttention11,self).__init__()
        self.attend_dim = attend_dim
        self.query_dim = query_dim
        self.att_dim = att_dim
        self.linear_pre = nn.Linear(attend_dim, att_dim, bias=True)
        self.linear_q = nn.Linear(query_dim, att_dim, bias=False)
        self.linear_v = nn.Linear(att_dim, 1, bias=False)
        if torch.__version__[:6] == '0.1.12':
            self.sm = nn.Softmax()
        else:
            self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context, precompute=None):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        if precompute is None:
            precompute00 = self.linear_pre(context.contiguous().view(-1, context.size(2)))
            precompute = precompute00.view(context.size(0), context.size(1), -1)  # batch x sourceL x att_dim
        targetT = self.linear_q(input).unsqueeze(1)  # batch x 1 x att_dim

        tmp10 = precompute + targetT.expand_as(precompute)  # batch x sourceL x att_dim
        tmp20 = self.tanh(tmp10)  # batch x sourceL x att_dim
        energy = self.linear_v(tmp20.view(-1, tmp20.size(2))).view(tmp20.size(0), tmp20.size(1))  # batch x sourceL
        if self.mask is not None:
            # energy.data.masked_fill_(self.mask, -float('inf'))
            # energy.masked_fill_(self.mask, -float('inf'))   # TODO: might be wrong
            energy = energy * (1 - self.mask) + self.mask * (-1000000)
        score = self.sm(energy)
        score_m = score.view(score.size(0), 1, score.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(score_m, context).squeeze(1)  # batch x dim

        return weightedContext, score, precompute


class ConcatAttention(nn.Module):
    def __init__(self, attend_dim, query_dim, att_dim):
        super(ConcatAttention, self).__init__()
        #assert linear_q == linear_v
        self.attend_dim = attend_dim//3      #here 3 is the number of heads
        self.query_dim = query_dim//3
        self.att_dim = att_dim//3
       #self.attn_heads = nn.ModuleList([
        #    ConcatAttention11(attend_dim, query_dim, att_dim,dropout)
        #self.attention = ConcatAttention11(attend_dim, query_dim, att_dim)
       #self.projection = nn.Linear(attend_dim, query_dim, att_dim)

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, attend_dim, query_dim, att_dim):
        
        #xe , xp =[attn(linear_pre,linear_q,linear_v)
         #   for i,attn in enumerate(self.attn_heads) ]


       # wc,xe,xp = self.attention()
        wc,xe,xp = ConcatAttention11(attend_dim, query_dim, att_dim)
        energy = torch.cat(xe,dim = linear_q)
        precompute = torch.cat(xp,dim = linear_v)
        weightedContext = torch.cat(wc,dim = linear_v)
       # x= self.projection(x)

        return weightedContext,energy, precompute

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.att_dim) + ' * ' + '(' \
               + str(self.attend_dim) + '->' + str(self.att_dim) + ' + ' \
               + str(self.query_dim) + '->' + str(self.att_dim) + ')' + ')'
