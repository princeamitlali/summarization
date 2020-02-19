import torch
import torch.nn as nn
import math

try:
    import ipdb
except ImportError:
    pass

class ConcatAttention11():
    def __init__(self, attend_dim, query_dim, att_dim):
        super(ConcatAttention, self).__init__()
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

    def forward(self, input, context, precompute_1=None):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        if precompute_1 is None:
            precompute_100 = self.linear_pre(context.contiguous().view(-1, context.size(2)))
            precompute_1 = precompute_100.view(context.size(0), context.size(1), -1)  # batch x sourceL x att_dim
        targetT = self.linear_q(input).unsqueeze(1)  # batch x 1 x att_dim

        tmp10 = precompute_1 + targetT.expand_as(precompute_1)  # batch x sourceL x att_dim
        tmp20 = self.tanh(tmp10)  # batch x sourceL x att_dim
        energy_1 = self.linear_v(tmp20.view(-1, tmp20.size(2))).view(tmp20.size(0), tmp20.size(1))  # batch x sourceL
        if self.mask is not None:
            # energy_1.data.masked_fill_(self.mask, -float('inf'))
            # energy_1.masked_fill_(self.mask, -float('inf'))   # TODO: might be wrong
            energy_1 = energy_1 * (1 - self.mask) + self.mask * (-1000000)
        score = self.sm(energy_1)
        score_m = score.view(score.size(0), 1, score.size(1))  # batch x 1 x sourceL

        weightedContext_1 = torch.bmm(score_m, context).squeeze(1)  # batch x dim

        return weightedContext_1, score, precompute_1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.att_dim) + ' * ' + '(' \
               + str(self.attend_dim) + '->' + str(self.att_dim) + ' + ' \
               + str(self.query_dim) + '->' + str(self.att_dim) + ')' + ')'



class ConcatAttention(nn.Module):
    def __init__(self,attend_dim, query_dim, att_dim):
        super(ScoreAttention, self).__init__()
        #assert linear_q == linear_v
        global d_model
        d_model = 256
        global heads
        heads = 8
        global d_k
        d_k = d_model // heads
        self.attend_dim_linear = nn.Linear(d_model, d_model)     #here 3 is the number of heads
        self.query_dim_linear = nn.Linear(d_model, d_model)
        self.att_dim_linear = nn.Linear(d_model, d_model)
       #self.attn_heads = nn.ModuleList([
        #    ConcatAttention11(attend_dim, query_dim, att_dim,dropout)
        #self.attention = ConcatAttention11(attend_dim, query_dim, att_dim)
       #self.projection = nn.Linear(attend_dim, query_dim, att_dim)

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, attend_dim, query_dim, att_dim):
        
        #xe , xp =[attn(linear_pre,linear_q,linear_v)
         #   for i,attn in enumerate(self.attn_heads) ]
         self.attend_dim = attend_dim
         self.query_dim = query_dim
         #self.att_dim = att_dim
         self.att_dim = att_dim

       # wc,xe,xp = self.attention()
         bs = query_dim.size(0)
         #k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
         print("..............................................................")
         print(heads,d_k,bs)
         print("..............................................................")

         attend_dim = self.attend_dim_linear(attend_dim).view(bs, -1,heads, d_k)
         query_dim = self.query_dim_linear(query_dim).view(bs, -1,heads, d_k)
         #att_dim = self.att_dim_linear(att_dim).view(bs,-1, heads, d_k)


         attend_dim = attend_dim.transpose(1,2)
         query_dim = query_dim.transpose(1,2)
         #att_dim = att_dim.transpose(1,2)

         wc,xe,xp = ConcatAttention11(attend_dim, query_dim, att_dim)
         energy_1 = xe.transpose(1,2).contiguous().view(bs, -1, d_model)
         precompute_1 = xp.transpose(1,2).contiguous().view(bs, -1, d_model)
         weightedContext_1 = wc.transpose(1,2).contiguous().view(bs, -1, d_model)
       # x= self.projection(x)

         return weightedContext,energy, precompute

