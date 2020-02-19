import torch
import torch.nn as nn
import math
import torch.nn.functional as F

try:
    import ipdb
except ImportError:
    pass

class ScoreAttention11(nn.Module):
    def __init__(self, attend_dim, query_dim, att_dim):
        super(ScoreAttention, self).__init__()
        self.attend_dim = attend_dim
        self.query_dim = query_dim
        self.att_dim = att_dim
        self.linear_pre = nn.Linear(attend_dim, att_dim, bias=True)
        # self.linear_2 = nn.Linear(att_dim, att_dim, bias=True)
        self.linear_q = nn.Linear(query_dim, att_dim, bias=True)
        self.linear_v = nn.Linear(att_dim, 1, bias=True)
        if torch.__version__[:6] == '0.1.12':
            self.sm = nn.Softmax()
        else:
            self.sm = nn.Softmax(dim=1)
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
        tmp20 = F.tanh(tmp10)
        energy_1 = self.linear_v(tmp20.view(-1, tmp20.size(2))).view(tmp20.size(0), tmp20.size(1))  # batch x sourceL

        if self.mask is not None:
            energy_1 = energy_1 * (1 - self.mask) + self.mask * (-1e8)
        energy_1 = F.softmax(energy_1, dim=1)

        return energy_1, precompute_1
        

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.att_dim) + ' * ' + '(' \
               + str(self.attend_dim) + '->' + str(self.att_dim) + ' + ' \
               + str(self.query_dim) + '->' + str(self.att_dim) + ')' + ')'


class ScoreAttention(nn.Module):
    def __init__(self,attend_dim, query_dim, att_dim):
        super(ScoreAttention, self).__init__()
        #assert linear_q == linear_v
        self.d_model = 256
        self.heads = 8
        self.d_k = d_model // heads
        self.attend_dim_linear = nn.Linear(d_model, d_model)     #here 3 is the number of heads
        self.query_dim_linear = nn.Linear(d_model, d_model)
        self.att_dim_linear = nn.Linear(d_model, d_model)
       # self.attn_heads = nn.ModuleList([
        #    ScoreAttention(attend_dim, query_dim, att_dim,dropout)
        #self.attention = ScoreAttention11(attend_dim, query_dim, att_dim)
       #self.projection = nn.Linear(attend_dim, query_dim, att_dim)

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, attend_dim, query_dim, att_dim):
        
        #xe , xp =[attn(linear_pre,linear_q,linear_v)
         #   for i,attn in enumerate(self.attn_heads) ]

         bs = query_dim.size(0)
         #k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
         attend_dim = self.attend_dim_linear(attend_dim).view(bs, -1, self.h, self.d_k)
         query_dim = self.query_dim_linear(query_dim).view(bs, -1, self.h, self.d_k)
         att_dim = self.attend_dim_linear(att_dim).view(bs, -1, self.h, self.d_k)


         attend_dim = attend_dim.transpose(1,2)
         query_dim = query_dim.transpose(1,2)
         att_dim = att_dim.transpose(1,2)


                                                      #xe,xp = self.attention()
         xe,xp = ScoreAttention11(attend_dim, query_dim, att_dim)

         energy = xe.transpose(1,2).contiguous().view(bs, -1, self.d_model)                                    #torch.cat(xe,dim = linear_q)
         precompute = xp.transpose(1,2).contiguous().view(bs, -1, self.d_model)                                #torch.cat(xp,dim = linear_v)
       # x= self.projection(x)

         return energy, precompute






               