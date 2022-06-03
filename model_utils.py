import torch
import torch.nn as nn
from torch.nn.functional import normalize as l2
import math
import torch.nn.functional as F
import itertools
import numpy as np
from random import sample as random_sample
from env_config import *

'''
    ##############################
    #                            #
    #    Custom Loss Functions   #
    #                            #
    ##############################

'''

def masking_tensor(input_dim, input_lengths, max_length):
    input_lengths = input_lengths.cpu().tolist() 
    mask_tensor = torch.zeros((len(input_lengths), max_length, input_dim)).cuda()
    for idx, input_length in enumerate(input_lengths):
        mask_tensor[idx, :input_length, :] = 1.
                
    return mask_tensor

''''
    #############################################
    #                                           #
    #    Building Blocks for Set Transformer    #
    #                                           #
    #############################################

    Following implementations are based on Juho Lee's Set Transformer
    Citated from paper: An Effective Pretrained Model for Recipe Representation Learning, Li et al., 2020
    Link to citation: http://proceedings.mlr.press/v97/lee19d/lee19d.pdf
    Link to github: https://github.com/juho-lee/set_transformer
'''


class DotProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, queries, keys):

        return torch.bmm(queries, keys.transpose(1, 2))

class ScaledDotProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, queries, keys):

        return torch.bmm(queries, keys.transpose(1, 2)) / (queries.size(2)**0.5)

class GeneralDotProduct(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        torch.nn.init.orthogonal_(self.W)

    def forward(self, queries, keys):

        return torch.bmm(queries @ self.W, keys.transpose(1,2))


class Attention(nn.Module):
    def __init__(self, similarity, hidden_dim=1024):
        super().__init__()
        self.softmax = nn.Softmax(dim=2)
        self.attention_maps = []

        assert similarity in ['dot', 'scaled_dot', 'general_dot']
        if similarity == 'dot':
            self.similarity = DotProduct()
        elif similarity == 'scaled_dot':
            self.similarity = ScaledDotProduct()
        elif similarity == 'general_dot':
            self.similarity = GeneralDotProduct(hidden_dim)
        else:
            raise

    def forward(self, queries, keys, qmasks=None, kmasks=None):
        if torch.is_tensor(qmasks) and not torch.is_tensor(kmasks):
            dim0, dim1 = qmasks.size(0), keys.size(1)
            kmasks = torch.ones(dim0,dim1).cuda()

        elif not torch.is_tensor(qmasks) and torch.is_tensor(kmasks):
            dim0, dim1 = kmasks.size(0), queries.size(1)
            qmasks = torch.ones(dim0,dim1).cuda()
        else:
            pass

        attention = self.similarity(queries, keys)
        if torch.is_tensor(qmasks) and torch.is_tensor(kmasks):
            qmasks = qmasks.repeat(queries.size(0)//qmasks.size(0),1).unsqueeze(2)
            kmasks = kmasks.repeat(keys.size(0)//kmasks.size(0),1).unsqueeze(2)
            attnmasks = torch.bmm(qmasks, kmasks.transpose(1, 2))
            attention = torch.clip(attention, min=-5, max=5)
            attention = attention.exp() * attnmasks
            attention = attention / (attention.sum(2).unsqueeze(2) + 1e-5)
        else:
            attention = self.softmax(attention)

        return attention

def save_attention_maps(self, input, output):
    
    self.attention_maps.append(output.data.detach().cpu().numpy())

class MultiheadAttention(nn.Module):
    def __init__(self, d, h, sim='dot', analysis=False):
        super().__init__()
        assert d % h == 0, f"{d} dimension, {h} heads"
        self.h = h
        p = d // h

        self.project_queries = nn.Linear(d, d)
        self.project_keys    = nn.Linear(d, d)
        self.project_values  = nn.Linear(d, d)
        self.concatenation   = nn.Linear(d, d)
        self.attention       = Attention(sim, p)

        if analysis:
            self.attention.register_forward_hook(save_attention_maps)

    def forward(self, queries, keys, values, qmasks=None, kmasks=None):
        h = self.h
        b, n, d = queries.size()
        _, m, _ = keys.size()
        p = d // h

        queries = self.project_queries(queries)  # shape [b, n, d]
        keys = self.project_keys(keys)  # shape [b, m, d]
        values = self.project_values(values)  # shape [b, m, d]

        queries = queries.view(b, n, h, p)
        keys = keys.view(b, m, h, p)
        values = values.view(b, m, h, p)

        queries = queries.permute(2, 0, 1, 3).contiguous().view(h * b, n, p)
        keys = keys.permute(2, 0, 1, 3).contiguous().view(h * b, m, p)
        values = values.permute(2, 0, 1, 3).contiguous().view(h * b, m, p)

        output = self.attention(queries, keys, qmasks, kmasks)  # shape [h * b, n, p]
        output = torch.bmm(output, values)
        output = output.view(h, b, n, p)
        output = output.permute(1, 2, 0, 3).contiguous().view(b, n, d)
        output = self.concatenation(output)  # shape [b, n, d]

        return output

class MultiheadAttentionExpanded(nn.Module):
    def __init__(self, d, h, sim='dot', analysis=False):
        super().__init__()
        self.project_queries = nn.ModuleList([nn.Linear(d,d) for _ in range(h)])
        self.project_keys    = nn.ModuleList([nn.Linear(d,d) for _ in range(h)])
        self.project_values  = nn.ModuleList([nn.Linear(d,d) for _ in range(h)])
        self.concatenation   = nn.Linear(h*d, d)
        self.attention       = Attention(sim, d)

        if analysis:
            self.attention.register_forward_hook(save_attention_maps)

    def forward(self, queries, keys, values, qmasks=None, kmasks=None):
        output = []
        for Wq, Wk, Wv in zip(self.project_queries, self.project_keys, self.project_values):
            Pq, Pk, Pv = Wq(queries), Wk(keys), Wv(values)
            output.append(torch.bmm(self.attention(Pq, Pk, qmasks, kmasks), Pv))

        output = self.concatenation(torch.cat(output, 1))

        return output

class EmptyModule(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, x):
        return 0.


class RFF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.rff = nn.Sequential(nn.Linear(args.hidden_dim,args.hidden_dim),nn.ReLU(),
                                 nn.Linear(args.hidden_dim,args.hidden_dim),nn.ReLU(),
                                 nn.Linear(args.hidden_dim,args.hidden_dim),nn.ReLU())

    def forward(self, x):

        return self.rff(x)


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d, h, rff, similarity='dot', full_head=False, analysis=False):
        super().__init__()

        self.multihead = MultiheadAttention(d, h, similarity, analysis) if not full_head else MultiheadAttentionExpanded(d, h, similarity, analysis)
        self.layer_norm1 = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.rff = rff

    def forward(self, x, y, xm=None, ym=None, layer_norm=True):
        if layer_norm:
            h = self.layer_norm1(x + self.multihead(x, y, y, xm, ym))
            return self.layer_norm2(h + self.rff(h))
        else:
            h = x + self.multihead(x, y, y, xm, ym)
            return h + self.rff(h)


class SetAttentionBlock(nn.Module):
    def __init__(self, d, h, rff, similarity='dot', full_head=False, analysis=False):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff, similarity, full_head, analysis)

    def forward(self, x, m=None, ln=True):

        return self.mab(x, x, m, m, ln)

class InducedSetAttentionBlock(nn.Module):
    def __init__(self, d, m, h, rff1, rff2, similarity='dot', full_head=False, analysis=False):

        super().__init__()
        self.mab1 = MultiheadAttentionBlock(d, h, rff1, similarity, full_head, analysis)
        self.mab2 = MultiheadAttentionBlock(d, h, rff2, similarity, full_head, analysis)
        self.inducing_points = nn.Parameter(torch.randn(1, m, d))

    def forward(self, x, m=None, ln=True):
        b = x.size(0)
        p = self.inducing_points
        p = p.repeat([b, 1, 1])  # shape [b, m, d]
        h = self.mab1(p, x, None, m, ln)  # shape [b, m, d]

        return self.mab2(x, h, m, None, ln)

class PoolingMultiheadAttention(nn.Module):
    def __init__(self, d, k, h, rff, similarity='dot', full_head=False, analysis=False):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff, similarity, full_head, analysis)
        self.seed_vectors = nn.Parameter(torch.randn(1, k, d))
        torch.nn.init.xavier_uniform_(self.seed_vectors)

    def forward(self, z, m=None, ln=True):
        b = z.size(0)
        s = self.seed_vectors
        s = s.repeat([b, 1, 1])  # random seed vector: shape [b, k, d]
        output = self.mab(s, z, None, m, ln)

        return output

class PoolingMultiheadCrossAttention(nn.Module):
    def __init__(self, d, h, rff, similarity='dot', full_head=False, analysis=False):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff, similarity, full_head, analysis)

    def forward(self, X, Y, Xm=None, Ym=None, ln=True):
        
        return self.mab(X, Y, Xm, Ym, ln)


'''
    ##############################
    #                            #
    #    Custom Loss Functions   #
    #                            #
    ##############################

'''


def numpify(tensor):
    if isinstance(tensor, torch.Tensor): 
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray): 
        return tensor
    else:
        raise


def show_model_params(model):
    for n, p in model.named_parameters():
        print(n, p.data)




##################### From Old Set Transformer

class MAB_mk2(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, nm=None):
        super(MAB_mk2, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        if nm:
            self.block_name = nm

    def forward(self, Q, K, Qm, Km):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        
        if Qm != None: Q = Q * Qm.repeat(1,1,Q.size(2))
        if Km != None: K = K * Km.repeat(1,1,K.size(2))

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = masked_softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), Qm, Km)

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB_mk2(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False, nm=None):
        super(SAB_mk2, self).__init__()
        self.mab = MAB_mk2(dim_in, dim_in, dim_out, num_heads, ln=ln, nm=nm)

    def forward(self, X, masks):
        return self.mab(X, X, masks, masks)

class ISAB_mk2(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, nm=None):
        super(ISAB_mk2, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB_mk2(dim_out, dim_in, dim_out, num_heads, ln=ln, nm=nm+'_IN')
        self.mab1 = MAB_mk2(dim_in, dim_out, dim_out, num_heads, ln=ln, nm=nm+'_OUT')

    def forward(self, X, masks=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, None, masks) # Compression
        return self.mab1(X, H, masks, None) # Expansion

class PMA_mk2(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, nm=None):
        super(PMA_mk2, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB_mk2(dim, dim, dim, num_heads, ln=ln, nm=nm)

    def forward(self, X, masks):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, None, masks)

def masked_softmax(X, m1, m2):
    if m2 != None:
        rpt = X.size(0) // m2.size(0)
        m2 = m2.transpose(1,2).repeat(rpt,1,1)
        X = X.exp() * m2

    if m1 != None:
        rpt = X.size(0) // m1.size(0)
        m1 = m1.repeat(rpt,1,1)
        X = X.exp() * m1

    X_denom = X.sum(2).unsqueeze(2)
    X = X / (X_denom + 1e-8)
    return X