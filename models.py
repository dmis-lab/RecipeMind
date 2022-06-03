from env_config import *
import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np
import collections
from model_utils import *
import torch.optim as optim
from sklearn.metrics import pairwise as pw
import os.path

torch.set_printoptions(threshold=10_000)

def load_recipe_model(args):
    if args.model_struct.startswith('recipemind'):
        return RecipeMind(args)
    elif args.model_struct.startswith('baseline_kim'):
        args.sab_num_aheads = 4
        args.sab_num_blocks = 1
        args.pma_num_aheads = 2
        args.pma_num_blocks = 1
        args.pma_num_sdvecs = 1
        args.model_struct   = 'recipemind_isab_pma_cat_' + args.model_struct.split('_')[-1]
        return RecipeMind(args)
    elif args.model_struct.startswith('baseline_li'):
        args.sab_num_aheads = 4
        args.sab_num_blocks = 2
        args.pma_num_aheads = 4
        args.pma_num_blocks = 1
        args.pma_num_sdvecs = 2
        args.model_struct   = 'recipemind_isab_pma_cat_' + args.model_struct.split('_')[-1]
        return RecipeMind(args)


    elif args.model_struct == 'kitchenette':
        return Kitchenette(args)
    else:
        print(args.model_struct)
        raise

def load_element_encoder(args):
    arg = args.model_struct.split('_')[1]
    if arg == 'non':
        return ElementEncoder(args)
    elif arg == 'mlp':
        return DeepSets(args)
    elif arg == 'sab':
        return SetAttention(args)
    elif arg == 'asab':
        return AsymSetAttention(args)
    elif arg == 'isab':
        return InducedSetAttention(args)
    elif arg == 'ars':
        return ApproxRepTheSet(args)
    elif arg == 'cpmx':
        return CascadedCrossAttention(args)
    elif arg == 'raw':
        return EmptyEncoder(args)
    else:
        print(args.model_struct)
        raise

def load_set_pooling(args):
    arg = args.model_struct.split('_')[2]
    if arg == 'pmx':
        return PmxPooling(args)
    elif arg == 'pma':
        return PmaPooling(args)
    elif arg == 'avg':
        return AvgPooling(args)
    elif arg == 'sum':
        return SumPooling(args)
    elif arg == 'max':
        return MaxPooling(args)
    else:
        print(args.model_struct)
        raise

def load_score_predictor(args):
    arg = args.model_struct.split('_')[-1]
    if arg == 'cat':
        return ConcatPredictor(args)
    elif arg == 'sff':
        return SimplePredictor(args)
    elif arg == 'cos':
        return CosinePredictor(args)
    elif arg == 'euc':
        return EuclideanPredictor(args)
    else:
        print(args.model_struct)
        raise

def save_element_vectors(self, input, output):
    #
    #
    if input[0].dim() == 2:
        self.element_vectors.append(output.data.detach().cpu().numpy())

def save_context_vectors(self, input, output):
    #
    #
    self.context_vectors.append(output.data.detach().cpu().numpy())

def check_gradients(module, grad_input, grad_output):
    print(grad_output)
    for grad in grad_input:    
        print('Inside ' + module.__class__.__name__ + ' backward')
        print('Inside class:' + module.__class__.__name__)
        print('')
        print('grad_output: ', type(grad_output))
        print('grad_output[0]: ', type(grad))
        print('')
        print('grad_input size:', grad.size())
        print('grad_input norm:', grad.norm())


class ElementEncoder(nn.Module):
    def __init__(self, args=0):
        super(ElementEncoder, self).__init__()
        self.in_dim = args.lang_dim['J']
        self.out_dim = args.hidden_dim
        self.mlp = nn.Sequential(nn.Linear(self.in_dim, args.hidden_dim), # 300 x 300
                                 nn.Dropout(args.dropout_rate),
                                 nn.ReLU(),
                                 nn.Linear(args.hidden_dim, args.hidden_dim),
                                 nn.Dropout(args.dropout_rate),
                                 nn.ReLU())

    def activate_shallow(self, X):

        return self.mlp(X)

    def activate(self, X, m):
        X = self.mlp(X)

        return X, m

    def forward(self, batch):
        batch['eQ'] = self.mlp(batch['xQ'])
        batch['eA'] = self.mlp(batch['xA'])

        return batch


class EmptyEncoder(ElementEncoder):
    def __init__(self, args=0):
        super(EmptyEncoder, self).__init__(args)
        self.mlp = None
        self.out_dim = self.in_dim

    def activate(self, X, m):

        return X, m

    def forward(self, batch):
        batch['eQ'] = batch['xQ']
        batch['eA'] = batch['xA']

        return batch


class SetAttention(ElementEncoder):
    def __init__(self, args=0):
        super(SetAttention, self).__init__(args)
        self.num_attn_heads = args.sab_num_aheads
        sab_args = (args.hidden_dim, self.num_attn_heads, RFF(args), 
                    args.multihead_sim, args.multihead_big, args.model_analysis)
        self.sab = nn.ModuleList([SetAttentionBlock(*sab_args) for _ in range(args.sab_num_blocks)])

    def activate(self, X, m):
        X = self.mlp(X)
        for sab in self.sab:
            if isinstance(m, torch.Tensor): X = sab(X, m) * m.unsqueeze(2)
            else: X = sab(X, m)

        return X, m

    def forward(self, batch):
        batch['eQ'] = self.mlp(batch['xQ'])
        batch['eA'] = self.mlp(batch['xA'])
        eQ, mQ = batch['eQ'], batch['mQ']
        eA = batch['eA'].unsqueeze(1)
        mA = torch.ones(batch['bs'],1).cuda()

        for sab in self.sab:
            if isinstance(mQ, torch.Tensor): eQ = sab(eQ, mQ) * mQ.unsqueeze(2)
            else: eQ = sab(eQ, mQ)
            eA = sab(eA, mA)
        batch['eQ'], batch['eA'] = eQ, eA.squeeze(1)

        return batch

class DeepSets(ElementEncoder):
    def __init__(self, args=0):
        super(DeepSets, self).__init__(args)
        mlp = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), 
                            nn.Dropout(args.dropout_rate),
                            nn.ReLU())
        self.sab = nn.ModuleList([mlp for _ in range(args.sab_num_blocks)])

    def activate(self, X, m):
        X = self.mlp(X)
        for sab in self.sab:
            if isinstance(m, torch.Tensor): X = sab(X, m) * m.unsqueeze(2)
            else: X = sab(X, m)

        return X, m

    def forward(self, batch):
        batch['eQ'] = self.mlp(batch['xQ'])
        batch['eA'] = self.mlp(batch['xA'])
        eQ, mQ = batch['eQ'], batch['mQ']
        eA = batch['eA'].unsqueeze(1)
        mA = torch.ones(batch['bs'],1).cuda()

        for sab in self.sab:
            if isinstance(mQ, torch.Tensor): eQ = sab(eQ) * mQ.unsqueeze(2)
            else: eQ = sab(eQ)
            eA = sab(eA)
        batch['eQ'], batch['eA'] = eQ, eA.squeeze(1)

        return batch


class AsymSetAttention(SetAttention):
    def __init__(self, args=0):
        super(AsymSetAttention, self).__init__(args)
        sab_args = (args.hidden_dim, self.num_attn_heads, RFF(args), 
                    args.multihead_sim, args.multihead_big, args.model_analysis)
        self.aab = nn.ModuleList([SetAttentionBlock(*sab_args) for _ in range(args.sab_num_blocks)])

    def activate(self, X, m):
        X = self.mlp(X)
        for sab in self.sab:
            if isinstance(m, torch.Tensor): X = sab(X, m) * m.unsqueeze(2)
            else: X = sab(X, m)

        return X, m

    def forward(self, batch):
        batch['eQ'] = self.mlp(batch['xQ'])
        batch['eA'] = self.mlp(batch['xA'])
        eQ, mQ = batch['eQ'], batch['mQ']
        eA = batch['eA'].unsqueeze(1)
        mA = torch.ones(batch['bs'],1).cuda()

        for sab, aab in zip(self.sab, self.aab):
            if isinstance(mQ, torch.Tensor): eQ = sab(eQ, mQ) * mQ.unsqueeze(2)
            else: eQ = sab(eQ, mQ)
            eA = aab(eA, mA)
        batch['eQ'], batch['eA'] = eQ, eA.squeeze(1)

        return batch


class InducedSetAttention(SetAttention):
    def __init__(self, args=0):
        super(InducedSetAttention, self).__init__(args)
        self.num_attn_heads = args.sab_num_aheads
        isab_args = (args.hidden_dim, 16, self.num_attn_heads, RFF(args), RFF(args),
                     args.multihead_sim, args.multihead_big, args.model_analysis)
        self.sab = nn.ModuleList([InducedSetAttentionBlock(*isab_args) for _ in range(args.sab_num_blocks)])


class CascadedCrossAttention(SetAttention):
    def __init__(self, args=0):
        super(CascadedCrossAttention, self).__init__(args)
        pmx_args = (args.hidden_dim, self.num_attn_heads, RFF(args), 
                    args.multihead_sim, args.multihead_big, args.model_analysis)
        self.pmx = nn.ModuleList([PoolingMultiheadCrossAttention(*pmx_args) for _ in range(args.sab_num_blocks)])

    def activate(self, X, m):
        
        raise

    def forward(self, batch):
        batch['eQ'] = self.mlp(batch['xQ'])
        batch['eA'] = self.mlp(batch['xA'])
        eQ, mQ = batch['eQ'], batch['mQ']
        eA = batch['eA'].unsqueeze(1)

        for sab, pmx in zip(self.sab, self.pmx):
            eA = pmx(Y=eQ,Ym=mQ,X=eA)
            if isinstance(mQ, torch.Tensor): eQ = sab(eQ, mQ) * mQ.unsqueeze(2)
            else: eQ = sab(eQ, mQ)
        batch['eQ'], batch['eA'] = eQ, eA.squeeze(1)

        return batch    


class ApproxRepTheSet(ElementEncoder):
    def __init__(self, args=0):
        super(ApproxRepTheSet, self).__init__(args)
        self.num_hidden_sets = args.ars_num_hsets
        self.num_hidden_elms = args.ars_num_helms
        self.ars = nn.Sequential(nn.Linear(args.hidden_dim, self.num_hidden_sets*self.num_hidden_elms, bias=False), nn.ReLU())

    def activate(self, X, m):
        print("this is not a code")
        raise

    def forward(self, batch):
        batch['eQ'] = self.mlp(batch['xQ'])
        batch['eA'] = self.mlp(batch['xA'])
        eQ, mQ = batch['eQ'], batch['mQ']
        if isinstance(mQ, torch.Tensor): eQ = self.ars(eQ) * mQ.unsqueeze(2)
        else: eQ = self.ars(eQ)
        eQ = eQ.view(eQ.size(0), eQ.size(1), self.num_hidden_elms, self.num_hidden_sets)
        eQ, _ = torch.max(eQ, dim=2)
        eA = batch['eA'].unsqueeze(1)
        eA = self.ars(eA)
        eA = eA.view(eA.size(0), eA.size(1), self.num_hidden_elms, self.num_hidden_sets)
        eA, _ = torch.max(eA, dim=2)

        batch['eQ'], batch['eA'] = eQ, eA
        return batch


class PmaPooling(nn.Module):
    def __init__(self, args=0):
        super(PmaPooling, self).__init__()
        self.num_attn_heads = args.pma_num_aheads
        self.num_seed_vecs  = args.pma_num_sdvecs
        # self.context_dim    = self.num_seed_vecs * args.hidden_dim

        # Pooling if there are lots of seed vectors
        assert len(args.model_struct.split('_')) > 3
        aggr = args.model_struct.split('_')[3]
        if   aggr == 'sum': self.pool2, self.smv = SumPooling(args), False
        elif aggr == 'avg': self.pool2, self.smv = AvgPooling(args), False
        elif aggr == 'max': self.pool2, self.smv = MaxPooling(args), False
        elif aggr == 'cat': self.pool2, self.smv = CatPooling(args), False
        elif aggr == 'smv': self.pool2, self.smv = None, True
        else:               
            self.pool2, self.smv, self.num_seed_vecs = None, False, 1

        pma_args = (args.hidden_dim, self.num_seed_vecs, self.num_attn_heads, RFF(args), 
                    args.multihead_sim, args.multihead_big, args.model_analysis)
        self.pool1 = PoolingMultiheadAttention(*pma_args)

        if self.num_seed_vecs > 1:
            sab_args = (args.hidden_dim, self.num_attn_heads, RFF(args), 
                        args.multihead_sim, args.multihead_big, args.model_analysis)
            self.sab = nn.ModuleList([SetAttentionBlock(*sab_args) for _ in range(args.sab_num_blocks)])

        else:
            self.sab   = None

    def activate(self, X, m):
        print("UNFINISHED CODE")
        raise

        X = self.pool1(X, m)
        for sab in self.sab:
            X = sab(X)

        return self.lin(X.view(-1, self.context_dim))

    def forward(self, batch):
        if batch['eA'].dim() == 2: batch['eA'] = batch['eA'].unsqueeze(1)
        batch['sQ'] = self.pool1(batch['eQ'], batch['mQ'])
        batch['tA'] = batch['eA']
        # If there are at least two seed vectors, propagate them thru sabs and perform pooling
        if self.num_seed_vecs > 1:
            for sab in self.sab: 
                batch['sQ'] = sab(batch['sQ'])
            batch['sQ'] = self.pool2.activate(batch['sQ'])
        else:
            batch['sQ'] = batch['sQ'].squeeze(1)
            batch['sQ'] = batch['sQ'] * batch['xL'].sum(1) if self.smv else batch['sQ']

        return batch

class PmxPooling(nn.Module):
    def __init__(self, args=0):
        super(PmxPooling, self).__init__()
        self.num_attn_heads = args.pma_num_aheads
        self.num_seed_vecs  = args.pma_num_sdvecs
        self.context_dim    = args.hidden_dim

        assert len(args.model_struct.split('_')) > 3
        aggr = args.model_struct.split('_')[3]
        self.smv = True if aggr == 'smv' else False

        pma_args = (args.hidden_dim, self.num_attn_heads, RFF(args), 
                    args.multihead_sim, args.multihead_big, args.model_analysis)
        self.pool = PoolingMultiheadCrossAttention(*pma_args)

    def activate(self, X, m):

        raise

    def forward(self, batch):
        if batch['eA'].dim() == 2: batch['eA'] = batch['eA'].unsqueeze(1)
        batch['sQ'] = self.pool(Y=batch['eQ'], Ym=batch['mQ'], X=batch['eA']).squeeze(1)
        batch['sQ'] = batch['sQ'] * batch['xL'].sum(1).unsqueeze(1) if self.smv else batch['sQ']
        batch['tA'] = batch['eA'].squeeze(1)

        return batch

class SumPooling(nn.Module):
    def __init__(self, args=0):
        super(SumPooling, self).__init__()

    def activate(self, X, m=None):
        if isinstance(m, torch.Tensor):
            return (X * m.unsqueeze(2)).sum(1)
        else:
            return X.sum(1)

    def forward(self, batch):
        batch['sQ'] = (batch['eQ'] * batch['mQ'].unsqueeze(2)).sum(1)
        batch['tA'] = batch['eA']

        return batch

class AvgPooling(nn.Module):
    def __init__(self, args=0):
        super(AvgPooling, self).__init__()

    def activate(self, X, m):
        if isinstance(m, torch.Tensor):
            return (X * m.unsqueeze(2)).sum(1) / m.sum(1).view(-1,1)
        else:
            return X.mean(1)

    def forward(self, batch):
        batch['sQ'] = (batch['eQ'] * batch['mQ'].unsqueeze(2)).sum(1) / batch['mQ'].sum(1).view(-1,1)
        batch['tA'] = batch['eA']

        return batch

class MaxPooling(nn.Module):
    def __init__(self, args=0):
        super(MaxPooling, self).__init__()

    def activate(self, X, m=None):
        if isinstance(m, torch.Tensor):
            return X.max(1)[0]
        else:
            return (X * m.unsqueeze(2)).max(1)[0]

    def forward(self, batch):
        batch['sQ'] = (batch['eQ'] * batch['mQ'].unsqueeze(2)).max(1)[0]
        batch['tA'] = batch['eA']

        return batch

class CatPooling(nn.Module):
    def __init__(self, args=0):
        super(CatPooling, self).__init__()
        h = args.hidden_dim 
        s = args.pma_num_sdvecs
        self.lin = nn.Linear(h*s, h)

    def activate(self, X, m=None):
        if isinstance(m, torch.Tensor):
            return self.lin(X.view(X.size(0),X.size(1)*X.size(2))) * m.unsqueeze(2)
        else:
            return self.lin(X.view(X.size(0),X.size(1)*X.size(2))) 

    def forward(self, batch):

        raise


class SimplePredictor(nn.Module):
    def __init__(self, args=0):
        super(SimplePredictor, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim//2),
                                 nn.Dropout(args.dropout_rate),
                                 nn.ReLU(),
                                 nn.Linear(args.hidden_dim//2, 1))

    def forward(self, batch):
        batch['xS'] = self.mlp(batch['sQ'])

        return batch


class ConcatPredictor(nn.Module):
    def __init__(self, args=0):
        super(ConcatPredictor, self).__init__()
        self.catmlp = nn.Sequential(nn.Linear(args.hidden_dim*2, args.hidden_dim),
                                    nn.Dropout(args.dropout_rate),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_dim, 1))
        
    def forward(self, batch):
        if batch['tA'].dim() == 3: batch['tA'] = batch['tA'].squeeze(1)
        batch['xS'] = self.catmlp(torch.cat([batch['tA'],batch['sQ']],axis=1))

        return batch

class CosinePredictor(nn.Module):
    def __init__(self, args=0):
        super(CosinePredictor, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim),
                                 nn.Dropout(args.dropout_rate),
                                 nn.ReLU(),
                                 nn.Linear(args.hidden_dim, args.hidden_dim))
        self.cos = nn.CosineSimilarity()

    def forward(self, batch):
        if batch['tA'].dim() == 3: batch['tA'] = batch['tA'].squeeze(1)
        batch['xS']= self.cos(self.mlp(batch['tA']),self.mlp(batch['sQ']))

        return batch

class EuclideanPredictor(nn.Module):
    def __init__(self, args=0):
        super(EuclideanPredictor, self).__init__()
        self.euc = nn.PairwiseDistance(p=2)

    def forward(self, batch):
        if batch['tA'].dim() == 3: batch['tA'] = batch['tA'].squeeze(1)
        batch['xS']= self.euc(batch['tA'],batch['sQ'])

        return batch


class WideDeepPredictor(nn.Module):
    def __init__(self, args=0):
        super(WideDeepPredictor, self).__init__()
        self.catmlp = nn.Sequential(nn.Linear(args.hidden_dim*2, args.hidden_dim),
                                    nn.Dropout(args.dropout_rate),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_dim, args.hidden_dim),
                                    nn.Dropout(args.dropout_rate),
                                    nn.ReLU())
        self.wndmlp = nn.Sequential(nn.Linear(args.hidden_dim**2+args.hidden_dim, args.hidden_dim),
                                    nn.Dropout(args.dropout_rate),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_dim, 1))

    def forward(self, batch):
        if batch['tA'].dim() == 3: batch['tA'] = batch['tA'].squeeze(1)
        wQ = torch.bmm(batch['sQ'].unsqueeze(2), batch['tA'].unsqueeze(1))
        wQ = wQ.reshape(batch['bs'], -1)
        dQ = torch.cat([batch['sQ'], batch['tA']], 1)
        dQ = self.catmlp(dQ)
        batch['xS'] = self.wndmlp(torch.cat([wQ,dQ],dim=1))

        return batch

class Kitchenette(nn.Module):
    def __init__(self, args=0):
        super(Kitchenette, self).__init__()
        self.encoder = ElementEncoder(args)
        self.deep_encoder = nn.Sequential(nn.Linear(args.hidden_dim*2, args.hidden_dim),
                                          nn.Dropout(args.dropout_rate),
                                          nn.ReLU(),
                                          nn.Linear(args.hidden_dim, args.hidden_dim),
                                          nn.Dropout(args.dropout_rate),
                                          nn.ReLU())

        self.predictor = nn.Linear(args.hidden_dim**2+args.hidden_dim, 1)

    def forward(self, batch):
        batch_size = batch['xQ'].size(0)
        batch = self.encoder(batch)

        batch['wQ'] = torch.bmm(batch['eQ'].unsqueeze(2), batch['eA'].unsqueeze(1))
        batch['wQ'] = batch['wQ'].reshape(batch_size, -1)

        batch['dQ'] = torch.cat([batch['eQ'], batch['eA']], 1)
        batch['dQ'] = self.deep_encoder(batch['dQ'])

        batch['eQ'] = torch.cat([batch['wQ'], batch['dQ']], 1)
        batch['xS'] = self.predictor(batch['eQ'])

        return batch


class RecipeMind(nn.Module):
    def __init__(self, args=0):
        super(RecipeMind, self).__init__()
        self.ele_encoder     = load_element_encoder(args)
        if 'raw' in args.model_struct: args.hidden_dim = args.lang_dim['J']
        if 'ars' in args.model_struct: args.hidden_dim = args.ars_num_hsets
        self.set_pooling     = load_set_pooling(args)
        self.score_predictor = load_score_predictor(args)

    def set_hybrid_coef(self, epsilon):
        self.score_predictor.eps = epsilon

    def make_representations(self, batch):
        batch['xL'] = batch['mQ']
        batch = self.ele_encoder(batch)
        batch = self.set_pooling(batch)

        return batch

    def make_predictions(self, batch):
        batch = self.score_predictor(batch)
  
        return batch

    @torch.no_grad()
    def v_infer(self, batch, n_passes=10):
        repeated = []
        for _ in range(n_passes):
            batch = self.make_predictions(self.make_representations(batch))
            repeated.append(batch['xS'].unsqueeze(1))
        repeated = torch.cat(repeated, dim=1)
        batch['xS'] = repeated.mean(1)
        batch['vS'] = repeated.std(1)

        return batch

    def forward(self, batch):
        batch = self.make_representations(batch)
        batch = self.make_predictions(batch)
        
        return batch
