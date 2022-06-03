
from env_config import *
from trainer import *
from models import *
import wandb
import torch
import numpy as np
import os
import random
import json
import copy
from itertools import chain, combinations
import setproctitle
import itertools
from sklearn.metrics import *
from scipy.stats import pearsonr
import math

def rank(topN, df, df_baselines):
    df_target = df.sort_values(by=['target'], ascending=False)
    list_target = df_target['target'][:topN].index.values.tolist()

    df_knet = df.sort_values(by=['prediction'], ascending=False)
    list_knet = df_knet['prediction'][:topN].index.values.tolist()

def dcg(list_pred, list_target, topN):
    output = []

    for d in list_target[:topN]:
        if d in list_pred[:topN]:
            denominator = list_target[:topN].index(d)+2
            denominator = math.log(denominator,2)
            #output.append(1/math.log(list_target[:topN].index(d),2))
            output.append(1/denominator)
        else:
            output.append(0)

    return sum(output)

def jaccard_sim(i, j, y_pred_ranked):
    i_row = set(y_pred_ranked[i,:].tolist())
    j_row = set(y_pred_ranked[j,:].tolist())
    return len(i_row & j_row) / len(i_row | j_row)

def pearson_cor(i, j, y_pred_scores):
    i_row = y_pred_scores[i,:]
    j_row = y_pred_scores[j,:]
    return pearsonr(i_row,j_row)[0]


class RecipeIdeationPipeline(object):
    def __init__(self, model, collate, session_name):
        self.model = model
        self.collate = collate
        self.session_name = session_name

    def __call__(self, seed_ingred):
        import pdb; pdb.set_trace()


    def get_ranked_list(self, query_ingredients, query_tags, mc_dropout=False):
        x = 0


class FoodPairingPipeline(RecipeIdeationPipeline):
    def __init__(self, model, collate, session_name=None, dataset_name=None):
        super(FoodPairingPipeline, self).__init__(model, collate, session_name)
        self.collate.fn_inference([])
        self.idx2ingred = self.collate.idx2ingred
        self.idx2ingred.remove('[PAD]')
        self.dataset_name = dataset_name

    def get_evaluation_results(self, df, label='doublets'):
        results = dict()
        results['mse']       = df['squared error'].mean()
        results['mae']       = df['absolute error'].mean()
        results['rmse']      = math.sqrt(df['squared error'].mean())
        results['pcorr']     = pearsonr(df['pred scores'], df['true scores'])[0]    
        results['r2']        = r2_score(df['pred scores'], df['true scores'])

        def get_ndcg_score(df, topN):
            def dcg(list_pred, list_target, topN):
                output = []
                for d in list_target[:topN]:
                    if d in list_pred[:topN]:
                        denominator = list_target[:topN].index(d)+2
                        denominator = math.log(denominator,2)
                        output.append(1/denominator)
                    else:
                        output.append(0)
                return sum(output)

            df_true = df.sort_values(by=['true scores'], ascending=False)
            y_true = df_true['true scores'][:topN].index.values.tolist()
            df_pred = df.sort_values(by=['pred scores'], ascending=False)
            y_pred = df_pred['pred scores'][:topN].index.values.tolist()

            return dcg(y_pred, y_true, topN) / dcg(y_true, y_true, topN)

        return results

    @torch.no_grad()
    def __call__(self, data, mc_dropout=False):
        results = dict()
        print("Variational Dropout ", mc_dropout)
        def extract_metadata(x_tuple_list, meta_col=None):
            x_list = []
            if meta_col == 'query_element':
                for x_tuple in x_tuple_list:
                    x_list.append(x_tuple[0])
            elif meta_col == 'target_element':
                for x_tuple in x_tuple_list:
                    x_list.append(x_tuple[1])
            elif meta_col == 'affinity_score':
                for x_tuple in x_tuple_list:
                    x_list.append(x_tuple[2])
            elif meta_col == 'target_index':
                for x_tuple in x_tuple_list:
                    x_list.append(self.idx2ingred.index(x_tuple[1].strip('<J>/')))
            return x_list

        if mc_dropout: self.model.train()
        print("Extracting Meta-Info from Test Dataset")
        x_item_names   = list(sum([extract_metadata(batch['uS'], 'query_element') for batch in data],[]))
        y_item_names   = list(sum([extract_metadata(batch['uS'], 'target_element') for batch in data],[]))
        y_scores       = list(sum([extract_metadata(batch['uS'], 'affinity_score') for batch in data],[]))
        y_item_indices = list(sum([extract_metadata(batch['uS'], 'target_index') for batch in data],[]))
        
        meta_info = (x_item_names, y_item_names, y_scores, y_item_indices)
        pred_scores, pred_indices = [], []

        for batch in tqdm(data):
            batch = self.model(batch)
            pred_scores.append(numpify(batch['xS']).reshape(-1,1))
        pred_scores = np.vstack(pred_scores)

        df = pd.DataFrame(pred_scores, columns=['pred scores'])
        df['true scores']= pd.Series(meta_info[2])
        df['query']      = pd.Series(meta_info[0])
        df['target']     = pd.Series(meta_info[1])
        df['absolute error'] = (df['pred scores'] - df['true scores']).abs()
        df['squared error']  = (df['pred scores'] - df['true scores']).pow(2)

        df['true ranks']     = df['true scores'].rank(method='first', ascending=False).astype(int)
        df['pred ranks']     = df['pred scores'].rank(method='first', ascending=False).astype(int)
        df.to_csv(f'./results/foodpairing_{self.session_name}_{self.model.__class__.__name__}.csv')
        
        return self.get_evaluation_results(df, 'doublets')

class NtupletScoringPipeline(FoodPairingPipeline):
    def __init__(self, model, collate, session_name=None, dataset_name=None):
        super(NtupletScoringPipeline, self).__init__(model, collate, session_name, dataset_name)

    @torch.no_grad()
    def __call__(self, data, ntuplets, mc_dropout=False):
        print("Variational Dropout ", mc_dropout)
        def extract_metadata(x_tuple_list, meta_col=None):
            x_list = []
            if meta_col == 'query_elements':
                for x_tuple in x_tuple_list:
                    x_list.append(' [&] '.join(x_tuple[0]))
            elif meta_col == 'target_element':
                for x_tuple in x_tuple_list:
                    x_list.append(x_tuple[1])
            elif meta_col == 'affinity_score':
                for x_tuple in x_tuple_list:
                    x_list.append(x_tuple[2])
            elif meta_col == 'target_index':
                for x_tuple in x_tuple_list:
                    x_list.append(self.idx2ingred.index(x_tuple[1].strip('<J>/')))
            return x_list

        if mc_dropout: self.model.train()
        print("Extracting Meta-Info from Test Dataset") 
        if not os.path.isfile(f'./cache/{self.dataset_name}_metainfo.pkl'):
            x_item_names   = list(sum([extract_metadata(batch['uS'], 'query_elements') for batch in data],[]))
            y_item_names   = list(sum([extract_metadata(batch['uS'], 'target_element') for batch in data],[]))
            y_scores       = list(sum([extract_metadata(batch['uS'], 'affinity_score') for batch in data],[]))
            y_item_indices = list(sum([extract_metadata(batch['uS'], 'target_index') for batch in data],[]))
            meta_info = (x_item_names, y_item_names, y_scores, y_item_indices)
            pickle.dump(meta_info, open(f'./cache/{self.dataset_name}_metainfo.pkl', 'wb'))

        meta_info = pickle.load(open(f'./cache/{self.dataset_name}_metainfo.pkl', 'rb'))
        pred_scores, pred_indices = [], []

        print("Running Model Inference ", self.model.__class__.__name__)
        for batch in tqdm(data):
            batch = self.model(batch)
            pred_scores.append(numpify(batch['xS']).reshape(-1,1))
        pred_scores = np.vstack(pred_scores)

        print("Evaluating Inference Results ", pred_scores.shape)
        df = pd.DataFrame(pred_scores, columns=['pred scores'])
        df['true scores']= pd.Series(meta_info[2])
        df['query']      = pd.Series(meta_info[0])
        df['target']     = pd.Series(meta_info[1])

        df['absolute error'] = (df['pred scores'] - df['true scores']).abs()
        df['squared error'] = (df['pred scores'] - df['true scores']).pow(2)
        df['true ranks']     = df['true scores'].rank(method='first', ascending=False).astype(int)
        df['pred ranks']     = df['pred scores'].rank(method='first', ascending=False).astype(int)
        df.to_csv(f'{CSV_PATH}{ntuplets}_{self.session_name}_{self.model.__class__.__name__}.csv')
        
        return self.get_evaluation_results(df, ntuplets)    