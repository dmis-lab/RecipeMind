from data_utils import *
import numpy as np
import pickle
from env_config import *
import itertools
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import time

from models import *
from model_utils import *

import wandb
import os 
import argparse
from tqdm import tqdm, trange
# import plotter 
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.metrics import *
from scipy.stats import pearsonr

def load_recipe_trainer(args):
    if 'recipemind' in args.model_struct or args.model_struct.startswith('baseline'):
        return RecipeMindTrainer(args)
    elif args.model_struct == 'kitchenette':
        return RecipeMindTrainer(args)
    else:
        raise

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class SetNormLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()

    def forward(self, x, c):

        return self.mae(x, c)

class Trainer(object):
    def __init__(self, args=0):
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.num_epochs = args.num_epochs
        self.weight_decay = args.weight_decay

        self.loss_function = args.loss_function 
        self.best_valid_loss = 9999.9999
        self.checkpoint_path = f'{OUT_PATH}{args.project_name}_{args.session_name}_{args.random_seed}/'
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.grad_update = args.grad_update
        self.early_stopping = 5
        self.grad_norm = 10.0
        self.model_struct = args.model_struct

        self.mc_dropout = args.mc_dropout
        # self.pretrained_recipebowl = args.pretrained_recipebowl

        self.rm_args = args

    def wandb_log(self, report, epoch=-1, label='train'):
        wandb_dict = {f'{label}/step': epoch} if epoch >=0 else dict()
        for k,v in report.items():
            if isinstance(v, float):
                wandb_dict[f'{label}/{k}'] = v
        wandb.log(wandb_dict)

class RecipeMindTrainer(Trainer):
    def __init__(self, args=0):
        super(RecipeMindTrainer, self).__init__(args)
        self.lookup_table = dict({'set_words':    [], 'set_vectors': []})
        self.lookup_values = dict({'true_scores': [], 'pred_scores': [], 'pred_variances': [], 'true_norms': [], 'pred_norms': []})
        self.debug_mode = args.debug_mode

    def reset_lookup_values(self):
        self.lookup_values = dict({'true_scores': [], 'pred_scores': [], 'pred_variances': [], 'true_norms': [], 'pred_norms': []})

    def calculate_losses(self, batch):
        if self.loss_function == 'rmse':
            criterion = RMSELoss()
            loss = criterion(batch['yS'].view(-1), batch['xS'].view(-1))
        elif self.loss_function == 'rmse_setnorm':
            raise
            criterion1 = RMSELoss()
            criterion2 = SetNormLoss()
            loss1 = criterion1(batch['yS'].view(-1), batch['xS'].view(-1))
            loss2 = criterion2(torch.norm(batch['sQ'],dim=1), batch['xL'].sum(1))
            loss = loss1 + 0.1*loss2
        else:
            raise

        if loss != loss:
            batch['xS'][batch['xS']!=batch['xS']] = 0.0
            print("Corrupted Loss")
            exit()
            
        return loss

    def retrieve_regression_check(self):
        report = dict()
        yhat, y = self.lookup_values['pred_scores'], self.lookup_values['true_scores']
        yhat, y = np.array(yhat), np.array(y)
        try:
            report['mse'] = mean_squared_error(y, yhat)
            report['pcorr'] = pearsonr(y, yhat)[0]
        except:
            import pdb; pdb.set_trace()
            report['mse'] = 'nan'
            report['pcorr'] = 'nan'

        return report

    def retrieve_regression_eval(self, detailed_eval=False):
        report = dict()
        yhat, y = self.lookup_values['pred_scores'], self.lookup_values['true_scores']
        yhat, y = np.array(yhat), np.array(y)
        report['mse'] = mean_squared_error(y, yhat)
        report['rmse'] = mean_squared_error(y, yhat, squared=False)
        report['mae'] = mean_absolute_error(y, yhat)
        report['mape'] = mean_absolute_percentage_error(y, yhat)
        report['r2'] = r2_score(y, yhat)
        report['pcorr'] = pearsonr(y, yhat)[0]

        return report

    def plot_scatter_scores(self):
        # df2 = pd.read_csv(ROOT_PATH+'recipemind_subset_2_scores.csv')[['input singleton', 'input singleton count', 'npmi score']]
        # df3 = pd.read_csv(ROOT_PATH+'recipemind_subset_3_scores.csv')[['input singleton', 'input singleton count', 'npmi score']]
        # dff = pd.concat([df2,df3],axis=0)
        # dff = dff.groupby('input singleton').agg(['mean', 'std'])

        yhat, y = self.lookup_values['pred_scores'], self.lookup_values['true_scores']
        if len(self.lookup_values['pred_variances']) > 0:
            ystd = self.lookup_values['pred_variances']  
        else:
            ystd = [0.0 for _ in self.lookup_values['pred_scores']]

        data = [[a,b,c] for (a,b,c) in zip(yhat, y, ystd)]
        wandb_table, csv_table = [], []
        csv_table_columns = ['qry', 'tgt', 'true score', 'pred score', 'pred stddev', 'absolute error']
        # csv_table_columns = ['qry', 'tgt', 'tgt count','tgt scores global mean', 'tgt score global std', 'true score', 'pred score', 'pred stddev', 'absolute error']
        wandb_table_columns = ["input", "predicted scores", "actual scores"]
        # import pdb; pdb.set_trace()
        for w, s in zip(self.lookup_table['set_words'], data):
            query_set = w[0]
            query_elm = w[1]
            if isinstance(query_set, str):
                query_set = [query_set]
            query_set = sorted(query_set)
            query_set = ' [&] '.join(query_set)
            query_elm = query_elm
            wandb_table.append((query_set+' >> '+query_elm, s[0], s[1]))
            csv_table.append([query_set, 
                              query_elm, 
                              # dff.loc[query_elm, ('input singleton count', 'mean')],
                              # dff.loc[query_elm, ('npmi score', 'mean')],
                              # dff.loc[query_elm, ('npmi score', 'std')],
                              s[1],
                              s[0],
                              s[2],
                              abs(s[0]-s[1])])
        if len(wandb_table) > 100000:
            from random import sample
            table_sampled = sample(wandb_table, 100000)
        else:
            table_sampled = wandb_table
        df = pd.DataFrame(csv_table, columns=csv_table_columns)
        wandb_table = wandb.Table(data=table_sampled, columns=wandb_table_columns)
        wandb.log({'Final Evaluation': wandb.plot.scatter(wandb_table, "predicted scores", "actual scores", title="Scatter Plot")})
        df.to_csv(self.checkpoint_path+f'final_evaluation_results.csv')

    def status_update(self, pbar, batch, n_iter):
        self.lookup_values['pred_scores'] += numpify(batch['xS']).reshape(-1).tolist()
        self.lookup_values['true_scores'] += numpify(batch['yS']).reshape(-1).tolist()
        report = self.retrieve_regression_check()
        pbar.set_description(f"Iter: {n_iter}, MSE: {report['mse']:.4f}, PCORR: {report['pcorr']:.4f}")

    def save_tsne_preparations(self, idx, batch):
        np.save(self.checkpoint_path+f'set_vecs_{idx}', batch['sQ'].detach().cpu().numpy())
        np.save(self.checkpoint_path+f'set_lens_{idx}', batch['mQ'].sum(1).detach().cpu().numpy())
        set_names = [ ' [&] '.join(sorted(b[0])) for b in batch['uS']]
        pickle.dump(set_names, open(self.checkpoint_path+f'set_names_{idx}', 'wb'))

    def train_model(self, model, trainloader, validloader, finetuning=False):
        # validloader = iter(validloader)
        if model.__class__.__name__ == 'Kitchenette': self.early_stopping = self.num_epochs

        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
        model.apply(weights_init)
        opti = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)

        for epoch in range(self.num_epochs):
            start_time = time.time()
            for idx, batch in enumerate(trainloader):
                model.train(True)
                opti.zero_grad(set_to_none=True)
                batch = model(batch)
                loss = self.calculate_losses(batch)
                loss.backward()
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), self.grad_norm)
                opti.step()
                if self.debug_mode: break
                
            model.eval()
            with torch.no_grad():
                print("Calculating Scores for Validation Dataset")
                for idx, batch in enumerate(validloader):
                    batch = model(batch)
                    self.lookup_values['pred_scores'] += numpify(batch['xS']).reshape(-1).tolist()
                    self.lookup_values['true_scores'] += numpify(batch['yS']).reshape(-1).tolist()
                    # self.save_tsne_preparations(idx, batch)
                    del batch; torch.cuda.empty_cache()
                report = self.retrieve_regression_eval()
                self.wandb_log(report, epoch, 'valid')
                self.reset_lookup_values()
                
                if self.best_valid_loss > report['rmse']:
                    self.best_valid_loss = report['rmse']
                    torch.save({'epoch': epoch, 
                                'model_state_dict': model.state_dict(),
                                'loss': report['mse']}, 
                                self.checkpoint_path+f'epoch_best.mdl')
                else:
                    if epoch > 10: self.early_stopping -= 1

            os.system('clear')
            print(f"Epoch: {epoch}, Current Validation: {report['rmse']:.4f}, Best Validation: {self.best_valid_loss:.4f}, Early Stopping: {self.early_stopping}, Minutes: {round(time.time()-start_time,2)//60}")
            if self.early_stopping == 0: break
            if self.debug_mode: break

        torch.save({'model_state_dict': model.state_dict()}, 
                    self.checkpoint_path+f'epoch_final.mdl')

        return model

    @torch.no_grad()
    def test_model(self, model, dataloader, tsne=False):
        if self.mc_dropout: model.train() 
        else: model.eval()
        pbar = tqdm(dataloader)

        with torch.no_grad():            
            for batch in pbar:
                self.lookup_values['true_scores'] += numpify(batch['yS']).reshape(-1).tolist()
                if not self.mc_dropout:
                    batch = model(batch)
                    self.lookup_values['pred_scores'] += numpify(batch['xS']).reshape(-1).tolist()
                else:
                    batch = model.v_infer(batch)
                    self.lookup_values['pred_scores'] += numpify(batch['xS']).reshape(-1).tolist()
                    self.lookup_table['pred_variances'] += numpify(batch['vS']).reshape(-1).tolist()
                self.lookup_table['set_words'] += batch['uS']
                del batch; torch.cuda.empty_cache()

        wandb_dict = self.retrieve_regression_eval(detailed_eval=True)
        self.plot_scatter_scores()
        wandb.log({k: v for k,v in wandb_dict.items() if isinstance(v, float)})