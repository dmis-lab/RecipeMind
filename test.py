from env_config import *
from trainer import *
from models import *
import wandb
import torch
import numpy as np
import os
import random
import json
import random
import copy
from itertools import chain, combinations
import setproctitle
from pipeline import *
# faiss.omp_set_num_threads(20)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_vector_dimensions(args):
    lang_dim = dict()

    dim_dict = {
        'reciptor': 768,
        'bert-base-uncased': 768,
        'flavorgraph': 300,
        'im2recipe': 300
    }

    # temp
    if args.model_struct == 'recipebowl':
        lang_dim['J'] = 300
        lang_dim['T'] = 630
        lang_dim['R'] = 600
        args.initial_vectors_J = 'flavorgraph'
        args.initial_vectors_T = 'binary'
        args.initial_vectors_R = 'reciptor'
    else:
        lang_dim['J'] = dim_dict[args.initial_vectors_J]
        lang_dim['T'] = dim_dict[args.initial_vectors_T]
        lang_dim['R'] = dim_dict[args.initial_vectors_R]
        lang_dim['S'] = dim_dict[args.initial_vectors_J]
        lang_dim['H'] = dim_dict[args.initial_vectors_J]

    args.lang_dim = lang_dim
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', '-pn', default='Test', type=str)
    parser.add_argument('--group_name', '-gn', default='Test', type=str)
    parser.add_argument('--session_name', '-sn', default='Test', type=str)
    parser.add_argument('--ideation_score', '-is', default='sPMId02', type=str)
    parser.add_argument('--random_seed', default=911012, type=int)

    parser.add_argument('--dataset_index', default=5, type=int)
    parser.add_argument('--dataset_version', default='211210', type=str)
    parser.add_argument('--dataset_name', default='recipemind', type=str)
    parser.add_argument('--initial_vectors_J', default='flavorgraph', type=str)
    parser.add_argument('--initial_vectors_T', default='nothing', type=str)
    parser.add_argument('--initial_vectors_R', default='nothing', type=str)
    parser.add_argument('--model_struct', default='recipemind', type=str)
    parser.add_argument('--model_analysis', default=False, action='store_true')

    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--dropout_rate', default=0.0, type=float)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--batch_size', default=500, type=int)

    parser.add_argument('--downstream_task', '-dt', default='food_pairing', type=str)
    args = parser.parse_args()

    print(f"[1] ======================================= Setting Random Seed {args.random_seed}")
    setup_seed(args.random_seed)

    print(f"[2] ======================================= Getting Vector Dimensions")
    pn, sn = args.project_name, args.session_name
    downstream_task = args.downstream_task
    ideation_score  = args.ideation_score
    batch_size      = args.batch_size
    args = pickle.load(open(f'{OUT_PATH}{pn}_{sn}_{args.random_seed}/model_config.pkl', 'rb'))
    args = get_vector_dimensions(args)
    args.batch_size = batch_size
    if 'ars' in args.model_struct: args.hidden_dim = 128 # temporary code

    print(f"[3] ======================================= Setting Up Wandb.AI")
    wandb_init_args = {'project': args.project_name, 
                       'group'  : args.session_name, 
                       'name'   : f'{downstream_task}_{args.random_seed}', 
                       'config' : args}
    for k ,v in wandb_init_args.items(): print(k, v)
    wandb.init(**wandb_init_args)
    setproctitle.setproctitle(f'{args.session_name}')

    print(f"[4] ======================================= Loading CollateFn")
    collate = CollateFn(args)

    print(f"[5] ======================================= Loading Model and Running Pipeline")
    if downstream_task == 'scoring_subset2':
        args.dataset_name = f'recipemind_subset2_{ideation_score}'
        model = load_recipe_model(args).cuda()
        test = get_test_loader(args, collate)
        checkpoint = torch.load(f'{OUT_PATH}{pn}_{sn}_{args.random_seed}/epoch_best.mdl')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        test = get_test_loader(args, collate)
        pipeline = FoodPairingPipeline(model=model, collate=collate, session_name=f'{args.session_name}_{args.random_seed}', dataset_name=args.dataset_name)
        results = pipeline(test, 'subset2')

    elif downstream_task == 'scoring_subset3':
        args.dataset_name = f'recipemind_subset3_{ideation_score}'
        model = load_recipe_model(args).cuda()
        test = get_test_loader(args, collate)
        checkpoint = torch.load(f'{OUT_PATH}{pn}_{sn}_{args.random_seed}/epoch_best.mdl')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        test = get_test_loader(args, collate)
        pipeline = NtupletScoringPipeline(model=model, collate=collate, session_name=f'{args.session_name}_{args.random_seed}', dataset_name=args.dataset_name)
        results = pipeline(test, 'subset3')

    elif downstream_task == 'scoring_subset4':
        args.dataset_name = f'recipemind_subset4_{ideation_score}'
        model = load_recipe_model(args).cuda()
        test = get_test_loader(args, collate)
        checkpoint = torch.load(f'{OUT_PATH}{pn}_{sn}_{args.random_seed}/epoch_best.mdl')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        test = get_test_loader(args, collate)
        pipeline = NtupletScoringPipeline(model=model, collate=collate, session_name=f'{args.session_name}_{args.random_seed}', dataset_name=args.dataset_name)
        results = pipeline(test, 'subset4')

    elif downstream_task == 'scoring_subset5':
        args.dataset_name = f'recipemind_subset5_{ideation_score}'
        model = load_recipe_model(args).cuda()
        test = get_test_loader(args, collate)
        checkpoint = torch.load(f'{OUT_PATH}{pn}_{sn}_{args.random_seed}/epoch_best.mdl')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        test = get_test_loader(args, collate)
        pipeline = NtupletScoringPipeline(model=model, collate=collate, session_name=f'{args.session_name}_{args.random_seed}', dataset_name=args.dataset_name)
        results = pipeline(test, 'subset5')

    elif downstream_task == 'scoring_subset6':
        args.dataset_name = f'recipemind_subset6_{ideation_score}'
        model = load_recipe_model(args).cuda()
        test = get_test_loader(args, collate)
        checkpoint = torch.load(f'{OUT_PATH}{pn}_{sn}_{args.random_seed}/epoch_best.mdl')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        test = get_test_loader(args, collate)
        pipeline = NtupletScoringPipeline(model=model, collate=collate, session_name=f'{args.session_name}_{args.random_seed}', dataset_name=args.dataset_name)
        results = pipeline(test, 'subset6')

    elif downstream_task == 'scoring_subset7':
        args.dataset_name = f'recipemind_subset7_{ideation_score}'
        model = load_recipe_model(args).cuda()
        test = get_test_loader(args, collate)
        checkpoint = torch.load(f'{OUT_PATH}{pn}_{sn}_{args.random_seed}/epoch_best.mdl')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        test = get_test_loader(args, collate)
        pipeline = NtupletScoringPipeline(model=model, collate=collate, session_name=f'{args.session_name}_{args.random_seed}', dataset_name=args.dataset_name)
        results = pipeline(test, 'subset7')

    else:
        raise

    wandb.log(results)
    wandb.finish()