# train.py -pn RecipeMind_CIKM2022 -sn ablated_cpmx --model_struct recipemind_cpmx_sum_cat --random_seed 1001 --batch_size 1024 --dataset_index ver1 --dataset_name recipemind_mixed_sPMId02

from env_config import *
from trainer import *
from models import *
import wandb
import torch
import numpy as np
import os
import random
import json
import setproctitle

torch.set_num_threads(1)
os.environ["MKL_NUM_THREADS"] = "20" 
os.environ["NUMEXPR_NUM_THREADS"] = "20" 
os.environ["OMP_NUM_THREADS"] = "20" 
os.environ['OPENBLAS_NUM_THREADS'] = "20"
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# torch.autograd.set_detect_anomaly(True) 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_vector_dimensions(args):
    lang_dim = dict()

    dim_dict = {
        'reciptor': 600,
        'bert-base-uncased': 768,
        'flavorgraph': 300,
        'im2recipe': 300,
        'binary': 630
    }

    lang_dim['J'] = dim_dict[args.initial_vectors_J]
    lang_dim['T'] = dim_dict[args.initial_vectors_T]
    lang_dim['R'] = dim_dict[args.initial_vectors_R]

    args.lang_dim = lang_dim
    return args

def baseline_arguments(args):
    if args.model_struct == 'kitchenette':
        print("Kitchenette Baseline Model")
        args.dataset_name = 'recipemind_doublets'
        args.initial_vectors_J = 'im2recipe'
        args.hidden_dim = 1024
        args.dropout_rate = 0.2
        args.learning_rate = 1e-4
        args.weight_decay = 1e-5
        args.num_epochs = 60
        args.batch_size = 32
    # elif 'recipebowl' in args.model_struct:
    #     print("RecipeBowl Pretraining Model")
    #     args.dataset_name = 'recipebowl_original'
    #     args.initial_vectors_J = 'flavorgraph'
    #     args.initial_vectors_T = 'binary'
    #     args.initial_vectors_R = 'reciptor'
    #     args.hidden_dim = 1024
    #     args.weight_decay = 0.0
    #     args.dropout_rate = 0.2
    #     args.learning_rate = 0.0003
    #     args.num_epochs = 60
    #     args.batch_size = 64
    else:
        pass
    
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', '-pn', default='Test SonyAI', type=str)
    # parser.add_argument('--group_name',   '-gn', default='Test SonyAI', type=str)
    parser.add_argument('--session_name', '-sn', default='Test SonyAI', type=str)
    parser.add_argument('--random_seed',         default=911012, type=int)
    # parser.add_argument('--test_mode',           default='none', type=str)
    parser.add_argument('--fine_tuning',         default=None, type=str)
    parser.add_argument('--debug_mode',   '-dm', default=False, action='store_true')

    parser.add_argument('--dataset_index', default='ver1', type=str)
    parser.add_argument('--dataset_version', default='211210', type=str)
    parser.add_argument('--dataset_name', default='recipemind_mixed_sPMId02', type=str)
    parser.add_argument('--initial_vectors_J', default='flavorgraph', type=str)
    parser.add_argument('--initial_vectors_T', default='bert-base-uncased', type=str)
    parser.add_argument('--initial_vectors_R', default='bert-base-uncased', type=str)
    parser.add_argument('--model_struct', default='recipemind', type=str)
    parser.add_argument('--model_analysis', default=False, action='store_true')

    parser.add_argument('--hidden_dim', default=128, type=int)      # 1024
    parser.add_argument('--dropout_rate', default=0.025, type=float)   # 0.2
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--loss_function', default='rmse', type=str)
    parser.add_argument('--grad_update', default='default', type=str)
    parser.add_argument('--train_eval', default=False, action='store_true')
    # parser.add_argument('--pretrained_recipebowl', default='none', type=str)

    parser.add_argument('--hybrid_coef', default=0.5, type=float)
    parser.add_argument('--mc_dropout', default=False, action='store_true')

    # Set Attention Blocks as Element Encoder
    parser.add_argument('--sab_num_aheads', default=8, type=int)
    parser.add_argument('--sab_num_blocks', default=3, type=int)

    # ApproxRepSet as Element Encoder
    parser.add_argument('--ars_num_hsets', default=256, type=int)
    parser.add_argument('--ars_num_helms', default=8,  type=int)

    # Pooling By Multihead Attention as Set Encoder
    parser.add_argument('--pma_num_aheads', default=8, type=int)
    parser.add_argument('--pma_num_sdvecs', default=4, type=int)
    parser.add_argument('--pma_num_blocks', default=2, type=int)

    # Multihead Attention related parameters
    parser.add_argument('--multihead_sim', default='general_dot', type=str)
    parser.add_argument('--multihead_big', default=False, action='store_true')

    args = parser.parse_args()
    args = baseline_arguments(args)
    if 'wnd' in args.model_struct: args.batch_size = 32
    print(f"[1] ======================================= Setting Random Seed {args.random_seed}")
    setup_seed(args.random_seed)

    print(f"[2] ======================================= Getting Vector Dimensions")
    args = get_vector_dimensions(args)

    print(f"[3] ======================================= Setting Up Wandb.AI")
    wandb_init_args = {'project': args.project_name, 
                       'group'  : args.session_name, 
                       'name'   : f'training_{args.random_seed}',
                       'config' : args}
    for k ,v in wandb_init_args.items(): print(k, v)
    wandb.init(**wandb_init_args)
    setproctitle.setproctitle(f'{args.session_name}')
    wandb.define_metric('train/step'); wandb.define_metric('train/*', step_metric='train/step')
    wandb.define_metric('valid/step'); wandb.define_metric('valid/*', step_metric='valid/step')

    print(f"[4] ======================================= Loading Model, Trainer and CollateFn")
    trainer = load_recipe_trainer(args)
    collate = CollateFn(args)
    # args.model_analysis = True
    model = load_recipe_model(args).cuda()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[5] ======================================= Number of Trainable Parameters for {args.model_struct}: {num_params}")
    wandb.watch(model, log='gradients', log_freq=1000)
    pickle.dump(args, open(trainer.checkpoint_path+'model_config.pkl', 'wb'))
    

    print(f"[6] ======================================= Loading Train/Valid Dataset and Dataloader")
    train = get_train_loader(args, collate)
    valid = get_valid_loader(args, collate)
    
    print(f"[7] ======================================= Training the Model {trainer.checkpoint_path}")
    model = trainer.train_model(model, train, valid, args.fine_tuning)
    del train; torch.cuda.empty_cache()

    if not args.debug_mode:
        print(f"[8] ======================================= Evaluating the Model on Full Validation Set")
        trainer.test_model(model, valid, True)




    # if args.fine_tuning:
    #     print(f"[6] ======================================= Loading the Pretrained Model {args.fine_tuning}")
    #     checkpoint = torch.load(f'{OUT_PATH}{args.fine_tuning}/epoch_final.mdl')
    #     model.load_state_dict(checkpoint['model_state_dict'])

    # if args.test_mode == 'none':
    #     print(f"[7] ======================================= Loading Train/Valid Dataset and Dataloader")
    #     train = get_train_loader(args, collate)
    #     valid = get_valid_loader(args, collate)
        
    #     print(f"[8] ======================================= Training the Model {trainer.checkpoint_path}")
    #     model = trainer.train_model(model, train, valid, args.fine_tuning)
    #     del train; torch.cuda.empty_cache()

    #     if not args.debug_mode:
    #         print(f"[9] ======================================= Evaluating the Model on Full Validation Set")
    #         trainer.test_model(model, valid, True)

    # else:
    #     print(f"[7] ======================================= Loading Test Dataset and Dataloader")
    #     checkpoint = torch.load(trainer.checkpoint_path+f'{args.test_mode}.mdl')
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     test = get_test_loader(args, collate)

    #     print(f"[8] ======================================= Evaluating the Model on Full Test Set")
    #     test_session = f'final_eval_{args.dataset_index}_{args.dataset_name}'
    #     trainer.test_model(model, test, True)

    wandb.finish()