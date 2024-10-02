import os
import random
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from transformers import set_seed
import wandb

# import os 
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0, 2, 3"

# # setting from terminal 
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

def setup_device(device: Optional[str] = None) -> torch.device:

    if torch.cuda.is_available():
        device = "cuda"
    else:
        try:
            if torch.backends.mps.is_available():
                device = "mps"
        except:
            device = "cpu"

    print(f'Device set to {device}\n')

    # return torch.device("cuda:0")
    return torch.device(device)


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class WandbLogger:
    def __init__(self, wandb_args, config_dict):

        self.wandb_args = wandb_args

        self.reset_wandb_env()
        wandb_init = self.setup_wandb()
        
        run = wandb.init(**wandb_init, settings=wandb.Settings(start_method='thread'), config=config_dict)
        
        run.define_metric('train/epoch') # x-axis
        run.define_metric('train/step') # x-axis
        run.define_metric('train/loss/epoch', step_metric='train/epoch')
        run.define_metric('train/loss/step', step_metric='train/step')
        run.define_metric('eval/epoch') # x-axis
        run.define_metric('eval/step') # x-axis
        run.define_metric('eval/loss/epoch', step_metric='eval/epoch')
        run.define_metric('eval/metrics/epoch', step_metric='eval/epoch')
        run.define_metric('eval/loss/step', step_metric='eval/step')
        
        run.define_metric("KL Divergence", step_metric='eval/epoch')
        run.define_metric("F1 Score", step_metric='eval/epoch')
        run.define_metric("Precision", step_metric='eval/epoch')
        run.define_metric("Recall", step_metric='eval/epoch')
        
        # {"KL Divergence": kld_avg.item(), "F1 Score": f1_score.item(), "Precision": precision.item(), "Recall": recall.item()}
        
        self.wandb_run = run

    def setup_wandb(self):
        wandb_init = dict()
        wandb_init['project'] = self.wandb_args.project_name
        wandb_init['entity'] = self.wandb_args.wandb_entity
        wandb_init['group'] = self.wandb_args.session_name
        wandb_init['name'] = self.wandb_args.name # f'training_{conf.experiment.dataset}'

        wandb_init['notes'] = self.wandb_args.session_name 
        os.environ['WANDB_START_METHOD'] = 'thread' 

        return wandb_init
    
    # def log_hps(self, config_dict):

    #     # wandb.config = config_dict # self.wandb_run.config??
    #     self.wandb_run.config = config_dict


    def reset_wandb_env(self):
        exclude = {'WANDB_PROJECT', 'WANDB_ENTITY', 'WANDB_API_KEY',}
        for k, v in os.environ.items():
            if k.startswith('WANDB_') and k not in exclude:
                del os.environ[k]

    def info(self, wandb_dict): # method name changed from log_wandb so that it's compatible with python logger option. 
        '''
        wandb_dict['train/loss'] = train_loss
        wandb_dict['eval/loss'] = eval_loss
        '''
        self.wandb_run.log(wandb_dict)

    def finish_wandb(self):
        wandb.finish()



class EarlyStopping:
    def __init__(self, patience=50, delta=0.0001, save_dir='./outputs'):

        self.patience = int(patience)
        self.delta = float(delta)
        # self.path = path
        self.save_dir = save_dir
        self.counter = 0
        self.best_score = None
        # self.val_loss_min = np.Inf
        
        self.early_stop = False
        self.best_w_delta = False

    def __call__(self, val_loss): # called every epoch
        score = -val_loss
        self.best_w_delta, self.early_stop = False, False

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.best_w_delta = True

        # The main purpose of introducing patience in early stopping is to ensure that training is not prematurely halted due to minor fluctuations in the performance metric.
        elif score < self.best_score + self.delta: # score not improving anymore, no point training more epochs. 
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.best_w_delta = True
            self.counter = 0


def create_directory_if_not_exists(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)
        print(f"Directory '{path_name}' was created.")
    else:
        print(f"Directory '{path_name}' already exists.")