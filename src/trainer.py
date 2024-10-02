import json
import math
import os
import numpy as np
import pandas as pd
import pickle
import random
import argparse
from tqdm import tqdm
import time

import wandb
import hydra
from omegaconf import OmegaConf

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset

from .configs.configs import TrainingConfigs

from torcheval.metrics.functional import multiclass_f1_score, multiclass_precision, multiclass_recall
from .utils.utils import setup_seed, setup_device, WandbLogger, EarlyStopping
# from .configs import TrainingConfigs
from .dataset.datasets import EEGDataset
from .dataset.datasets_cam import EEGMontageDataset
# from .models.eegnet import EEGNet

num_cpus = os.cpu_count()
ngpus = torch.cuda.device_count()

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)

# print()
# print(f"{ngpus} Available")
# print()

from braindecode.models import ShallowFBCSPNet, EEGNetv4, EEGConformer, EEGResNet, ATCNet, Deep4Net

model_class_map = {
    "eegnet": EEGNetv4, 
    "eegconformer": EEGConformer,
    "atcnet": ATCNet,
    "deep4net": Deep4Net, 
    # "examplenet": ShallowFBCSPNet,
}

# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# loss_fn_map = {
#     "cross_entropy": torch.nn.CrossEntropyLoss,
#     "kl_divergence": torch.nn.KLDivLoss, #(reduction="batchmean"),
#     # KL, etc
# }

loss_fn_map = {
    "cross_entropy": torch.nn.CrossEntropyLoss,
    "kl_divergence": torch.nn.KLDivLoss, #(reduction="batchmean"),
    "multi_class_F1": multiclass_f1_score,
    "multi_class_precision": multiclass_precision,
    "multi_class_recall": multiclass_recall
}

'''
TO DO: check for to(self.device)'s missing (removed DDP logic)
- could add scheduler logic -> not adding for lr dependency reporting
- early stopping - done
- compute_eval_metrics - KL
- dataloader = self.dataloaders['test'] - cam's added test split
- add models
- revise config hierarchy
- wandb entity to group
- get eval metrics -> save to wandb if true

- validation set
'''

import os

def create_directory_if_not_exists(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)
        print(f"Directory '{path_name}' was created.")
    else:
        print(f"Directory '{path_name}' already exists.")


class Trainer:
    def __init__(self, configs: TrainingConfigs):

        self.configs = configs
        self.hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        self.output_dir = self.hydra_cfg["runtime"]["output_dir"]
        # print(f"output_dir = {self.output_dir}")

        self.trainer_configs = self.configs.trainer
        self.model_configs = self.configs.model
        # self.hp_settings = self.configs.hp_settings
        
        create_directory_if_not_exists(self.trainer_configs.save_checkpoint_path)
        create_directory_if_not_exists(self.trainer_configs.save_preds_path)

        self.device = setup_device()
        setup_seed()

        self.dataloaders = self._get_dataloaders(toy_dataset=self.configs.debug)

        if not configs.debug:
            self._setup_logger()

        # self._setup_logger() # delete this. 


        # self.dataloaders = self._get_dataloaders()
        self.model = self.initialize_model()
        self.optimizer = self.initialize_optimizer() # python main.py override=test_experiment
        
        self.early_stopping = EarlyStopping(patience=self.trainer_configs.patience, delta=self.trainer_configs.delta) # change this to config var

    def _setup_logger(self, logger='wandb'):

        if logger=='wandb':
            
            config_dict = {}
            config_dict.update(self.trainer_configs)
            config_dict.update(self.model_configs)
            
            self.logger = WandbLogger(self.trainer_configs.wandb_args, config_dict)

    def _get_dataloaders(self, toy_dataset=True) -> dict:

        dataloaders = {}
        splits = ["train", "test"] if toy_dataset else ["train", "validation", "test"]
        for split in splits: # fix. validation set?
            
            data_dir = self.trainer_configs.data_dir.format(split=split) # f"/home/co-chae/rds/hpc-work/home/camwheeler/camerons_datasets/mlp_eeg_data/{split}/data" # fix 
            target_dir = self.trainer_configs.target_dir.format(split=split) # f"/home/co-chae/rds/hpc-work/home/camwheeler/camerons_datasets/mlp_eeg_data/{split}/targets" # fix 
            
            if toy_dataset:

                dataset = EEGDataset(
                    'CamWheeler135/MLP_EEG_Tiny_Dataset',
                    split=split,
                )
            else:
                dataset = EEGMontageDataset(data_dir=data_dir, target_dir=target_dir)

            dataloaders[split] = DataLoader(
                dataset,
                shuffle=True,
                num_workers=self.trainer_configs.num_workers,
                batch_size=self.trainer_configs.batch_size,
                # **self.data_configs.data_loader_configs,
            )

        return dataloaders

    def initialize_model(self): 
        
        model = model_class_map[self.model_configs.model_name](**self.model_configs.model_args)
        model = model.to(self.device) # or to device
        # model = DDP(model, device_ids=[rank])
        
        self.start_epoch = False

        if self.model_configs.configs.from_pretrained:
            checkpoint_path = self.model_configs.configs.pretrained_checkpoint
            ckpt = torch.load(checkpoint_path)#, map_location=f'cuda:{rank}')
            model.load_state_dict(ckpt)
            model.to(self.device)
            self.start_epoch = int(os.path.basename(checkpoint_path).split('_')[-2]) # eegnet_epoch_41_best.pt
            
            print(f"\n******** Pretrained checkpoint (epoch {self.start_epoch}) loaded !! ********\n")
        else:
            pass
            # model.init_weights() # weight initialization done internally by braindecode # https://github.com/braindecode/braindecode/blob/65870a3e09ef5ecf5817d12806e7a3a00d67662b/braindecode/models/eegnet.py
        
        return model

    def initialize_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.trainer_configs.lr, betas=(0.9, 0.98), eps=1e-9)
        return optimizer


    def _save_checkpoint(self, epoch_idx): # rank argument not needed
        
        # import pdb; pdb.set_trace()
        
        n_filts = {}
        n_filts['eegnet']='F1'
        n_filts['eegconformer']='n_filters_time'
        n_filts['atcnet']='conv_block_n_filters'
        n_filts['deep4net']='n_filters_time'

        n_heads = {}
        n_heads['eegconformer']='att_heads'
        n_heads['atcnet']='att_num_heads'
        
        n_filts_name = n_filts[self.model_configs.model_name]
        
        tag = "best" if self.early_stopping.best_w_delta else "early_stop"
        
        
        checkpoint_filename = f'{self.model_configs.model_name}_'
        checkpoint_filename += f'n_filts_{getattr(self.model_configs.model_args, n_filts_name)}_'
        if self.model_configs.model_name in n_heads.keys():
            n_heads_name = n_heads[self.model_configs.model_name]
            checkpoint_filename += f'n_heads_{getattr(self.model_configs.model_args, n_heads_name)}_'
        checkpoint_filename += f'epoch_{epoch_idx}_{tag}.pt'
        
        save_path = os.path.join(self.trainer_configs.save_checkpoint_path, checkpoint_filename)
        
        print(f"\n******** saving checkpoint at {save_path}... ********\n")
        
        torch.save(self.model.state_dict(), save_path) 
        
        print(f"\n******** Checkpoint saved at {save_path} !! ********\n")
        

    def _train_step(self, loss):
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

    def _train_loop_epoch(self, show_progress=True):
        '''
        TO DO: look at last batch size
        '''

        self.model.train()
        
        wandb_dict_step = None

        losses, epoch_loss = 0.0, 0.0
        num_batches = len(self.dataloaders["train"])
        
        progress = tqdm(self.dataloaders["train"], desc=f'\nTraining progress ({num_batches} batches for a single epoch): ') if show_progress else self.dataloaders["train"]
        for batch_idx, batch in enumerate(progress): # batch_idx is step
            
            batch_data, batch_label = batch[0].to(self.device), batch[1].to(self.device)

            _, batch_loss = self._batch_forward(batch_data, batch_label) # (batch_size, self.n_outputs), (bs, n_output)

            wandb_dict_step = {'train/step': batch_idx, 'train/loss/step': batch_loss.item()}
            self.logger.info(wandb_dict_step)
            
            losses += batch_loss.item()
            epoch_loss = losses / num_batches   
            self._train_step(batch_loss)
        
        return epoch_loss

    def _batch_forward(self, batch_data, batch_label, return_preds=False): 
        
        # batch_data, batch_label = batch
        # batch_data, batch_label = batch_data.to(self.device), batch_label.to(self.device)
        # print(f"batch_label.shape = {batch_label.shape}")
        # print(f"batch_data.shape = {batch_data.shape}")
        batch_preds = None

        model_output = self.model(batch_data) # , batch_label) # Your method needs to return a tensor either of shape (batch_size, self.n_outputs), or, in special cases, of shape (batch_size, self.n_outputs, n_out_times).
        
        
        batch_preds = model_output # 128 by 6 -> so the model output is in log space. 
        # (Pdb) batch_preds[0, :].sum()
        # tensor(-0.1064, device='cuda:0', grad_fn=<SumBackward0>)
        # import pdb; pdb.set_trace()
        if return_preds:
            batch_preds = self._get_preds(model_output) 

        batch_loss = self._compute_batch_loss(model_output, batch_label) # first dim has to be batch_size
        
        return batch_preds, batch_loss
    
    def _compute_batch_loss(self, model_output, label): # model_output.shape = 8, 6 = bs, n_output

        loss_fn = loss_fn_map[self.trainer_configs.loss_fn]()
        # import pdb; pdb.set_trace()
        batch_loss = loss_fn(model_output, label) # batch_size x num_classes

        return batch_loss
    
    def _get_preds(self, model_out): 
        """
        Post-process raw model output to get final prediction values. 
        e.g. for most NLP tasks, this maps token logits to tok ids. 

        Perhaps not relevant to EEG challenge. 
        """

        pass
    

    def train(self):
        
        wandb_dict = None

        for epoch_idx in range(1, self.trainer_configs.num_epochs+1):
            
            if self.start_epoch:
                epoch_idx = self.start_epoch + epoch_idx # if epoch_idx =1 and start_epoch =50, 

            start_time = time.time()

            train_loss_epoch = self._train_loop_epoch() # train_epoch
            eval_loss_epoch, eval_metrics_epoch = self._eval_loop_epoch(get_eval_metrics=True) # evaluate
            
            end_time = time.time()
            epoch_time = end_time - start_time
        
            # if rank==0:
            print(f"Epoch: {epoch_idx}, Train loss: {train_loss_epoch:.6f}, Val loss: {eval_loss_epoch:.6f}, Epoch time = {epoch_time:.3f}s")#, Rank: {rank}\n")
            
            # wandb_dict['train/loss'], wandb_dict['eval/loss'] = train_loss_epoch, eval_loss_epoch
            wandb_dict = {'train/epoch': epoch_idx, 'train/loss/epoch': train_loss_epoch, 'eval/epoch': epoch_idx, 'eval/loss/epoch': eval_loss_epoch}#, 'eval/metrics/epoch': eval_metrics_epoch}
            wandb_dict.update(eval_metrics_epoch)
            self.logger.info(wandb_dict)

            print(f"Log saved for epoch {epoch_idx}")
            
            # print(f'Saving checkpoint for epoch {epoch_idx}...')
            # self._save_checkpoint(epoch_idx)
            self.early_stopping(eval_loss_epoch)

            if self.early_stopping.early_stop: # or self.early_stopping.best_w_delta:
                print(f"Early stopping! (epoch {epoch_idx})")
                self._save_checkpoint(epoch_idx)
                break
            
            if self.early_stopping.best_w_delta:
                print(f"Saving best checkpoint... (epoch {epoch_idx})")
                self._save_checkpoint(epoch_idx)

        return
    
    def evaluate(self):
        
        wandb_dict = None
        start_time = time.time()

        eval_loss_epoch, eval_metrics_epoch = self._eval_loop_epoch(get_eval_metrics=True) # self._eval_loop_epoch(self.dataloaders['validation'], get_eval_metrics=True) # evaluate
        
        end_time = time.time()
        epoch_time = end_time - start_time
    
        # if rank==0:
        print(f"Val loss: {eval_loss_epoch}, Accuracy: {eval_metrics_epoch}, Epoch time = {epoch_time:.3f}s\n")#, Rank: {rank}\n")
        
        # wandb_dict['train/loss'], wandb_dict['eval/loss'] = train_loss_epoch, eval_loss_epoch
        wandb_dict = {'eval/epoch': 1, 'eval/loss/epoch': eval_loss_epoch}#, 'eval/metrics/epoch': eval_metrics_epoch}
        wandb_dict.update(eval_metrics_epoch)
        self.logger.info(wandb_dict)

    @torch.no_grad()
    def _eval_loop_epoch(self, log_metrics=False, get_eval_metrics=False, show_progress=True): 
        """
        
        """

        self.model.eval()

        dataloader = self.dataloaders['validation'] # self.dataloaders['valid'] if self.dataloaders['valid'] else self.dataloaders['test']

        wandb_dict_step = {}
        num_batches = len(dataloader)

        losses, epoch_loss = 0, 0
        eval_metrics = None

        all_preds, all_tgt = [], [] 

        # if dist.get_rank()==0:
        dataloader = tqdm(dataloader, desc=f'Evaluation Loop Progress: ') if show_progress else dataloader

        for batch_idx, batch in enumerate(dataloader):
            
            batch_data, batch_label = batch[0].to(self.device), batch[1].to(self.device)

            batch_preds, batch_loss = self._batch_forward(batch_data, batch_label) # (batch_size, self.n_outputs), (bs, n_output)

            all_tgt.append(batch_label) # batch_data, batch_label = batch
            all_preds.append(batch_preds) # (bs, n_output)
            
            # ## DDP logic
            if log_metrics: # done in def train(self):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                wandb_dict_step = {'eval/step': batch_idx, 'eval/loss/step': batch_loss}
                self.logger.info(wandb_dict_step)

            losses += batch_loss.item()
            epoch_loss = losses / num_batches
            
        if get_eval_metrics: # potential fix
            eval_metrics = self.compute_eval_metrics(all_preds, all_tgt)

        return epoch_loss, eval_metrics 
    
    def compute_eval_metrics(self, all_preds, all_tgt): # metric calculated across the entire dataset
        """
        could do accuracy here instead of cross entropy or KL-divergence. 
        
        filtered_preds
        """
        
        # import pdb; pdb.set_trace()
        
        all_preds, all_tgt = [pred for pred in all_preds if pred is not None], [pred for pred in all_tgt if pred is not None]
        
        all_preds_tensor, all_tgt_tensor = torch.cat(all_preds, dim=0).detach(), torch.cat(all_tgt, dim=0).detach()
        
    
        
        # print(type(all_preds)) # both list, type(all_tgt[0]): tensor all_tgt[0].shape: torch.Size([16, 6]) (bs, n_classes) # torch.save(all_tgt[0], 'list_of_tensors.pt') # all_tgt[0][0,:].sum() -> not in logspace
        # print(type(all_tgt))
        # print(all_preds)
        # print(all_tgt)
        all_preds_tensor_log_probs = F.log_softmax(all_preds_tensor, dim=1)
        kld_fn = loss_fn_map["kl_divergence"](reduction='batchmean')
        # import pdb; pdb.set_trace()

        kld_avg = kld_fn(all_preds_tensor_log_probs, all_tgt_tensor) # batch_size x num_classes
        f1_score = loss_fn_map["multi_class_F1"](all_preds_tensor, torch.argmax(all_tgt_tensor, axis=1), num_classes=6) # We take the argmax across the target tensor rows to get our class labels
        precision = loss_fn_map['multi_class_precision'](all_preds_tensor, torch.argmax(all_tgt_tensor, axis=1), num_classes=6) 
        recall = loss_fn_map['multi_class_recall'](all_preds_tensor, torch.argmax(all_tgt_tensor, axis=1), num_classes=6)

        metrics = {"KL Divergence": kld_avg.item(), "F1 Score": f1_score.item(), "Precision": precision.item(), "Recall": recall.item()} # ex. {"acc": , "recall": , ...}
        return metrics



    def test(self, rank):
        
        start_time = time.time()
        
        test_accuracy = self._test_loop_epoch()
        
        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Test Accuracy: {test_accuracy}, Epoch time = {epoch_time:.3f}s, Rank: {rank}\n")


    @torch.no_grad()
    def _test_loop_epoch(self):
        self.model.eval()

        len_dataloader = len(self.dataloaders[''])
        
        test_accuracy = 0
        
        all_preds = [] 
        all_tgt = []
        
        dataloader = tqdm(dataloader, desc=f'Testing progress ({len_dataloader} batches for one epoch): ')

        for batch_idx, batch in enumerate(dataloader): 

            inferred_tensors = self._infer_batch(batch) 
            all_preds.append(inferred_tensors)
            all_tgt.append(batch[1])  
            
        dataloader.close()

        test_accuracy = self.compute_eval_metrics(all_preds, all_tgt) 
            
        return test_accuracy
    

    def _infer_batch(self, batch):
        """
        For NLP tasks, this would have decoding logic. Pooling is often done here. 
        """
        
        source, label = batch
        inference_output = ...

        return inference_output
            