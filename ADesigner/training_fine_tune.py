#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn.functional as F  # Import torch.nn.functional as F

# import wandb
from utils.logger import print_log
from parser import create_parser

from data.dataset import EquiAACDataset
from torch.utils.data import DataLoader
from models.adesigner import ADesigner


class TrainConfig:


    def __init__(self, save_dir, lr, max_epoch, metric_min_better, early_stop, patience, grad_clip, anneal_base):
        self.save_dir = save_dir
        self.lr = lr
        self.max_epoch = max_epoch
        self.metric_min_better = metric_min_better
        self.patience = patience if early_stop else max_epoch + 1
        self.grad_clip = grad_clip
        self.anneal_base = anneal_base


    def __str__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)


def get_training_config(is_finetuning=False):
    """Get configuration parameters based on training mode"""
    if is_finetuning:
        return {
            'lr': 0.0001,           # 10x lower learning rate
            'max_epoch': 15,        # Fewer epochs but not too few
            'early_stop': True,     # Enable early stopping
            'patience': 5,          # More patience for convergence
            'grad_clip': 0.5,       # Lower gradient clipping
            'anneal_base': 0.98,    # Slower decay
            'batch_size': 8         # Smaller batch for stability
        }
    else:
        return {
            'lr': 0.001,
            'max_epoch': 20,
            'early_stop': False,
            'patience': 3,
            'grad_clip': 1.0,
            'anneal_base': 0.95,
            'batch_size': 16
        }


class Trainer:
    def __init__(self, model, train_loader, valid_loader, config, save_dir, cdr=None, fold=None, is_finetuning=False):
        self.model = model
        self.config = config
        self.save_dir = save_dir
        self.is_finetuning = is_finetuning  # Track if we're fine-tuning
        
        # Apply layer freezing if fine-tuning
        if is_finetuning:
            self.freeze_backbone_layers(model)
        
        self.optimizer = self.get_optimizer()
        sched_config = self.get_scheduler(self.optimizer)
        if sched_config is None:
            sched_config = {'scheduler': None, 'frequency': None}
        self.scheduler = sched_config['scheduler']
        self.sched_freq = sched_config['frequency']
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # distributed training
        self.local_rank = -1

        # log
        self.model_dir = os.path.join(save_dir, 'checkpoint')
        self.writer_buffer = {}

        # training process recording
        self.global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.last_valid_metric = None
        self.patience = config.patience

        self.init_step = (1 - 1) * 10 * int(config.max_epoch) + 0 * int(config.max_epoch)
        self.ex_name = 'CDR{0}_fold{1}'.format(cdr, fold)
        
        # Fine-tuning monitoring
        if is_finetuning:
            self.training_losses = []
            self.validation_losses = []

    def freeze_backbone_layers(self, model):
        """Freeze specific layers for fine-tuning"""
        frozen_count = 0
        total_count = 0
        
        for name, param in model.named_parameters():
            total_count += 1
            # Freeze all layers except the final output/generator layers
            # Adjust these conditions based on your model architecture
            if any(keyword in name.lower() for keyword in ['output', 'generator', 'final', 'head']):
                param.requires_grad = True
            else:
                param.requires_grad = False
                frozen_count += 1
        
        print(f"Fine-tuning mode: Frozen {frozen_count}/{total_count} parameters")
        print("Only training final layers for domain adaptation")


    @classmethod
    def to_device(cls, data, device):
        if isinstance(data, dict):
            for key in data:
                data[key] = cls.to_device(data[key], device)
        elif isinstance(data, list) or isinstance(data, tuple):
            res = [cls.to_device(item, device) for item in data]
            data = type(data)(res)
        elif hasattr(data, 'to'):
            data = data.to(device)
        return data

    def _is_main_proc(self):
        return self.local_rank == 0 or self.local_rank == -1

    def _train_epoch(self, device):
        if self.train_loader.sampler is not None and self.local_rank != -1:  # distributed
            self.train_loader.sampler.set_epoch(self.epoch)
        t_iter = tqdm(self.train_loader) if self._is_main_proc() else self.train_loader
        for batch in t_iter:
            batch = self.to_device(batch, device)
            loss = self.train_step(batch, self.global_step)
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            # try_catch_oom(self.optimizer.step)
            self.optimizer.step()
            if hasattr(t_iter, 'set_postfix'):
                t_iter.set_postfix(loss=loss.item())
            self.global_step += 1
            if self.sched_freq == 'batch':
                self.scheduler.step()
        if self.sched_freq == 'epoch':
            self.scheduler.step()
        # write training log
        mean_writer = {}
        for name in self.writer_buffer:
            value = np.mean(self.writer_buffer[name])
            mean_writer[name] = value
        # if self.wandb:
        #     wandb.log(mean_writer, step=self.init_step + self.epoch)
        self.writer_buffer = {}

    def _valid_epoch(self, device):
        metric_arr = []
        self.model.eval()
        with torch.no_grad():
            t_iter = tqdm(self.valid_loader) if self._is_main_proc() else self.valid_loader
            for batch in t_iter:
                batch = self.to_device(batch, device)
                metric = self.valid_step(batch, self.valid_global_step)
                metric_arr.append(metric.cpu().item())
                self.valid_global_step += 1
        self.model.train()
        # judge
        ckpt_saved, save_path = False, None
        valid_metric = np.mean(metric_arr)
        if self._metric_better(valid_metric):
            self.patience = self.config.patience
            if self._is_main_proc():
                save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
                module_to_save = self.model.module if self.local_rank == 0 else self.model
                torch.save(module_to_save, save_path)
                # torch.save(module_to_save, os.path.join(self.config.save_dir, f'best.ckpt'))
                torch.save(module_to_save, os.path.join(self.save_dir, f'best.ckpt'))
                ckpt_saved = True
            self.last_valid_metric = valid_metric
        else:
            self.patience -= 1
        # write validation log
        mean_writer = {}
        for name in self.writer_buffer:
            value = np.mean(self.writer_buffer[name])
            mean_writer[name] = value
        # if self.wandb:
        #     wandb.log(mean_writer, step=self.init_step + self.epoch)
        self.writer_buffer = {}
        return ckpt_saved, save_path
    
    def _metric_better(self, new):
        old = self.last_valid_metric
        if old is None:
            return True
        if self.config.metric_min_better:
            return new < old
        else:
            return old < new

    def train(self):
        # Assuming a single GPU setup here, GPU ID 0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
    # def train(self, device_ids, local_rank):
    #     # set local rank  
    #     self.local_rank = local_rank
        # init writer
        if self._is_main_proc():
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            # with open(os.path.join(self.config.save_dir, 'train_config.json'), 'w') as fout:
            with open(os.path.join(self.save_dir, 'train_config.json'), 'w') as fout:
                json.dump(self.config.__dict__, fout)


        # Now training with a single GPU
        for _ in range(self.config.max_epoch):
            print_log(f'epoch{self.epoch} starts')
            self._train_epoch(device)
            print_log(f'validating ...')
            ckpt_saved, save_path = self._valid_epoch(device)
            if ckpt_saved:
                print_log(f'checkpoint saved to {save_path}')
            self.epoch += 1
            if self.patience <= 0:
                print_log('early stopping')
                break
        print_log('finished training')

    def log(self, name, value, step, val=False):
        if self._is_main_proc():
            if isinstance(value, torch.Tensor):
                value = value.cpu().item()
            if name not in self.writer_buffer:
                self.writer_buffer[name] = []
            self.writer_buffer[name].append(value)

    ########## Overload these functions below ##########
    # define optimizer
    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        return optimizer

    # scheduler example: linear. Return None if no scheduler is needed.
    def get_scheduler(self, optimizer):
        lam = lambda epoch: self.config.anneal_base ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)
        return {
            'scheduler': scheduler,
            'frequency': 'epoch' # or batch
        }

    # train step, note that batch should be dict/list/tuple or other objects with .to(device) attribute
    def train_step(self, batch, batch_idx):
        return self.share_forward(batch, batch_idx, 'train/' + self.ex_name)

    # validation step
    def valid_step(self, batch, batch_idx):
        return self.share_forward(batch, batch_idx, 'validation/' + self.ex_name, val=True)
    
    def share_forward(self, batch, batch_idx, _type, val=False):
        loss, snll, closs = self.model(
            batch['X'], batch['S'], batch['L'], batch['offsets']
        )
        ppl = snll.exp().item()
        self.log(f'Loss/{_type}', loss, batch_idx, val=val)
        self.log(f'SNLL/{_type}', snll, batch_idx, val=val)
        self.log(f'Closs/{_type}', closs, batch_idx, val=val)
        self.log(f'PPL/{_type}', ppl, batch_idx, val=val)
        return loss



# Create the argument parser
parser = argparse.ArgumentParser(description='Parse arguments required for training')
parser.add_argument('--train_set', type=str, required=True, help='Path')
parser.add_argument('--valid_set', type=str, required=True, help='Path')
parser.add_argument('--save_dir', type=str, required=True, help='Path')
parser.add_argument('--fold', type=int, required=True, help='Path')

parser.add_argument('--pretrained_model', type=str, default=None, 
                   help='Path to pretrained model checkpoint for fine-tuning')
parser.add_argument('--freeze_layers', action='store_true', 
                   help='Freeze early layers during fine-tuning')
parser.add_argument('--finetune_mode', action='store_true',
                   help='Enable fine-tuning mode with adjusted hyperparameters')


# Parse the arguments
args = parser.parse_args()


def load_pretrained_model(model, checkpoint_path):
    """Safely load pretrained weights with compatibility checks"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading pretrained model from: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if hasattr(checkpoint, 'state_dict'):
            pretrained_dict = checkpoint.state_dict()
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            # Assume the checkpoint is the state dict itself
            pretrained_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        
        # Get current model state
        model_dict = model.state_dict()
        
        # Filter out incompatible keys and size mismatches
        compatible_dict = {}
        incompatible_keys = []
        
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    compatible_dict[k] = v
                else:
                    incompatible_keys.append(f"{k}: {v.shape} vs {model_dict[k].shape}")
            else:
                incompatible_keys.append(f"{k}: not found in current model")
        
        # Load compatible weights
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict)
        
        print(f"Successfully loaded {len(compatible_dict)}/{len(pretrained_dict)} layers")
        if incompatible_keys:
            print(f"Skipped {len(incompatible_keys)} incompatible layers")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

def main(args):
    # Determine if we're in fine-tuning mode
    is_finetuning = args.pretrained_model is not None or args.finetune_mode
    
    # Get configuration based on training mode
    config_params = get_training_config(is_finetuning)
    
    # Base model parameters
    embed_size = 64
    hidden_size = 128
    shuffle = False
    num_workers = 4
    
    # Apply configuration
    lr = config_params['lr']
    max_epoch = config_params['max_epoch']
    early_stop = config_params['early_stop']
    patience = config_params['patience']
    grad_clip = config_params['grad_clip']
    anneal_base = config_params['anneal_base']
    batch_size = config_params['batch_size']
    
    print(f"=== Training Mode: {'FINE-TUNING' if is_finetuning else 'FROM SCRATCH'} ===")
    print(f"Learning rate: {lr}")
    print(f"Max epochs: {max_epoch}")
    print(f"Batch size: {batch_size}")
    print(f"Early stopping: {early_stop}")

    # Initialize the model
    model = ADesigner(embed_size=embed_size, hidden_size=hidden_size, n_channel=4,
                     n_layers=3, cdr_type='3', alpha=0.1, dropout=0.1, args=None)

    # Load pretrained weights if provided
    if args.pretrained_model:
        load_pretrained_model(model, args.pretrained_model)

    # Initialize datasets with the updated collate function
    _collate_fn = EquiAACDataset.collate_fn

    train_dataset = EquiAACDataset(args.train_set, args.save_dir, num_entry_per_file=-1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, 
                             num_workers=num_workers, collate_fn=_collate_fn)

    valid_dataset = EquiAACDataset(args.valid_set, args.save_dir, num_entry_per_file=-1)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle, 
                             num_workers=num_workers, collate_fn=_collate_fn)

    # Create training configuration
    config = TrainConfig(
        save_dir=args.save_dir,
        lr=lr,
        max_epoch=max_epoch,
        metric_min_better=True,
        early_stop=early_stop,
        patience=patience,
        grad_clip=grad_clip,
        anneal_base=anneal_base
    )

    # Initialize trainer
    trainer = Trainer(model, train_loader, valid_loader, config, 
                     save_dir=args.save_dir, cdr=3, fold=args.fold, 
                     is_finetuning=is_finetuning)

    # Start training
    trainer.train()

if __name__ == '__main__':
    main(args)