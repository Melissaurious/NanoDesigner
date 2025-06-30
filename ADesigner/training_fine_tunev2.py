#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import sys
import traceback
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import glob
import psutil
from utils.logger import print_log
from parser import create_parser

from data.dataset import EquiAACDataset
from torch.utils.data import DataLoader
from models.adesigner import ADesigner


def check_system_resources():
    """Check GPU and memory availability"""
    print("=== SYSTEM RESOURCES ===")
    
    # GPU Check
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✓ CUDA available with {gpu_count} GPU(s)")
        
        # Test GPU access
        try:
            device = torch.cuda.current_device()
            print(f"✓ Current GPU device: {device}")
            
            # Basic GPU memory info
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
                print(f"GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
                
        except Exception as e:
            print(f"✗ GPU access error: {e}")
            return False
    else:
        print("✗ CUDA not available")
        return False
    
    # Memory check
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.available // (1024**3)}GB available / {memory.total // (1024**3)}GB total")
    
    return True

def quick_metainfo_check(base_dir):
    """Quick check for metainfo files without extensive fixing"""
    print("Quick metainfo check...")
    
    metainfo_pattern = os.path.join(base_dir, "**/_metainfo")
    metainfo_files = glob.glob(metainfo_pattern, recursive=True)
    
    if not metainfo_files:
        print("ℹ No metainfo files found - first run")
        return True
    
    # Just check first few files
    for metainfo_path in metainfo_files[:3]:
        try:
            with open(metainfo_path, 'r') as f:
                metainfo = json.load(f)
            print(f"✓ Metainfo OK: {os.path.dirname(metainfo_path)}")
        except Exception as e:
            print(f"⚠ Metainfo issue: {metainfo_path} - {e}")
            return False
    
    print(f"✓ Checked {min(3, len(metainfo_files))} metainfo files")
    return True


class TrainConfig:
    def __init__(self, save_dir, lr, max_epoch, metric_min_better, early_stop, patience, 
                 grad_clip, anneal_base, weight_decay=0.0, dropout_scale=1.0):
        self.save_dir = save_dir
        self.lr = lr
        self.max_epoch = max_epoch
        self.metric_min_better = metric_min_better
        self.patience = patience if early_stop else max_epoch + 1
        self.grad_clip = grad_clip
        self.anneal_base = anneal_base
        self.weight_decay = weight_decay
        self.dropout_scale = dropout_scale

    def __str__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)


def get_training_config(is_finetuning=False, freeze_strategy='none', custom_lr=None, custom_weight_decay=None):
    """Get configuration parameters based on training mode"""
    if is_finetuning:
        return {
            'lr': custom_lr if custom_lr else 0.0003,
            'max_epoch': 15,
            'early_stop': True,     
            'patience': 4,
            'grad_clip': 1.0,
            'anneal_base': 0.96,
            'batch_size': 16,
            'weight_decay': custom_weight_decay if custom_weight_decay else 1e-5,
            'dropout_scale': 0.8,
            'freeze_strategy': freeze_strategy
        }
    else:
        return {
            'lr': 0.001,
            'max_epoch': 20,
            'early_stop': False,
            'patience': 3,
            'grad_clip': 1.0,
            'anneal_base': 0.95,
            'batch_size': 16,
            'weight_decay': 0.0,
            'dropout_scale': 1.0,
            'freeze_strategy': 'none'
        }


class Trainer:
    def __init__(self, model, train_loader, valid_loader, config, save_dir, cdr=None, fold=None, 
                 is_finetuning=False, freeze_strategy='none', dataset_sizes=None):
        self.model = model
        self.config = config
        self.save_dir = save_dir
        self.is_finetuning = is_finetuning
        self.freeze_strategy = freeze_strategy
        self.dataset_sizes = dataset_sizes or {'train': 'unknown', 'valid': 'unknown'}
        
        print(f"\n=== TRAINER INITIALIZATION ===")
        print(f"Fine-tuning mode: {is_finetuning}")
        print(f"Freeze strategy: {freeze_strategy}")
        
        # Apply layer freezing if fine-tuning and strategy is not 'none'
        if is_finetuning and freeze_strategy != 'none':
            self.freeze_backbone_layers(model, freeze_strategy)
        elif is_finetuning:
            print("Fine-tuning ALL parameters (no freezing)")
        
        # Verify we have trainable parameters
        self.verify_trainable_params()
        
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
        
        # Enhanced monitoring for fine-tuning
        if is_finetuning:
            self.training_losses = []
            self.validation_losses = []
            self.learning_rates = []

    def freeze_backbone_layers(self, model, strategy='partial'):
        """Enhanced freezing strategy with better layer detection"""
        frozen_count = 0
        trainable_count = 0
        total_count = 0
        
        print(f"Applying freeze strategy: {strategy}")
        
        # Get all parameters with their full context
        all_params = list(model.named_parameters())
        total_count = len(all_params)
        
        if strategy == 'minimal':
            # Only freeze early embedding/positional layers
            freeze_patterns = ['embed', 'pos', 'position']
            for name, param in all_params:
                if any(pattern in name.lower() for pattern in freeze_patterns):
                    param.requires_grad = False
                    frozen_count += 1
                else:
                    param.requires_grad = True
                    trainable_count += 1
                    
        elif strategy == 'partial':
            # Freeze roughly first half, train second half
            freeze_count = total_count // 2
            for i, (name, param) in enumerate(all_params):
                if i < freeze_count:
                    param.requires_grad = False
                    frozen_count += 1
                else:
                    param.requires_grad = True
                    trainable_count += 1
                    
        elif strategy == 'aggressive':
            # Only train final layers (last 25%)
            train_count = max(1, total_count // 4)  # At least 1 layer
            freeze_count = total_count - train_count
            for i, (name, param) in enumerate(all_params):
                if i < freeze_count:
                    param.requires_grad = False
                    frozen_count += 1
                else:
                    param.requires_grad = True
                    trainable_count += 1
        
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Freeze strategy '{strategy}':")
        print(f"  Frozen layers: {frozen_count}/{total_count}")
        print(f"  Frozen parameters: {frozen_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable ratio: {trainable_params/(frozen_params+trainable_params)*100:.1f}%")

    def verify_trainable_params(self):
        """Verify we have trainable parameters"""
        trainable_params = []
        total_trainable_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
                total_trainable_params += param.numel()
        
        if len(trainable_params) == 0:
            raise RuntimeError(
                "No trainable parameters found! All parameters are frozen. "
                "Try using freeze_strategy='none' or 'minimal'."
            )
        
        print(f"✓ Verified: {len(trainable_params)} trainable parameter groups")
        print(f"✓ Total trainable parameters: {total_trainable_params:,}")
        return len(trainable_params)

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
        self.model.train()
        if self.train_loader.sampler is not None and self.local_rank != -1:
            self.train_loader.sampler.set_epoch(self.epoch)
        t_iter = tqdm(self.train_loader) if self._is_main_proc() else self.train_loader
        
        epoch_losses = []
        
        for batch_idx, batch in enumerate(t_iter):
            try:
                batch = self.to_device(batch, device)
                
                loss = self.train_step(batch, self.global_step)
                
                if not loss.requires_grad:
                    print(f"⚠ Warning: Loss doesn't require grad at batch {batch_idx}")
                    continue
                
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.optimizer.step()
                
                loss_item = loss.item()
                epoch_losses.append(loss_item)
                
                if hasattr(t_iter, 'set_postfix'):
                    t_iter.set_postfix(loss=f"{loss_item:.4f}")
                    
                self.global_step += 1
                
                if self.sched_freq == 'batch':
                    self.scheduler.step()
                    
            except RuntimeError as e:
                print(f"✗ Error at batch {batch_idx}: {e}")
                if "out of memory" in str(e).lower():
                    print("💾 GPU Out of Memory! Try reducing batch size.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                raise e
        
        # Record training metrics
        if self.is_finetuning and epoch_losses:
            avg_loss = np.mean(epoch_losses)
            self.training_losses.append(avg_loss)
            if self.scheduler:
                current_lr = self.scheduler.get_last_lr()[0]
                self.learning_rates.append(current_lr)
                print(f"Epoch {self.epoch}: Avg Loss = {avg_loss:.4f}, LR = {current_lr:.2e}")
                
        if self.sched_freq == 'epoch' and self.scheduler:
            self.scheduler.step()

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
        
        # judge
        ckpt_saved, save_path = False, None
        valid_metric = np.mean(metric_arr)
        
        # Record validation metrics
        if self.is_finetuning:
            self.validation_losses.append(valid_metric)
        
        if self._metric_better(valid_metric):
            self.patience = self.config.patience
            if self._is_main_proc():
                save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                module_to_save = self.model.module if self.local_rank == 0 else self.model
                torch.save(module_to_save, save_path)
                torch.save(module_to_save, os.path.join(self.save_dir, f'best.ckpt'))
                ckpt_saved = True
            self.last_valid_metric = valid_metric
        else:
            self.patience -= 1
            
        print(f"✓ Validation metric: {valid_metric:.4f} (best: {self.last_valid_metric:.4f}), patience: {self.patience}")
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
        try:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(f" Using device: {device}")
            self.model.to(device)
            
            # Print training summary
            print(f"\n=== TRAINING STARTED ===")
            print(f"Mode: {'Fine-tuning' if self.is_finetuning else 'From scratch'}")
            print(f"Dataset sizes: Train={self.dataset_sizes['train']}, Valid={self.dataset_sizes['valid']}")
            print(f"Max epochs: {self.config.max_epoch}")
            print(f"Learning rate: {self.config.lr}")
            print(f"Weight decay: {self.config.weight_decay}")
            print(f"Patience: {self.config.patience}")
            print("======================\n")
            
            # init writer
            if self._is_main_proc():
                os.makedirs(self.model_dir, exist_ok=True)
                with open(os.path.join(self.save_dir, 'train_config.json'), 'w') as fout:
                    json.dump(self.config.__dict__, fout, indent=2)

            # Training loop
            for _ in range(self.config.max_epoch):
                print(f'\n--- Epoch {self.epoch} ---')
                self._train_epoch(device)
                print('Validating...')
                ckpt_saved, save_path = self._valid_epoch(device)
                if ckpt_saved:
                    print(f'✓ Checkpoint saved to {save_path}')
                self.epoch += 1
                if self.patience <= 0:
                    print('Early stopping triggered')
                    break
            
            # Final summary for fine-tuning
            if self.is_finetuning:
                self.print_training_summary()
                
            print('Training completed successfully!')
            return True
            
        except Exception as e:
            print(f" Training failed with error: {e}")
            print("Full traceback:")
            traceback.print_exc()
            return False

    def print_training_summary(self):
        """Print summary of fine-tuning progress"""
        print(f"\n=== TRAINING SUMMARY ===")
        if self.training_losses:
            print(f"Initial training loss: {self.training_losses[0]:.4f}")
            print(f"Final training loss: {self.training_losses[-1]:.4f}")
            improvement = ((self.training_losses[0] - self.training_losses[-1])/self.training_losses[0]*100)
            print(f"Training loss improvement: {improvement:.1f}%")
        
        if self.validation_losses:
            print(f"Best validation loss: {min(self.validation_losses):.4f}")
            print(f"Final validation loss: {self.validation_losses[-1]:.4f}")
        
        print(f"Total epochs trained: {self.epoch}")
        print(f"Best epoch: {np.argmin(self.validation_losses) if self.validation_losses else 'unknown'}")
        print("========================\n")

    def get_optimizer(self):
        # Only optimize parameters that require gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters for optimizer!")
        
        optimizer = torch.optim.Adam(trainable_params, lr=self.config.lr, 
                                   weight_decay=self.config.weight_decay)
        print(f"✓ Optimizer created with {len(trainable_params)} parameter groups")
        return optimizer

    def get_scheduler(self, optimizer):
        lam = lambda epoch: self.config.anneal_base ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)
        return {
            'scheduler': scheduler,
            'frequency': 'epoch'
        }

    def train_step(self, batch, batch_idx):
        return self.share_forward(batch, batch_idx, 'train/' + self.ex_name)

    def valid_step(self, batch, batch_idx):
        return self.share_forward(batch, batch_idx, 'validation/' + self.ex_name, val=True)
    
    def share_forward(self, batch, batch_idx, _type, val=False):
        loss, snll, closs = self.model(
            batch['X'], batch['S'], batch['L'], batch['offsets']
        )
        ppl = snll.exp().item()
        return loss


def get_dataset_size(json_path):
    """Get dataset size from JSON file"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            return len(data)
    except Exception as e:
        print(f"⚠ Could not read dataset size from {json_path}: {e}")
        return 'unknown'


def load_pretrained_model(model, checkpoint_path):
    """Safely load pretrained weights with enhanced compatibility checks"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"🔄 Loading pretrained model from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if hasattr(checkpoint, 'state_dict'):
            pretrained_dict = checkpoint.state_dict()
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        
        model_dict = model.state_dict()
        
        # Filter compatible weights
        compatible_dict = {}
        size_mismatches = []
        
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    compatible_dict[k] = v
                else:
                    size_mismatches.append(f"{k}: {v.shape} vs {model_dict[k].shape}")
        
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict)
        
        print(f"✓ Successfully loaded {len(compatible_dict)}/{len(pretrained_dict)} layers")
        
        if size_mismatches:
            print(f"⚠ Skipped {len(size_mismatches)} layers due to size mismatch")
        
        return True
            
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return False


def main(args):
    print("Starting improved fine-tuning script...")
    
    # STEP 1: System checks
    if not check_system_resources():
        print(" System resource check failed!")
        return False
    
    # STEP 2: Quick metainfo check (much faster)
    base_dir = "/ibex/user/rioszemm"
    if not quick_metainfo_check(base_dir):
        print("⚠ Metainfo issues detected but continuing...")
    
    # STEP 3: Validate input files
    print("\n=== INPUT VALIDATION ===")
    for file_path, name in [(args.train_set, "Training set"), (args.valid_set, "Validation set")]:
        if not os.path.exists(file_path):
            print(f"{name} not found: {file_path}")
            return False
        print(f"✓ {name}: {file_path}")
    
    if args.pretrained_model and not os.path.exists(args.pretrained_model):
        print(f"Pretrained model not found: {args.pretrained_model}")
        return False
    
    # STEP 4: Configuration
    is_finetuning = args.pretrained_model is not None or args.finetune_mode
    freeze_strategy = args.freeze_strategy if is_finetuning else 'none'
    
    config_params = get_training_config(
        is_finetuning, 
        freeze_strategy,
        custom_lr=args.finetune_lr,
        custom_weight_decay=args.weight_decay
    )
    
    if args.batch_size is not None:
        config_params['batch_size'] = args.batch_size
    
    print(f"\n=== CONFIGURATION ===")
    print(f"Mode: {'FINE-TUNING' if is_finetuning else 'FROM SCRATCH'}")
    print(f"Freeze strategy: {freeze_strategy}")
    print(f"Learning rate: {config_params['lr']}")
    print(f"Batch size: {config_params['batch_size']}")
    print(f"Max epochs: {config_params['max_epoch']}")
    print(f"Weight decay: {config_params['weight_decay']}")
    
    # STEP 5: Dataset size analysis
    dataset_sizes = {
        'train': get_dataset_size(args.train_set),
        'valid': get_dataset_size(args.valid_set)
    }
    print(f"Dataset sizes: Train={dataset_sizes['train']}, Valid={dataset_sizes['valid']}")
    
    # STEP 6: Model initialization
    print("\n=== MODEL SETUP ===")
    try:
        model = ADesigner(embed_size=64, hidden_size=128, n_channel=4,
                         n_layers=3, cdr_type='3', alpha=0.1, dropout=0.1, args=None)
        print("✓ Model initialized")
        
        # Load pretrained weights if provided
        if args.pretrained_model:
            if not load_pretrained_model(model, args.pretrained_model):
                print("Failed to load pretrained model")
                return False
    except Exception as e:
        print(f"Model initialization failed: {e}")
        return False

    # STEP 7: Dataset initialization
    print("\n=== DATASET SETUP ===")
    try:
        _collate_fn = EquiAACDataset.collate_fn
        batch_size = config_params['batch_size']

        print("Loading training dataset...")
        train_dataset = EquiAACDataset(args.train_set, args.save_dir, num_entry_per_file=-1)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                                 num_workers=2, collate_fn=_collate_fn)  # Reduced workers

        print("Loading validation dataset...")
        valid_dataset = EquiAACDataset(args.valid_set, args.save_dir, num_entry_per_file=-1)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                                 num_workers=2, collate_fn=_collate_fn)  # Reduced workers
        print("✓ Datasets loaded successfully")
        
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        traceback.print_exc()
        return False

    # STEP 8: Training setup
    print("\n=== TRAINING SETUP ===")
    try:
        config = TrainConfig(
            save_dir=args.save_dir,
            lr=config_params['lr'],
            max_epoch=config_params['max_epoch'],
            metric_min_better=True,
            early_stop=config_params['early_stop'],
            patience=config_params['patience'],
            grad_clip=config_params['grad_clip'],
            anneal_base=config_params['anneal_base'],
            weight_decay=config_params['weight_decay'],
            dropout_scale=config_params['dropout_scale']
        )

        trainer = Trainer(model, train_loader, valid_loader, config, 
                         save_dir=args.save_dir, cdr=3, fold=args.fold, 
                         is_finetuning=is_finetuning, freeze_strategy=freeze_strategy,
                         dataset_sizes=dataset_sizes)

        print("✓ Trainer initialized")
        
    except Exception as e:
        print(f"❌ Trainer setup failed: {e}")
        traceback.print_exc()
        return False

    # STEP 9: Start training
    print("\n" + "="*50)
    print(" STARTING TRAINING")
    print("="*50)
    
    success = trainer.train()
    
    if success:
        print("Training completed successfully!")
        return True
    else:
        print(" Training failed!")
        return False


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Improved fine-tuning script with better debugging')
    parser.add_argument('--train_set', type=str, required=True, help='Path to training set')
    parser.add_argument('--valid_set', type=str, required=True, help='Path to validation set')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save directory')
    parser.add_argument('--fold', type=int, required=True, help='Fold number')

    # Fine-tuning specific arguments
    parser.add_argument('--pretrained_model', type=str, default=None, 
                       help='Path to pretrained model checkpoint for fine-tuning')
    parser.add_argument('--freeze_strategy', type=str, default='none',
                       choices=['none', 'minimal', 'partial', 'aggressive'],
                       help='Strategy for freezing layers during fine-tuning')
    parser.add_argument('--finetune_mode', action='store_true',
                       help='Enable fine-tuning mode with adjusted hyperparameters')

    # Additional fine-tuning control arguments
    parser.add_argument('--finetune_lr', type=float, default=None,
                       help='Custom learning rate for fine-tuning (overrides default)')
    parser.add_argument('--weight_decay', type=float, default=None,
                       help='Weight decay for regularization')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Custom batch size')

    args = parser.parse_args()
    
    # Run main function and exit with appropriate code
    success = main(args)
    sys.exit(0 if success else 1)