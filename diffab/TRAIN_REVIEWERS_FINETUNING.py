import torch
torch.cuda.empty_cache()
import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import re
import sys
import time
import gc
import copy
# Ensure TF32 is enabled for better performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Add the DiffAB directory to path
sys.path.append('/ibex/user/rioszemm/NanobodiesProject/diffab')

from diffab.datasets import get_dataset
from diffab.models import get_model
from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.train import *
from diffab.datasets.sabdab_copy import SAbDabDataset
from torchvision.transforms import Compose
import time
import gc


def get_args():
    parser = argparse.ArgumentParser(description='Train DiffAb model for antibody/nanobody design')
    
    # Basic configuration
    parser.add_argument('--config', type=str, default="/ibex/user/rioszemm/NanobodiesProject/diffab/configs/train/codesign_single.yml", help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--auto_resume', action='store_true', default=True, help='Automatically resume from latest checkpoint')
    parser.add_argument('--logdir', type=str, default='/ibex/user/rioszemm/diffab/logs', help='Log directory')
    parser.add_argument('--processed_dir', type=str, required=True, help='Directory for processed data')
    
    # Dataset configuration
    parser.add_argument('--fold', type=str, required=True, help='Cross-validation fold number')
    parser.add_argument('--train_json', type=str, required=True, help='Path to training json file')
    parser.add_argument('--valid_json', type=str, required=True, help='Path to validation json file')
    parser.add_argument('--chothia_dir', type=str, default="/ibex/user/rioszemm/all_structures/chothia", help='Directory with structure files')
    parser.add_argument('--tsv_file', type=str, default="/ibex/user/rioszemm/Final_dataset_may_2025_paper/sabdab_summary_all-5.tsv", help='Path to SAbDab summary file')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=4, help='Override batch size from config')
    parser.add_argument('--max_iters', type=int, default=None, help='Override max iterations from config')
    parser.add_argument('--val_freq', type=int, default=None, help='Override validation frequency')
    parser.add_argument('--special_filter', default=True, help='Set true to use the produced datasets (json files)')
    parser.add_argument('--design_mode', choices=['single', 'multiple'], default='single', 
                        help='CDR design mode (single or multiple)')
    parser.add_argument('--keep_checkpoints', type=int, default=5, 
                        help='Number of recent checkpoints to keep')
    
    # System configuration
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loader workers')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')
    parser.add_argument('--tag', type=str, default='', help='Additional tag for logging')
    parser.add_argument('--reset', action='store_true', default=False, help='Reset cache')
    
    # Fine-tuning configuration
    parser.add_argument('--finetune', type=str, default=None, help='Path to model to finetune from')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    
    return parser.parse_args()



#ADDED
def cleanup_checkpoints(checkpoint_dir, keep=5, finetune=False, logger=None, finetune_interval=5000):
    """
    Keep only the most recent 'keep' checkpoints and delete others.
    If finetune is True, keep checkpoints at specified intervals (default: every 5000).
    """
    # Check if checkpoint directory exists first
    if not os.path.exists(checkpoint_dir):
        if logger:
            logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        else:
            print(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return  # Exit gracefully if directory doesn't exist
    
    # Get all checkpoint files
    try:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') and not f.startswith('checkpoint_nan_')]
    except OSError as e:
        if logger:
            logger.error(f"Error accessing checkpoint directory {checkpoint_dir}: {e}")
        else:
            print(f"Error accessing checkpoint directory {checkpoint_dir}: {e}")
        return
    
    # If no checkpoints found, nothing to clean up
    if not checkpoint_files:
        if logger:
            logger.info("No checkpoints found for cleanup")
        return
    
    # Extract iteration numbers and sort
    checkpoint_iters = []
    for f in checkpoint_files:
        match = re.match(r'(\d+)\.pt', f)
        if match:
            checkpoint_iters.append((int(match.group(1)), f))
    
    # Sort by iteration number (ascending for easier processing)
    checkpoint_iters.sort()
    
    if finetune:
        # FINE-TUNING MODE: Keep checkpoints at specified intervals + most recent
        if logger:
            logger.info(f"Fine-tuning mode: Keeping checkpoints every {finetune_interval} iterations")
        
        # Find the most recent checkpoint
        if checkpoint_iters:
            most_recent_iter = checkpoint_iters[-1][0]
        
        # Determine which checkpoints to keep
        checkpoints_to_keep = set()
        for iter_num, filename in checkpoint_iters:
            # Keep if it's a multiple of the interval OR it's the most recent
            if iter_num % finetune_interval == 0 or iter_num == most_recent_iter:
                checkpoints_to_keep.add(filename)
        
        # Delete checkpoints that are not in the keep set
        for iter_num, filename in checkpoint_iters:
            if filename not in checkpoints_to_keep:
                filepath = os.path.join(checkpoint_dir, filename)
                try:
                    os.remove(filepath)
                    if logger:
                        logger.info(f"Removed checkpoint: {filename}")
                    else:
                        print(f"Removed checkpoint: {filename}")
                except OSError as e:
                    if logger:
                        logger.error(f"Error removing checkpoint {filename}: {e}")
                    else:
                        print(f"Error removing checkpoint {filename}: {e}")
        
        if logger:
            logger.info(f"Kept {len(checkpoints_to_keep)} checkpoints during fine-tuning")
    
    else:
        # REGULAR TRAINING MODE: Keep only the most recent 'keep' checkpoints
        # Sort by iteration number (descending) for regular training
        checkpoint_iters.sort(reverse=True)
        
        # Keep only the most recent 'keep' checkpoints
        for i, (iter_num, filename) in enumerate(checkpoint_iters):
            if i >= keep:  # Delete older checkpoints
                filepath = os.path.join(checkpoint_dir, filename)
                try:
                    os.remove(filepath)
                    if logger:
                        logger.info(f"Removed old checkpoint: {filename}")
                    else:
                        print(f"Removed old checkpoint: {filename}")
                except OSError as e:
                    if logger:
                        logger.error(f"Error removing checkpoint {filename}: {e}")
                    else:
                        print(f"Error removing checkpoint {filename}: {e}")




def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the given directory
    Returns the path to the latest checkpoint or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    try:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                          if f.endswith('.pt') and not f.startswith('checkpoint_nan_')]
    except OSError:
        return None
    
    if not checkpoint_files:
        return None
    
    # Extract iteration numbers and find the highest
    latest_iter = -1
    latest_file = None
    
    for f in checkpoint_files:
        match = re.match(r'(\d+)\.pt', f)
        if match:
            iter_num = int(match.group(1))
            if iter_num > latest_iter:
                latest_iter = iter_num
                latest_file = f
    
    if latest_file:
        return os.path.join(checkpoint_dir, latest_file)
    return None

# ADDED MELISSA FOR FINETUNNIG
def freeze_layers(model, freeze_patterns, logger):
    """
    Freeze layers based on pattern matching
    """
    if not freeze_patterns:
        return
        
    frozen_count = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        for pattern in freeze_patterns:
            if pattern in name:
                param.requires_grad = False
                frozen_count += 1
                logger.info(f'Freezing parameter: {name}')
                break
    
    logger.info(f'Frozen {frozen_count}/{total_params} parameters for fine-tuning')


def safe_gpu_cleanup():
    """Safely clean up GPU memory"""
    try:
        if torch.cuda.is_available():
            # First, try gentle cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            # Check if memory is still high
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            usage_percent = (allocated / total) * 100
            
            if usage_percent > 50:  # If more than 50% memory used
                logger.warning(f"High GPU memory usage detected: {usage_percent:.1f}%")
                
                # Force more aggressive cleanup
                for obj in gc.get_objects():
                    if torch.is_tensor(obj):
                        if obj.is_cuda:
                            del obj
                
                torch.cuda.empty_cache()
                gc.collect()
                
                # Final check
                new_allocated = torch.cuda.memory_allocated(0)
                new_usage = (new_allocated / total) * 100
                logger.info(f"After cleanup: {new_usage:.1f}% GPU memory used")
                
    except Exception as e:
        logger.error(f"Error during GPU cleanup: {e}")
        # Fall back to more drastic measures only if needed
        try:
            os.system('nvidia-smi --gpu-reset')
        except:
            logger.error("Failed to reset GPU via nvidia-smi")


def main():
    # Parse arguments
    args = get_args()

    # Clear GPU cache first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    safe_gpu_cleanup()


    start_time = time.time()
    
    # Load configs
    config, config_name = load_config(args.config)

    # Important: Handle fine-tuning config
    if args.finetune is not None:
        if 'finetune' in config:
            logger_temp = get_logger('config', None)  # Temporary logger for config messages
            logger_temp.info('Using fine-tuning configuration section')
            # Store original train config
            original_train_config = copy.deepcopy(config.train)
            # Replace train config with finetune config
            config.train = config.finetune
        else:
            logger_temp = get_logger('config', None)
            logger_temp.warning('Fine-tuning mode but no finetune section found. Using train config.')
    

    seed_all(config.train.seed)



    # Override config with command line arguments if provided
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.max_iters is not None:
        config.train.max_iters = args.max_iters
    if args.val_freq is not None:
        config.train.val_freq = args.val_freq
    if args.lr is not None:
        config.train.optimizer.lr = args.lr
    
    # Set up output directory
    if args.output_dir:
        current_folder = args.output_dir
    else:
        current_folder = f"/ibex/user/rioszemm/NB_AB_DIFFAB_fold_{args.fold}"


    
    # Set up logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
        
        # Fix: Add _fine_tuned suffix when fine-tuning
        if args.finetune is not None:
            ckpt_dir = current_folder + f"/fold_{args.fold}_fine_tuned/checkpoints"
        else:
            ckpt_dir = current_folder + f"/fold_{args.fold}/checkpoints"
            
        if not os.path.exists(ckpt_dir): 
            os.makedirs(ckpt_dir)
        
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        
        # Copy config file
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    logger.info(args)
    logger.info(config)

    
    # Set up processed directory
    processed_dir = os.path.join(current_folder, f"fold_{args.fold}/processed_entries")
    os.makedirs(processed_dir, exist_ok=True)
    
    
    args.reset = False
    if args.resume:
        reset_dataset = False  # Don't reset data when resuming
        print("Resuming training - using existing processed data")




    logger.info('Loading datasets...')
    # Create datasets
    if args.design_mode == "single":
        train_dataset = SAbDabDataset(
            summary_path=args.tsv_file,
            chothia_dir=args.chothia_dir,
            json_file=args.train_json,
            fold=args.fold,
            processed_dir= args.processed_dir, 
            split='train',
            split_seed=2022,
            transform = get_dataset({'type': 'sabdab', 'summary_path': args.tsv_file, 
                                    'chothia_dir': args.chothia_dir, 'processed_dir': args.processed_dir, 
                                    'split': 'train', 'transform': [{'type': 'mask_single_cdr'}, {'type': 'merge_chains'}, {'type': 'patch_around_anchor'}]}),
            reset=args.reset,
            # special_filter=args.special_filter
            special_filter=True
        )

        val_dataset = SAbDabDataset(
            summary_path=args.tsv_file,
            chothia_dir=args.chothia_dir,
            json_file=args.valid_json,
            fold=args.fold,
            processed_dir= args.processed_dir,
            split='val',
            split_seed=2022,
            transform=get_dataset({'type': 'sabdab', 'summary_path': args.tsv_file,
                                    'chothia_dir':args.chothia_dir, 'processed_dir': args.processed_dir,
                                        'split': 'val', 'transform': [{'type': 'mask_single_cdr', 'selection': 'CDR3'}, {'type': 'merge_chains'}, {'type': 'patch_around_anchor'}]}),
            # special_filter=args.special_filter
            reset=args.reset,
            special_filter=True
        )
    elif args.design_mode == "multiple":

        train_dataset = SAbDabDataset(
            summary_path=args.tsv_file,
            chothia_dir=args.chothia_dir,
            processed_dir= args.processed_dir, 
            split='train',
            split_seed=2022,
            transform = get_dataset({'type': 'sabdab', 'summary_path': args.tsv_file, 
                                    'chothia_dir': args.chothia_dir, 'processed_dir': args.processed_dir, 
                                    'split': 'train', 'transform': [{'type': 'mask_multiple_cdrs'}, {'type': 'merge_chains'}, {'type': 'patch_around_anchor'}]}),
            reset=args.reset,
            special_filter=args.special_filter
        )

        val_dataset = SAbDabDataset(
            summary_path=args.tsv_file,
            chothia_dir=args.chothia_dir,
            processed_dir= args.processed_dir,
            split='val',
            split_seed=2022,
            transform=get_dataset({'type': 'sabdab', 'summary_path': args.tsv_file,
                                    'chothia_dir':args.chothia_dir, 'processed_dir': args.processed_dir,
                                        'split': 'val', 'transform': [{'type': 'mask_single_cdr', 'selection': 'CDR3'}, {'type': 'merge_chains'}, {'type': 'patch_around_anchor'}]}),
            reset=args.reset,
            special_filter=args.special_filter
        )                                                                   #Mask only CDR3 at validation

    

    logger.info('Loading dataset...')
    # train_dataset = get_dataset(config.dataset.train)
    # val_dataset = get_dataset(config.dataset.val)
    ### comment these two lines if personalized running
    train_iterator = inf_iterator(DataLoader(
        train_dataset, 
        batch_size=config.train.batch_size, 
        collate_fn=PaddingCollate(), 
        shuffle=True,
        num_workers=args.num_workers
    ))
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, collate_fn=PaddingCollate(), shuffle=False, num_workers=args.num_workers)
    print(f"Validation dataset size: {len(val_dataset)}")
    print(val_dataset)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))


    # Build model
    logger.info('Building model...')
    try:
        model = get_model(config.model).to(args.device)
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("GPU out of memory during model loading. Trying to free memory...")
            torch.cuda.empty_cache()
            # Try again
            model = get_model(config.model).to(args.device)
        else:
            raise e



    logger.info(f'Number of parameters: {count_parameters(model)}')
    
    # Resume or finetune
    # FIXED VERSION - Replace your checkpoint loading section with this:

    # Resume or finetune
    if args.resume is not None:
        # RESUME MODE: Load everything including optimizer state
        ckpt_path = args.resume
        logger.info(f'Loading checkpoint for resume: {ckpt_path}')
        
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        
        # Create and load optimizer state
        logger.info('RESUME MODE - Creating and loading optimizer/scheduler states...')
        optimizer = get_optimizer(config.train.optimizer, model)
        scheduler = get_scheduler(config.train.scheduler, optimizer)
        
        it_first = ckpt['iteration'] + 1
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        
        # Override learning rate if specified
        if args.lr is not None:
            logger.info(f'Overriding learning rate to {args.lr}')
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

    # elif args.finetune is not None:
    #     # FINE-TUNING MODE: Load model weights only, create fresh optimizer
    #     ckpt_path = args.finetune
    #     logger.info(f'Loading checkpoint for fine-tuning: {ckpt_path}')
    #     logger.info('FINE-TUNING MODE')
        
    #     # Load only what we need (model weights) to CPU first
    #     ckpt = torch.load(ckpt_path, map_location='cpu')
        
    #     # Extract the iteration number from the checkpoint
    #     if 'iteration' in ckpt:
    #         pretrained_iters = ckpt['iteration']
    #         it_first = pretrained_iters + 1  # Continue from next iteration internally
    #         logger.info(f'Resuming fine-tuning from internal iteration {it_first}')
            
    #         # 🛠️ FIX: Make fine-tuning max_iters additive
    #         original_max_iters = config.train.max_iters
    #         config.train.max_iters = pretrained_iters + original_max_iters
    #         logger.info(f'Fine-tuning: Running for {original_max_iters} additional iterations')
    #         logger.info(f'Total internal iterations will be: {pretrained_iters} + {original_max_iters} = {config.train.max_iters}')
            
    #         # NEW: Track fine-tuning iterations separately for checkpoint naming
    #         finetune_start_iter = pretrained_iters + 1
            
    #     else:
    #         # If no iteration info, extract from filename
    #         import re
    #         match = re.search(r'(\d+)\.pt$', ckpt_path)
    #         if match:
    #             pretrained_iters = int(match.group(1))
    #             it_first = pretrained_iters + 1
    #             logger.info(f'Extracted iteration {it_first} from filename')
                
    #             original_max_iters = config.train.max_iters
    #             config.train.max_iters = pretrained_iters + original_max_iters
    #             logger.info(f'Fine-tuning: Running for {original_max_iters} additional iterations')
    #             logger.info(f'Total internal iterations will be: {pretrained_iters} + {original_max_iters} = {config.train.max_iters}')
                
    #             # NEW: Track fine-tuning iterations separately
    #             finetune_start_iter = pretrained_iters + 1
    #         else:
    #             it_first = 1
    #             finetune_start_iter = 1
    #             logger.info('Could not determine iteration, starting from 1')
        
    #     # Load model weights while still on CPU
    #     model.load_state_dict(ckpt['model'])
        
    #     # Clear checkpoint from memory - we don't need optimizer/scheduler states
    #     del ckpt
    #     torch.cuda.empty_cache()
        
    #     # Apply layer freezing while model is still on CPU
    #     if hasattr(config.train, 'freeze_layers') and config.train.freeze_layers:
    #         logger.info('Applying layer freezing...')
    #         freeze_layers(model, config.train.freeze_layers, logger)
        
    #     # Now move model to GPU
    #     logger.info('Moving model to GPU...')
    #     model = model.to(args.device)
        
    #     # Create fresh optimizer and scheduler (these will be much smaller)
    #     logger.info('Creating fresh optimizer and scheduler')
    #     optimizer = get_optimizer(config.train.optimizer, model)
    #     scheduler = get_scheduler(config.train.scheduler, optimizer)

    #     # FORCE the learning rate to be correct
    #     target_lr = config.train.optimizer.lr
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = target_lr
    #     logger.info(f'Set learning rate to: {target_lr}')

    #     # Reset scheduler state completely for fine-tuning
    #     if config.train.scheduler.type == 'plateau':
    #         scheduler.best = float('inf')
    #         scheduler.num_bad_epochs = 0
    #         if hasattr(scheduler, 'cooldown_counter'):
    #             scheduler.cooldown_counter = 0
    #         if hasattr(scheduler, '_last_lr'):
    #             scheduler._last_lr = [target_lr]
    #         logger.info("Reset scheduler state for fine-tuning")

    #     # ADDED VERIFICATION LOGIC
    #     logger.info("=== OPTIMIZER VERIFICATION ===")
    #     for i, param_group in enumerate(optimizer.param_groups):
    #         logger.info(f"Param group {i}: lr = {param_group['lr']}")
            
    #     # Additional verification - this should now be redundant but kept for safety
    #     if optimizer.param_groups[0]['lr'] == 0 or optimizer.param_groups[0]['lr'] != config.train.optimizer.lr:
    #         logger.warning(f"Learning rate mismatch! Expected: {config.train.optimizer.lr}, Got: {optimizer.param_groups[0]['lr']}")
    #         logger.warning("Setting to fine-tuning rate...")
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = config.train.optimizer.lr
    #         logger.info(f"Learning rate after fix: {optimizer.param_groups[0]['lr']}")
        
    #     logger.info(f'Fine-tuning learning rate: {config.train.optimizer.lr}')
        
    #     # If learning rate is overridden via command line, apply it
    #     if args.lr is not None:
    #         logger.info(f'Overriding learning rate to {args.lr}')
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = args.lr

    # else:
    #     # Normal training from scratch
    #     logger.info('Starting training from scratch')
    #     optimizer = get_optimizer(config.train.optimizer, model)
    #     scheduler = get_scheduler(config.train.scheduler, optimizer)
    #     it_first = 1


    # Resume or finetune - MODIFIED VERSION WITH AUTO-RESUME
# Replace your fine-tuning section with this corrected version:

    elif args.finetune is not None:
        # FINE-TUNING MODE: Load model weights only, create fresh optimizer
        ckpt_path = args.finetune
        logger.info(f'Loading checkpoint for fine-tuning: {ckpt_path}')
        logger.info('FINE-TUNING MODE')
        
        # Check if we should auto-resume fine-tuning from latest checkpoint
        fine_tuned_ckpt_dir = current_folder + f"/fold_{args.fold}_fine_tuned/checkpoints"
        latest_finetuned_ckpt = find_latest_checkpoint(fine_tuned_ckpt_dir)
        
        if latest_finetuned_ckpt:
            logger.info(f'Found existing fine-tuned checkpoint: {latest_finetuned_ckpt}')
            logger.info('AUTO-RESUMING FINE-TUNING from latest checkpoint')
            
            # Load the latest fine-tuned checkpoint (full state)
            ckpt = torch.load(latest_finetuned_ckpt, map_location=args.device)
            model.load_state_dict(ckpt['model'])
            
            # Create and load optimizer/scheduler state (resume mode for fine-tuning)
            optimizer = get_optimizer(config.train.optimizer, model)
            scheduler = get_scheduler(config.train.scheduler, optimizer)
            
            # FIXED: Use checkpoint_iteration for resuming fine-tuning
            if 'checkpoint_iteration' in ckpt:
                it_first = ckpt['checkpoint_iteration'] + 1
                logger.info(f'Resuming fine-tuning from checkpoint iteration {it_first}')
            else:
                # Fallback: extract from filename
                import re
                match = re.search(r'(\d+)\.pt$', latest_finetuned_ckpt)
                if match:
                    it_first = int(match.group(1)) + 1
                else:
                    it_first = 1
                logger.info(f'Resuming fine-tuning from extracted iteration {it_first}')
            
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            
            # Override learning rate if specified
            if args.lr is not None:
                logger.info(f'Overriding learning rate to {args.lr}')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                    
        else:
            # No existing fine-tuned checkpoint, start fresh fine-tuning
            logger.info('No existing fine-tuned checkpoint found, starting fresh fine-tuning')
            
            # Load only model weights from pre-trained checkpoint
            ckpt = torch.load(ckpt_path, map_location='cpu')
            
            # FIXED: Always start fine-tuning from iteration 1
            it_first = 1
            logger.info('Starting fresh fine-tuning from iteration 1')
            logger.info(f'Fine-tuning will run for {config.train.max_iters} iterations')
            
            # Load model weights while still on CPU
            model.load_state_dict(ckpt['model'])
            
            # Clear checkpoint from memory
            del ckpt
            torch.cuda.empty_cache()
            
            # Apply layer freezing while model is still on CPU
            if hasattr(config.train, 'freeze_layers') and config.train.freeze_layers:
                logger.info('Applying layer freezing...')
                freeze_layers(model, config.train.freeze_layers, logger)
            
            # Now move model to GPU
            logger.info('Moving model to GPU...')
            model = model.to(args.device)
            
            # Create fresh optimizer and scheduler
            logger.info('Creating fresh optimizer and scheduler for fine-tuning')
            optimizer = get_optimizer(config.train.optimizer, model)
            scheduler = get_scheduler(config.train.scheduler, optimizer)

            # Set correct learning rate
            target_lr = config.train.optimizer.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = target_lr
            logger.info(f'Set fine-tuning learning rate to: {target_lr}')

            # Reset scheduler state for fine-tuning
            if config.train.scheduler.type == 'plateau':
                scheduler.best = float('inf')
                scheduler.num_bad_epochs = 0
                if hasattr(scheduler, 'cooldown_counter'):
                    scheduler.cooldown_counter = 0
                if hasattr(scheduler, '_last_lr'):
                    scheduler._last_lr = [target_lr]
                logger.info("Reset scheduler state for fine-tuning")

            # If learning rate is overridden via command line, apply it
            if args.lr is not None:
                logger.info(f'Overriding learning rate to {args.lr}')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr

    else:
        # REGULAR TRAINING MODE with AUTO-RESUME
        regular_ckpt_dir = current_folder + f"/fold_{args.fold}/checkpoints"
        latest_regular_ckpt = find_latest_checkpoint(regular_ckpt_dir)
        
        if latest_regular_ckpt and not args.resume:
            # AUTO-RESUME from latest checkpoint
            logger.info(f'Found existing checkpoint: {latest_regular_ckpt}')
            logger.info('AUTO-RESUMING from latest checkpoint')
            
            ckpt = torch.load(latest_regular_ckpt, map_location=args.device)
            model.load_state_dict(ckpt['model'])
            
            # Create and load optimizer state
            logger.info('RESUME MODE - Creating and loading optimizer/scheduler states...')
            optimizer = get_optimizer(config.train.optimizer, model)
            scheduler = get_scheduler(config.train.scheduler, optimizer)
            
            it_first = ckpt['iteration'] + 1
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            
            # Override learning rate if specified
            if args.lr is not None:
                logger.info(f'Overriding learning rate to {args.lr}')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                    
            logger.info(f'Auto-resumed from iteration {it_first}')
            
        elif args.resume is not None:
            # EXPLICIT RESUME MODE (keep existing functionality)
            ckpt_path = args.resume
            logger.info(f'Loading checkpoint for explicit resume: {ckpt_path}')
            
            ckpt = torch.load(ckpt_path, map_location=args.device)
            model.load_state_dict(ckpt['model'])
            
            # Create and load optimizer state
            logger.info('RESUME MODE - Creating and loading optimizer/scheduler states...')
            optimizer = get_optimizer(config.train.optimizer, model)
            scheduler = get_scheduler(config.train.scheduler, optimizer)
            
            it_first = ckpt['iteration'] + 1
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            
            # Override learning rate if specified
            if args.lr is not None:
                logger.info(f'Overriding learning rate to {args.lr}')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
        else:
            # START FROM SCRATCH
            logger.info('Starting training from scratch')
            optimizer = get_optimizer(config.train.optimizer, model)
            scheduler = get_scheduler(config.train.scheduler, optimizer)
            it_first = 1


    # Make sure to zero gradients before training
    optimizer.zero_grad()

    # Add one final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Debug: Verify everything is set up correctly
    logger.info("=== FINAL SETUP VERIFICATION ===")
    logger.info(f"Training mode: {'Resume' if args.resume else 'Fine-tune' if args.finetune else 'From scratch'}")
    logger.info(f"Starting iteration: {it_first}")
    logger.info(f"Max iterations: {config.train.max_iters}")
    logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    logger.info(f"Optimizer type: {type(optimizer).__name__}")
    logger.info(f"Scheduler type: {type(scheduler).__name__}")



    def train(it):
        time_start = current_milli_time()
        model.train()
        
        # PERIODIC LEARNING RATE CHECK (ADDED)
        if it % 50 == 1:  # Check every 50 iterations
            actual_lr = optimizer.param_groups[0]['lr']
            expected_lr = config.train.optimizer.lr
            
            logger.info(f"=== Learning Rate Check at iteration {it}:")
            logger.info(f"    Current LR: {actual_lr}")
            logger.info(f"    Expected LR: {expected_lr}")
            
            if actual_lr == 0:
                logger.warning("Learning rate is 0! This means the model is not learning.")
                logger.info("Checking if scheduler has reduced it...")
                
                # Check scheduler state
                if hasattr(scheduler, '_last_lr'):
                    logger.info(f"    Scheduler last_lr: {scheduler._last_lr}")
                if hasattr(scheduler, 'best'):
                    logger.info(f"    Scheduler best: {scheduler.best}")
                    logger.info(f"    Scheduler bad epochs: {scheduler.num_bad_epochs}")
        
        # Prepare data
        try:
            batch = recursive_to(next(train_iterator), args.device)
            
            # Validate batch before passing to model
            if batch is None:
                logger.error(f"Received None batch in iteration {it}, skipping")
                return  # Use return instead of continue since we're in a function
                
            # Check if required keys exist
            if 'generate_flag' not in batch:
                logger.error(f"Batch missing 'generate_flag' in iteration {it}, skipping")
                return  # Use return instead of continue
                
            # Forward pass
            loss_dict = model(batch)
            loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
            loss_dict['overall'] = loss
            time_forward_end = current_milli_time()
            
            # Backward pass
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            time_backward_end = current_milli_time()
            
            # Logging
            log_losses(loss_dict, it, 'train', logger, writer, others={
                'grad': orig_grad_norm,
                'lr': optimizer.param_groups[0]['lr'],
                'time_forward': (time_forward_end - time_start) / 1000,
                'time_backward': (time_backward_end - time_forward_end) / 1000,
            })
            
            if not torch.isfinite(loss):
                logger.error('NaN or Inf detected.')
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    'batch': recursive_to(batch, 'cpu'),
                }, os.path.join(log_dir, f'checkpoint_nan_{it}.pt'))
                raise KeyboardInterrupt()
        
        except Exception as e:
            logger.error(f"Error in training iteration {it}: {e}")
            # Optionally print traceback for debugging
            import traceback
            logger.error(traceback.format_exc())
            # Skip this iteration but don't crash the whole training
            return
    
    # Validation function
    def validate(it):
        loss_tape = ValidationLossTape()
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                # Prepare data
                batch = recursive_to(batch, args.device)
                
                # Forward pass
                loss_dict = model(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                loss_dict['overall'] = loss
                loss_tape.update(loss_dict, 1)
        
        avg_loss = loss_tape.log(it, logger, writer, 'val')
        
        # Update scheduler
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        
        return avg_loss
    
    try:
        for it in range(it_first, config.train.max_iters + 1):
            train(it)
            
            if it % config.train.val_freq == 0:
                avg_val_loss = validate(it)
                
                if not args.debug:
                    # For fine-tuning: always use the actual training iteration (starts from 1)
                    checkpoint_iter = it
                    
                    if args.finetune is not None:
                        logger.info(f'Saving fine-tuning checkpoint {checkpoint_iter}')
                    else:
                        logger.info(f'Saving regular training checkpoint {checkpoint_iter}')
                    
                    ckpt_path = os.path.join(ckpt_dir, f'{checkpoint_iter}.pt')
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,  # This is the training iteration (starts from 1 for fine-tuning)
                        'checkpoint_iteration': checkpoint_iter,  # Same as iteration now
                        'avg_val_loss': avg_val_loss,
                        'is_finetuned': args.finetune is not None,
                        'pretrained_from': args.finetune if args.finetune else None,
                    }, ckpt_path)
                    
                    # Clean up old checkpoints
                    cleanup_checkpoints(ckpt_dir,
                                    keep=args.keep_checkpoints,
                                    finetune=args.finetune is not None,
                                    logger=logger,
                                    finetune_interval=5000)
            
            # Free up memory
            gc.collect()
            if it % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except KeyboardInterrupt:
        logger.info('Training terminated by user')

    
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Total training time: {execution_time:.2f} seconds")
    
    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()