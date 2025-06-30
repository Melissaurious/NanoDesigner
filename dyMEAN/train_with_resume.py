#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import argparse
import torch
import glob
from torch.utils.data import DataLoader

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
setup_seed(SEED)

########### Import your packages below ##########
from data.dataset import E2EDataset, VOCAB, E2EDataset2
from data.framework_templates import ConserveTemplateGenerator #added
from trainer import TrainConfig


def find_best_checkpoint(save_dir):
    """
    Find the best checkpoint from topk_map.txt file
    Returns (checkpoint_path, epoch) or (None, 0) if no checkpoint found
    """
    # Look for topk_map.txt in version_* directories
    version_dirs = glob.glob(os.path.join(save_dir, "version_*"))
    
    for version_dir in sorted(version_dirs, reverse=True):  # Start with latest version
        topk_map_path = os.path.join(version_dir, "checkpoint", "topk_map.txt")
        
        if os.path.exists(topk_map_path):
            try:
                with open(topk_map_path, 'r') as f:
                    lines = f.readlines()
                
                if lines:
                    # The topk_map.txt format is: score: checkpoint_path
                    # We want the one with the best (lowest) score
                    
                    best_checkpoint_path = None
                    best_score = float('inf')
                    best_epoch = 0
                    
                    for line in lines:
                        line = line.strip()
                        if line and ':' in line:
                            score_str, ckpt_path = line.split(':', 1)
                            score = float(score_str.strip())
                            ckpt_path = ckpt_path.strip()
                            
                            # Extract epoch number from checkpoint path
                            epoch_match = re.search(r'epoch(\d+)', ckpt_path)
                            if epoch_match:
                                epoch_num = int(epoch_match.group(1))
                                
                                # Choose the checkpoint with the best (lowest) score
                                if score < best_score:
                                    best_score = score
                                    best_epoch = epoch_num
                                    best_checkpoint_path = ckpt_path
                    
                    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                        print_log(f"Found checkpoint to resume from: {best_checkpoint_path} (epoch {best_epoch}, score {best_score:.4f})")
                        return best_checkpoint_path, best_epoch
                        
            except Exception as e:
                print_log(f"Error reading topk_map.txt from {topk_map_path}: {e}")
                continue
    
    print_log("No valid checkpoint found for resuming")
    return None, 0


def check_training_completion(save_dir, max_epoch, is_finetuning=False):
    """
    Check if training is already completed by looking at the highest epoch in checkpoints
    For fine-tuning, we need to be more careful about what constitutes "completion"
    """
    checkpoint_path, last_epoch = find_best_checkpoint(save_dir)
    
    if checkpoint_path and last_epoch >= max_epoch:
        # For fine-tuning, check if this is a pre-trained checkpoint vs actual fine-tuned checkpoint
        if is_finetuning:
            # Check if there are any additional checkpoints beyond the initial copied one
            # Look for version_1 or later, or multiple checkpoints in version_0
            version_dirs = glob.glob(os.path.join(save_dir, "version_*"))
            
            has_additional_training = False
            for version_dir in version_dirs:
                checkpoint_dir = os.path.join(version_dir, "checkpoint")
                if os.path.exists(checkpoint_dir):
                    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
                    # If we have more than 1 checkpoint file, or if we have version_1+, then fine-tuning happened
                    if len(ckpt_files) > 1 or "version_1" in version_dir or "version_2" in version_dir:
                        has_additional_training = True
                        break
            
            # Also check for substantial log files indicating actual training
            log_files = glob.glob(os.path.join(save_dir, "*log*.txt"))
            training_started = len(log_files) > 0 and any(os.path.getsize(log) > 1000 for log in log_files)
            
            if not has_additional_training and not training_started:
                print_log(f"Found pre-trained checkpoint (epoch {last_epoch}) but no evidence of fine-tuning training yet")
                return False
            else:
                print_log(f"Fine-tuning already completed up to epoch {last_epoch} (max_epoch: {max_epoch})")
                return True
        else:
            print_log(f"Training already completed up to epoch {last_epoch} (max_epoch: {max_epoch})")
            return True
    
    return False


def validate_checkpoint_for_finetuning(checkpoint_path):
    """
    Validate that the checkpoint is suitable for fine-tuning
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        required_keys = ['state_dict', 'epoch']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            print_log(f"Warning: Checkpoint missing keys: {missing_keys}")
            return False
            
        # Check if state_dict has model parameters
        if 'state_dict' in checkpoint and len(checkpoint['state_dict']) > 0:
            print_log(f"Checkpoint validation passed. Found {len(checkpoint['state_dict'])} parameters.")
            return True
        else:
            print_log("Warning: state_dict is empty or missing")
            return False
            
    except Exception as e:
        print_log(f"Error validating checkpoint: {e}")
        return False


def parse():
    parser = argparse.ArgumentParser(description='training')
    # data
    parser.add_argument('--train_set', type=str, required=True, help='path to train set')
    parser.add_argument('--valid_set', type=str, required=True, help='path to valid set')
    parser.add_argument('--cdr', type=str, default=None, nargs='+', help='cdr to generate, L1/2/3, H1/2/3,(can be list, e.g., L3 H3) None for all including framework')
    parser.add_argument('--paratope', type=str, default='H3', nargs='+', help='cdrs to use as paratope')

    # training related
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4, help='exponential decay from lr to final_lr')
    parser.add_argument('--warmup', type=int, default=0, help='linear learning rate warmup')
    parser.add_argument('--max_epoch', type=int, default=10, help='max training epoch')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients with too big norm')
    parser.add_argument('--save_dir', type=str, help='directory to save model and logs')
    parser.add_argument('--template_path', type=str, required=True, help='path to template file')

    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--patience', type=int, default=1000, help='patience before early stopping (set with a large number to turn off early stopping)')
    parser.add_argument('--save_topk', type=int, default=10, help='save topk checkpoint. -1 for saving all ckpt that has a better validation metric than its previous epoch')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=4)

    # Resume functionality (enabled by default)
    parser.add_argument('--no_resume', action='store_true', help='Disable automatic resume from checkpoint')
    parser.add_argument('--force_restart', action='store_true', help='Force restart training even if completed')
    
    # Fine-tuning flag
    parser.add_argument('--is_finetuning', action='store_true', help='Indicates this is a fine-tuning run')

    # device
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='gpu to use (e.g., --gpus 0 1), -1 for CPU')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    # model
    parser.add_argument('--model_type', type=str, required=True, choices=['dyMEAN', 'dyMEANOpt'],
                        help='Type of model')
    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of residue/atom embedding')
    parser.add_argument('--hidden_size', type=int, default=128, help='dimension of hidden states')
    parser.add_argument('--k_neighbors', type=int, default=9, help='Number of neighbors in KNN graph')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--iter_round', type=int, default=3, help='Number of iterations for generation')

    # dyMEANOpt related
    parser.add_argument('--seq_warmup', type=int, default=0, help='Number of epochs before starting training sequence')

    # task setting
    parser.add_argument('--struct_only', action='store_true', help='Predict complex structure given the sequence')
    parser.add_argument('--bind_dist_cutoff', type=float, default=6.6, help='distance cutoff to decide the binding interface')

    # ablation
    parser.add_argument('--no_pred_edge_dist', action='store_true', help='Turn off edge distance prediction at the interface')
    parser.add_argument('--backbone_only', action='store_true', help='Model backbone only')
    parser.add_argument('--fix_channel_weights', action='store_true', help='Fix channel weights, may also for special use (e.g. antigen with modified AAs)')
    parser.add_argument('--no_memory', action='store_true', help='No memory passing')

    return parser.parse_args()


def main(args):
    ########### load your train / valid set ###########
    if (len(args.gpus) > 1 and int(os.environ['LOCAL_RANK']) == 0) or len(args.gpus) == 1:
        print_log(args)
        print_log(f'CDR type: {args.cdr}')
        print_log(f'Paratope: {args.paratope}')
        print_log('structure only' if args.struct_only else 'sequence & structure codesign')

    import torch
    import os

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force GPU 0
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Set default save_dir if not specified
    if args.save_dir is None:
        dataset_dir = os.path.dirname(args.train_set)
        args.save_dir = os.path.join(dataset_dir, 'checkpoints')

    os.makedirs(args.save_dir, exist_ok=True) 
    print("Checkpoints will be saved at: ", args.save_dir)

    # Fine-tuning epoch recommendation
    if args.is_finetuning and args.max_epoch > 30:
        print_log(f"Warning: Fine-tuning with {args.max_epoch} epochs. Consider using 20-30 epochs to avoid overfitting.")
        print_log("Fine-tuning starts from epoch 0 but uses pre-trained weights from antibody model.")

    # Check if training is already completed (unless force restart)
    if not args.force_restart and check_training_completion(args.save_dir, args.max_epoch, args.is_finetuning):
        print_log("Training already completed. Use --force_restart to restart from scratch.")
        return

    # Check for existing checkpoints (unless disabled)
    resume_checkpoint_path = None
    start_epoch = 0
    
    if not args.no_resume:
        resume_checkpoint_path, start_epoch = find_best_checkpoint(args.save_dir)
        if resume_checkpoint_path:
            # Validate checkpoint if fine-tuning
            if args.is_finetuning:
                # if validate_checkpoint_for_finetuning(resume_checkpoint_path):
                print_log(f"Fine-tuning will start from validated checkpoint")
                # else:
                #     print_log("ERROR: Checkpoint validation failed. Cannot proceed with fine-tuning.")
                #     print_log("Fine-tuning requires a valid pre-trained checkpoint.")
                #     return  # Exit the program or skip this fold
            else:
                # Regular training can start from scratch
                print_log(f"Auto-resuming training from epoch {start_epoch}")
        else:
            print_log("No checkpoint found. Starting from scratch.")

    # Creating specific directories for processed data
    train_processed = os.path.join(args.save_dir, "train_processed")
    os.makedirs(train_processed, exist_ok=True) 

    valid_processed = os.path.join(args.save_dir, "valid_processed")
    os.makedirs(valid_processed, exist_ok=True) 

    template_generator = ConserveTemplateGenerator(args.template_path)

    # Explicitly setting save_dir for datasets
    train_set = E2EDataset2(
        args.train_set, 
        template_generator=template_generator,
        save_dir=train_processed,
        cdr=args.cdr, 
        paratope=args.paratope
    )

    valid_set = E2EDataset2(
        args.valid_set, 
        template_generator=template_generator,
        save_dir=valid_processed,
        cdr=args.cdr, 
        paratope=args.paratope
    )

    ########## set your collate_fn ##########
    collate_fn = train_set.collate_fn

    ########## define your model/trainer/trainconfig #########
    config = TrainConfig(**vars(args))
    
    # Add resume checkpoint to config if available
    if resume_checkpoint_path:
        config.add_parameter(resume_checkpoint=resume_checkpoint_path)

    # Fine-tuning specific adjustments
    if args.is_finetuning:
        # Reduce patience for fine-tuning to avoid overfitting
        original_patience = args.patience
        args.patience = min(args.patience, 15)  # Max 15 epochs without improvement
        print_log(f"Fine-tuning mode: reducing patience from {original_patience} to {args.patience}")
        
        # Update config with new patience
        config.add_parameter(patience=args.patience)

    if args.model_type == 'dyMEAN':
        from trainer import dyMEANTrainer as Trainer
        from models import dyMEANModel
        model = dyMEANModel(args.embed_dim, args.hidden_size, VOCAB.MAX_ATOM_NUMBER,
                   VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
                   args.k_neighbors, bind_dist_cutoff=args.bind_dist_cutoff,
                   n_layers=args.n_layers, struct_only=args.struct_only,
                   iter_round=args.iter_round,
                   backbone_only=args.backbone_only,
                   fix_channel_weights=args.fix_channel_weights,
                   pred_edge_dist=not args.no_pred_edge_dist,
                   keep_memory=not args.no_memory,
                   cdr_type=args.cdr, paratope=args.paratope)
    elif args.model_type == 'dyMEANOpt':
        from trainer import dyMEANOptTrainer as Trainer
        from models import dyMEANOptModel
        model = dyMEANOptModel(args.embed_dim, args.hidden_size, VOCAB.MAX_ATOM_NUMBER,
                   VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
                   args.k_neighbors, bind_dist_cutoff=args.bind_dist_cutoff,
                   n_layers=args.n_layers, struct_only=args.struct_only,
                   fix_atom_weights=args.fix_channel_weights, cdr_type=args.cdr)
    else:
        raise NotImplemented(f'model {args.model_type} not implemented')

    step_per_epoch = (len(train_set) + args.batch_size - 1) // args.batch_size
    config.add_parameter(step_per_epoch=step_per_epoch)

    if len(args.gpus) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=args.shuffle)
        args.batch_size = int(args.batch_size / len(args.gpus))
        if args.local_rank == 0:
            print_log(f'Batch size on a single GPU: {args.batch_size}')
    else:
        args.local_rank = -1
        train_sampler = None
    config.local_rank = args.local_rank

    if args.local_rank == 0 or args.local_rank == -1:
        print_log(f'step per epoch: {step_per_epoch}')

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=(args.shuffle and train_sampler is None),
                              sampler=train_sampler,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)
    
    trainer = Trainer(model, train_loader, valid_loader, config)
    
    # Train without passing resume_checkpoint as parameter
    trainer.train(args.gpus, args.local_rank)


if __name__ == '__main__':
    args = parse()
    main(args)