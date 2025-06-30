import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import sys
sys.path.append('/home/rioszemm/NanobodiesProject/diffab')

from diffab.datasets import get_dataset
from diffab.models import get_model
from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.train import *
# from diffab.datasets import SAbDabDataset  me
from diffab.datasets.sabdab import SAbDabDataset
from torchvision.transforms import Compose
import time
import wandb


if __name__ == '__main__':
    start_time = time.time()
    wandb.login()
    
    now_training = "nano_fold_0"
    fold = "0"
    wandb.init(project=f"diffab-training_{now_training}")
    trainin = "nano_fold_0"
    # wandb.init(project="diffab-training_antibodies_final_4")
    # wandb.init(project="diffab-training_Ab_Nb_final_4")


    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default="/home/rioszemm/NanobodiesProject/diffab/configs/train/codesign_single.yml")
    parser.add_argument('--logdir', type=str, default='/ibex/user/rioszemm/diffab/logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)  # TO CONTINUE TRAINING FROM LAST CKPT
    # parser.add_argument('--resume', type=str, default='/ibex/user/rioszemm/diffab/logs/checkpoints_antibodies_2/135000.pt')
    parser.add_argument('--finetune', type=str, default=None)
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # print("config", config)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
        # ckpt_dir = os.path.join(log_dir, 'checkpoints') 
        ckpt_dir =  f"/ibex/user/rioszemm/NANOBODY_DATASET_CLUSTERED_Ag_DIFFAB/fold_{fold}/checkpoints_{now_training}" # check I am providing the full path to the last ckpt
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)


    # processed_dir = "/ibex/user/rioszemm/diffab/data/processed_nanobodies_2"
    # processed_dir = "processed_dir': '/ibex/user/rioszemm/diffab/data/processed_Ab_Nb_2"
    processed_dir = f"/ibex/user/rioszemm/NANOBODY_DATASET_CLUSTERED_Ag_DIFFAB/fold_{fold}/processed_{now_training}"
    tsv_file = '/ibex/user/rioszemm/all_structures/sabdab_summary_all-5.tsv'
    chothia_dir = '/ibex/user/rioszemm/all_structures/chothia'

    # Training dataset
    # train_transform = get_dataset({'type': 'sabdab', 'summary_path': tsv_file, 
    #                                'chothia_dir': chothia_dir, 'processed_dir': processed_dir, 
    #                                'split': 'train', 'transform': [{'type': 'mask_single_cdr'}, {'type': 'merge_chains'}, {'type': 'patch_around_anchor'}]})
    

    train_dataset = SAbDabDataset(
        summary_path=tsv_file,
        chothia_dir=chothia_dir,
        processed_dir= processed_dir, 
        split='train',
        split_seed=2022,
        # transform=[{'type': 'mask_single_cdr'}, {'type': 'merge_chains'}, {'type': 'patch_around_anchor'}],
        transform = get_dataset({'type': 'sabdab', 'summary_path': tsv_file, 
                                   'chothia_dir': chothia_dir, 'processed_dir': processed_dir, 
                                   'split': 'train', 'transform': [{'type': 'mask_single_cdr'}, {'type': 'merge_chains'}, {'type': 'patch_around_anchor'}]}),
        reset=False,
        special_filter=True
    )


    # Validation dataset
    # val_transform = get_dataset({'type': 'sabdab', 'summary_path': tsv_file,
    #                               'chothia_dir':chothia_dir, 'processed_dir': processed_dir,
    #                                 'split': 'val', 'transform': [{'type': 'mask_single_cdr', 'selection': 'CDR3'}, {'type': 'merge_chains'}, {'type': 'patch_around_anchor'}]})


    val_dataset = SAbDabDataset(
        summary_path=tsv_file,
        chothia_dir=chothia_dir,
        processed_dir= processed_dir,
        split='val',
        split_seed=2022,
        transform=get_dataset([{'type': 'mask_single_cdr', 'selection': 'CDR3'}, {'type': 'merge_chains'}, {'type': 'patch_around_anchor'}]),
        reset=False,
        special_filter=True
    )



    # Data
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
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # Model
    logger.info('Building model...')
    model = get_model(config.model).to(args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))

    # Optimizer & scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    # Resume
    if args.resume is not None or args.finetune is not None:
        ckpt_path = args.resume if args.resume is not None else args.finetune
        logger.info('Resuming from checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device)
        it_first = ckpt['iteration']  + 1
        model.load_state_dict(ckpt['model'])
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])


    # Train
    def train(it):
        time_start = current_milli_time()
        model.train()

        # Prepare data
        batch = recursive_to(next(train_iterator), args.device)

        # Forward
        # if args.debug: torch.set_anomaly_enabled(True)
        loss_dict = model(batch)
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        loss_dict['overall'] = loss
        time_forward_end = current_milli_time()

        # Backward
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

        # wandb.log(loss_dict)

        if not torch.isfinite(loss):
            logger.error('NaN or Inf detected.')
            torch.save({
                'config': config,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iteration': it,
                'batch': recursive_to(batch, 'cpu'),
            }, os.path.join(log_dir, 'checkpoint_nan_%d.pt' % it))
            raise KeyboardInterrupt()

    # Validate
    def validate(it):
        loss_tape = ValidationLossTape()
        # metrics = []  # Added, this will be a list of dictionaries, each sample will be evaluated Gt vs Predicted
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                # print("batch", batch)
                # Prepare data
                batch = recursive_to(batch, args.device) #moves the batch of data to the specified device (e.g., GPU) for computation
                # Forward
                loss_dict = model(batch) #computes the loss values for the batch by passing it through the model
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)  # overall loss by summing the individual losses from the loss_dict dictionary
                loss_dict['overall'] = loss
                loss_tape.update(loss_dict, 1)

                # Generate predicted structure
                # predicted_structure = model.sample(batch)

                # #Get ground truth sequence
                # ground_truth_structure = batch['ground_truth_structure']


        avg_loss = loss_tape.log(it, logger, writer, 'val')
        # Trigger scheduler
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        return avg_loss

    try:
        for it in range(it_first, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0:   #val_freq: 1000  every iterations a validation step will be conducted
                avg_val_loss = validate(it)

                if not args.debug:
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        'avg_val_loss': avg_val_loss,
                    }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time taken: {execution_time} seconds")


#        # if train_step%10000==0:
            #sample it to get the generated CDR_ region
            # gen_cdr_strucr,gen_cdr_seq = model.sample(batch)
            # @write mod_pbd
            #calculate metrics
            # metrics={}
            # metrics:{'TMSCORe':get TMSCORE(Mod_pdb, Ref pdb}
            # get LDDT(Mod_pdb, Ref pdb
            # get DDG(Mod_pdb, Ref pdb
            # get DockQ(Mod_pdb, Ref pdb
            # get RMSD(Mod_pdb, Ref pdb
            # AAR(ground_seq, Modi_seq)
            # wand.lod(metrics)

