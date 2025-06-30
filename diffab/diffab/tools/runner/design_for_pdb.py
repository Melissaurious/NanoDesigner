import os
import argparse
import copy
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from diffab.diffab.datasets.custom import preprocess_antibody_structure
from diffab.diffab.models import get_model
from diffab.diffab.modules.common.geometry import reconstruct_backbone_partially
from diffab.diffab.modules.common.so3 import so3vec_to_rotation
from diffab.diffab.utils.inference import RemoveNative
from diffab.diffab.utils.protein.writers import save_pdb
from diffab.diffab.utils.train import recursive_to
from diffab.diffab.utils.misc import *
from diffab.diffab.utils.data import *
from diffab.diffab.utils.transforms import *
from diffab.diffab.utils.inference import *
import time

# from diffab.tools.renumber import renumber as renumber_antibody


def create_data_variants(config, structure_factory):
    structure = structure_factory()
    structure_id = structure['id']

    data_variants = []
    if config.mode == 'single_cdr':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            data_variants.append({
                'data': data_var,
                'name': f'{structure_id}-{cdr_name}',
                'tag': f'{cdr_name}',
                'cdr': cdr_name,
                'residue_first': residue_first,
                'residue_last': residue_last,
            })
    elif config.mode == 'multiple_cdrs':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        transform = Compose([
            MaskMultipleCDRs(selection=cdrs, augmentation=False),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        data_variants.append({
            'data': data_var,
            'name': f'{structure_id}-MultipleCDRs',
            'tag': 'MultipleCDRs',
            'cdrs': cdrs,
            'residue_first': None,
            'residue_last': None,
        })
    elif config.mode == 'full':
        transform = Compose([
            MaskAntibody(),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        data_variants.append({
            'data': data_var,
            'name': f'{structure_id}-Full',
            'tag': 'Full',
            'residue_first': None,
            'residue_last': None,
        })
    elif config.mode == 'abopt':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            for opt_step in config.sampling.optimize_steps:
                data_variants.append({
                    'data': data_var,
                    'name': f'{structure_id}-{cdr_name}-O{opt_step}',
                    'tag': f'{cdr_name}-O{opt_step}',
                    'cdr': cdr_name,
                    'opt_step': opt_step,
                    'residue_first': residue_first,
                    'residue_last': residue_last,
                })
    else:
        raise ValueError(f'Unknown mode: {config.mode}.')
    return data_variants



def design_for_pdb(args):

    start_time = time.time() 
    # Load configs
    print("args.config", args.config)
    config, config_name = load_config(args.config)
    seed_all(args.seed if args.seed is not None else config.sampling.seed)

    # Structure loading
    data_id = os.path.basename(args.pdb_path)
    if args.no_renumber:
        pdb_path = args.pdb_path

    if args.heavy is None and args.light is None:
        raise ValueError("Neither heavy chain id (--heavy) or light chain id (--light) is specified.")
    get_structure = lambda: preprocess_antibody_structure({
        'id': data_id,
        'pdb_path': pdb_path,
        'heavy_id': args.heavy,
        # If the input is a nanobody, the light chain will be ignored
        'light_id': args.light,
    })

    # Skip logging setup - no log files needed
    structure_ = get_structure()
    structure_id = structure_['id']
    log_dir = args.out_root
    
    # Skip saving reference.pdb and metadata.json
    data_native = MergeChains()(structure_)
    # save_pdb(data_native, os.path.join(log_dir, 'reference.pdb'))  # COMMENTED OUT

    if config.mode == "single_cdr":
        cdr_type = ["H3"]
    else:
        cdr_type = ["H3", "H2", "H1"]

    # Load checkpoint and model
    print('Loading model config and checkpoints: %s' % (config.model.checkpoint))
    ckpt = torch.load(config.model.checkpoint, map_location='cpu')
    cfg_ckpt = ckpt['config']
    model = get_model(cfg_ckpt.model).to(args.device)
    lsd = model.load_state_dict(ckpt['model'])
    print(str(lsd))

    # Make data variants
    data_variants = create_data_variants(
        config = config,
        structure_factory = get_structure,
    )

    # Skip saving metadata.json
    # metadata = {
    #     'identifier': structure_id,
    #     'index': data_id,
    #     'config': args.config,
    #     'items': [{kk: vv for kk, vv in var.items() if kk != 'data'} for var in data_variants],
    # }
    # with open(os.path.join(log_dir, 'metadata.json'), 'w') as f:
    #     json.dump(metadata, f, indent=2)

    # Start sampling
    collate_fn = PaddingCollate(eight=False)
    inference_tfm = [ PatchAroundAnchor(), ]
    if 'abopt' not in config.mode:  # Don't remove native CDR in optimization mode
        inference_tfm.append(RemoveNative(
            remove_structure = config.sampling.sample_structure,
            remove_sequence = config.sampling.sample_sequence,
        ))
    inference_tfm = Compose(inference_tfm)

    if int(args.iteration) == 1:
        if hasattr(config.sampling, 'num_samples_iter_1'):
            num_samples = config.sampling.num_samples_iter_1
        else:
            num_samples = 3

    else:
        if hasattr(config.sampling, 'num_samples_iter_x'):
            num_samples = config.sampling.num_samples_iter_x
        else:
            num_samples = 10



    for variant in data_variants:
        os.makedirs(os.path.join(log_dir), exist_ok=True)

        print(f"Start sampling for: {variant['tag']}")
        
        # Skip saving REF1.pdb
        # save_pdb(data_native, os.path.join(log_dir, 'REF1.pdb'))  # COMMENTED OUT

        data_cropped = inference_tfm(
            copy.deepcopy(variant['data'])
        )

        data_list_repeat = [data_cropped] * int(2 * num_samples)
        loader = DataLoader(data_list_repeat, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        count = 0  # Counter for successful samples
        
        for batch in tqdm(loader, desc=variant['name'], dynamic_ncols=True):
            if count >= num_samples:
                break  # Exit the loop if enough samples are processed

            torch.set_grad_enabled(False)
            model.eval()
            batch = recursive_to(batch, args.device)
            
            if 'abopt' in config.mode:
                # Antibody optimization starting from native
                traj_batch = model.optimize(batch, opt_step=variant['opt_step'], optimize_opt={
                    'pbar': True,
                    'sample_structure': config.sampling.sample_structure,
                    'sample_sequence': config.sampling.sample_sequence,
                })
            else:
                # De novo design
                traj_batch = model.sample(batch, sample_opt={
                    'pbar': True,
                    'sample_structure': config.sampling.sample_structure,
                    'sample_sequence': config.sampling.sample_sequence,
                })

            aa_new = traj_batch[0][2]   # 0: Last sampling step. 2: Amino acid.
            pos_atom_new, mask_atom_new = reconstruct_backbone_partially(
                pos_ctx = batch['pos_heavyatom'],
                R_new = so3vec_to_rotation(traj_batch[0][0]),
                t_new = traj_batch[0][1],
                aa = aa_new,
                chain_nb = batch['chain_nb'],
                res_nb = batch['res_nb'],
                mask_atoms = batch['mask_heavyatom'],
                mask_recons = batch['generate_flag'],
            )
            aa_new = aa_new.cpu()
            pos_atom_new = pos_atom_new.cpu()
            mask_atom_new = mask_atom_new.cpu()

            for i in range(aa_new.size(0)):
                if count >= num_samples:
                    break  # Additional check to ensure no extra samples

                data_tmpl = variant['data']
                aa = apply_patch_to_tensor(data_tmpl['aa'], aa_new[i], data_cropped['patch_idx'])
                mask_ha = apply_patch_to_tensor(data_tmpl['mask_heavyatom'], mask_atom_new[i], data_cropped['patch_idx'])
                pos_ha  = (
                    apply_patch_to_tensor(
                        data_tmpl['pos_heavyatom'], 
                        pos_atom_new[i] + batch['origin'][i].view(1, 1, 3).cpu(), 
                        data_cropped['patch_idx']
                    )
                )

                import sys 
                sys.path.append(args.dymean_code_dir)

                save_path = os.path.join(log_dir, '%04d.pdb' % (count, ))
                try:
                    save_pdb({
                        'chain_nb': data_tmpl['chain_nb'],
                        'chain_id': data_tmpl['chain_id'],
                        'resseq': data_tmpl['resseq'],
                        'icode': data_tmpl['icode'],
                        # Generated
                        'aa': aa,
                        'mask_heavyatom': mask_ha,
                        'pos_heavyatom': pos_ha,
                    }, path=save_path)

                    from utils.renumber import renumber_pdb
                    renumber_pdb(save_path, save_path, scheme = "imgt")
                    
                    # Modified summary format - only the design results
                    summary = {
                        "mod_pdb": save_path,
                        "ref_pdb": args.pdb_path,
                        'heavy_chain': args.heavy,
                        'light_chain': args.light,
                        'antigen_chains': args.antigen,
                        'cdr_type': cdr_type,
                        'entry_id': args.pdb_code,
                        'cdr_model': "DiffAb"
                    }

                    with open(args.summary_dir, 'a') as f:
                        f.write(json.dumps(summary) + '\n')
                    count += 1

                except Exception as e:
                    print(f"Error processing file: {e}")
                    # Check if the file exists before attempting to delete it
                    if os.path.exists(save_path):
                        os.remove(save_path)  # Delete the problematic file
                    continue  # Proceed with the next batch




def args_from_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_path', type=str)
    parser.add_argument('--heavy', type=str, default=None, help='Chain id of the heavy chain.')
    parser.add_argument('--light', type=str, default=None, help='Chain id of the light chain.')
    parser.add_argument('--antigen', type=list, default=None, help='Chain id of the light chain.')
    # parser.add_argument('--no_renumber', action='store_true', default=False)
    parser.add_argument('--no_renumber', action='store_true', default=True) # Do not do renumbering
    parser.add_argument('-c', '--config', type=str, default='./configs/test/codesign_single.yml')
    parser.add_argument('-o', '--logger.info', type=str, default='./results')
    parser.add_argument('-t', '--tag', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('--summary_dir', type=str, help="Path to save the summary file")
    # parser.add_argument('--cdr_type', type=str, default='H3', help='Type of CDR',
    #                     choices=['H3'])
    parser.add_argument('--cdr_type', choices=['H1', 'H2', 'H3', '-'], nargs='+', help='CDR types to randomize')
    parser.add_argument('--pdb_code', type=str, default=None, help='4 code of pdb')
    parser.add_argument('--model', type=str, default=None, help='hdock model id')

    args = parser.parse_args()
    return args


    #parser.add_argument('--cdr_type', choices=['H1', 'H2', 'H3', '-'], nargs='+', help='CDR types to randomize')


def args_factory(**kwargs):
    default_args = EasyDict(
        heavy = 'H',
        light = 'L',
        no_renumber = False,
        config = './configs/test/codesign_single.yml',
        out_root = './results',
        tag = '',
        seed = None,
        device = 'cuda',
        batch_size = 16
    )
    default_args.update(kwargs)
    return default_args


if __name__ == '__main__':
    design_for_pdb(args_from_cmdline())
