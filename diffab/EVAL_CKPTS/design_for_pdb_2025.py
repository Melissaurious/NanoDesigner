import path_setup 
import os
import argparse
import copy
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import sys
import logging
from diffab.datasets.custom import preprocess_antibody_structure
from diffab.models import get_model
# from diffab._base import get_model
from diffab.modules.common.geometry import reconstruct_backbone_partially
from diffab.modules.common.so3 import so3vec_to_rotation
from diffab.utils.inference import RemoveNative
from diffab.utils.protein.writers import save_pdb
from diffab.utils.train import recursive_to
from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.transforms import *
from diffab.utils.inference import *


def get_item_by_entry_id(pdb_dict, entry_id):
    # Returns the item in the pdb_dict based on the entry_id.
    return pdb_dict.get(entry_id, None)



def create_data_variants(structure_factory, design_mode='single_cdr'):
    """
    Create data variants for CDR design
    
    Args:
        structure_factory: Function that returns the structure
        design_mode: 'single_cdr' for H_CDR3 only, 'multiple_cdrs' for H_CDR1,2,3
    """
    structure = structure_factory()
    structure_id = structure['id']
    data_variants = []
    
    if design_mode == 'single_cdr':
        # Single CDR mode - design only H_CDR3 (default)
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(["H_CDR3"])))
        print(f"Single CDR mode - CDRs found: {cdrs}")
        
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
    
    elif design_mode == 'multiple_cdrs':
        # Multiple CDRs mode - design H_CDR1, H_CDR2, H_CDR3 simultaneously
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(["H_CDR3","H_CDR2","H_CDR1"])))
        print(f"Multiple CDRs mode - CDRs found: {cdrs}")
        
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
    
    return data_variants


def design_for_pdb(args):
    # Load configs
    # print("args.config", args.config)
    # config, config_name = load_config(args.config)
    seed_all(args.seed if args.seed is not None else 2022)# according to config json files 


    selected_item = get_item_by_entry_id(args.pdb_dict,args.pdb_code)

    # Structure loading
    # data_id = os.path.basename(args.pdb_path)
    data_id = args.pdb_code

    if args.no_renumber:
        pdb_path = args.pdb_path
    else:
        in_pdb_path = args.pdb_path
        out_pdb_path = os.path.splitext(in_pdb_path)[0] + '_chothia.pdb'
        heavy_chains, light_chains = renumber_antibody(in_pdb_path, out_pdb_path)
        pdb_path = out_pdb_path

        if args.heavy is None and len(heavy_chains) > 0:
            args.heavy = heavy_chains[0]
        if args.light is None and len(light_chains) > 0:
            args.light = light_chains[0]
    
    if args.heavy is None and args.light is None:
        raise ValueError("Neither heavy chain id (--heavy) or light chain id (--light) is specified.")
    get_structure = lambda: preprocess_antibody_structure({
        'id': data_id,
        'pdb_path': pdb_path,
        'heavy_id': args.heavy,
        # If the input is a nanobody, the light chain will be ignored
        'light_id': args.light,
    })

    # Logging
    structure_ = get_structure()
    structure_id = structure_['id']

    log_dir = args.out_dir
    logger = get_logger('sample', log_dir)
    logger.info(f'Data ID: {structure_["id"]}')
    # logger.info(f'Results will be saved to {log_dir}')
    data_native = MergeChains()(structure_)

    save_pdb(data_native, os.path.join(log_dir, f"{args.pdb_code}_ref.pdb"))

    # Load checkpoint and model
    # logger.info('Loading model config and checkpoints: %s' % (args.checkpoint))
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg_ckpt = ckpt['config']
    # logger.info(f'Checkpoint config: {cfg_ckpt}')
    model = get_model(cfg_ckpt.model).to(args.device)
    lsd = model.load_state_dict(ckpt['model'])
    logger.info(str(lsd))

    # Make data variants
    data_variants = create_data_variants(
        structure_factory = get_structure,
        design_mode = args.design_mode  
    )

    # Save metadata
    metadata = {
        'identifier': structure_id,
        'index': data_id,
        # 'config': args.config,
        'items': [{kk: vv for kk, vv in var.items() if kk != 'data'} for var in data_variants],
    }
    with open(os.path.join(log_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Start sampling
    collate_fn = PaddingCollate(eight=False)
    inference_tfm = [ PatchAroundAnchor(), ]

    inference_tfm = Compose(inference_tfm)

    for variant in data_variants:
        # os.makedirs(os.path.join(log_dir, variant['tag']), exist_ok=True)
        os.makedirs(os.path.join(log_dir), exist_ok=True)

        logger.info(f"Start sampling for: {variant['tag']}")
        
        data_cropped = inference_tfm(
            copy.deepcopy(variant['data'])
        )

        # In case there is something wrong with the first sample, get 5 times more to select the next good one
        # data_list_repeat = [ data_cropped ] * int(args.num_samples)*5   #config.sampling.num_samples
        data_list_repeat = [data_cropped] * int(args.num_samples)
        loader = DataLoader(data_list_repeat, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
        count = 0  # Counter for successful samples

        for batch in tqdm(loader, desc=variant['name'], dynamic_ncols=True):
            if count >= args.num_samples:
                break  # Exit the loop if enough samples are processed

            torch.set_grad_enabled(False)
            model.eval()
            batch = recursive_to(batch, args.device)

            # De novo design
            traj_batch = model.sample(batch, sample_opt={
                'pbar': True,
                'sample_structure': True,
                'sample_sequence': True,
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

                try: 
                    if int(args.num_samples) != 1:
                        save_path = os.path.join(log_dir, args.pdb_code + f"_{count}.pdb")
                    else:
                        save_path = os.path.join(log_dir, args.pdb_code + ".pdb")
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


                    from dyMEAN.utils.renumber import renumber_pdb
                    renumber_pdb(save_path, save_path, scheme = "imgt")

                    # also renumnber ref_pdb as in the next stages is also required as imgt
                    ref_path = os.path.join(log_dir, f"{args.pdb_code}_ref.pdb")
                    renumber_pdb(ref_path, ref_path, scheme = "imgt")
                

                    summary ={
                    "mod_pdb": save_path,
                    "ref_pdb": args.pdb_path,
                    'heavy_chain':args.heavy,
                    'light_chain': args.light,
                    'antigen_chains': args.antigen,
                    'cdr_type': ["H3"],
                    }


                    if selected_item is not None:
                        selected_item.update(summary)
                    else:
                        selected_item = summary

                    with open(args.summary_file, 'a') as f:
                        f.write(json.dumps(selected_item) + '\n')
                    count += 1

                except Exception as e:
                    print(f"Error processing file: {e}")
                    # Check if the file exists before attempting to delete it
                    if os.path.exists(save_path):
                        os.remove(save_path)  # Delete the problematic file
                    continue  # Proceed with the next batch

        logger.info('Finished.\n')

    # return summary


def main(): 
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained checkpoint file')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to the results')
    parser.add_argument('--test_set', type=str, required=True, help='Path to test set')
    parser.add_argument('--FOLD', type=int, default=None)
    parser.add_argument('--no_renumber', action='store_true', default=True) # Do not do renumbering
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--max_num_test_entries', type=int, default=100)
    parser.add_argument('--filter_2_Ag_entries', action='store_true',
                    help='Filter entries with single antigen (length 1)')
    parser.add_argument('--design_mode', type=str, default='single_cdr', 
                    choices=['single_cdr', 'multiple_cdrs'],
                    help='Design mode: single_cdr (H_CDR3 only) or multiple_cdrs (H_CDR1,2,3)')


    args = parser.parse_args()



    with open(args.test_set, 'r') as fin:
        data = [json.loads(entry) for entry in fin] 

    data = [entry for entry in data if not entry.get('light_chain')]
    print(f"Initial nanobody filtering: {len(data)} entries remaining - only evaluating on nanobodies")

    # Add filtering logic here
    if args.filter_2_Ag_entries:
        # Filter entries with exactly 1 antigen chain
        data = [entry for entry in data if len(entry.get('antigen_chains', [])) == 1]
        print(f"After filtering for single antigen entries: {len(data)} entries remaining")

    # Random sampling if max_num_test_entries is specified
    if args.max_num_test_entries and len(data) > args.max_num_test_entries:
        import random
        random.seed(args.seed if args.seed is not None else 2022)
        data = random.sample(data, args.max_num_test_entries)
        print(f"Randomly sampled {args.max_num_test_entries} entries from {len(data)} available entries")

    print(f"Final number of entries to process: {len(data)}")


    # create a dictionary of {pdb:whole dictionary correspinding to such pdb}
    pdb_dict = {}
    for json_obj in data:
        pdb = json_obj.get("entry_id", json_obj.get("pdb"))
        pdb_dict[pdb] = json_obj


    checkpoint_full_path = args.checkpoint
    # write down the summary
    summary_file = os.path.join(args.out_dir, "summary.json")
    

    if not os.path.exists(summary_file):
        with open(summary_file, 'w') as fout:
            pass
    elif os.path.exists(summary_file) and os.stat(summary_file).st_size != 0:
        with open(summary_file, 'r') as f:
            processed_entries = [json.loads(entry) for entry in f] 


        processed_mod_pdbs = set()
        for entry in processed_entries:
            processed_mod_pdbs.add(entry["pdb_data_path"])

        missing_entries_to_design = [element for element in data if element['pdb_data_path'] not in processed_mod_pdbs]
        print(f"number of missing entries to design {len(missing_entries_to_design)}")
        data = missing_entries_to_design[:]



    for json_obj in data:
        pdb_unique_code = json_obj.get("entry_id", json_obj.get("pdb"))

        #Extract the imgt pdb that will be used as ref_pdb at the next stages (evaluation), it can stay as "imgt" as the next stages requires it as that
        pdb_path= json_obj['pdb_data_path']  # this path is to the imgt directory from test.json file
        path_ref_pdb = os.path.join(args.out_dir, pdb_unique_code + "_ref.pdb") # ref_pdb, input to inference, renumber once used

        import sys 
        # sys.path.append('/ibex/user/rioszemm/NanobodiesProject/dyMEAN')
        from dyMEAN.utils.renumber import renumber_pdb
        renumber_pdb(pdb_path, path_ref_pdb, scheme = "chothia")  # inferece input must be in chothia


        args = EasyDict(
            # pdb_path=json_obj['pdb_data_path'], # this is in imgt numbering, change to the chothia folder.
            pdb_path=path_ref_pdb,
            pdb=json_obj.get("pdb"),
            heavy=json_obj['heavy_chain'],
            light=json_obj['light_chain'],
            antigen=json_obj['antigen_chains'],
            num_samples=args.num_samples,
            no_renumber =args.no_renumber,
            seed=args.seed,
            out_dir=args.out_dir,
            checkpoint=checkpoint_full_path,
            device=args.device,
            batch_size=args.batch_size,
            tag=args.tag,
            pdb_code=pdb_unique_code,
            summary_file=summary_file,
            pdb_dict=pdb_dict,
            design_mode=args.design_mode
        )

        try:
            design_for_pdb(args)
        except Exception as e:
            error_message = f"Error processing {json_obj['entry_id']}: {e}"
            print(error_message)
            continue


if __name__ == '__main__':
    main()
