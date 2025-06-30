#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('/ibex/user/rioszemm/NanobodiesProject/dyMEAN')
from data.dataset import E2EDataset2
from data.pdb_utils import VOCAB, Residue, Peptide, Protein, AgAbComplex, AgAbComplex2
from data.framework_templates import ConserveTemplateGenerator 

from utils.logger import print_log
from utils.random_seed import setup_seed


def to_cplx(ori_cplx, ab_x, ab_s) -> AgAbComplex2:
    heavy_chain, light_chain = [], []
    chain = None
    for residue, residue_x in zip(ab_s, ab_x):
        residue = VOCAB.idx_to_symbol(residue)
        if residue == VOCAB.BOA:
            continue
        elif residue == VOCAB.BOH:
            chain = heavy_chain
            continue
        elif residue == VOCAB.BOL:
            chain = light_chain
            continue
        if chain is None:  # still in antigen region
            continue
        coord, atoms = {}, VOCAB.backbone_atoms + VOCAB.get_sidechain_info(residue)

        for atom, x in zip(atoms, residue_x):
            coord[atom] = x
        chain.append(Residue(
            residue, coord, _id=(len(chain), ' ')
        ))
    heavy_chain = Peptide(ori_cplx.heavy_chain, heavy_chain)
    # light_chain = Peptide(ori_cplx.light_chain, light_chain)
    for res, ori_res in zip(heavy_chain, ori_cplx.get_heavy_chain()):
        res.id = ori_res.id
    # for res, ori_res in zip(light_chain, ori_cplx.get_light_chain()):
    #     res.id = ori_res.id

    peptides = {
        ori_cplx.heavy_chain: heavy_chain,
        # ori_cplx.light_chain: light_chain
    }
    antibody = Protein(ori_cplx.pdb_id, peptides)
    cplx = AgAbComplex2(
        ori_cplx.antigen, antibody, ori_cplx.heavy_chain,
        skip_epitope_cal=True,
        skip_validity_check=True
    )
    cplx.cdr_pos = ori_cplx.cdr_pos
    return cplx


# def load_and_filter_test_data(test_set_path, filter_2_Ag_entries=False, max_num_test_entries=None, seed=2022):
#     """Load test data from JSON file and apply filtering/sampling"""
    
#     # Load test data
#     with open(test_set_path, 'r') as fin:
#         # Handle both single JSON object and JSONL format
#         content = fin.read().strip()
#         if content.startswith('['):
#             # Standard JSON array
#             data = json.loads(content)
#         else:
#             # JSONL format - each line is a separate JSON object
#             fin.seek(0)
#             data = [json.loads(line.strip()) for line in fin if line.strip()]
    
#     print(f"Loaded {len(data)} entries from test set")
    
#     # Apply filtering logic
#     if filter_2_Ag_entries:
#         # Filter entries with exactly 1 antigen chain
#         original_count = len(data)
#         data = [entry for entry in data if len(entry.get('antigen_chains', [])) == 1]
#         print(f"After filtering for single antigen entries: {len(data)} entries remaining (filtered out {original_count - len(data)})")
    
#     # Random sampling if max_num_test_entries is specified
#     if max_num_test_entries and len(data) > max_num_test_entries:
#         random.seed(seed)
#         data = random.sample(data, max_num_test_entries)
#         print(f"Randomly sampled {max_num_test_entries} entries")
    
#     print(f"Final number of entries to process: {len(data)}")
#     return data


def load_and_filter_test_data(test_set_path, filter_2_Ag_entries=False, max_num_test_entries=None, seed=2022):
    """Load test data from JSON file and apply filtering/sampling"""
    
    # Load test data
    with open(test_set_path, 'r') as fin:
        # Handle both single JSON object and JSONL format
        content = fin.read().strip()
        if content.startswith('['):
            # Standard JSON array
            data = json.loads(content)
        else:
            # JSONL format - each line is a separate JSON object
            fin.seek(0)
            data = [json.loads(line.strip()) for line in fin if line.strip()]
    
    print(f"Loaded {len(data)} entries from test set")
    
    # First filter: keep only nanobody entries (no light chain)
    original_count = len(data)
    data = [entry for entry in data if not entry.get('light_chain')]
    print(f"After nanobody filtering: {len(data)} entries remaining (filtered out {original_count - len(data)})")
    
    # Apply filtering logic
    if filter_2_Ag_entries:
        # Filter entries with exactly 1 antigen chain
        original_count = len(data)
        data = [entry for entry in data if len(entry.get('antigen_chains', [])) == 1]
        print(f"After filtering for single antigen entries: {len(data)} entries remaining (filtered out {original_count - len(data)})")
    
    # Random sampling if max_num_test_entries is specified
    if max_num_test_entries and len(data) > max_num_test_entries:
        random.seed(seed)
        data = random.sample(data, max_num_test_entries)
        print(f"Randomly sampled {max_num_test_entries} entries")
    
    print(f"Final number of entries to process: {len(data)}")
    return data


# def generate(test_loader, test_set, ref_pdb, save_dir, model, device, cdr_type):
#     idx = 0
#     summary_items = []
#     for batch in tqdm(test_loader):
#         with torch.no_grad():
#             # move data
#             for k in batch:
#                 if hasattr(batch[k], 'to'):
#                     batch[k] = batch[k].to(device)
#             # generate
#             del batch['xloss_mask']
#             X, S, pmets = model.sample(**batch)

#             X, S, pmets = X.tolist(), S.tolist(), pmets.tolist()
#             X_list, S_list = [], []
#             cur_bid = -1
#             if 'bid' in batch:
#                 batch_id = batch['bid']
#             else:
#                 lengths = batch['lengths']
#                 batch_id = torch.zeros_like(batch['S'])  # [N]
#                 batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
#                 batch_id.cumsum_(dim=0)  # [N], item idx in the batch
#             for i, bid in enumerate(batch_id):
#                 if bid != cur_bid:
#                     cur_bid = bid
#                     X_list.append([])
#                     S_list.append([])
#                 X_list[-1].append(X[i])
#                 S_list[-1].append(S[i])
                


#         for i, (x, s) in enumerate(zip(X_list, S_list)):
#             ori_cplx = test_set.data[idx]
#             cplx = to_cplx(ori_cplx, x, s)
#             pdb_id = cplx.get_id().split('(')[0]
#             mod_pdb = os.path.join(save_dir, pdb_id + '.pdb')
#             cplx.to_pdb(mod_pdb)
            
#             # Get the entry-specific reference PDB
#             ref_pdb = test_data[idx].get("pdb_data_path", "")
            
#             summary_items.append({
#                 'mod_pdb': mod_pdb,
#                 'ref_pdb': ref_pdb,  # <-- Now this is entry-specific
#                 'heavy_chain': cplx.heavy_chain,
#                 'light_chain': "",
#                 'antigen_chains': cplx.antigen.get_chain_names(),
#                 'cdr_type': cdr_type,
#                 'cdr_model': "dyMEAN",
#                 'pdb': pdb_id,
#                 'pmetric': pmets[i]
#             })
#             idx += 1


#     # write done the summary
#     summary_file = os.path.join(save_dir, 'summary.json')
#     with open(summary_file, 'w') as fout:
#         fout.writelines(list(map(lambda item: json.dumps(item) + '\n', summary_items)))
#     print_log(f'Summary of generated complexes written to {summary_file}')

def generate(test_loader, test_set, test_data, save_dir, model, device, cdr_type):
    idx = 0
    summary_items = []
    for batch in tqdm(test_loader):
        with torch.no_grad():
            # move data
            for k in batch:
                if hasattr(batch[k], 'to'):
                    batch[k] = batch[k].to(device)
            # generate
            del batch['xloss_mask']
            X, S, pmets = model.sample(**batch)

            X, S, pmets = X.tolist(), S.tolist(), pmets.tolist()
            X_list, S_list = [], []
            cur_bid = -1
            if 'bid' in batch:
                batch_id = batch['bid']
            else:
                lengths = batch['lengths']
                batch_id = torch.zeros_like(batch['S'])  # [N]
                batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
                batch_id.cumsum_(dim=0)  # [N], item idx in the batch
            for i, bid in enumerate(batch_id):
                if bid != cur_bid:
                    cur_bid = bid
                    X_list.append([])
                    S_list.append([])
                X_list[-1].append(X[i])
                S_list[-1].append(S[i])
                
        for i, (x, s) in enumerate(zip(X_list, S_list)):
            ori_cplx = test_set.data[idx]
            cplx = to_cplx(ori_cplx, x, s)
            pdb_id = cplx.get_id().split('(')[0]
            mod_pdb = os.path.join(save_dir, pdb_id + '.pdb')
            cplx.to_pdb(mod_pdb)
            
            # Get the entry-specific reference PDB from test_data
            ref_pdb = ""
            if idx < len(test_data):
                ref_pdb = test_data[idx].get("pdb_data_path", "")
                if not ref_pdb:
                    ref_pdb = test_data[idx].get("nano_source", "")
            
            # Fallback if no reference found
            if not ref_pdb:
                ref_pdb = f"/default/path/{pdb_id}.pdb"
                print(f"Warning: No reference PDB found for {pdb_id}, using fallback: {ref_pdb}")
            
            summary_items.append({
                'mod_pdb': mod_pdb,
                'ref_pdb': ref_pdb,
                'heavy_chain': cplx.heavy_chain,
                'light_chain': "",
                'antigen_chains': cplx.antigen.get_chain_names(),
                'cdr_type': cdr_type,
                'cdr_model': "dyMEAN",
                'pdb': pdb_id,
                'pmetric': pmets[i]
            })
            idx += 1

    # write done the summary
    summary_file = os.path.join(save_dir, 'summary.json')
    with open(summary_file, 'w') as fout:
        fout.writelines(list(map(lambda item: json.dumps(item) + '\n', summary_items)))
    print_log(f'Summary of generated complexes written to {summary_file}')


def main(args):
    setup_seed(args.seed)


    # template = args.template if args.template else "/ibex/user/rioszemm/april_2024_dyMEAN_NANO_clst_Ag/fold_5/template.json"
    # template_generator = ConserveTemplateGenerator(template)

    if args.save_dir:
        summary_file = os.path.join(args.save_dir, "summary.json")
        if os.path.exists(summary_file) and os.path.getsize(summary_file) > 0:
            print(f"Output already exists: {summary_file}")
            print("Skipping...")
            return


    if os.path.exists(args.template):
        print(f"Using template: {args.template}")
        template_generator = ConserveTemplateGenerator(args.template)
    else:
        template = "/ibex/user/rioszemm/april_2024_dyMEAN_NANO_clst_Ag/fold_5/template.json"
        print(f"Template file not found: {template}")
        template_generator = ConserveTemplateGenerator(template)

        # Handle the error case here - you might want to exit or raise an exception
        raise FileNotFoundError(f"Template file not found: {template}")


    # load model
    model = torch.load(args.ckpt, map_location='cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    # model_type
    print_log(f'Model type: {type(model)}')

    # cdr type
    cdr_type = model.cdr_type
    print_log(f'CDR type: {cdr_type}')
    print_log(f'Paratope definition: {model.paratope}')

    # Load and filter test data
    test_data = load_and_filter_test_data(
        args.test_set, 
        filter_2_Ag_entries=args.filter_2_Ag_entries,
        max_num_test_entries=args.max_num_test_entries,
        seed=args.seed
    )

    # Extract reference PDB path (assuming it's in the first entry or provided separately)
    if test_data:
        ref_pdb = test_data[0].get("pdb_data_path", "")
    else:
        ref_pdb = ""
    
    # Create a filtered dataset file for E2EDataset2
    filtered_entries_file = os.path.join(os.path.dirname(args.test_set), "filtered_entries.json")
    with open(filtered_entries_file, 'w') as f:
        for entry in test_data:
            f.write(json.dumps(entry) + '\n')
    
    # load test set with filtered data
    test_set = E2EDataset2(filtered_entries_file, template_generator=template_generator, cdr="H3", paratope="H3")
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             collate_fn=E2EDataset2.collate_fn)
    
    # create save dir
    if args.save_dir is None:
        save_dir = '.'.join(args.ckpt.split('.')[:-1]) + '_results'
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # generate(test_loader, test_set, ref_pdb, save_dir, model, device, cdr_type)
    generate(test_loader, test_set, test_data, save_dir, model, device, cdr_type)
    
    # Clean up temporary file
    if os.path.exists(filtered_entries_file):
        os.remove(filtered_entries_file)


def parse():
    parser = argparse.ArgumentParser(description='Generate antibodies given epitopes')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--test_set', type=str, required=True, help='Path to test set JSON file')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated antibodies')
    parser.add_argument('--template', type=str, default=None, help='Path to template JSON file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    
    # New filtering parameters
    parser.add_argument('--filter_2_Ag_entries', action='store_true', 
                        help='Filter entries to only include those with exactly 1 antigen chain')
    parser.add_argument('--max_num_test_entries', type=int, default=None,
                        help='Maximum number of test entries to process (random sampling if exceeded)')
    
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())