#!/usr/bin/python
# -*- coding:utf-8 -*-
from functools import partial
import json
import os
import random
from copy import deepcopy
import logging
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import torch
from torch.utils.data import DataLoader

from data import AAComplex,AAComplex2, EquiAACDataset
from evaluation.rmsd import kabsch
from utils import check_dir
from utils.logger import print_log
from evaluation import compute_rmsd, tm_score
from argparse import ArgumentParser

import gc


def get_item_by_entry_id(pdb_dict, entry_id):
    # Returns the item in the pdb_dict based on the entry_id.
    return pdb_dict.get(entry_id, None)


def set_cdr(cplx, seq, x, cdr='H3'):
    cdr = cdr.upper()
    cplx: AAComplex2 = deepcopy(cplx)
    chains = cplx.peptides
    cdr_chain_key = cplx.heavy_chain if 'H' in cdr else cplx.light_chain
    refined_chain = chains[cdr_chain_key]
    start, end = cplx.get_cdr_pos(cdr)
    start_pos, end_pos = refined_chain.get_ca_pos(start), refined_chain.get_ca_pos(end)
    start_trans, end_trans = x[0][1] - start_pos, x[-1][1] - end_pos
    # left to start of cdr
    for i in range(0, start):
        refined_chain.set_residue_translation(i, start_trans)
    # end of cdr to right
    for i in range(end + 1, len(refined_chain)):
        refined_chain.set_residue_translation(i, end_trans)
    # cdr 
    for i, residue_x, symbol in zip(range(start, end + 1), x, seq):
        center = residue_x[4] if len(residue_x) > 4 else None
        refined_chain.set_residue(i, symbol,
            {
                'N': residue_x[0],
                'CA': residue_x[1],
                'C': residue_x[2],
                'O': residue_x[3]
            }, center, gen_side_chain=False
        )
    new_cplx = AAComplex2(cplx.pdb_id, chains, cplx.heavy_chain,
                         cplx.antigen_chains, cplx.light_chain,numbering=None, cdr_pos=cplx.cdr_pos,
                         skip_cal_interface=True)
    return new_cplx


def eval_one(tup, out_dir, cdr='H3', pdb_dict=None):
    cplx, seq, x, true_x, aligned, k_index = tup

    pdb_path = os.path.join(out_dir, f"{cplx.get_id()}.pdb") #of the new cplx

    entry_id = cplx.get_id()
    pdb = entry_id.split('_')[0]
    summary = {
        'pdb': pdb,
        'heavy_chain': cplx.heavy_chain,
        'light_chain': cplx.light_chain if cplx.light_chain else "",  # if nanobody, this will be None, set to empty
        'antigen_chains': cplx.antigen_chains,
        'mod_pdb': pdb_path,
        "cdr_type":[cdr],
        "entry_id":entry_id
    }

    selected_item = get_item_by_entry_id(pdb_dict, entry_id)
    if selected_item is not None:
        selected_item.update(summary)
        selected_item["ref_pdb"] = os.path.join(out_dir, cplx.get_id() + '_ref.pdb')
    else:
        selected_item = summary

    # kabsch
    if aligned:
        ca_aligned = x[:, 1, :]
    else:
        ca_aligned, rotation, t = kabsch(x[:, 1, :], true_x[:, 1, :])
        x = np.dot(x - np.mean(x, axis=0), rotation) + t
    summary['RMSD'] = compute_rmsd(ca_aligned, true_x[:, 1, :], aligned=True)
    # set cdr
    new_cplx = set_cdr(cplx, seq, x, cdr)
    pdb_path = os.path.join(out_dir, cplx.get_id() + '.pdb')
    new_cplx.to_pdb(pdb_path)

    return selected_item


def load_and_filter_test_data(test_set_path, filter_2_Ag_entries=False, max_num_test_entries=None, seed=42):
    """Load test data from JSON file and apply filtering/sampling"""
    
    # Load test data
    with open(test_set_path, 'r') as fin:
        data = fin.read().strip().split('\n')
        data = [json.loads(item) for item in data if item.strip()]
    
    print_log(f"Loaded {len(data)} entries from test set")
    
    # First filter: keep only nanobody entries (no light chain)
    original_count = len(data)
    data = [entry for entry in data if not entry.get('light_chain')]
    print_log(f"After nanobody filtering: {len(data)} entries remaining (filtered out {original_count - len(data)})")
    
    # Apply filtering logic
    if filter_2_Ag_entries:
        # Filter entries with exactly 1 antigen chain
        original_count = len(data)
        data = [entry for entry in data if len(entry.get('antigen_chains', [])) == 1]
        print_log(f"After filtering for single antigen entries: {len(data)} entries remaining (filtered out {original_count - len(data)})")
    
    # Random sampling if max_num_test_entries is specified
    if max_num_test_entries and len(data) > max_num_test_entries:
        random.seed(seed)
        data = random.sample(data, max_num_test_entries)
        print_log(f"Randomly sampled {max_num_test_entries} entries")
    
    print_log(f"Final number of entries to process: {len(data)}")
    return data


def rabd_test(args, model, test_set, test_loader, out_dir, device, pdb_dict):
    args.rabd_topk = min(args.rabd_topk, args.rabd_sample)
    global_best_ppl = [[1e10 for _ in range(args.rabd_topk)] for _ in range(len(test_set))]
    global_best_results = [[None for _ in range(args.rabd_topk)] for _ in range(len(test_set))]

    k_ids = [k for k in range(args.rabd_topk)]
    with torch.no_grad():
        for _ in tqdm(range(args.rabd_sample)):
            results, ppl = [], []
            for i, batch in enumerate(test_loader):
                ppls, seqs, xs, true_xs, aligned = model.infer(batch, device)
                ppl.extend(ppls)
                results.extend([(seqs[i], xs[i], true_xs[i], aligned) for i in range(len(seqs))])
            for i, p in enumerate(ppl):
                max_ppl_id = max(k_ids, key=lambda k: global_best_ppl[i][k])
                if p < global_best_ppl[i][max_ppl_id]:
                    global_best_ppl[i][max_ppl_id] = p
                    global_best_results[i][max_ppl_id] = results[i]
                        
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print_log(f'dumped to {args.save_dir}')

    heads = ['PPL']
    eval_res = [[] for _ in heads]
    
    for k in range(args.rabd_topk):
        inputs = []
        for cplx, item in zip(test_set.data, global_best_results):
            inputs.append((cplx, ) + tuple(item[k]) + (k,))
        
        summaries = process_map(partial(eval_one, out_dir=args.save_dir, cdr='H3', pdb_dict=pdb_dict), inputs, max_workers=args.num_workers, chunksize=10)

        summary_fout = open(os.path.join(args.save_dir, 'summary.json'), 'w')
        for i, summary in enumerate(summaries):
            summary['PPL'] = global_best_ppl[i][k]
            summary_fout.write(json.dumps(summary) + '\n')
        summary_fout.close()

        for i, h in enumerate(heads):
            eval_res[i].extend([summary[h] for summary in summaries if h in summary])
    
    eval_res = np.array(eval_res, dtype=float)
    means = np.mean(eval_res, axis=1)
    stdvars = np.std(eval_res, axis=1)
    print_log(f'Results for top {args.rabd_topk} candidates:')
    for i, h in enumerate(heads):
        print_log(f'{h}: mean {means[i]}, std {stdvars[i]}')


def average_test(args, model, test_set, test_loader, out_dir, device):
    heads, eval_res = ['PPL', 'RMSD', 'TMscore', 'AAR'], []
    for _round in range(args.run):
        print_log(f'round {_round}')
        results, ppl = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                ppls, seqs, xs, true_xs, aligned = model.infer(batch, device)
                ppl.extend(ppls)
                results.extend([(seqs[i], xs[i], true_xs[i], aligned) for i in range(len(seqs))])

        assert len(test_set) == len(results)

        inputs = [(cplx, ) + item for cplx, item in zip(test_set.data, results)]  # cplx, seq, x 

        check_dir(out_dir)
        print_log(f'dumped to {out_dir}')
        
        cdr_type = 'H' + model.cdr_type
        summaries = process_map(partial(eval_one, out_dir=out_dir, cdr=cdr_type), inputs, max_workers=args.num_workers, chunksize=10)

        summary_fout = open(os.path.join(out_dir, 'summary.json'), 'w')
        for i, summary in enumerate(summaries):
            summary['PPL'] = ppl[i]
            summary_fout.write(json.dumps(summary) + '\n')
        summary_fout.close()

        rmsds = [summary['RMSD'] for summary in summaries]
        tm_scores = [summary['TMscore'] for summary in summaries]
        aars = [summary['AAR'] for summary in summaries]
        ppl, rmsd, tm, aar = np.mean(ppl), np.mean(rmsds), np.mean(tm_scores), np.mean(aars)
        print_log(f'ppl: {ppl}, rmsd: {rmsd}, TM score: {tm}, AAR: {aar}')
        eval_res.append([ppl, rmsd, tm, aar])

    eval_res = np.array(eval_res)
    means = np.mean(eval_res, axis=0)
    stdvars = np.std(eval_res, axis=0)
    print_log(f'Results after {args.run} runs:')
    report_means = {'PPL': [], 'RMSD': [], 'TMscore': [], 'AAR': []}
    for i, h in enumerate(heads):
        report_means[h] = means[i]
        print_log(f'{h}: mean {means[i]}, std {stdvars[i]}')
    return report_means


def main(args):
    print(str(args))

    summary_file = os.path.join(args.save_dir, "summary.json")
    if os.path.exists(summary_file) and os.stat(summary_file).st_size != 0:
        print(f"Summary file already exists and not empty, Skipping inference step...")
        exit()

    model = torch.load(args.ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu[0] == -1 else f'cuda:{args.gpu[0]}')

    print("model cdr type", model.cdr_type)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(str(args))
    logger.info("model cdr type: %s", model.cdr_type)

    # Load and filter test data
    filtered_data = load_and_filter_test_data(
        args.test_set, 
        filter_2_Ag_entries=args.filter_2_Ag_entries,
        max_num_test_entries=args.max_num_test_entries,
        seed=args.seed
    )

    # Create pdb_dict from filtered data
    pdb_dict = {}
    for item in filtered_data:
        item["model"] = args.model
        pdb = item.get("entry_id", item.get("pdb"))
        pdb_dict[pdb] = item

    # Create filtered dataset file first
    filtered_entries_file = os.path.join(os.path.dirname(args.test_set), "filtered_entries_temp.json")
    with open(filtered_entries_file, 'w') as f:
        for entry in filtered_data:
            f.write(json.dumps(entry) + '\n')

    # Use preprocessed data if available
    if args.preprocessed_path and os.path.exists(args.preprocessed_path):
        print_log(f"Using preprocessed data from: {args.preprocessed_path}")
        # Load the dataset with filtered entries
        test_set = EquiAACDataset(filtered_entries_file)
        test_set.preprocessed_dir = os.path.dirname(args.preprocessed_path)
        print_log(f"Loaded {len(test_set.data)} filtered entries using preprocessed data")
    else:
        # Create a filtered dataset file for EquiAACDataset (fallback to reprocessing)
        print_log("No preprocessed path provided, will reprocess data...")
        test_set = EquiAACDataset(filtered_entries_file)

    test_set.mode = args.mode
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=False,
                             collate_fn=test_set.collate_fn)
    model.to(device)
    model.eval()
    
    rabd_test(args, model, test_set, test_loader, args.save_dir, device, pdb_dict)
    
    # Writing original structures
    print_log(f'Writing original structures')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    for cplx in tqdm(test_set.data):
        pdb_path = os.path.join(args.save_dir, cplx.get_id() + '_ref.pdb')
        cplx.to_pdb(pdb_path)
    
    # Clean up temporary file
    filtered_entries_file = os.path.join(os.path.dirname(args.test_set), "filtered_entries_temp.json")
    if os.path.exists(filtered_entries_file):
        os.remove(filtered_entries_file)
    
    gc.collect()


def parse():
    parser = ArgumentParser(description='Generate antibody')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--test_set', type=str, required=True, help='Path to test set JSON file')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save generated antibodies')
    parser.add_argument('--preprocessed_path', type=str, default=None, 
                        help='Path to preprocessed .pkl file (e.g., /path/to/part_0.pkl)')
    parser.add_argument('--mode', type=str, default='1*1')
    parser.add_argument('--rabd_topk', type=int, default=1)
    parser.add_argument('--rabd_sample', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=[0], nargs='+', help='gpu(s) to use, -1 for cpu')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='Seed to use in training')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--run', type=int, default=1, help='Number of runs for evaluation')
    parser.add_argument('--model', type=str, default='ADesigner', help='Model name')
    
    # New filtering parameters
    parser.add_argument('--filter_2_Ag_entries', action='store_true', 
                        help='Filter entries to only include those with exactly 1 antigen chain')
    parser.add_argument('--max_num_test_entries', type=int, default=None,
                        help='Maximum number of test entries to process (random sampling if exceeded)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    main(args)