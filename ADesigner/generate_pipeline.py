#!/usr/bin/python
# -*- coding:utf-8 -*-
from functools import partial
import json
import os
from copy import deepcopy
import logging
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import torch
from torch.utils.data import DataLoader

import sys
current_working_dir = os.getcwd()
sys.path.append(current_working_dir)

from models import ADesigner
from data import AAComplex,AAComplex2, EquiAACDataset
from evaluation.rmsd import kabsch
from utils import check_dir
from utils.logger import print_log
from evaluation import compute_rmsd, tm_score
from argparse import ArgumentParser
import glob
import gc

import yaml
from easydict import EasyDict


def load_config(config_path):
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name

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



def eval_one(tup, out_dir, ref_pdb, pdb_n,cdr='H3', pdb_dict=None,processed_models=None):
    cplx, seq, x, true_x, aligned, k_index = tup


    model_name = f"{k_index:04d}.pdb"
    pdb_path = os.path.join(out_dir, model_name) #of the new cplx


    if pdb_path in processed_models:
        return None

    # if os.path.exists(pdb_path):
    #     return None

    # pdb_path = os.path.join(out_dir, f"{cplx.get_id()}.pdb") #of the new cplx

    entry_id = cplx.get_id()
    # pdb = entry_id.split('_')[0]

    selected_item = get_item_by_entry_id(pdb_dict, entry_id)
    
    summary = {
        'pdb': pdb_n,
        'heavy_chain': cplx.heavy_chain,
        'light_chain': cplx.light_chain if cplx.light_chain else "",  # if nanobody, this will be None, set to empty
        'antigen_chains': cplx.antigen_chains,
        'mod_pdb': pdb_path,
        'ref_pdb':ref_pdb,
        'cdr_model': "ADesign",
        "cdr_type":[cdr],
        "entry_id":entry_id
        }

    if selected_item != None:
        selected_item.update(summary)

    # selected_item = get_item_by_entry_id(pdb_dict, entry_id)
    # selected_item.update(summary)
    # selected_item = get_item_by_entry_id(pdb_dict, entry_id)
    # if selected_item is not None:
    #     selected_item.update(summary)
    #     # ref_pdb = selected_item["pdb_data_path"]
    #     # selected_item["ref_pdb"] = os.path.join(out_dir, cplx.get_id() + '_ref.pdb')
    # else:
    #     selected_item = summary

    # Merge to pdb_dict additional data from summary
    # Ensure pdb_dict is a dictionary; if not, initialize as empty dict
    # combined_summary = pdb_dict if isinstance(pdb_dict, dict) else {}
    # combined_summary.update(summary)


    # kabsch
    if aligned:
        ca_aligned = x[:, 1, :]
    else:
        ca_aligned, rotation, t = kabsch(x[:, 1, :], true_x[:, 1, :])
        x = np.dot(x - np.mean(x, axis=0), rotation) + t
    summary['RMSD'] = compute_rmsd(ca_aligned, true_x[:, 1, :], aligned=True)
    # set cdr
    new_cplx = set_cdr(cplx, seq, x, cdr)
    # pdb_path = os.path.join(out_dir, cplx.get_id() + '.pdb')
    new_cplx.to_pdb(pdb_path)

    processed_models.add(pdb_path)

    return summary



def rabd_test(args, model, test_set, test_loader, out_dir, device, pdb_dict, ref_pdb,pdb_n):
    if int(args.iteration) == 1:
        args.rabd_topk = 3
    else:
        args.rabd_topk = 7

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

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    heads = ['PPL']
    eval_res = [[] for _ in heads]

    # Ensure top models are selected and ordered by best PPL values
    for i in range(len(global_best_ppl)):
        sorted_indices = sorted(range(len(global_best_ppl[i])), key=lambda k: global_best_ppl[i][k])
        global_best_ppl[i] = [global_best_ppl[i][k] for k in sorted_indices][:args.rabd_topk]
        global_best_results[i] = [global_best_results[i][k] for k in sorted_indices][:args.rabd_topk]

    summary_fout_path = os.path.join(args.out_dir, f"summary_iter_{args.iteration}.json")
    processed_models = set()  # Initialize the set to track processed models
    with open(summary_fout_path, 'a') as summary_fout:
        for k in range(args.rabd_topk):
            inputs = []
            for cplx, items in zip(test_set.data, global_best_results):
                item = items[k]
                if item is not None:
                    inputs.append((cplx, ) + tuple(item) + (k,))

            inputs = inputs[:args.rabd_topk]
            summaries = process_map(partial(eval_one, out_dir=out_dir, ref_pdb=ref_pdb, pdb_n=pdb_n, cdr='H3', pdb_dict=pdb_dict, processed_models=processed_models), inputs, max_workers=args.num_workers, chunksize=10)

            for i, summary in enumerate(summaries):
                if summary is not None:
                    summary['PPL'] = global_best_ppl[i][k]
                    summary_fout.write(json.dumps(summary) + '\n')

            for i, h in enumerate(heads):
                eval_res[i].extend([summary[h] for summary in summaries if summary and h in summary])

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

    config, config_name = load_config(args.config)

    if int(args.iteration) == 1:
        args.rabd_topk = config['sampling']['num_samples_iter_1']
    else:
        args.rabd_topk = config['sampling']['num_samples_iter_x']

    from models.adesigner import ADesigner

    print(str(args))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    summary_file = os.path.join(args.out_dir, f"summary_iter_{args.iteration}.json")
    if not os.path.exists(summary_file):
        with open(summary_file, 'w') as f:
            pass

    if os.path.exists(summary_file) and os.stat(summary_file).st_size != 0:
        print(f"Summary file for fold {args.iteration} already exists and not empty, Skipping inference step...")
        exit()

    ckpt_path = args.ckpt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(ckpt_path, map_location=device)
    model.to(device)

    print("model cdr typ", model.cdr_type)


    logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO
    logger = logging.getLogger(__name__)
    logger.info(str(args))
    logger.info("model cdr type: %s", model.cdr_type)

    with open(args.test_set, 'r') as fin:
        data = fin.read().strip().split('\n')

    # create a dictionary of {pdb:whole dictionary correspinding to such pdb}
    pdb_dict = {}
    for item in data:
        json_obj = json.loads(item)
        json_obj["model"] = args.model #add model
        # pdb_dict[json_obj["entry_id"]] = json_obj
        pdb = json_obj.get("entry_id", json_obj.get("pdb"))
        pdb_dict[pdb] = json_obj


    for line in tqdm(data):
        item = json.loads(line)
        heavy, light, antigen = item['heavy_chain'], item['light_chain'], item['antigen_chains']
        ref_pdb = item.get("pdb_data_path")
        entry_id = item.get("entry_id")

        items_in_hdock_models = os.listdir(args.hdock_models)

        print("items_in_hdock_models ",items_in_hdock_models )

        dir_l = [item for item in items_in_hdock_models if os.path.isdir(os.path.join(args.hdock_models, item))]
        new_dir = []
        for dir_ in dir_l:
            if "tmp_dir" in dir_:
                continue
            new_dir.append(dir_)

        print("new_dir ",new_dir )

        for pdb_folder in new_dir: 
            pdb_folder_path = os.path.join(args.hdock_models, pdb_folder) 
            print("pdb_folder_path",pdb_folder_path) 
            if os.path.isdir(pdb_folder_path) and "tmp_dir_binding_computations" not in pdb_folder_path:
                pdb_model = os.path.basename(pdb_folder_path) #3g9a_2
                pdb_n = pdb_model

                hdock_models = []

                top_file = os.path.join(pdb_folder_path, "top_models.json")
                if os.path.exists(top_file):
                    with open(top_file, 'r') as f:
                        data = [json.loads(line) for line in f]
    
                    for entry in data:
                        path_pdb_to_design = entry.get("hdock_model","pdb_data_path")
                        model_hdock = path_pdb_to_design.split("/")[-1].split(".")[0]
                        entry["model"] = model_hdock
                        entry["ref_pdb"] = ref_pdb
                        entry["heavy_chain"] = heavy
                        entry["light_chain"] = light
                        entry["antigen_chains"] = antigen
                        entry["pdb_data_path"] = path_pdb_to_design
                        entry["entry_id"] = entry_id
                        hdock_models.append(entry)


                else:
                    pdb_files = glob.glob(os.path.join(pdb_folder_path, 'model_*.pdb'))
                    for pdb_file in pdb_files:
                        print(pdb_file)
                        model_hdock = pdb_file.split("/")[-1].split(".")[0]
                        item = {
                            "pdb_data_path": pdb_file,
                            "ref_pdb": ref_pdb,
                            "heavy_chain": heavy,
                            "light_chain": light,
                            "antigen_chains": antigen,
                            "entry_id": entry_id,
                            "model": model_hdock
                        }
                        hdock_models.append(item)

                processed_models = set()

                for hdock_model in hdock_models:

                    hmodel = hdock_model.get("model")

                    if hmodel in processed_models:
                        print(f"Skipping already processed model: {hmodel}")
                        continue
                    processed_models.add(hmodel)
                    
                    # design_file = os.path.join(pdb_folder_path, f"{hmodel}_for_inference.json")
                    # print("design_file",design_file)

                    # if not os.path.exists(design_file):
                    #     with open(design_file, 'w') as f:
                    #         f.write(json.dumps(hdock_model) + '\n')


                    # test_set = EquiAACDataset(design_file)#(args.test_set)
                    test_set = EquiAACDataset(args.test_set)

                    test_set.mode = args.mode
                    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            shuffle=False,
                                            collate_fn=test_set.collate_fn)

                    out_dir_n = os.path.join(args.out_dir,pdb_model,hmodel)

                    if not os.path.exists(out_dir_n):
                        os.makedirs(out_dir_n, exist_ok=True)
                    rabd_test(args, model, test_set, test_loader, out_dir_n, device, pdb_dict,ref_pdb, pdb_n)

        
    gc.collect()


# added:
def parse():
    parser = ArgumentParser(description='Generate antibody')
    parser.add_argument('--out_dir', type=str, default='./summaries')
    parser.add_argument('--test_set', type=str, default='./summaries')
    parser.add_argument('--hdock_models', type=str, default='./summaries')
    parser.add_argument('--mode', type=str, default='1*1')
    parser.add_argument('--rabd_topk', type=int, default=1)
    parser.add_argument('--rabd_sample', type=int, default=100)
    # parser.add_argument('--gpu', type=int, default=[0], nargs='+', help='gpu to use, -1 for cpu')
    parser.add_argument('--gpu', type=int, default=[0], nargs='+', help='gpu(s) to use, -1 for cpu')
    parser.add_argument('--ckpt', type=str, default='./summaries')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='Seed to use in training')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--iteration', type=int, default=4)
    parser.add_argument('--run', type=int, default=1, help='Number of runs for evaluation')
    parser.add_argument('--model', type=str, default='ADesigner', help='Number of runs for evaluation')
    parser.add_argument('--config', type=str, default='./NanoDesigner/config_files/NanoDesigner_ADesigner_codesign_single.yml')
    



    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    main(args)