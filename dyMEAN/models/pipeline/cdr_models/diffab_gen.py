#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
from tqdm import tqdm
from shutil import rmtree
import logging

logging.disable('INFO')

import sys
sys.path.append('/home/rioszemm/NanobodiesProject/diffab')

from diffab.tools.runner.design_for_pdb_original import design_for_pdb


class Arg:
    def __init__(self, pdb, heavy, light,antigen, config, out_root, pdb_code,summary_dir):
        self.pdb_path = pdb
        self.heavy = heavy
        self.light = light
        self.antigen=antigen
        self.no_renumber = True
        self.config = config
        self.out_root = out_root
        self.tag = ''
        self.seed = 0
        self.device = 'cuda'
        self.batch_size = 16
        self.pdb_code = pdb_code
        self.summary_dir = summary_dir


def main(args):
    # load dataset
    with open(args.dataset, 'r') as fin:
        lines = fin.read().strip().split('\n')

    # out_dir = os.path.join(os.path.split(args.config)[0], 'results') if args.out_dir is None else args.out_dir
    # out_dir = "/ibex/user/rioszemm/diffab_test_Nb/results"
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if not os.path.exists(args.summary_dir):
        with open(args.summary_dir, 'w') as f:
            pass

    #--config /home/rioszemm/NanobodiesProject/diffab/configs/test/codesign_single_200000_ab.yml
    # parts = args.config.split("/")
    # config_name = parts[-1].split(".")[0]
    # config_dir = os.path.join(args.out_dir,config_name)
    # if not os.path.exists(config_dir):
    #     os.makedirs(config_dir)

    # folder_results = os.listdir(config_dir)

    # pdb_list = [folder[:4] for folder in folder_results]

    # new_lines = []

    # for line in lines:
    #     try:
    #         data = json.loads(line)
    #         pdb_code = data.get('pdb')
    #         if pdb_code and pdb_code not in pdb_list:
    #             new_lines.append(line)
    #     except json.JSONDecodeError:
    #         print(f"Error decoding line: {line}")


    # for line in tqdm(lines):
    for line in tqdm(lines):
        item = json.loads(line)
        pdb = item['pdb_data_path']
        pdb_code = item["pdb"]
        heavy, light,antigen= item['heavy_chain'], item['light_chain'],item['antigen_chains']
        out_dir = os.path.join(args.out_dir,pdb_code)
        design_for_pdb(Arg(pdb, heavy, light,antigen, args.config, out_dir, pdb_code ,args.summary_dir))

    tmp_dir = os.path.join(args.out_dir, 'codesign_single')
    for f in os.listdir(tmp_dir):
        pdb_id = f[:4]
        pdb_file = os.path.join(tmp_dir, f, 'H_CDR3', '0000.pdb')
        tgt_file = os.path.join(args.out_dir, pdb_id + '_generated.pdb')
        os.system(f'cp {pdb_file} {tgt_file}')

    rmtree(tmp_dir)


def parse():
    parser = argparse.ArgumentParser(description='generation by diffab')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--summary_dir', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--config', type=str, required=True, help='config to the diffab model')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())


