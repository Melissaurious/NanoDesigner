#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import pickle
import argparse
from typing import List

import numpy as np
import torch
import glob
from utils.logger import print_log

########## import your packages below ##########
from tqdm import tqdm

from .pdb_utils import AgAbComplex, VOCAB, AgAbComplex_mod, AgAbComplex2
from .framework_templates import ConserveTemplateGenerator


def _generate_chain_data(residues, start):
    backbone_atoms = VOCAB.backbone_atoms
    # Coords, Sequence, residue positions, mask for loss calculation (exclude missing coordinates)
    X, S, res_pos, xloss_mask = [], [], [], []
    # global node
    # coordinates will be set to the center of the chain
    X.append([[0, 0, 0] for _ in range(VOCAB.MAX_ATOM_NUMBER)])  
    S.append(VOCAB.symbol_to_idx(start))
    res_pos.append(0)
    xloss_mask.append([0 for _ in range(VOCAB.MAX_ATOM_NUMBER)])
    # other nodes
    for residue in residues:
        residue_xloss_mask = [0 for _ in range(VOCAB.MAX_ATOM_NUMBER)]
        bb_atom_coord = residue.get_backbone_coord_map()
        sc_atom_coord = residue.get_sidechain_coord_map()
        if 'CA' not in bb_atom_coord:
            for atom in bb_atom_coord:
                ca_x = bb_atom_coord[atom]
                print_log(f'no ca, use {atom}', level='DEBUG')
                break
        else:
            ca_x = bb_atom_coord['CA']
        x = [ca_x for _ in range(VOCAB.MAX_ATOM_NUMBER)]
        
        i = 0
        for atom in backbone_atoms:
            if atom in bb_atom_coord:
                x[i] = bb_atom_coord[atom]
                residue_xloss_mask[i] = 1
            i += 1
        for atom in residue.sidechain:
            if atom in sc_atom_coord:
                x[i] = sc_atom_coord[atom]
                residue_xloss_mask[i] = 1
            i += 1

        X.append(x)
        S.append(VOCAB.symbol_to_idx(residue.get_symbol()))
        res_pos.append(residue.get_id()[0])
        xloss_mask.append(residue_xloss_mask)
    X = np.array(X)
    center = np.mean(X[1:].reshape(-1, 3), axis=0)
    X[0] = center  # set center
    if start == VOCAB.BOA:  # epitope does not have position encoding
        res_pos = [0 for _ in res_pos]
    data = {'X': X, 'S': S, 'residue_pos': res_pos, 'xloss_mask': xloss_mask}
    return data


# use this class to splice the dataset and maintain only one part of it in RAM
# Antibody-Antigen Complex dataset
class E2EDataset_mine(torch.utils.data.Dataset):
    def __init__(self, file_path, save_dir=None, cdr=None, paratope='H3', full_antigen=False, num_entry_per_file=-1, random=False):
        '''
        file_path: path to the dataset
        save_dir: directory to save the processed data
        cdr: which cdr to generate (L1/2/3, H1/2/3) (can be list), None for all including framework
        paratope: which cdr to use as paratope (L1/2/3, H1/2/3) (can be list)
        full_antigen: whether to use the full antigen information
        num_entry_per_file: number of entries in a single file. -1 to save all data into one file 
                            (In-memory dataset)
        '''
        super().__init__()
        self.cdr = cdr
        self.paratope = paratope
        self.full_antigen = full_antigen
        if save_dir is None:
            if not os.path.isdir(file_path):
                save_dir = os.path.split(file_path)[0]
            else:
                save_dir = file_path
            prefix = os.path.split(file_path)[1]
            if '.' in prefix:
                prefix = prefix.split('.')[0]
            save_dir = os.path.join(save_dir, f'{prefix}_processed')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        metainfo_file = os.path.join(save_dir, '_metainfo')
        # modification
        self.data: List[AgAbComplex2] = []  # list of ABComplex

        # try loading preprocessed files
        need_process = False
        try:
            with open(metainfo_file, 'r') as fin:
                metainfo = json.load(fin)
                self.num_entry = metainfo['num_entry']
                self.file_names = metainfo['file_names']
                self.file_num_entries = metainfo['file_num_entries']
        except FileNotFoundError:
            print_log('No meta-info file found, start processing', level='INFO')
            need_process = True
        except Exception as e:
            print_log(f'Faild to load file {metainfo_file}, error: {e}', level='WARN')
            need_process = True

        if need_process:
            # preprocess
            self.file_names, self.file_num_entries = [], []
            self.preprocess(file_path, save_dir, num_entry_per_file)
            self.num_entry = sum(self.file_num_entries)

            metainfo = {
                'num_entry': self.num_entry,
                'file_names': self.file_names,
                'file_num_entries': self.file_num_entries
            }
            with open(metainfo_file, 'w') as fout:
                json.dump(metainfo, fout)

        self.random = random
        self.cur_file_idx, self.cur_idx_range = 0, (0, self.file_num_entries[0])  # left close, right open
        self._load_part()

        # user defined variables
        self.idx_mapping = [i for i in range(self.num_entry)]
        self.mode = '1*1'  # H/L/Antigen, 1 for include, 0 for exclude 

    def _save_part(self, save_dir, num_entry):
        file_name = os.path.join(save_dir, f'part_{len(self.file_names)}.pkl')
        print_log(f'Saving {file_name} ...')
        file_name = os.path.abspath(file_name)
        if num_entry == -1:
            end = len(self.data)
        else:
            end = min(num_entry, len(self.data))
        with open(file_name, 'wb') as fout:
            pickle.dump(self.data[:end], fout)
        self.file_names.append(file_name)
        self.file_num_entries.append(end)
        self.data = self.data[end:]

    def _load_part(self):
        f = self.file_names[self.cur_file_idx]
        print_log(f'Loading preprocessed file {f}, {self.cur_file_idx + 1}/{len(self.file_names)}')
        with open(f, 'rb') as fin:
            del self.data
            self.data = pickle.load(fin)
        self.access_idx = [i for i in range(len(self.data))]
        if self.random:
            np.random.shuffle(self.access_idx)

    def _check_load_part(self, idx):
        if idx < self.cur_idx_range[0]:
            while idx < self.cur_idx_range[0]:
                end = self.cur_idx_range[0]
                self.cur_file_idx -= 1
                start = end - self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        elif idx >= self.cur_idx_range[1]:
            while idx >= self.cur_idx_range[1]:
                start = self.cur_idx_range[1]
                self.cur_file_idx += 1
                end = start + self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        idx = self.access_idx[idx - self.cur_idx_range[0]]
        return idx
     
    def __len__(self):
        return self.num_entry

    ########### load data from file_path and add to self.data ##########
    def preprocess(self, file_path, save_dir, num_entry_per_file):
        '''
        Load data from file_path and add processed data entries to self.data.
        Remember to call self._save_data(num_entry_per_file) to control the number
        of items in self.data (this function will save the first num_entry_per_file
        data and release them from self.data) e.g. call it when len(self.data) reaches
        num_entry_per_file.
        '''
        with open(file_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        # line_id = 0
        for line in tqdm(lines):
            # if line_id < 206:
            #     line_id += 1
            #     continue
            item = json.loads(line)
            try:
                #modification
                cplx = AgAbComplex2.from_pdb(
                    item['pdb_data_path'], item['heavy_chain'], item['light_chain'],
                    item['antigen_chains'])
            except AssertionError as e:
                print_log(e, level='ERROR')
                print_log(f'parse {item["pdb"]} pdb failed, skip', level='ERROR')
                continue

            self.data.append(cplx)
            if num_entry_per_file > 0 and len(self.data) >= num_entry_per_file:
                self._save_part(save_dir, num_entry_per_file)
        if len(self.data):
            self._save_part(save_dir, num_entry_per_file)

########## override get item ##########
    def __getitem__(self, idx):
        idx = self.idx_mapping[idx]
        idx = self._check_load_part(idx)
        item = self.data[idx]

        # Antigen processing
        ag_residues = []
        if self.full_antigen:
            ag = item.get_antigen()
            for chain in ag.get_chain_names():
                chain = ag.get_chain(chain)
                for i in range(len(chain)):
                    residue = chain.get_residue(i)
                    ag_residues.append(residue)
        else:
            for residue, chain, i in item.get_epitope():
                ag_residues.append(residue)
                
        ag_data = _generate_chain_data(ag_residues, VOCAB.BOA)

        hc = item.get_heavy_chain()
        hc_residues = []
        # generate heavy chain data
        for i in range(len(hc)):
            hc_residues.append(hc.get_residue(i))
        hc_data = _generate_chain_data(hc_residues, VOCAB.BOH)


        # Initialize data and masks
        data = {key: np.concatenate([ag_data[key], hc_data[key]], axis=0) for key in hc_data}
        cmask = [0 for _ in ag_data['S']] + [0] + [1 for _ in hc_data['S'][1:]]
        smask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']))]

        # Light chain processing (if present)
        if item.get_light_chain() is not None:
            lc = item.get_light_chain()
            lc_residues = []
            for i in range(len(lc)):
                lc_residues.append(lc.get_residue(i))
            lc_data = _generate_chain_data(lc_residues, VOCAB.BOL)

            # Append light chain data and extend masks
            for key in lc_data:
                data[key] = np.concatenate([data[key], lc_data[key]], axis=0)
            cmask += [0] + [1 for _ in lc_data['S'][1:]]
            smask += [0] + [1 for _ in lc_data['S'][1:]]

        # according to the setting of cdr
        if self.cdr is None:
            smask = cmask
        else:
            cdrs = [self.cdr] if isinstance(self.cdr, str) else self.cdr
            for cdr in cdrs:
                cdr_range = item.get_cdr_pos(cdr)
                offset = len(ag_data['S']) + 1 + (0 if cdr[0] == 'H' else len(hc_data['S']))
                for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
                    smask[idx] = 1

        paratope = [self.paratope] if isinstance(self.paratope, str) else self.paratope
        # paratope_mask = [0 for _ in range(len(data['S']))]
        # Initialize paratope_mask
        # If light chain is present, include its length in the paratope_mask
        if item.get_light_chain() is not None:
            paratope_mask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']) + len(lc_data['S']))]
        else:
            paratope_mask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']))]

        for cdr in paratope:
            cdr_range = item.get_cdr_pos(cdr)
            offset = len(ag_data['S']) + 1 + (0 if cdr[0] == 'H' else len(hc_data['S']))
            for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
                paratope_mask[idx] = 1

        data['cmask'], data['smask'], data['paratope_mask'] = cmask, smask, paratope_mask

        # Template generation
        template = ConserveTemplateGenerator().construct_template(item, align=False)
        data['template'] = template

        return data


    @classmethod
    def collate_fn(cls, batch):
        keys = ['X', 'S', 'smask', 'cmask', 'paratope_mask', 'residue_pos', 'template', 'xloss_mask']
        types = [torch.float, torch.long, torch.bool, torch.bool, torch.bool, torch.long, torch.float, torch.bool]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)
        lengths = [len(item['S']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        return res


class E2EDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, save_dir=None, cdr=None, paratope='H3', full_antigen=False, num_entry_per_file=-1, random=False):
        '''
        file_path: path to the dataset
        save_dir: directory to save the processed data
        cdr: which cdr to generate (L1/2/3, H1/2/3) (can be list), None for all including framework
        paratope: which cdr to use as paratope (L1/2/3, H1/2/3) (can be list)
        full_antigen: whether to use the full antigen information
        num_entry_per_file: number of entries in a single file. -1 to save all data into one file 
                            (In-memory dataset)
        '''
        super().__init__()
        self.cdr = cdr
        self.paratope = paratope
        self.full_antigen = full_antigen
        if save_dir is None:
            if not os.path.isdir(file_path):
                save_dir = os.path.split(file_path)[0]
            else:
                save_dir = file_path
            prefix = os.path.split(file_path)[1]
            if '.' in prefix:
                prefix = prefix.split('.')[0]
            save_dir = os.path.join(save_dir, f'{prefix}_processed')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        metainfo_file = os.path.join(save_dir, '_metainfo')
        self.data: List[AgAbComplex2] = []  # list of ABComplex

        # try loading preprocessed files
        need_process = False
        try:
            with open(metainfo_file, 'r') as fin:
                metainfo = json.load(fin)
                self.num_entry = metainfo['num_entry']
                self.file_names = metainfo['file_names']
                self.file_num_entries = metainfo['file_num_entries']
        except FileNotFoundError:
            print_log('No meta-info file found, start processing', level='INFO')
            need_process = True
        except Exception as e:
            print_log(f'Faild to load file {metainfo_file}, error: {e}', level='WARN')
            need_process = True

        if need_process:
            # preprocess
            self.file_names, self.file_num_entries = [], []
            self.preprocess(file_path, save_dir, num_entry_per_file)
            self.num_entry = sum(self.file_num_entries)

            metainfo = {
                'num_entry': self.num_entry,
                'file_names': self.file_names,
                'file_num_entries': self.file_num_entries
            }
            with open(metainfo_file, 'w') as fout:
                json.dump(metainfo, fout)

        self.random = random
        self.cur_file_idx, self.cur_idx_range = 0, (0, self.file_num_entries[0])  # left close, right open
        self._load_part()

        # user defined variables
        self.idx_mapping = [i for i in range(self.num_entry)]
        self.mode = '111'  # H/L/Antigen, 1 for include, 0 for exclude

    def _save_part(self, save_dir, num_entry):
        file_name = os.path.join(save_dir, f'part_{len(self.file_names)}.pkl')
        print_log(f'Saving {file_name} ...')
        file_name = os.path.abspath(file_name)
        if num_entry == -1:
            end = len(self.data)
        else:
            end = min(num_entry, len(self.data))
        with open(file_name, 'wb') as fout:
            pickle.dump(self.data[:end], fout)
        self.file_names.append(file_name)
        self.file_num_entries.append(end)
        self.data = self.data[end:]

    def _load_part(self):
        f = self.file_names[self.cur_file_idx]
        print_log(f'Loading preprocessed file {f}, {self.cur_file_idx + 1}/{len(self.file_names)}')
        with open(f, 'rb') as fin:
            del self.data
            self.data = pickle.load(fin)
        self.access_idx = [i for i in range(len(self.data))]
        if self.random:
            np.random.shuffle(self.access_idx)

    def _check_load_part(self, idx):
        if idx < self.cur_idx_range[0]:
            while idx < self.cur_idx_range[0]:
                end = self.cur_idx_range[0]
                self.cur_file_idx -= 1
                start = end - self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        elif idx >= self.cur_idx_range[1]:
            while idx >= self.cur_idx_range[1]:
                start = self.cur_idx_range[1]
                self.cur_file_idx += 1
                end = start + self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        idx = self.access_idx[idx - self.cur_idx_range[0]]
        return idx
     
    def __len__(self):
        return self.num_entry

    ########### load data from file_path and add to self.data ##########
    def preprocess(self, file_path, save_dir, num_entry_per_file):
        '''
        Load data from file_path and add processed data entries to self.data.
        Remember to call self._save_data(num_entry_per_file) to control the number
        of items in self.data (this function will save the first num_entry_per_file
        data and release them from self.data) e.g. call it when len(self.data) reaches
        num_entry_per_file.
        '''
        with open(file_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        # line_id = 0
        for line in tqdm(lines):
            # if line_id < 206:
            #     line_id += 1
            #     continue
            item = json.loads(line)
            try:
                cplx = AgAbComplex2.from_pdb(
                    item['pdb_data_path'], item['heavy_chain'], item['light_chain'],
                    item['antigen_chains'])
            except AssertionError as e:
                print_log(e, level='ERROR')
                print_log(f'parse {item["pdb"]} pdb failed, skip', level='ERROR')
                continue

            self.data.append(cplx)
            if num_entry_per_file > 0 and len(self.data) >= num_entry_per_file:
                self._save_part(save_dir, num_entry_per_file)
        if len(self.data):
            self._save_part(save_dir, num_entry_per_file)

    ########## override get item ##########
    def __getitem__(self, idx):
        '''
        an example of the returned data
        {
            'X': [n, n_channel, 3],
            'S': [n],
            'cmask': [n],
            'smask': [n],
            'paratope_mask': [n],
            'xloss_mask': [n, n_channel],
            'template': [n, n_channel, 3]
        }
        '''
        idx = self.idx_mapping[idx]
        idx = self._check_load_part(idx)
        item = self.data[idx]

        # antigen
        ag_residues = []

        if self.full_antigen:
            # get antigen residues
            ag = item.get_antigen()
            for chain in ag.get_chain_names():
                chain = ag.get_chain(chain)
                for i in range(len(chain)):
                    residue = chain.get_residue(i)
                    ag_residues.append(residue)
        else:
            # get antigen residues (epitope only)
            for residue, chain, i in item.get_epitope():
                ag_residues.append(residue)
    
        # generate antigen data
        ag_data = _generate_chain_data(ag_residues, VOCAB.BOA)


        hc, lc = item.get_heavy_chain(), item.get_light_chain()
        hc_residues = []

        # Heavy chain data generation
        for i in range(len(hc)):
            hc_residues.append(hc.get_residue(i))
        hc_data = _generate_chain_data(hc_residues, VOCAB.BOH)

        # Initialize light chain data
        lc_data = None

        # Check if light chain is present
        if lc is not None:
            lc_residues = []
            for i in range(len(lc)):
                lc_residues.append(lc.get_residue(i))
            lc_data = _generate_chain_data(lc_residues, VOCAB.BOL)

        # Data concatenation with condition on light chain
        if lc_data is not None:
            data = {key: np.concatenate([ag_data[key], hc_data[key], lc_data[key]], axis=0) 
                    for key in hc_data}
        else:
            data = {key: np.concatenate([ag_data[key], hc_data[key]], axis=0) 
                    for key in hc_data}

        # Generate cmask and smask with condition on light chain
        cmask = [0 for _ in ag_data['S']] + [0] + [1 for _ in hc_data['S'][1:]]
        if lc_data is not None:
            cmask += [0] + [1 for _ in lc_data['S'][1:]]

        if self.cdr is None:
            smask = cmask.copy()
        else:
            smask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']))]
            if lc_data is not None:
                smask += [0 for _ in lc_data['S']]
            cdrs = [self.cdr] if type(self.cdr) == str else self.cdr
            for cdr in cdrs:
                cdr_range = item.get_cdr_pos(cdr)
                offset = len(ag_data['S']) + 1 + (0 if cdr[0] == 'H' else len(hc_data['S']))
                for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
                    smask[idx] = 1

        data['cmask'], data['smask'] = cmask, smask

        # Initialize paratope_mask with antigen and heavy chain lengths
        paratope_mask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']))]

        # Extend paratope_mask if light chain is present
        if lc_data is not None:
            paratope_mask += [0 for _ in lc_data['S']]

        # Set paratope
        paratope = [self.paratope] if type(self.paratope) == str else self.paratope
        for cdr in paratope:
            cdr_range = item.get_cdr_pos(cdr)
            offset = len(ag_data['S']) + 1 + (0 if cdr[0] == 'H' else len(hc_data['S']))
            for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
                paratope_mask[idx] = 1

        data['paratope_mask'] = paratope_mask


        template = ConserveTemplateGenerator().construct_template(item, align=False)
        data['template'] = template

        return data

    @classmethod
    def collate_fn(cls, batch):
        keys = ['X', 'S', 'smask', 'cmask', 'paratope_mask', 'residue_pos', 'template', 'xloss_mask']
        types = [torch.float, torch.long, torch.bool, torch.bool, torch.bool, torch.long, torch.float, torch.bool]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)
        lengths = [len(item['S']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        return res

# use this class to splice the dataset and maintain only one part of it in RAM
# Antibody-Antigen Complex dataset
class my_E2EDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, save_dir=None, cdr=None,
                 force_reinit:bool = False, 
                 paratope='H3', full_antigen=False, num_entry_per_file=-1,
                 random=False, hdock_models_path:str = None):
        '''
        file_path: path to the dataset
        save_dir: directory to save the processed data
        cdr: which cdr to generate (L1/2/3, H1/2/3) (can be list), None for all including framework
        paratope: which cdr to use as paratope (L1/2/3, H1/2/3) (can be list)
        full_antigen: whether to use the full antigen information
        num_entry_per_file: number of entries in a single file. -1 to save all data into one file 
                            (In-memory dataset)
        '''
        super().__init__()
        self.cdr = cdr
        self.hdock_models_path = hdock_models_path
        assert hdock_models_path is not None, f'Please provide file location for hdock models'
        self.paratope = paratope
        self.full_antigen = full_antigen
        if save_dir is None:
            if not os.path.isdir(file_path):
                save_dir = os.path.split(file_path)[0]
            else:
                save_dir = file_path
            prefix = os.path.split(file_path)[1]
            if '.' in prefix:
                prefix = prefix.split('.')[0]
            save_dir = os.path.join(save_dir, f'{prefix}_processed')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.metainfo_file = metainfo_file = os.path.join(save_dir, '_metainfo')
        # modification
        self.data: List[AgAbComplex_mod] = []  # list of ABComplex

        # try loading preprocessed files
        need_process = False
        try:
            with open(metainfo_file, 'r') as fin:
                self.metainfo = metainfo = json.load(fin)
                self.num_entry = metainfo['num_entry']
                self.file_names = metainfo['file_names']
                self.file_num_entries = metainfo['file_num_entries']
        except FileNotFoundError:
            print_log('No meta-info file found, start processing', level='INFO')
            need_process = True
        except Exception as e:
            print_log(f'Faild to load file {metainfo_file}, error: {e}', level='WARN')
            need_process = True
        if force_reinit:
            print_log('Doing reprocessing as specified')
            need_process = True
        if need_process:
            # preprocess
            self.file_names, self.file_num_entries = [], []
            self.preprocess(file_path, save_dir, num_entry_per_file)
            self.num_entry = sum(self.file_num_entries)

            metainfo = {
                'num_entry': self.num_entry,
                'file_names': self.file_names,
                'file_num_entries': self.file_num_entries
            }
            with open(metainfo_file, 'w') as fout:
                json.dump(metainfo, fout)

        self.random = random
        self.cur_file_idx, self.cur_idx_range = 0, (0, self.file_num_entries[0])  # left close, right open
        self._load_part()

        # user defined variables
        self.idx_mapping = [i for i in range(self.num_entry)]
        self.mode = '101'  # H/L/Antigen, 1 for include, 0 for exclude 

    def _save_part(self, save_dir, num_entry):
        file_name = os.path.join(save_dir, f'part_{len(self.file_names)}.pkl')
        print_log(f'Saving {file_name} ...')
        file_name = os.path.abspath(file_name)
        if num_entry == -1:
            end = len(self.data)
        else:
            end = min(num_entry, len(self.data))
        with open(file_name, 'wb') as fout:
            pickle.dump(self.data[:end], fout)
        self.file_names.append(file_name)
        self.file_num_entries.append(end)
        self.data = self.data[end:]

    def _load_part(self):
        f = self.file_names[self.cur_file_idx]
        print_log(f'Loading preprocessed file {f}, {self.cur_file_idx + 1}/{len(self.file_names)}')
        with open(f, 'rb') as fin:
            del self.data
            self.data = pickle.load(fin)
        self.access_idx = [i for i in range(len(self.data))]
        if self.random:
            np.random.shuffle(self.access_idx)

    def _check_load_part(self, idx):
        if idx < self.cur_idx_range[0]:
            while idx < self.cur_idx_range[0]:
                end = self.cur_idx_range[0]
                self.cur_file_idx -= 1
                start = end - self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        elif idx >= self.cur_idx_range[1]:
            while idx >= self.cur_idx_range[1]:
                start = self.cur_idx_range[1]
                self.cur_file_idx += 1
                end = start + self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        idx = self.access_idx[idx - self.cur_idx_range[0]]
        return idx
     
    def __len__(self):
        return self.num_entry

    ########### load data from file_path and add to self.data ##########
    def preprocess(self, file_path, save_dir, num_entry_per_file):
        '''
        Load data from file_path and add processed data entries to self.data.
        Remember to call self._save_data(num_entry_per_file) to control the number
        of items in self.data (this function will save the first num_entry_per_file
        data and release them from self.data) e.g. call it when len(self.data) reaches
        num_entry_per_file.
        '''
        with open(file_path, 'r') as fin: #reads dataset.json
            lines = fin.read().strip().split('\n')
        # line_id = 0
        for line in tqdm(lines):
            # if line_id < 206:
            #     line_id += 1
            #     continue
            item = json.loads(line)
            print(f"at my_E2EDataset {item} used to extract heavy and antigen information")
            for hdock_model in glob.glob(os.path.join(self.hdock_models_path, item['pdb'], f'model_*.pdb')):
                print("preparing information for", hdock_model)
                try:
                    #modification
                    cplx = AgAbComplex_mod.from_pdb(
                        hdock_model, item['heavy_chain'],
                        item['antigen_chains'])
                except AssertionError as e:
                    print_log(e, level='ERROR')
                    print_log(f'parse {item["pdb"]} pdb failed, skip', level='ERROR')
                    continue

                self.data.append(cplx)
                if num_entry_per_file > 0 and len(self.data) >= num_entry_per_file:
                    self._save_part(save_dir, num_entry_per_file)
        if len(self.data):
            self._save_part(save_dir, num_entry_per_file)

    ########## override get item ##########
    def __getitem__(self, idx):
        '''
        an example of the returned data
        {
            'X': [n, n_channel, 3],
            'S': [n],
            'cmask': [n],
            'smask': [n],
            'paratope_mask': [n],
            'xloss_mask': [n, n_channel],
            'template': [n, n_channel, 3]
        }
        '''
        idx = self.idx_mapping[idx]
        idx = self._check_load_part(idx)
        item = self.data[idx]

        # antigen
        ag_residues = []

        if self.full_antigen:
            # get antigen residues
            ag = item.get_antigen()
            for chain in ag.get_chain_names():
                chain = ag.get_chain(chain)
                for i in range(len(chain)):
                    residue = chain.get_residue(i)
                    ag_residues.append(residue)
        else:
            # get antigen residues (epitope only)
            for residue, chain, i in item.get_epitope():
                ag_residues.append(residue)
    
        # generate antigen data
        ag_data = _generate_chain_data(ag_residues, VOCAB.BOA)

        # hc, lc = item.get_heavy_chain(), item.get_light_chain()
        # hc_residues, lc_residues = [], []

        # modification
        hc = item.get_heavy_chain()
        hc_residues, lc_residues = [], []

        # generate heavy chain data
        for i in range(len(hc)):
            hc_residues.append(hc.get_residue(i))
        hc_data = _generate_chain_data(hc_residues, VOCAB.BOH)

        # generate light chain data
        # modification
        # for i in range(len(lc)):
        #     lc_residues.append(lc.get_residue(i))
        # lc_data = _generate_chain_data(lc_residues, VOCAB.BOL)

        # modification
        # data = { key: np.concatenate([ag_data[key], hc_data[key], lc_data[key]], axis=0) \
        #          for key in hc_data}
        data = { key: np.concatenate([ag_data[key], hc_data[key]], axis=0) for key in hc_data}

        # smask (sequence) and cmask (coordinates): 0 for fixed, 1 for generate
        # not generate coordinates of global node and antigen 
        # modification
        cmask = [0 for _ in ag_data['S']] + [0] + [1 for _ in hc_data['S'][1:]]# + [0] + [1 for _ in lc_data['S'][1:]]
        # according to the setting of cdr
        if self.cdr is None:
            smask = cmask
        else:
            # modification
            # smask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']) + len(lc_data['S']))]
            smask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']))]
            cdrs = [self.cdr] if type(self.cdr) == str else self.cdr
            for cdr in cdrs:
                cdr_range = item.get_cdr_pos(cdr)
                offset = len(ag_data['S']) + 1 + (0 if cdr[0] == 'H' else len(hc_data['S']))
                for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
                    smask[idx] = 1

        data['cmask'], data['smask'] = cmask, smask

        # modification
        # paratope_mask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']) + len(lc_data['S']))]
        paratope_mask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']) )]
        paratope = [self.paratope] if type(self.paratope) == str else self.paratope
        for cdr in paratope:
            cdr_range = item.get_cdr_pos(cdr)
            offset = len(ag_data['S']) + 1 + (0 if cdr[0] == 'H' else len(hc_data['S']))
            for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
                paratope_mask[idx] = 1
        data['paratope_mask'] = paratope_mask

        template = ConserveTemplateGenerator().construct_template(item, align=False)
        data['template'] = template

        # print("data from my_E2EDataset", data) # this is a huge tensor

        return data

    @classmethod
    def collate_fn(cls, batch):
        keys = ['X', 'S', 'smask', 'cmask', 'paratope_mask', 'residue_pos', 'template', 'xloss_mask']
        types = [torch.float, torch.long, torch.bool, torch.bool, torch.bool, torch.long, torch.float, torch.bool]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)
        lengths = [len(item['S']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        return res



# use this class to splice the dataset and maintain only one part of it in RAM
# Antibody-Antigen Complex dataset
class E2EDataset2(torch.utils.data.Dataset):
    # def __init__(self, file_path, save_dir=None, cdr=None, paratope='H3', full_antigen=False, num_entry_per_file=-1, random=False):
    def __init__(self, file_path, template_generator=None, save_dir=None, cdr=None, paratope='H3', full_antigen=False, num_entry_per_file=-1, random=False):  # to add my templates

        '''
        file_path: path to the dataset
        save_dir: directory to save the processed data
        cdr: which cdr to generate (L1/2/3, H1/2/3) (can be list), None for all including framework
        paratope: which cdr to use as paratope (L1/2/3, H1/2/3) (can be list)
        full_antigen: whether to use the full antigen information
        num_entry_per_file: number of entries in a single file. -1 to save all data into one file 
                            (In-memory dataset)
        '''
        super().__init__()
        self.cdr = cdr
        self.paratope = paratope
        self.template_generator = template_generator #
        self.full_antigen = full_antigen
        if save_dir is None:
            if not os.path.isdir(file_path):
                save_dir = os.path.split(file_path)[0]
            else:
                save_dir = file_path
            prefix = os.path.split(file_path)[1]
            if '.' in prefix:
                prefix = prefix.split('.')[0]
            save_dir = os.path.join(save_dir, f'{prefix}_processed')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        metainfo_file = os.path.join(save_dir, '_metainfo')
        self.data: List[AgAbComplex2] = []  # list of ABComplex

        # try loading preprocessed files
        need_process = False
        try:
            with open(metainfo_file, 'r') as fin:
                metainfo = json.load(fin)
                self.num_entry = metainfo['num_entry']
                self.file_names = metainfo['file_names']
                self.file_num_entries = metainfo['file_num_entries']
        except FileNotFoundError:
            print_log('No meta-info file found, start processing', level='INFO')
            need_process = True
        except Exception as e:
            print_log(f'Faild to load file {metainfo_file}, error: {e}', level='WARN')
            need_process = True

        if need_process:
            # preprocess
            self.file_names, self.file_num_entries = [], []
            self.preprocess(file_path, save_dir, num_entry_per_file)
            self.num_entry = sum(self.file_num_entries)

            metainfo = {
                'num_entry': self.num_entry,
                'file_names': self.file_names,
                'file_num_entries': self.file_num_entries
            }
            with open(metainfo_file, 'w') as fout:
                json.dump(metainfo, fout)

        self.random = random
        self.cur_file_idx, self.cur_idx_range = 0, (0, self.file_num_entries[0])  # left close, right open
        self._load_part()

        # user defined variables
        self.idx_mapping = [i for i in range(self.num_entry)]
        self.mode = '1*1'  # H/L/Antigen, 1 for include, 0 for exclude, * relaxed filter, may be present or not
        # this mode is never explicitly used it

    def __len__(self): # ADDED
        return len(self.data)
    
    def _save_part(self, save_dir, num_entry):
        file_name = os.path.join(save_dir, f'part_{len(self.file_names)}.pkl')
        print_log(f'Saving {file_name} ...')
        file_name = os.path.abspath(file_name)
        if num_entry == -1:
            end = len(self.data)
        else:
            end = min(num_entry, len(self.data))
        with open(file_name, 'wb') as fout:
            pickle.dump(self.data[:end], fout)
        self.file_names.append(file_name)
        self.file_num_entries.append(end)
        self.data = self.data[end:]

    def _load_part(self):
        f = self.file_names[self.cur_file_idx]
        print_log(f'Loading preprocessed file {f}, {self.cur_file_idx + 1}/{len(self.file_names)}')
        with open(f, 'rb') as fin:
            del self.data
            self.data = pickle.load(fin)
        self.access_idx = [i for i in range(len(self.data))]
        if self.random:
            np.random.shuffle(self.access_idx)

    def _check_load_part(self, idx):
        if idx < self.cur_idx_range[0]:
            while idx < self.cur_idx_range[0]:
                end = self.cur_idx_range[0]
                self.cur_file_idx -= 1
                start = end - self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        elif idx >= self.cur_idx_range[1]:
            while idx >= self.cur_idx_range[1]:
                start = self.cur_idx_range[1]
                self.cur_file_idx += 1
                end = start + self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        idx = self.access_idx[idx - self.cur_idx_range[0]]
        return idx     

    ########### load data from file_path and add to self.data ##########
    def preprocess(self, file_path, save_dir, num_entry_per_file):
        '''
        Load data from file_path and add processed data entries to self.data.
        Remember to call self._save_data(num_entry_per_file) to control the number
        of items in self.data (this function will save the first num_entry_per_file
        data and release them from self.data) e.g. call it when len(self.data) reaches
        num_entry_per_file.
        '''
        with open(file_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        for line in tqdm(lines):
            item = json.loads(line)
            try:
                cplx = AgAbComplex2.from_pdb(
                    item['pdb_data_path'], item['heavy_chain'], item['light_chain'],
                    item['antigen_chains'])
            except AssertionError as e:
                print_log(e, level='ERROR')
                print_log(f'parse {item["pdb"]} pdb failed, skip', level='ERROR')
                continue

            self.data.append(cplx)
            if num_entry_per_file > 0 and len(self.data) >= num_entry_per_file:
                self._save_part(save_dir, num_entry_per_file)
        if len(self.data):
            self._save_part(save_dir, num_entry_per_file)

    ########## override get item ##########
    def __getitem__(self, idx):
        '''
        an example of the returned data
        {
            'X': [n, n_channel, 3],
            'S': [n],
            'cmask': [n],
            'smask': [n],
            'paratope_mask': [n],
            'xloss_mask': [n, n_channel],
            'template': [n, n_channel, 3]
        }
        '''
        idx = self.idx_mapping[idx]
        idx = self._check_load_part(idx)
        item = self.data[idx]

        # antigen
        ag_residues = []

        if self.full_antigen:
            # get antigen residues
            ag = item.get_antigen()
            for chain in ag.get_chain_names():
                chain = ag.get_chain(chain)
                for i in range(len(chain)):
                    residue = chain.get_residue(i)
                    ag_residues.append(residue)
        else:
            # get antigen residues (epitope only)
            for residue, chain, i in item.get_epitope():
                ag_residues.append(residue)

        # generate antigen data
        ag_data = _generate_chain_data(ag_residues, VOCAB.BOA)

        
        hc, lc = item.get_heavy_chain(), item.get_light_chain() # lc will be none if there is not light chain

        hc_residues, lc_residues = [], []

        # generate heavy chain data
        for i in range(len(hc)):
            hc_residues.append(hc.get_residue(i))
        hc_data = _generate_chain_data(hc_residues, VOCAB.BOH)


        # generate light chain data
        lc_data = None
        if lc is not None:
            for i in range(len(lc)):
                lc_residues.append(lc.get_residue(i))
            lc_data = _generate_chain_data(lc_residues, VOCAB.BOL)


        # Combine antigen, heavy chain, and light chain data
        data = {key: np.concatenate([ag_data[key], hc_data[key]] +
                                    ([] if lc_data is None else [lc_data[key]]), axis=0)
                for key in hc_data}


        # Adjust cmask and smask for the light chain being optional
        cmask = [0 for _ in ag_data['S']] + [0] + [1 for _ in hc_data['S'][1:]]
        smask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']))]
        if lc_data is not None:
            cmask += [0] + [1 for _ in lc_data['S'][1:]]
            smask += [0 for _ in lc_data['S']]

        # smask (sequence) and cmask (coordinates): 0 for fixed, 1 for generate
        # not generate coordinates of global node and antigen 
        cmask = [0 for _ in ag_data['S']] + [0] + [1 for _ in hc_data['S'][1:]]
        if lc_data is not None:
            cmask += [0] + [1 for _ in lc_data['S'][1:]]
        # according to the setting of cdr
        if self.cdr is None:
            smask = cmask
        else:
            smask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']))]
            if lc_data is not None:
                smask += [0 for _ in lc_data['S']]
            cdrs = [self.cdr] if type(self.cdr) == str else self.cdr
            for cdr in cdrs:
                cdr_range = item.get_cdr_pos(cdr)
                offset = len(ag_data['S']) + 1 + (0 if cdr[0] == 'H' else len(hc_data['S']))
                for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
                    smask[idx] = 1

        data['cmask'], data['smask'] = cmask, smask

        # Adjust paratope_mask calculation to handle cases where light chain may be None
        paratope_mask_length = len(ag_data['S']) + len(hc_data['S'])
        if lc_data is not None:
            paratope_mask_length += len(lc_data['S'])
            
        paratope_mask = [0 for _ in range(paratope_mask_length)]

        # paratope_mask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']) + len(lc_data['S']))]
        paratope = [self.paratope] if type(self.paratope) == str else self.paratope
        for cdr in paratope:
            cdr_range = item.get_cdr_pos(cdr)
            offset = len(ag_data['S']) + 1 + (0 if cdr[0] == 'H' else len(hc_data['S']))
            for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
                paratope_mask[idx] = 1
        data['paratope_mask'] = paratope_mask

        # template = ConserveTemplateGenerator().construct_template(item, align=False)
        template = self.template_generator.construct_template(item, align=False)
        data['template'] = template

        return data

    @classmethod
    def collate_fn(cls, batch):
        keys = ['X', 'S', 'smask', 'cmask', 'paratope_mask', 'residue_pos', 'template', 'xloss_mask']
        types = [torch.float, torch.long, torch.bool, torch.bool, torch.bool, torch.long, torch.float, torch.bool]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)
        lengths = [len(item['S']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        return res


def parse():
    parser = argparse.ArgumentParser(description='Process data')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to save processed data')
    return parser.parse_args()
 

if __name__ == '__main__':
    args = parse()
    dataset = E2EDataset2(args.dataset, args.save_dir, num_entry_per_file=-1)
    print(len(dataset))




# def __getitem__(self, idx):
#     # ... [previous code remains unchanged] ...

#     hc, lc = item.get_heavy_chain(), item.get_light_chain()
#     hc_residues = []

#     # Heavy chain data generation
#     for i in range(len(hc)):
#         hc_residues.append(hc.get_residue(i))
#     hc_data = _generate_chain_data(hc_residues, VOCAB.BOH)

#     # Initialize light chain data
#     lc_data = None

#     # Check if light chain is present
#     if lc is not None:
#         lc_residues = []
#         for i in range(len(lc)):
#             lc_residues.append(lc.get_residue(i))
#         lc_data = _generate_chain_data(lc_residues, VOCAB.BOL)

#     # Data concatenation with condition on light chain
#     if lc_data is not None:
#         data = {key: np.concatenate([ag_data[key], hc_data[key], lc_data[key]], axis=0) 
#                 for key in hc_data}
#     else:
#         data = {key: np.concatenate([ag_data[key], hc_data[key]], axis=0) 
#                 for key in hc_data}

#     # Generate cmask and smask with condition on light chain
#     cmask = [0 for _ in ag_data['S']] + [0] + [1 for _ in hc_data['S'][1:]]
#     if lc_data is not None:
#         cmask += [0] + [1 for _ in lc_data['S'][1:]]

#     if self.cdr is None:
#         smask = cmask.copy()
#     else:
#         smask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']))]
#         if lc_data is not None:
#             smask += [0 for _ in lc_data['S']]
#         cdrs = [self.cdr] if type(self.cdr) == str else self.cdr
#         for cdr in cdrs:
#             cdr_range = item.get_cdr_pos(cdr)
#             offset = len(ag_data['S']) + 1 + (0 if cdr[0] == 'H' else len(hc_data['S']))
#             for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
#                 smask[idx] = 1

#     # ... [rest of your existing code for setting paratope mask and template] ...

#     return data
