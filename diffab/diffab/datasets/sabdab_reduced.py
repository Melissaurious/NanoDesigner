import os
import random
import logging
import datetime
import pandas as pd
import joblib
import lmdb
import pickle
import subprocess
import torch
from Bio import PDB, SeqRecord, SeqIO, Seq
from Bio.PDB import PDBExceptions
from Bio.PDB import Polypeptide
from torch.utils.data import Dataset
from tqdm.auto import tqdm
# from ..utils.protein import parsers, constants
# from ._base import register_dataset
import sys
sys.path.append('/ibex/user/rioszemm/NanobodiesProject/diffab')


from diffab.utils.protein import parsers, constants
from diffab.datasets._base import register_dataset
from torchvision.transforms import Compose
import json


ALLOWED_AG_TYPES = {
    'protein',
    'protein | protein',
    'protein | protein | protein',
    'protein | protein | protein | protein | protein',
    'protein | protein | protein | protein',
}

RESOLUTION_THRESHOLD = 4.0

TEST_ANTIGENS = [
    'sars-cov-2 receptor binding domain',
    'hiv-1 envelope glycoprotein gp160',
    'mers s',
    'influenza a virus',
    'cd27 antigen',
]


def nan_to_empty_string(val):
    if val != val or not val:
        return ''
    else:
        return val


def nan_to_none(val):
    if val != val or not val:
        return None
    else:
        return val


def split_sabdab_delimited_str(val):
    if not val:
        return []
    else:
        return [s.strip() for s in val.split('|')]


def parse_sabdab_resolution(val):
    if val == 'NOT' or not val or val != val:
        return None
    elif isinstance(val, str) and ',' in val:
        return float(val.split(',')[0].strip())
    else:
        return float(val)


def _aa_tensor_to_sequence(aa):
    return ''.join([Polypeptide.index_to_one(a.item()) for a in aa.flatten()])


def _label_heavy_chain_cdr(data, seq_map, max_cdr3_length=30):
    if data is None or seq_map is None:
        return data, seq_map

    # Add CDR labels
    cdr_flag = torch.zeros_like(data['aa'])
    for position, idx in seq_map.items():
        resseq = position[1]
        cdr_type = constants.ChothiaCDRRange.to_cdr('H', resseq)
        if cdr_type is not None:
            cdr_flag[idx] = cdr_type
    data['cdr_flag'] = cdr_flag

    # Add CDR sequence annotations
    data['H1_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H1] )
    data['H2_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H2] )
    data['H3_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H3] )

    cdr3_length = (cdr_flag == constants.CDR.H3).sum().item()
    # Remove too long CDR3
    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.H3] = 0
        logging.warning(f'CDR-H3 too long {cdr3_length}. Removed.')
        return None, None

    # Filter: ensure CDR3 exists
    if cdr3_length == 0:
        logging.warning('No CDR-H3 found in the heavy chain.')
        return None, None

    return data, seq_map


def _label_light_chain_cdr(data, seq_map, max_cdr3_length=30):
    if data is None or seq_map is None:
        return data, seq_map
    cdr_flag = torch.zeros_like(data['aa'])
    for position, idx in seq_map.items():
        resseq = position[1]
        cdr_type = constants.ChothiaCDRRange.to_cdr('L', resseq)
        if cdr_type is not None:
            cdr_flag[idx] = cdr_type
    data['cdr_flag'] = cdr_flag

    data['L1_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L1] )
    data['L2_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L2] )
    data['L3_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L3] )

    cdr3_length = (cdr_flag == constants.CDR.L3).sum().item()
    # Remove too long CDR3
    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.L3] = 0
        logging.warning(f'CDR-L3 too long {cdr3_length}. Removed.')
        return None, None

    # Ensure CDR3 exists
    if cdr3_length == 0:
        logging.warning('No CDRs found in the light chain.')
        return None, None

    return data, seq_map


def preprocess_sabdab_structure(task):
    entry = task['entry']
    pdb_path = task['pdb_path']

    parser = PDB.PDBParser(QUIET=True)
    model = parser.get_structure(id, pdb_path)[0]

    #Added to check if the chains described in the tsv file are indeed there and avoid future errors
    structure = parser.get_structure(entry['pdbcode'], pdb_path)[0]

    # Check if heavy chain is present in the PDB structure
    if entry['H_chain'] is not None:
        heavy_chain_present = entry['H_chain'] in [chain.id for chain in structure.get_chains()]
        if not heavy_chain_present:
            logging.warning(f'Heavy chain {entry["H_chain"]} not found for {entry["id"]}. Skipping entry.')
            return None



    # Extract all chain IDs from the structure
    structure_chain_ids = [chain.id for chain in structure.get_chains()]

    # Check if all antigen chains are present in the PDB structure
    antigen_chains_present = all(c in structure_chain_ids for c in entry['ag_chains'])
    if not antigen_chains_present:
        missing_chains = [c for c in entry['ag_chains'] if c not in structure_chain_ids]
        logging.warning(f'Antigen chains {missing_chains} not found for {entry["id"]}. Skipping entry.')
        return None
    

    # Check if heavy chain is present in the PDB structure
    if entry['L_chain'] is not None:
        heavy_chain_present = entry['L_chain'] in [chain.id for chain in structure.get_chains()]
        if not heavy_chain_present:
            logging.warning(f'Light chain {entry["L_chain"]} not found for {entry["id"]}. Skipping entry.')
            return None 

    parsed = {
        'id': entry['id'],
        'heavy': None,
        'heavy_seqmap': None,
        'light': None,
        'light_seqmap': None,
        'antigen': None,
        'antigen_seqmap': None,
    }
    try:
        if entry['H_chain'] is not None:
            (
                parsed['heavy'], 
                parsed['heavy_seqmap']
            ) = _label_heavy_chain_cdr(*parsers.parse_biopython_structure(
                model[entry['H_chain']],
                max_resseq = 250    # Chothia, end of Heavy chain Fv
                #original was 113
            ))
        
        if entry['L_chain'] is not None:
            (
                parsed['light'], 
                parsed['light_seqmap']
            ) = _label_light_chain_cdr(*parsers.parse_biopython_structure(
                model[entry['L_chain']],
                max_resseq = 107    # Chothia, end of Light chain Fv
            ))

        if parsed['heavy'] is None and parsed['light'] is None:
            raise ValueError('Neither valid H-chain or L-chain is found.')
    
        if len(entry['ag_chains']) > 0:
            chains = [model[c] for c in entry['ag_chains']]
            (
                parsed['antigen'], 
                parsed['antigen_seqmap']
            ) = parsers.parse_biopython_structure(chains)

    except (
        PDBExceptions.PDBConstructionException, 
        parsers.ParsingException, 
        KeyError,
        ValueError,
    ) as e:
        logging.warning('[{}] {}: {}'.format(
            task['id'], 
            e.__class__.__name__, 
            str(e)
        ))
        return None

    # print(parsed)
    return parsed



class SAbDabDataset(Dataset):

    MAP_SIZE = 32*(1024*1024*1024)  # 32GB
    def __init__(
        self, 
        summary_path,  
        chothia_dir,
        json_file,   
        processed_dir,  
        split,        
        split_seed,   
        transform,     
        reset,
        special_filter,
        fold
    ):
        super().__init__()

        self.fold = fold 

        print("conducting fold = ", self.fold)
        self.special_filter = special_filter
        self.summary_path = summary_path
        self.chothia_dir = chothia_dir
        self.split=split
        self.split_seed=split_seed
        self.json_file = json_file

        with open(self.json_file,'r') as f:
            data = f.read().strip().split('\n')
        
        ids = []
        for entry in data:
            try:
                entry_dict = json.loads(entry)
                id_ = entry_dict["entry_id"]
                ids.append(id_)
            except json.JSONDecodeError as e:
                ids.append(entry_dict["entry_id"])

        if self.split == 'train':
            # populate with pdbcode
            self.all_ = self.train_entries = ids
        elif self.split == 'val':
            self.all_ = self.valid_entries = ids


        if not os.path.exists(chothia_dir):
            raise FileNotFoundError(
                f"SAbDab structures not found in {chothia_dir}. "
                "Please download them from http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/"
            )
        
        # Decide which entries to process based on the split
        if self.split == 'train':
            self.entries_to_process = self.train_entries
        elif self.split == 'val':
            self.entries_to_process = self.valid_entries
        # elif split == 'test':
        #     self.entries_to_process = self.test_entries
        else:
            raise ValueError("Invalid split specified")

        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

        self.reset = reset
    
        self.sabdab_entries = None
        self._load_sabdab_entries()

        self.db_conn = None
        self.db_ids = None
        self._load_structures(reset)

        # self.clusters = None
        # self.id_to_cluster = None
        # self._load_clusters(reset)  #comment it and its functions if performing the special_filter = True

        self.ids_in_split = None
        self._load_split()

        self.transform = transform  # You can provide a transformation function
    

        print("self.sabdab_entries",len(self.sabdab_entries),chothia_dir)
        # print(len(self.sabdab_entries))

    def _load_sabdab_entries(self):
        print("performing_load_sabdab_entries")
        df = pd.read_csv(self.summary_path, sep='\t')
        # print("df", len(df))
        
        if not self.special_filter:
            entries_all = []
            for i, row in tqdm(
                df.iterrows(), 
                dynamic_ncols=True, 
                desc='Loading entries',
                total=len(df),
            ):
                entry_id = "{pdbcode}_{H}_{L}_{Ag}".format(
                    pdbcode = row['pdb'],
                    H = nan_to_empty_string(row['Hchain']),
                    L = nan_to_empty_string(row['Lchain']),
                    Ag = ''.join(split_sabdab_delimited_str(
                        nan_to_empty_string(row['antigen_chain'])
                    ))
                )
                ag_chains = split_sabdab_delimited_str(
                    nan_to_empty_string(row['antigen_chain'])
                )
                resolution = parse_sabdab_resolution(row['resolution'])
                entry = {
                    'id': entry_id,
                    'pdbcode': row['pdb'],
                    'H_chain': nan_to_none(row['Hchain']),
                    'L_chain': nan_to_none(row['Lchain']),
                    'ag_chains': ag_chains,
                    'ag_type': nan_to_none(row['antigen_type']),
                    'ag_name': nan_to_none(row['antigen_name']),
                    'date': datetime.datetime.strptime(row['date'], '%m/%d/%y'),
                    'resolution': resolution,
                    'method': row['method'],
                    'scfv': row['scfv'],
                }

                if (
                    (entry['ag_type'] in ALLOWED_AG_TYPES or entry['ag_type'] is None)
                    and (entry['resolution'] is not None and entry['resolution'] <= RESOLUTION_THRESHOLD)
                ):
                    entries_all.append(entry)

        else:
            # I already processed these entries, with the information needed, requires resolution and so on
            with open(self.json_file,'r') as f:
                data = f.read().strip().split('\n')
            
            entries_all = []
            for entry in data:
                try:
                    entry_dict = json.loads(entry)
                    #{"pdb": "8cyc", "heavy_chain": "D", "light_chain": "", "antigen_chains": ["A"], "antigen_type": ["protein"], "entry_id": "8cyc_D_X_A", "cluster": "8cyc"}
                    #check if the file is present before appending its index and continue with the rest of the process

                    old_pdb_path = entry_dict['pdb_data_path']
                    file_name = os.path.basename(old_pdb_path)
                    pdb_path = os.path.join(self.chothia_dir,file_name)
                    if not os.path.exists(pdb_path):
                        continue

                    entry_dict["id"] = entry_dict["entry_id"]
                    entry_dict["pdbcode"] = entry_dict["id"]
                    entry_dict["H_chain"] = entry_dict["heavy_chain"]
                    if entry_dict["light_chain"]:
                        entry_dict["L_chain"] = entry_dict["light_chain"]
                    else:
                        entry_dict["L_chain"] = None
                    entry_dict["ag_chains"] = entry_dict["antigen_chains"]
                    entries_all.append(entry_dict)
                except json.JSONDecodeError as e:

                    old_pdb_path = entry['pdb_data_path']
                    file_name = os.path.basename(old_pdb_path)
                    pdb_path = os.path.join(self.chothia_dir,file_name)
                    if not os.path.exists(pdb_path):
                        continue

                    entry["id"] = entry["entry_id"]
                    entry["pdbcode"] = entry["id"]
                    entry["H_chain"] = entry["heavy_chain"]
                    if entry["light_chain"]:
                        entry["L_chain"] = entry["light_chain"]
                    else:
                        entry["L_chain"] = None
                    entry["ag_chains"] = entry["antigen_chains"]
                    entries_all.append(entry)


        self.sabdab_entries = entries_all



        processed_entries = []
        for entry in self.sabdab_entries:
            processed_entries.append(entry)


    def _load_structures(self, reset):
        # if os.path.exists(self._structure_cache_path) and not reset:
        if not os.path.exists(self._structure_cache_path):
            # LMDB database does not exist, so preprocess structures
            self._preprocess_structures()
        
        try:
            with open(self._structure_cache_path + '-ids', 'rb') as f:
                self.db_ids = pickle.load(f)

            if self.db_ids is None:
                raise ValueError("Failed to load db_ids from file")

            # print(self.db_ids)
            self.sabdab_entries = list(
                filter(
                    lambda e: e['id'] in self.db_ids,
                    self.sabdab_entries
                )
            )
        except FileNotFoundError as e:
            print(f"File not found error: {e}")
        except ValueError as e:
            print(f"Value error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    @property
    def _structure_cache_path(self):
        filename = f"structures_{self.split}_fold_{self.fold}.lmdb"  # This will create a filename like structures_train.lmdb, structures_val.lmdb, etc.
        return os.path.join(self.processed_dir, filename)

        
    def _preprocess_structures(self):
        lmdb_exists = os.path.exists(self._structure_cache_path) and os.path.exists(self._structure_cache_path + '-ids')

        # Skip preprocessing if LMDB database already exists
        if lmdb_exists:
            print("LMDB database already exists, skipping preprocessing.")
            return
        tasks = []
        failed_entries = []  # List to store the entries that failed to preprocess,ADDED

        for entry in self.sabdab_entries:

            if not self.special_filter:
                if entry['pdbcode'] not in self.entries_to_process:
                    continue  # Skip entries not in the current split
                pdb_path = os.path.join(self.chothia_dir, '{}.pdb'.format(entry['pdbcode'])) # set it to all_structures/chothia
            if self.special_filter:
                if entry['entry_id'] not in self.entries_to_process:
                    continue  # Skip entries not in the current split
                old_pdb_path = entry['pdb_data_path']
                file_name = os.path.basename(old_pdb_path)
                pdb_path = os.path.join(self.chothia_dir,file_name)
                # print(pdb_path)
            if not os.path.exists(pdb_path):
                logging.warning(f"PDB not found: {pdb_path}")
                continue
            # 'id': entry['id'],
            tasks.append({
                'id': entry['entry_id'],
                'entry': entry,
                'pdb_path': pdb_path,
            })

        data_list = []
        for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess'):
            try:
                data = preprocess_sabdab_structure(task)
                if data is not None:
                    data_list.append(data)
            except Exception as e:
                logging.error(f"Error processing entry: {task['id']}, {e}")
                failed_entries.append(task['id'])
        # print("data_list", len(data_list))
        db_conn = lmdb.open(
            self._structure_cache_path,
            map_size = self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                ids.append(data['id'])
                txn.put(data['id'].encode('utf-8'), pickle.dumps(data))

        with open(self._structure_cache_path + '-ids', 'wb') as f:
            pickle.dump(ids, f)


    ############
        with open(self._structure_cache_path + '-ids', 'rb') as f:
            self.db_ids = pickle.load(f)
        self.sabdab_entries = list(
            filter(
                lambda e: e['id'] in self.db_ids,
                self.sabdab_entries
            )
        )
        

    @property
    def _cluster_path(self):
        return os.path.join(self.processed_dir, 'cluster_result_cluster.tsv')

    def _load_clusters(self, reset):
        
        if not os.path.exists(self._cluster_path) or reset:
            self._create_clusters()

        clusters, id_to_cluster = {}, {}
        with open(self._cluster_path, 'r') as f:
            for line in f.readlines():
                cluster_name, data_id = line.split()
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(data_id)
                id_to_cluster[data_id] = cluster_name
        self.clusters = clusters
        self.id_to_cluster = id_to_cluster

    def _create_clusters(self):
        cdr_records = []
        for id in self.db_ids:
            structure = self.get_structure(id)
            if structure['heavy'] is not None:
                cdr_records.append(SeqRecord.SeqRecord(
                    Seq.Seq(structure['heavy']['H3_seq']),
                    id = structure['id'],
                    name = '',
                    description = '',
                ))
            elif structure['light'] is not None:
                cdr_records.append(SeqRecord.SeqRecord(
                    Seq.Seq(structure['light']['L3_seq']),
                    id = structure['id'],
                    name = '',
                    description = '',
                ))
        fasta_path = os.path.join(self.processed_dir, 'cdr_sequences.fasta')
        SeqIO.write(cdr_records, fasta_path, 'fasta')

        cmd = ' '.join([
            'mmseqs', 'easy-cluster',
            os.path.realpath(fasta_path),
            'cluster_result', 'cluster_tmp',
            '--min-seq-id', '0.5',
            '-c', '0.8',
            '--cov-mode', '1',
        ])
        try:
            subprocess.run(cmd, cwd=self.processed_dir, shell=True, check=True)
        except:
            print("Error with mmseqs, pass")
            pass


    def _load_split(self):
        assert self.split in ('train', 'val', 'test')

        if self.special_filter:
            print("entered the special filter if")
            if self.split == 'val':
                # ids_val = [entry['id'] for entry in self.sabdab_entries if entry['pdbcode'] in self.valid_entries]
                ids_val = self.valid_entries
                # self.ids_in_split = self.valid_entries[:3]
                self.ids_in_split = ids_val
                # print("self.ids_in_split", self.ids_in_split)
            elif self.split == 'train':
                # ids_train = [entry['id'] for entry in self.sabdab_entries if entry['pdbcode'] in self.train_entries]
                ids_train = self.train_entries
                self.ids_in_split = ids_train
        else: #original implementation
            ids_test = [entry['id'] for entry in self.sabdab_entries if entry['ag_name'] in TEST_ANTIGENS]
            test_relevant_clusters = set([self.id_to_cluster[id] for id in ids_test])
            ids_train_val = [entry['id'] for entry in self.sabdab_entries if self.id_to_cluster[entry['id']] not in test_relevant_clusters]
            random.Random(self.split_seed).shuffle(ids_train_val)
            if self.split == 'test':
                self.ids_in_split = ids_test
            elif self.split == 'val':
                self.ids_in_split = ids_train_val[:20]
            else:
                self.ids_in_split = ids_train_val[20:]

    def _connect_db(self):
        if self.db_conn is not None:
            return
        self.db_conn = lmdb.open(
            self._structure_cache_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def get_structure(self, id):
        try:
            if id not in self.db_ids:
                raise ValueError(f"ID {id} not found in database IDs")
            
            self._connect_db()
            with self.db_conn.begin() as txn:
                encoded_id = id.encode()
                data = txn.get(encoded_id)
                if data is None:
                    raise ValueError(f"Data not found for ID: {id}")

                return pickle.loads(data)

        except ValueError as e:
            # Log the error and return None or dummy data
            print(f"Warning: Skipping ID {id} - {e}")
            return None  # You could also return some dummy data here if appropriate


    def __len__(self):
        return len(self.ids_in_split)


    def __getitem__(self, index):
        try:
            id = self.ids_in_split[index]
            data = self.get_structure(id)
            if self.transform is not None:
                data = self.transform(data)
        except Exception as e:
            logging.error(f"Error processing index {index} (ID: {id}): {e}")
            return None  # Or return some default value
        return data



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--processed_dir', type=str, default='/ibex/user/rioszemm/diffab/data/processed_Ab_Nb_december') #Modify
    parser.add_argument('--reset', action='store_true', default=True)
    args = parser.parse_args()
    if args.reset:
        sure = input('Sure to reset? (y/n): ')
        if sure != 'y':
            exit()
    dataset = SAbDabDataset(
        processed_dir=args.processed_dir,
        split=args.split, 
        reset=args.reset
    )
    # print(dataset[0])
    print(f"{len(dataset)}, num of clusters {len(dataset.clusters)}")
    