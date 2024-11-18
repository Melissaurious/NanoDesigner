from utils.logger import print_log
import concurrent
import time
from functools import partial 
from utils.renumber import renumber_pdb
from configs import IMGT
import os
import argparse
import json
import copy
import gc
from data.pdb_utils import VOCAB, Protein, Peptide
from Bio import PDB
import numpy as np
from scipy.spatial import cKDTree
import traceback
import sys
current_working_dir = os.getcwd()
parent_dir = os.path.dirname(current_working_dir)
sys.path.append(parent_dir)
from functionalities.complex_analysis import extract_antibody_info, get_cdr_pos, clean_extended
from functionalities.nanobody_antibody_interacting_residues import interacting_residues, binding_residues_analysis


parser = PDB.PDBParser(QUIET=True)
downloader = PDB.PDBList()



def get_default_item(pdb_dict):
    # Returns the first item in the pdb_dict. 
    # You can modify this function if you need to select a specific item instead of the first one.
    return next(iter(pdb_dict.values()))



def main(args):

    with open(args.test_set, 'r') as fin:
        data = fin.read().strip().split('\n')

    pdb_dict = {}
    for item in data:
        json_obj = json.loads(item)
        pdb_dict[json_obj["pdb"]] = json_obj


    # Create a list of all the hdock models for the current N randomized one
    items_in_hdock_models = os.listdir(args.hdock_models)
    dir_l = [item for item in items_in_hdock_models if os.path.isdir(os.path.join(args.hdock_models, item))]

    new_dir = []
    for dir_ in dir_l:
        if "tmp_dir" in dir_:
            continue
        new_dir.append(dir_)
    
    dir_l = new_dir[:]

    dir_l_ = []  # List to hold directories with more than 3 .pdb files
    new_l = []  # List to hold full directory paths with more than 3 .pdb files

    for directory in new_dir:
        full_dir_path = os.path.join(args.hdock_models, directory)
        hdock_models = [file for file in os.listdir(full_dir_path) if file.startswith('model_') and file.endswith('.pdb')]
        n = int(args.top_n) + 10
        if len(hdock_models) != n:
            dir_l_.append(directory)
            new_l.append(full_dir_path)

    for index, (pdb_n, full_dir_path) in enumerate(zip(dir_l_, new_l)):
        
        pdb_parts = pdb_n.rsplit('_')[0]
        ag_pdb = os.path.join(full_dir_path, pdb_parts + "_ag.pdb")
        ab_pdb = os.path.join(full_dir_path, pdb_parts + "_only_nb.pdb")
        if not os.path.exists(ab_pdb):
            ab_pdb = os.path.join(full_dir_path, pdb_parts + "_IgFold.pdb")
        
        original_item = pdb_dict.get(pdb_parts)

        if original_item is None:
            original_item = get_default_item(pdb_dict)
            print("Using a default item from pdb_dict")
        else:
            pass

        item_copy = copy.deepcopy(original_item)
        heavy_chain = item_copy["heavy_chain"]
        light_chain = item_copy["light_chain"]

        hdock_models = [os.path.join(full_dir_path, file) for file in os.listdir(full_dir_path) if file.startswith('model_') and file.endswith('.pdb')]

        track_file = os.path.join(full_dir_path, f"{pdb_n}_ep_and_dock_information.json")
        track_file_2 = os.path.join(full_dir_path, f"{pdb_n}_binding_information.json")
        parent_dir = os.path.dirname(track_file)


        top_file = os.path.join(parent_dir, "top_models.json")
        if os.path.exists(top_file) and os.stat(top_file).st_size > 0:
            print(f"Top models already selected for {pdb_n}, skipping...")
            continue


        if len(hdock_models) <= n:
            continue

        # Check if track_file exists and if it contains only entries with "cdr3_avg": 0.0
        if os.path.exists(track_file):
            with open(track_file, 'r') as f:
                track_data = [json.loads(line) for line in f]
                if len(track_data) >= len(hdock_models):
                    all_cdr3_avg_zero = all(float(entry.get("cdr3_avg", 0.0)) == 0.0 for entry in track_data)
                    if all_cdr3_avg_zero:
                        print(f"All cdr3_avg values are 0.0 for {pdb_n}, skipping...")
                        continue


        print("Processing",pdb_n )


        binding_data = []
        if not os.path.exists(track_file_2):
            with open(track_file_2, 'w') as f:
                pass
        else:
            with open(track_file_2, 'r') as f:
                binding_data = [json.loads(line) for line in f]
        

        gt_epitope = original_item["epitope"]
        epitope_ = [tuple(item) for item in gt_epitope] 
        epitope_ = list(set(epitope_))
        binding_rsite = [tuple(item[:2]) for item in epitope_]


        # I only require doing it once cause the randomized cdrh3 sequence is aconserved across all hdock models

        try:
            pdb = Protein.from_pdb(ab_pdb, heavy_chain)
            item_copy["heavy_chain_seq"] = ""

            for peptide_id, peptide in pdb.peptides.items():
                sequence = peptide.get_seq()
                item_copy['heavy_chain_seq'] += sequence
            
            # Extract dicitionary of cdrHs and cdrL (is existant) positions
            cdr_pos_dict = extract_antibody_info(pdb, heavy_chain, light_chain, "imgt")

            # Create Peptide objects per chain in Protein object
            peptides_dict = pdb.get_peptides()

            # Get the peptide of interest (for now only heavy chain)
            nano_peptide = peptides_dict.get(heavy_chain)

            # get the sequence of cdrs given the positions
            for i in range(1, 4):
                cdr_name = f'H{i}'.lower()
                cdr_pos= get_cdr_pos(cdr_pos_dict,cdr_name)
                item_copy[f'cdr{cdr_name}_pos_mod'] = cdr_pos
                start, end = cdr_pos 
                end += 1
                cdr_seq = nano_peptide.get_span(start, end).get_seq()
                item_copy[f'cdr{cdr_name}_seq_mod'] = cdr_seq

        except Exception as e:
            # Handle exceptions if needed
            print(f'Something went wrong for {ab_pdb}, {e}')
            # if something went wrong it means the IgFold and therefore all of the docked models had an issue, delete folder.
            try:
                import shutil
                shutil.rmtree(full_dir_path)
                break
            except Exception as e:
                print(e)
                break


        tmp_dir_for_interacting_aa = os.path.join(args.hdock_models, f"tmp_dir_binding_computations_{pdb_n}")
        os.makedirs(tmp_dir_for_interacting_aa, exist_ok=True)
        shared_args = ag_pdb, ab_pdb, binding_rsite, item_copy, full_dir_path, tmp_dir_for_interacting_aa


        if len(binding_data) != len(hdock_models):

            complete = False
            while not complete:
                with open(track_file_2, 'r') as f:
                    data = [json.loads(line) for line in f]

                hdock_models_already_processed = [item["hdock_model"] for item in data]
                new_list = [model for model in hdock_models if model not in hdock_models_already_processed]

                if not new_list:
                    print("All models have been processed.")
                    complete = True
                    continue

                process_with_args = partial(binding_residues_analysis, shared_args)
                skipped_tasks_count = 0 
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = [executor.submit(process_with_args, k) for k in new_list]
                    
                    results = []
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result(timeout=1000)
                            if result is not None:
                                results.append(result)
                        except concurrent.futures.TimeoutError:
                            print("A task exceeded the time limit and was skipped.")
                            skipped_tasks_count += 1

                for result in results:
                    with open(track_file_2, 'a') as f:
                        f.write(json.dumps(result) + '\n')

                #check lenght of the content of the track file
                with open(track_file_2, 'r') as f:
                    binding_data = [json.loads(line) for line in f]

                if len(binding_data) < len(hdock_models):
                    print("Still missing entries, continuing to process remaining models.")
                else:
                    print("All models have been processed.")
                    complete = True

        with open(track_file_2, 'r') as f:
            binding_data_new = [json.loads(line) for line in f]

        if len(binding_data) >= len(hdock_models):

            filtered_list = [item for item in binding_data_new if float(item["cdr3_avg"]) != 0.0]
            filtered_list.sort(key=lambda x: x['epitope_recall'], reverse=True)
            n = int(args.top_n) # + 10
            top_models = filtered_list[:n]
            parent_dir = os.path.dirname(track_file_2)
            top_file = os.path.join(parent_dir, "top_models.json")
            if not os.path.exists(top_file):
                with open(top_file, 'w') as f:
                    pass

            with open(top_file, 'a') as f:
                for item in top_models:
                    f.write(json.dumps(item) + '\n')

            top_model_paths = [item["hdock_model"] for item in top_models]

            for top_model in top_model_paths:
                renumber_pdb(top_model, top_model, "imgt")

            if int(args.iteration) > 1:
                for hdock_model in hdock_models:
                    if hdock_model not in top_model_paths:
                        try:
                            os.remove(hdock_model)
                        except FileNotFoundError:
                            pass  

            # 6. Delete all models not in the top models
            for hdock_model in hdock_models:
                if hdock_model not in top_model_paths:
                    if int(args.iteration) == 1:
                        pass
                    else:
                        try:
                            os.remove(hdock_model)
                        except FileNotFoundError:
                            pass  
                else:
                    renumber_pdb(hdock_model, hdock_model, "imgt")
            return top_models
        break

    gc.collect()

def parse():
    parser = argparse.ArgumentParser(description='inference of pipeline method')
    parser.add_argument('--test_set', type=str, required=True, help='Path to the summary file of dataset in json format')
    parser.add_argument('--hdock_models', type=str, required=True, help='Path to save generated PDBs from hdock')
    parser.add_argument('--top_n', type=str, required=True, help='Top n docked models to select ')
    parser.add_argument('--iteration', type=str, required=True, help='Top n docked models to select ')


    return parser.parse_args()

if __name__ == '__main__':
    main(parse())