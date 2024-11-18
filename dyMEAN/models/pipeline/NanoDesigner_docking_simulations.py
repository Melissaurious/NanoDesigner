#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
from time import sleep
import shutil
from multiprocessing import Process
import random
import copy
from tqdm import tqdm
from data.pdb_utils import VOCAB, Protein, AgAbComplex2
from configs import IMGT
from .hdock_api import dock
from .igfold_api import pred_nano
from utils.renumber import renumber_pdb
from utils.relax import openmm_relax
import torch 
import time
from Bio import PDB
import numpy as np
import sys
current_working_dir = os.getcwd()
parent_dir = os.path.dirname(current_working_dir)
sys.path.append(parent_dir)
from functionalities.complex_analysis import extract_seq_info_from_pdb, extract_antibody_info, get_cdr_pos, count_clashes


parser = PDB.PDBParser(QUIET=True)
downloader = PDB.PDBList()

def dock_wrap(args_in, track_file, track_item):
    H = track_item["heavy"]
    A = track_item["antigen"]
    chain_list = [H] + A
    valid_models = False
    retry_count = 0
    max_retries = 3

    while not valid_models and retry_count < max_retries:
        try:
            results, total_time = dock(*args_in)
        except Exception as e:
            print(f"Docking attempt {retry_count+1} failed with error: {str(e)}")
            retry_count += 1
            continue

        n_models = len(results)
        track_item["Hdock_time"] = total_time

        count = 0
        for result in results:
            try:
                mod_prot = Protein.from_pdb(result, chain_list)
            except Exception as e:
                print(f"Error reading model {result}: {str(e)}")
                count += 1

        if count >= 0.5 * n_models:
            print("Docking to be repeated")
            retry_count += 1
        else:
            valid_models = True

    if not valid_models:
        print(f"Docking failed after {max_retries} attempts")

    with open(track_file, 'a') as f:
        f.write(json.dumps(track_item) + '\n')

def randomize_cdr(cdr_type, item_copy, heavy_chain, light_chain):
    cdr_pos = f"cdr{cdr_type.lower()}_pos"
    cdr_start, cdr_end = item_copy[cdr_pos]
    pert_cdr = np.random.randint(low=0, high=VOCAB.get_num_amino_acid_type(), size=(cdr_end - cdr_start + 1,))
    pert_cdr = ''.join([VOCAB.idx_to_symbol(int(i)) for i in pert_cdr])
    if cdr_type[0] == 'H':
        l, r = heavy_chain[:cdr_start], heavy_chain[cdr_end + 1:]
        heavy_chain = l + pert_cdr + r
    else:
        l, r = light_chain[:cdr_start], light_chain[cdr_end + 1:]
        light_chain = l + pert_cdr + r
    return pert_cdr, heavy_chain, light_chain



def main(args):
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    numbering = "imgt" 

    with open(args.dataset_json, 'r') as fin:
        lines = fin.read().strip().split('\n')


    hdock_p = [] 
    parent_dir = os.path.dirname(os.path.dirname(args.hdock_models))
    os.makedirs(args.hdock_models, exist_ok=True)
    track_file = os.path.join(parent_dir, "track_file_hdock.json")
    if not os.path.exists(track_file):
        with open(track_file, 'w') as f:
            pass

    for line in tqdm(lines): # So far process one PDB at the time
        item = json.loads(line)
        pdb = item['pdb']
        heavy_chain = item['heavy_chain_seq']


        # perturb CDR only at first iteration or use original
        if int(args.iteration) == 1:

            if args.initial_cdr == "randomized":
                
                ids = args.randomized                                
                id_list = list(range(1, int(ids) + 1))

                for i in id_list:

                    # create a copy of the original item/dictionary with the information needed as it will be modified
                    item_copy = copy.deepcopy(item)  
                    tmp_dir_for_interacting_aa = os.path.join(args.hdock_models, f"tmp_dir_binding_computations_{pdb}_{i}")  
                    os.makedirs(tmp_dir_for_interacting_aa, exist_ok=True)

                    pdb_dir_hdock = os.path.join(args.hdock_models, pdb + f"_{i}")  
                    if os.path.exists(pdb_dir_hdock) and any(fname.startswith('model_') and fname.endswith('.pdb') for fname in os.listdir(pdb_dir_hdock)):
                        print(f"Directory for {pdb_dir_hdock} already has model files. Skipping...")
                        continue

                    else:
                        os.makedirs(pdb_dir_hdock, exist_ok=True)
        
                    tmp_dir = args.hdock_models 

                    ag_pdb = os.path.join(tmp_dir, pdb + f"_{i}", f'{pdb}_ag.pdb') 
                    ab_pdb = os.path.join(tmp_dir, pdb + f"_{i}", f'{pdb}_only_nb.pdb')
                    

                    user_key = "epitope_user_input"

                    # Save antigen pdb, the nanobody/antibody pdb will come from IgFold
                    if user_key in item_copy and "antigen_source" in item_copy:
                        # means we already have an "isolated" antigen pdb at:
                        antigen_pdb = item_copy["antigen_source"]
                        #create copy at target directory
                        shutil.copy(antigen_pdb, ag_pdb)
                    else:
                        try:
                            protein = Protein.from_pdb(item_copy['pdb_data_path'], item_copy['antigen_chains'])
                            protein.to_pdb(ag_pdb)
                        except Exception as e:
                            pdb_file = item_copy["pdb_data_path"]
                            print(f"Failed to process PDB file {pdb_file}: {e}")

                    # check if there are any clashes in the antigen structure, if True, then fix it with relaxation
                    try:
                        structure = parser.get_structure("1111", ag_pdb)
                        tuple_result = count_clashes(structure, clash_cutoff=0.63)
                        num_clashes = tuple_result[0]
                        print(f"Number of clashes = {num_clashes} in {ag_pdb}")

                        if num_clashes > 0:
                            print("Conducting relaxation")
                            openmm_relax(ag_pdb,ag_pdb)  # Decorated with timeout

                    except TimeoutError as te:
                        print(f"Relaxation for {ag_pdb} timed out: {te}")
                        continue  # Handle the timeout case as needed
                    except Exception as e:
                        print(f"Refinement for {ag_pdb} failed: {e}")
                        continue


                    # valid_ig_fold = False
                    # while not valid_ig_fold:
                    if args.cdr_type != '-':  # no cdr to generate, i.e. structure prediction
                        cdr_types = ['H1', 'H2', 'H3'] if args.cdr_type == 'all' else [args.cdr_type]

                        for cdr_type in cdr_types:
                            cdr_pos = f"cdr{cdr_type.lower()}_pos"
                            cdr_start, cdr_end = item_copy[cdr_pos]

                            pert_cdr = np.random.randint(low=0, high=VOCAB.get_num_amino_acid_type(), size=(cdr_end - cdr_start + 1,))
                            pert_cdr = ''.join([VOCAB.idx_to_symbol(int(i)) for i in pert_cdr])

                            if cdr_type[0] == 'H':
                                l, r = heavy_chain[:cdr_start], heavy_chain[cdr_end + 1:]
                                heavy_chain = l + pert_cdr + r
                            else:
                                l, r = light_chain[:cdr_start], light_chain[cdr_end + 1:]
                                light_chain = l + pert_cdr + r

                            # UPDATE THE CDR in the item, otherwise errors at hdock
                            original_cdr_seq = item[f"cdr{cdr_type.lower()}_seq"]
                            item_copy[f"cdr{cdr_type.lower()}_seq"] = pert_cdr
                            new_cdr_seq = item_copy[f"cdr{cdr_type.lower()}_seq"]
                            print(f"Previous sequence of {cdr_type}", original_cdr_seq)
                            print(f"New sequence of {cdr_type}", new_cdr_seq)

                    # 1.IgFold
                    ab_pdb = os.path.join(tmp_dir, pdb + f"_{i}", f'{pdb}_IgFold.pdb')
                    print(f"Starting IgFold for entry {pdb}")
                    start_igfold_time = time.time()  # Record IgFold start time
                                    
                    heavy_chain_seq = heavy_chain # must come from the ranzomization step above
                    heavy_chain_id = item_copy["heavy_chain"]
                    pred_nano(heavy_chain_id, heavy_chain_seq, ab_pdb, do_refine=False)  # IgFold also works for nanobodies according to the GitHub website
                    end_igfold_time = time.time()  # Record IgFold end time

                    #renumber and revise IgFold generated a nanobody that "makes sense"
                    try:
                        renumber_pdb(ab_pdb,ab_pdb, numbering)
                    except Exception as e:
                        print("Generated nanobody could not be renumbering, skipping...")

                    try:
                        heavy_chain_seq_for_filter = ""
                        pdb_igfold = Protein.from_pdb(ab_pdb, item_copy['heavy_chain'])

                        for peptide_id, peptide in pdb_igfold.peptides.items():
                            sequence = peptide.get_seq()
                            heavy_chain_seq_for_filter += sequence

                            # Extract dicitionary of cdrHs and cdrL (is existant) positions
                            cdr_pos_dict = extract_antibody_info(pdb_igfold, item_copy["heavy_chain"], item_copy["light_chain"], "imgt")

                            # Create Peptide objects per chain in Protein object
                            peptides_dict = pdb_igfold.get_peptides()

                            # Get the peptide of interest (for now only heavy chain)
                            nano_peptide = peptides_dict.get(item_copy["heavy_chain"])

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


                    # check if produced structure contains clashes, if yes..coduct relaxation
                    try:
                        structure = parser.get_structure("1111", ab_pdb)
                        tuple_result = count_clashes(structure, clash_cutoff=0.63)
                        num_clashes = tuple_result[0]
                        IgFold_clashes = tuple_result[0]
                        print(f"Number of clashes = {num_clashes} in {ab_pdb}")

                        if num_clashes > 0:
                            print("Conducting relaxation")
                            openmm_relax(ab_pdb,ab_pdb)  # Decorated with timeout

                    except TimeoutError as te:
                        print(f"Relaxation for {ab_pdb} timed out: {te}")
                        continue  # Handle the timeout case as needed
                    except Exception as e:
                        print(f"Refinement for {ab_pdb} failed: {e}")
                        continue

                    # check how many clashes we have after relaxing the IgFold results
                    structure = parser.get_structure("1111", ab_pdb)
                    tuple_result = count_clashes(structure, clash_cutoff=0.63)
                    num_clashes = tuple_result[0]
                    print(f"Number of clashes = {num_clashes} in {ab_pdb}")
                    print(f"IgFold for entry {pdb} took {end_igfold_time - start_igfold_time:.2f} seconds")

                    while len(hdock_p) >= 4:
                        removed = False
                        for p in hdock_p:
                            if not p.is_alive():
                                p.join()
                                p.close()
                                hdock_p.remove(p)
                                removed = True
                        if not removed:
                            sleep(10)

                    # epitipe has to always be the first original
                    epitope = item_copy["epitope"]  # when extracted from the json file looks like a list of lists [['A', 26, 'N'], ['A', 27, 'H'], ...]
                    # put into the required format for hdock
                    epitope_ = [tuple(item) for item in epitope] #[('A', 26, 'N'), ('A', 27, 'H'), ('A', 28, 'F'), ('A', 29, 'V')]
                    epitope_ = list(set(epitope_))
                    binding_rsite = [tuple(item[:2]) for item in epitope_]
                    print("binding_rsite (epitope)",binding_rsite)

                    # Paratope
                    lsite3 = extract_seq_info_from_pdb(ab_pdb, item_copy["heavy_chain"], item_copy["cdrh3_seq"])
                    lsite2 = extract_seq_info_from_pdb(ab_pdb, item_copy["heavy_chain"], item_copy["cdrh2_seq"])
                    lsite1 = extract_seq_info_from_pdb(ab_pdb, item_copy["heavy_chain"], item_copy["cdrh1_seq"])


                    binding_lsite = []

                    # Append valid lsites to binding_lsite
                    if lsite3 is not None:
                        binding_lsite.extend([tup[:2] for tup in lsite3])
                    if lsite2 is not None:
                        binding_lsite.extend([tup[:2] for tup in lsite2])
                    if lsite1 is not None:
                        binding_lsite.extend([tup[:2] for tup in lsite1])


                    print(f"Starting Hdock for entry {pdb}_{i}")


                    track_item = {}
                    track_item["heavy"] = item_copy["heavy_chain"]
                    track_item["light"] = item_copy["light_chain"]
                    track_item["antigen"] = item_copy["antigen_chains"]
                    track_item["pdb"] = f"{pdb}_{i}"
                    # track_item["Hdock_time"] = dock_time
                    track_item["Hdock_n_models"] = args.n_docked_models
                    track_item["IgFold_time"] = end_igfold_time - start_igfold_time
                    track_item["IgFold_clashes"] = IgFold_clashes
                    track_item["IgFold_clashes_after_relax"] = num_clashes
                    track_item["iteration"] = args.iteration


                    args_in = (ag_pdb, ab_pdb, pdb_dir_hdock, args.n_docked_models, binding_rsite, binding_lsite)

                    p = Process(target=dock_wrap, args=(args_in, track_file, track_item))
                    p.start()
                    hdock_p.append(p)


            elif args.initial_cdr == "original":
                # TO DO, no randomization of the CDRS, keep original sequence/structure
                pass

        else:
            # open the file of the top_n of the previous iteration and process it, if the folder for the corresponding
            # entry exists, then skip
            print("entered the if")
            pdb_dir_hdock = os.path.join(args.hdock_models)
            os.makedirs(pdb_dir_hdock, exist_ok=True)
            previous_iter = int(args.iteration) - 1
            desired_path = os.path.dirname(os.path.dirname(args.hdock_models))
            best_candidates_file = os.path.join(desired_path, f"best_candidates_iter_{previous_iter}.json")


            with open(best_candidates_file, 'r') as f:
                data = [json.loads(line) for line in f]

            if len(data) > int(args.best_mutants):
                # Ensure only the top_n entries are kept and processed
                top_n = int(args.best_mutants)
                data = [entry for entry in data if 1 <= entry["rank"] <= top_n]

                # Rewriting the file to keep only the top_n entries
                with open(best_candidates_file, 'w') as f:
                    for entry in data:
                        f.write(json.dumps(entry) + '\n')

            for element in data:
                rank = element["rank"]
                pdb = element["pdb"]  # e.g "3g9a_6" 
                pdb_parts = pdb.rsplit('_')[0] # 3g9a
                new_pdb_id =  pdb_parts + f"_{rank}"

                # create hdock dir path
                pdb_dir_hdock = os.path.join(args.hdock_models, new_pdb_id)  

                if os.path.exists(pdb_dir_hdock) and any(fname.startswith('model_') and fname.endswith('.pdb') for fname in os.listdir(pdb_dir_hdock)):
                    print(f"Directory for {pdb_dir_hdock} already has model files. Skipping...")
                    continue
                else:
                    os.makedirs(pdb_dir_hdock, exist_ok=True)

                tmp_dir_for_interacting_aa = os.path.join(args.hdock_models, f"tmp_dir_binding_computations_{new_pdb_id}")  
                os.makedirs(tmp_dir_for_interacting_aa, exist_ok=True)

    
                tmp_dir = args.hdock_models # dyMEAN_iter_original_cdr/HDOCK_iter_i

                ag_pdb = os.path.join(tmp_dir, new_pdb_id, f'{pdb_parts}_ag.pdb') # save antigen and epitope information
                ab_pdb = os.path.join(tmp_dir, new_pdb_id , f'{pdb_parts}_only_nb.pdb')

                ag_pdb = pdb_dir_hdock
                if not os.path.exists(ag_pdb):
                    os.makedirs(ag_pdb)

                
                H,L,A = element["heavy_chain"], element["light_chain"], element["antigen_chains"]
                top_model_pdb = element["mod_pdb"]

                ag_pdb = os.path.join(pdb_dir_hdock, f'{pdb_parts}_ag.pdb') # save antigen and epitope information

                try:
                    cplx = AgAbComplex2.from_pdb(top_model_pdb, H, L, A ,numbering=numbering, skip_epitope_cal=False, skip_validity_check=False)
                except Exception as e:
                    print("WARNING!,Error creating mod AgAbComplex:", e)
                    pass

                print("ag and ab pdb files come from this pdb", top_model_pdb)

                # Extract the antigen structure from the pdb and save it in the required location
                cplx.get_antigen().to_pdb(ag_pdb)

                # extract nanobody chain from the complex:
                heavy = cplx.get_heavy_chain()
                antibody = Protein(
                    pdb_id='combined_protein',
                    peptides={
                        # 'light':light,
                        'heavy':heavy,
                    }
                )
                if element['light_chain']:
                    light = cplx.get_light_chain()
                    antibody = Protein(
                        pdb_id='combined_protein',
                        peptides={
                            'light':light,
                            'heavy':heavy,
                        }
                    )
                antibody.to_pdb(ab_pdb)


                while len(hdock_p) >= 4:
                    removed = False
                    for p in hdock_p:
                        if not p.is_alive():
                            p.join()
                            p.close()
                            hdock_p.remove(p)
                            removed = True
                    if not removed:
                        sleep(10)

                # epitipe has to always be the first original
                epitope = item["epitope"]  # when extracted from the json file looks like a list of lists [['A', 26, 'N'], ['A', 27, 'H'], ...]
                # put into the required format for hdock
                epitope_ = [tuple(item) for item in epitope] #[('A', 26, 'N'), ('A', 27, 'H'), ('A', 28, 'F'), ('A', 29, 'V')]
                epitope_ = list(set(epitope_))
                binding_rsite = [tuple(item[:2]) for item in epitope_]
                print("binding_rsite",binding_rsite)

                cdrh3_seq = element["cdrh3_seq_mod"]
                cdrh2_seq = element["cdrh2_seq_mod"]
                cdrh1_seq = element["cdrh1_seq_mod"]


                # Paratope
                lsite3 = extract_seq_info_from_pdb(ab_pdb, element["heavy_chain"], cdrh3_seq)
                lsite2 = extract_seq_info_from_pdb(ab_pdb, element["heavy_chain"], cdrh2_seq)
                lsite1 = extract_seq_info_from_pdb(ab_pdb, element["heavy_chain"], cdrh1_seq)

                binding_lsite = lsite3 + lsite2 + lsite1
                binding_lsite =[tup[:2] for tup in binding_lsite] # paratope, which comes from the current (designed) nanobody


                print(f"Starting Hdock for entry {new_pdb_id}")


                track_item = {}
                track_item["heavy"] = element["heavy_chain"]
                track_item["light"] = element["light_chain"]
                track_item["antigen"] = element["antigen_chains"]
                track_item["pdb"] = new_pdb_id
                # track_item["Hdock_time"] = dock_time
                track_item["Hdock_n_models"] = args.n_docked_models
                track_item["iteration"] = args.iteration

                args_in = (ag_pdb, ab_pdb, pdb_dir_hdock, args.n_docked_models, binding_rsite, binding_lsite)

                p = Process(target=dock_wrap, args=(args_in, track_file, track_item))
                p.start()
                hdock_p.append(p)


    while len(hdock_p):
        p = hdock_p[0]
        p.join()
        p.close()
        hdock_p = hdock_p[1:]



def parse():
    parser = argparse.ArgumentParser(description='inference of pipeline method')
    parser.add_argument('--cdr_model', type=str, required=True, help='Type of model that generates CDRs',
                        choices=['DiffAb', 'dyMEAN', 'ADesigner'])
    parser.add_argument('--dataset_json', type=str, required=True, help='Path to the summary file of dataset in json format') # TO DO, do i use it?
    parser.add_argument('--cdr_type', type=str, default='H3', help='Type of cdr to generate',
                        choices=['H3', 'all', '-'])
    parser.add_argument('--iteration', type=int, help='Iteration number')
    parser.add_argument('--n_docked_models', type=int, help='Hdock models to predict per entry')
    parser.add_argument("--randomized", type=int, required=True, help="Total number of randomized nanobodies from the CDR(s)")
    parser.add_argument("--best_mutants", type=int, required=True, help="Total number of randomized nanobodies from the CDR(s)")
    parser.add_argument('--hdock_models', type=str, required=True, help='Path to save generated PDBs from hdock')
    parser.add_argument('--csv_dir', type=str, required=True, help='Path where results are')
    parser.add_argument('--csv_dir_', type=str, required=True, help='Path where results are')
    parser.add_argument('--initial_cdr', type=str, required=True, default='randomized', help='Keep original or randomized CDR sequence/structure', choices=['randomized','original']) 


    #TO DO, consider number of workers? 

    return parser.parse_args()


if __name__ == '__main__':
        main(parse())
