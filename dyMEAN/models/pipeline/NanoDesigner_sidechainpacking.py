#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
import sys 
import shutil
current_working_dir = os.getcwd()
sys.path.append(current_working_dir)
from utils.renumber import renumber_pdb
from data.pdb_utils import AgAbComplex2, Protein
from data.pdb_utils import Peptide as Peptide_Class
from configs import CACHE_DIR
from utils.logger import print_log
from utils.relax import rosetta_sidechain_packing #, openmm_relax_no_decorator, openmm_relax
import gc
from Bio import PDB
import numpy as np
import concurrent.futures 
from functools import partial 

parent_dir = os.path.dirname(current_working_dir)
sys.path.append(parent_dir)
from functionalities.complex_analysis import count_clashes


parser = PDB.PDBParser(QUIET=True)
downloader = PDB.PDBList()


def get_default_item(pdb_dict):
    # Returns the first item in the pdb_dict. 
    # You can modify this function if you need to select a specific item instead of the first one.
    return next(iter(pdb_dict.values()))


def side_ch_packing(args, pdb_dict, item):

    try:
        # Try treating 'item' as a dictionary and access its keys
        H, L, A = item['heavy_chain'], item['light_chain'], item['antigen_chains']
        pdb = item['pdb']
        mod_pdb = item["mod_pdb"]

    except TypeError:
        # If 'item' is not a dictionary (TypeError on key access), parse it as JSON
        item = json.loads(item)
        H, L, A = item['heavy_chain'], item['light_chain'], item['antigen_chains']
        pdb = item['pdb']
        mod_pdb = item["mod_pdb"]
    
    try:
        mod_cplx = AgAbComplex2.from_pdb(mod_pdb, H, L, A, numbering="imgt", skip_epitope_cal=True)
    except Exception as e:
        print(f"{mod_pdb} raised an error: {e}")
        renumber_pdb(mod_pdb,mod_pdb,scheme = "imgt")

    item["side_chain_packed"] = "No"

    if args.hdock_models:
        # pdb_parts = pdb.rsplit('_')[0]
        pdb_parts = pdb.rsplit('_',1)[0]
        original_item = pdb_dict.get(pdb_parts)
    else:
        pdb = item.get("entry_id", item.get("pdb"))
        original_item = pdb_dict.get(pdb)


    if original_item is None:
        original_item = get_default_item(pdb_dict)
        print("Using a default item from pdb_dict")
    else:
        pass

    if not os.path.exists(mod_pdb):
        print_log(f'{mod_pdb} not exists!', level='ERROR')
        print(f'{mod_pdb} not exists!')
        # return None
        return item
    if os.path.getsize(mod_pdb) == 0:
        print_log(f'{mod_pdb} exists but is empty!', level='ERROR')
        print_log(f'{mod_pdb} exists but is empty!')
        # return None
        return item


    pdb_modified = pdb.rsplit('_', 1)[0]

    ref_pdb = item.get("ref_pdb",None)
    if ref_pdb is not None:
         ref_for_sanity_ck = ref_pdb
    else:
        ref_for_sanity_ck = original_item.get("pdb_data_path")
        if ref_for_sanity_ck == None:
            ref_for_sanity_ck = original_item["nano_source"]


    if args.hdock_models:
        ref_for_sanity_ck = os.path.join(args.hdock_models,pdb, pdb_modified + "_only_nb.pdb")
        if not os.path.exists(ref_for_sanity_ck):
            ref_for_sanity_ck = os.path.join(args.hdock_models,pdb, pdb_modified + "_IgFold.pdb")


    mod_revised, ref_revised = False, False
    chains_list = [H] + A
    if L != "":
        chains_list = [H] + A + [L]

    try:
        mod_prot = Protein.from_pdb(mod_pdb,chains_list)
    except Exception as e:
        print(f'parse {mod_pdb} failed for {e}')
        return item
    try:
        ref_prot = Protein.from_pdb(ref_for_sanity_ck,chains_list)
    except Exception as e:
        print(f'parse {ref_for_sanity_ck} failed for {e}')
        return item



    try:
        # Iterate through chains specified in the list
        for chain_name in chains_list:
            # Retrieve Peptide objects by chain name
            ref_chain = ref_prot.peptides.get(chain_name)
            mod_chain = mod_prot.peptides.get(chain_name)

            # Check if chains exist in both structures
            if mod_chain is None or ref_chain is None:
                print(f"Chain {chain_name} not found in one of the complexes. Skipping.")
                continue

            if ref_chain is None:
                continue

            # Align chains by position number if their lengths differ
            if len(mod_chain.residues) != len(ref_chain.residues):
                print(f"{mod_chain} chain {chain_name} length not consistent: {len(mod_chain.residues)} != {len(ref_chain.residues)}. Trying to align by position number.")
                # Prepare to store revised residues
                mod_residues, ref_residues = [], []
                pos_map = {'-'.join(str(a) for a in res.get_id()): res for res in ref_chain.residues}

                for res in mod_chain.residues:
                    res_id = '-'.join(str(a) for a in res.get_id())
                    if res_id in pos_map:
                        mod_residues.append(res)
                    else:
                        # Handle the case where the residue position does not match
                        print(f"Warning: Residue {res_id} in mod_chain not found in ref_chain. Skipping.")

                ref_residues = [pos_map[res_id] for res_id in [ '-'.join(str(a) for a in res.get_id()) for res in mod_residues ] if res_id in pos_map]

                # Update peptide chains within the Protein objects 
                mod_prot.peptides[chain_name] = Peptide_Class(chain_name, mod_residues)
                ref_prot.peptides[chain_name] = Peptide_Class(chain_name, ref_residues)

                mod_revised, ref_revised = True, False  # Only mod_prot is revised
                print(f"{mod_prot.peptides[chain_name]} chain {chain_name} length after aligned: {len(mod_prot.peptides[chain_name].residues)} == {len(ref_prot.peptides[chain_name].residues)}")

    except Exception as e:
        print(f"An exception was raised: {e}")

    if mod_revised:
        print("Entered the mod_revised mode")
        print("Saving aligned protein")
        mod_prot.to_pdb(mod_pdb)

    if args.cdr_model == 'dyMEAN':

        try:
            print("Skipping Side chain packing as dyMEAN is end-to-end")
            item["side_chain_packed"] = "No"
            structure = parser.get_structure(pdb, mod_pdb)
            total_clashes = count_clashes(structure, clash_cutoff=0.63)[0]

            chain_clashes = {}
            for chain in structure.get_chains():
                chain_id = chain.get_id()
                chain_clashes[chain_id] = count_clashes(chain, clash_cutoff=0.63)[0]

            inter_chain_clashes = total_clashes - sum(chain_clashes.values())

            item["inference_clashes"] = total_clashes
            item["clashes_per_chain_inference"] = chain_clashes
            item["inter_chain_clashes_inference"] = inter_chain_clashes
        

        except Exception as e:
            print(f"{mod_pdb} could not be refined, skipping the evaluation of this model")
            item["refined"] = "No"

    else:

        try:
            structure = parser.get_structure(pdb, mod_pdb)
            total_clashes = count_clashes(structure, clash_cutoff=0.63)[0]

            chain_clashes = {}
            for chain in structure.get_chains():
                chain_id = chain.get_id()
                chain_clashes[chain_id] = count_clashes(chain, clash_cutoff=0.63)[0]

            inter_chain_clashes = total_clashes - sum(chain_clashes.values())

            item["inference_clashes"] = total_clashes
            item["clashes_per_chain_inference"] = chain_clashes
            item["inter_chain_clashes_inference"] = inter_chain_clashes
        

        except Exception as e:
            # print(f"{mod_pdb} could not be refined, skipping the evaluation of this model")
            # item["refined"] = "No
            item["inference_clashes"] = np.nan
            item["clashes_per_chain_inference"] = np.nan
            item["inter_chain_clashes_inference"] = np.nan
        


        try:
            print(f"Conducting side chain packing...on {pdb}")
            mod_pdb, rosetta_time = rosetta_sidechain_packing(mod_pdb, mod_pdb)
            
            # Check if the output file was actually created
            if os.path.exists(mod_pdb):
                item["side_chain_packed"] = "Yes"
                item["side_ch_packing_time"] = rosetta_time
            else:
                print(f"Side chain packing for {mod_pdb} failed.")
                item["side_chain_packed"] = "No"
                
        except Exception as e:
            print(f"{mod_pdb} could not be side-chain-packed, revise correct installation of Rosetta")
            item["side_chain_packed"] = "No"


        try:
            structure = parser.get_structure(pdb, mod_pdb)
            total_clashes = count_clashes(structure, clash_cutoff=0.63)[0]

            chain_clashes = {}
            for chain in structure.get_chains():
                chain_id = chain.get_id()
                chain_clashes[chain_id] = count_clashes(chain, clash_cutoff=0.63)[0]

            inter_chain_clashes = total_clashes - sum(chain_clashes.values())

            item["side_chain_p_num_clashes"] = total_clashes
            item["clashes_per_chain_side_ch_packing"] = chain_clashes
            item["inter_chain_clashes_side_ch_packing"] = inter_chain_clashes
        

        except Exception as e:
            print_log(f'Clash analysis for {mod_pdb} failed for {e}', level='ERROR')

    
    # Update the ref file as diffab adds as reference the hdock model
    user_key = "epitope_user_input"
    if args.hdock_models:
        # Save antigen pdb, the nanobody/antibody pdb will come from IgFold
        if user_key in original_item and int(args.iteration) != 1:
            desired_path = os.path.dirname(os.path.dirname(args.hdock_models))
            best_candidates_file = os.path.join(desired_path, f"best_candidates_iter_{1}.json")  # always the best one from interation # 1
            with open(best_candidates_file,"r") as f:
                data = f.read().strip().split('\n')

            for entry in data:
                json_obj = json.loads(entry)
                if int(json_obj["rank"]) == 1:
                    item["ref_pdb"] = json_obj["mod_pdb"]
                    break
        elif user_key not in original_item:
            item["ref_pdb"] = original_item["pdb_data_path"]
    else:
        pass

    return item 



# Create the argument parser
arg_parser = argparse.ArgumentParser(description='do side chain packing and refinement')
arg_parser.add_argument('--summary_json', type=str, required=True, help='Path to the summary in json format providing H/L/antigen')
arg_parser.add_argument('--out_file', type=str, required=True, help='Path to the summary in json format providing H/L/antigen')
arg_parser.add_argument('--test_set', type=str, required=True, help='Path to the reference data')
arg_parser.add_argument('--iteration', type=str, default="0", help='Path to the reference data')
arg_parser.add_argument('--hdock_models', type=str, default="", help='Path to the reference data')
arg_parser.add_argument('--cdr_model', type=str, required=True, help='Type of model that generates CDRs',
                        choices=['DiffAb', 'dyMEAN', 'ADesigner'])


# Parse the arguments
args = arg_parser.parse_args()

def main(args):
    
    with open(args.test_set, 'r') as fin:
        data = fin.read().strip().split('\n')

    # create a dictionary of {pdb:whole dictionary correspinding to such pdb}
    pdb_dict = {}
    for item in data:
        json_obj = json.loads(item)
        pdb_id = json_obj.get("entry_id", json_obj.get("pdb"))
        pdb_dict[pdb_id] = json_obj


    with open(args.summary_json, 'r') as fin:
        summary = fin.read().strip().split('\n')

    # Check if the file doesn't exist or is empty
    if not os.path.exists(args.out_file) or os.stat(args.out_file).st_size == 0:
        # The file is either non-existent or empty, proceed with processing

        with open(args.out_file, 'w') as f:
            pass

        process_with_args = partial(side_ch_packing, args, pdb_dict)
        skipped_tasks_count = 0 
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_with_args, k) for k in summary]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=1500)  # Adjust timeout as needed
                    if result is not None:
                        results.append(result)
                except concurrent.futures.TimeoutError:
                    print("A task exceeded the time limit and was skipped.")
                    skipped_tasks_count += 1

        print(f"Total tasks skipped due to time limit: {skipped_tasks_count}")

        with open(args.out_file, 'a') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')



    elif os.path.exists(args.out_file) and os.stat(args.out_file).st_size != 0:
        # File exists and is not empty, check what is in there to not repeat the whole process
        with open(args.out_file, 'r') as f:
            data = f.read().strip().split('\n')

        # Create a set of all 'mod_pdb' values from processed files
        processed_mod_pdbs = set()
        for entry in data:
            try:
                entry_json = json.loads(entry)
            except json.JSONDecodeError:
                entry_json = entry  # Use the raw entry as a fallback if JSON parsing fails
                print(f"Invalid JSON entry: {entry_json}")

            # Check if entry_json is a dictionary before trying to access "mod_pdb"
            if isinstance(entry_json, dict):
                processed_mod_pdbs.add(entry_json.get("mod_pdb"))
            else:
                processed_mod_pdbs.add(entry_json)
            
        missing_entries_to_process = [element for element in summary if json.loads(element)['mod_pdb'] not in processed_mod_pdbs and json.loads(element).get('side_chain_packed', 'No') != 'Yes']

        print("missing_entries_to_process", len(missing_entries_to_process))

        results = []
        if missing_entries_to_process:
            print(f"Missing entries to add side chains {len(missing_entries_to_process)}")
            process_with_args = partial(side_ch_packing, args, pdb_dict)
            skipped_tasks_count = 0 
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_with_args, k) for k in missing_entries_to_process]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=1500)  # Adjust timeout as needed
                        if result is not None:
                            results.append(result)
                    except concurrent.futures.TimeoutError:
                        print("A task exceeded the time limit and was skipped.")
                        skipped_tasks_count += 1

            print(f"Total tasks skipped due to time limit: {skipped_tasks_count}")
        else:
            print(f"Already processed. Skipping....")
            print("To repeat the process, simply delete", args.out_file)

        with open(args.out_file, 'a') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')

    gc.collect()

    # Clean up the cache directory after processing
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f"Cleaning {CACHE_DIR}.")

if __name__ == "__main__":
    main(args)
