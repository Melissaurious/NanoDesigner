#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
import sys 
from dyMEAN.utils.renumber import renumber_pdb
from dyMEAN.data.pdb_utils import AgAbComplex2, Protein

from dyMEAN.data.pdb_utils import Peptide as Peptide_Class

from dyMEAN.utils.logger import print_log
from dyMEAN.utils.relax import rosetta_sidechain_packing 
import gc

import time
import os.path as osp

"""
SOURCE: https://www.blopig.com/blog/2023/05/checking-your-pdb-file-for-clashing-atoms/
with additional information about the clashes
"""

from Bio import PDB
import numpy as np
from scipy.spatial import cKDTree
import concurrent.futures  # to paralelize the processes
from functools import partial # required for my paralelization 

parser = PDB.PDBParser(QUIET=True)
downloader = PDB.PDBList()

# For developability metrics
# import peptides

# Atomic radii for various atom types. 
# You can comment out the ones you don't care about or add new ones
atom_radii = {
#    "H": 1.20,  # Who cares about hydrogen??
    "C": 1.70, 
    "N": 1.55, 
    "O": 1.52,
    "S": 1.80,
    "F": 1.47, 
    "P": 1.80, 
    "CL": 1.75, 
    "MG": 1.73,
}


def get_default_item(pdb_dict):

    return next(iter(pdb_dict.values()))






def clean_extended(origin_antigen_pdb, origin_antibody_pdb, template_pdb, out_pdb,
                   chain_mapping=None, remove_hydrogens=False, preserve_insertion_codes=True):
    """
    Clean and align template structure using original reference structures.
    
    Keeps template coordinates with original residue numbering for positions that 
    exist in both structures. Removes hydrogens and handles insertion codes properly.
    
    Args:
        origin_antigen_pdb: Path to original antigen PDB
        origin_antibody_pdb: Path to original antibody PDB  
        template_pdb: Path to template/predicted PDB
        out_pdb: Output path for cleaned structure
        chain_mapping: Optional dict mapping original to template chains
        remove_hydrogens: Remove hydrogen atoms (default: True)
        preserve_insertion_codes: Keep insertion codes in numbering (default: True)
    """
    from Bio import PDB
    from Bio.PDB import PDBIO, Structure, Model, Chain
    from difflib import SequenceMatcher
    import traceback
    
    def extract_residue_info(pdb_file):
        """Extract residue information from PDB file"""
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("temp", pdb_file)
        
        chain_data = {}
        for model in structure:
            for chain in model:
                residues = []
                sequence = ""
                
                for residue in chain:
                    if PDB.is_aa(residue):
                        res_id = residue.id
                        
                        # Create position key
                        if preserve_insertion_codes:
                            pos_key = f"{res_id[1]}{res_id[2].strip()}" if res_id[2].strip() else str(res_id[1])
                        else:
                            pos_key = str(res_id[1])
                        
                        # Build sequence
                        try:
                            aa_code = PDB.Polypeptide.three_to_one(residue.get_resname())
                            sequence += aa_code
                        except KeyError:
                            sequence += 'X'
                        
                        residues.append({
                            'position': pos_key,
                            'residue_id': res_id,
                            'residue': residue,
                            'resname': residue.get_resname()
                        })
                
                chain_data[chain.id] = {
                    'residues': residues,
                    'sequence': sequence
                }
        
        return chain_data
    
    def map_chains(orig_data, template_data, manual_mapping=None):
        """Map original chains to template chains"""
        if manual_mapping:
            return manual_mapping
        
        mappings = {}
        used_template_chains = set()
        
        # Match by sequence similarity (threshold 0.75)
        for orig_id, orig_info in orig_data.items():
            best_match = None
            best_score = 0
            
            for temp_id, temp_info in template_data.items():
                if temp_id in used_template_chains:
                    continue
                
                matcher = SequenceMatcher(None, orig_info['sequence'], temp_info['sequence'])
                score = matcher.ratio()
                
                if score > best_score and score >= 0.60:
                    best_score = score
                    best_match = temp_id
            
            if best_match:
                mappings[orig_id] = best_match
                used_template_chains.add(best_match)
        
        return mappings
    
    def clean_residue(residue):
        """Remove hydrogens from residue"""
        if not remove_hydrogens:
            return residue
        
        cleaned = residue.copy()
        atoms_to_remove = []
        
        for atom in cleaned:
            if atom.element == 'H' or atom.get_name().startswith('H'):
                atoms_to_remove.append(atom.get_id())
        
        for atom_id in atoms_to_remove:
            cleaned.detach_child(atom_id)
        
        return cleaned
    
    try:
        # Load structures
        antigen_data = extract_residue_info(origin_antigen_pdb)
        antibody_data = extract_residue_info(origin_antibody_pdb)
        original_data = {**antigen_data, **antibody_data}
        template_data = extract_residue_info(template_pdb)
        
        # Create chain mapping
        mappings = map_chains(original_data, template_data, chain_mapping)
        
        if not mappings:
            # Fallback: copy template if no mappings
            import shutil
            shutil.copy2(template_pdb, out_pdb)
            return
        
        # Create new structure
        new_structure = Structure.Structure("cleaned")
        new_model = Model.Model(0)
        new_structure.add(new_model)
        
        # Process each chain
        for orig_chain_id, template_chain_id in mappings.items():
            orig_residues = original_data[orig_chain_id]['residues']
            template_residues = template_data[template_chain_id]['residues']
            
            # Create position maps
            orig_pos_map = {r['position']: r for r in orig_residues}
            template_pos_map = {r['position']: r for r in template_residues}
            
            # Find common positions
            common_positions = set(orig_pos_map.keys()) & set(template_pos_map.keys())
            
            if not common_positions:
                continue
            
            # Create new chain
            new_chain = Chain.Chain(orig_chain_id)
            
            # Sort positions numerically
            sorted_positions = sorted(common_positions, key=lambda x: (
                int(''.join(filter(str.isdigit, x)) or '0'),
                ''.join(filter(str.isalpha, x))
            ))
            
            # Process residues
            for position in sorted_positions:
                orig_res_info = orig_pos_map[position]
                template_res_info = template_pos_map[position]
                
                # Use template coordinates with original numbering
                template_residue = template_res_info['residue']
                new_residue = clean_residue(template_residue)
                new_residue.id = orig_res_info['residue_id']  # Original numbering
                
                try:
                    new_chain.add(new_residue)
                except:
                    continue
            
            # Add chain if it has residues
            if len(new_chain) > 0:
                new_model.add(new_chain)
        
        # Save structure
        io = PDBIO()
        io.set_structure(new_structure)
        io.save(out_pdb)
        
    except Exception as e:
        # Fallback on error
        try:
            import shutil
            shutil.copy2(template_pdb, out_pdb)
        except:
            pass


def side_ch_packing(args, pdb_dict, item):
    # Extract PDB ID for tracking purposes
    try:
        if isinstance(item, dict):
            pdb = item.get("pdb", item.get("entry_id", "unknown_pdb"))
        elif isinstance(item, str):
            item_dict = json.loads(item)
            pdb = item_dict.get("pdb", item_dict.get("entry_id", "unknown_pdb"))
        else:
            pdb = "unknown_pdb"
    except:
        pdb = "unknown_pdb"
    
    try:
        # Try treating 'item' as a dictionary and access its keys
        H, L, A = item['heavy_chain'], item['light_chain'], item['antigen_chains']
        # Use safe key access with fallback
        pdb = item.get('pdb', item.get('entry_id', 'unknown_pdb'))
        mod_pdb = item["mod_pdb"]

    except (TypeError, KeyError):  # Catch both TypeError and KeyError
        # If 'item' is not a dictionary or missing keys, parse it as JSON
        try:
            item = json.loads(item)
            H, L, A = item['heavy_chain'], item['light_chain'], item['antigen_chains']
            # Use safe key access with fallback
            pdb = item.get('pdb', item.get('entry_id', 'unknown_pdb'))
            mod_pdb = item["mod_pdb"]
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing item: {e}")
            print(f"Item content: {item}")
            return None

    # Enhanced early validation - check file existence and content before proceeding
    item["side_chain_packed"] = "No"
    
    if not os.path.exists(mod_pdb):
        print_log(f'{mod_pdb} not exists!', level='ERROR')
        print(f'{mod_pdb} not exists!')
        item["error"] = f"mod_pdb file does not exist: {mod_pdb}"
        return item
        
    if os.path.getsize(mod_pdb) == 0:
        print_log(f'{mod_pdb} exists but is empty!', level='ERROR')
        print(f'{mod_pdb} exists but is empty!')
        item["error"] = f"mod_pdb file is empty: {mod_pdb}"
        return item
    
    # Additional check: verify it's a valid PDB file before heavy processing
    try:
        with open(mod_pdb, 'r') as f:
            first_lines = f.read(200)  # Read more lines for better validation
            if not any(keyword in first_lines for keyword in ['ATOM', 'HETATM', 'HEADER', 'REMARK']):
                print(f'{mod_pdb} does not appear to be a valid PDB file!')
                item["error"] = f"Invalid PDB file format: {mod_pdb}"
                return item
    except Exception as e:
        print(f'Error reading {mod_pdb}: {e}')
        item["error"] = f"Error reading PDB file: {mod_pdb} - {e}"
        return item

    
    try:
        mod_cplx = AgAbComplex2.from_pdb(mod_pdb, H, L, A, numbering="imgt", skip_epitope_cal=True)
    except Exception as e:
        print(f"{mod_pdb} raised an error: {e}, renumbering to see if it fixes it...")
        renumber_pdb(mod_pdb, mod_pdb, scheme="imgt")

    item["side_chain_packed"] = "No"

    if not os.path.exists(mod_pdb):
        print_log(f'{mod_pdb} not exists!', level='ERROR')
        print(f'{mod_pdb} not exists!')
        return item
    if os.path.getsize(mod_pdb) == 0:
        print_log(f'{mod_pdb} exists but is empty!', level='ERROR')
        print_log(f'{mod_pdb} exists but is empty!')
        return item

    pdb_modified = pdb.rsplit('_', 1)[0]

    ref_pdb = item.get("ref_pdb", None)
    if ref_pdb is not None:
        ref_for_sanity_ck = ref_pdb
    else:
        print("original_item", original_item)
        ref_for_sanity_ck = original_item.get("pdb_data_path")
        if ref_for_sanity_ck == None:
            ref_for_sanity_ck = original_item["nano_source"]
        print("ref_for_sanity_ck", ref_for_sanity_ck)

    if args.hdock_models:
        ref_for_sanity_ck = os.path.join(args.hdock_models, pdb, pdb_modified + "_only_nb.pdb")
        if not os.path.exists(ref_for_sanity_ck):
            ref_for_sanity_ck = os.path.join(args.hdock_models, pdb, pdb_modified + "_IgFold.pdb")

    mod_revised, ref_revised = False, False
    chains_list = [H] + A
    if L != "":
        chains_list = [H] + A + [L]

    try:
        mod_prot = Protein.from_pdb(mod_pdb, chains_list)
    except Exception as e:
        print(f'parse {mod_pdb} failed for {e}')
        return item
    try:
        ref_prot = Protein.from_pdb(ref_for_sanity_ck, chains_list)
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

                ref_residues = [pos_map[res_id] for res_id in ['-'.join(str(a) for a in res.get_id()) for res in mod_residues] if res_id in pos_map]

                # Update peptide chains within the Protein objects 
                mod_prot.peptides[chain_name] = Peptide_Class(chain_name, mod_residues)
                ref_prot.peptides[chain_name] = Peptide_Class(chain_name, ref_residues)

                mod_revised, ref_revised = True, False  # Only mod_prot is revised
                print(f"{mod_prot.peptides[chain_name]} chain {chain_name} length after aligned: {len(mod_prot.peptides[chain_name].residues)} == {len(ref_prot.peptides[chain_name].residues)}")

    except Exception as e:
        print(f"An exception was raised during alignment: {e}")
        print(traceback.format_exc())

    if mod_revised:
        print("Entered the mod_revised mode")
        mod_prot.to_pdb(mod_pdb)


    if args.cdr_model == 'dyMEAN':
        #dont do side chain packing, only refinement
        try:
            print("Skipping Side chain packing as dyMEAN is end-to-end")
            item["side_chain_packed"] = "No"
            structure = parser.get_structure(pdb, mod_pdb)
            total_clashes = count_clashes(structure, clash_cutoff=0.60)[0]

            chain_clashes = {}
            for chain in structure.get_chains():
                chain_id = chain.get_id()
                chain_clashes[chain_id] = count_clashes(chain, clash_cutoff=0.60)[0]

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
            total_clashes = count_clashes(structure, clash_cutoff=0.60)[0]

            chain_clashes = {}
            for chain in structure.get_chains():
                chain_id = chain.get_id()
                chain_clashes[chain_id] = count_clashes(chain, clash_cutoff=0.60)[0]

            inter_chain_clashes = total_clashes - sum(chain_clashes.values())

            item["inference_clashes"] = total_clashes
            item["clashes_per_chain_inference"] = chain_clashes
            item["inter_chain_clashes_inference"] = inter_chain_clashes
        

        except Exception as e:
            item["inference_clashes"] = np.nan
            item["clashes_per_chain_inference"] = np.nan
            item["inter_chain_clashes_inference"] = np.nan
        

        try:
            print(f"Conducting side chain packing...on {pdb}")
            mod_pdb, rosetta_time = rosetta_sidechain_packing(mod_pdb, mod_pdb)
            item["side_chain_packed"] = "Yes"
            item["side_ch_packing_time"] = rosetta_time
            # clean_pdb(mod_pdb, mod_pdb)
        except Exception as e:
            print(f"{mod_pdb} could not be side-chain-packed, skipping the evaluation of this model")
            item["side_chain_packed"] = "No"
            # return None

        try:
            structure = parser.get_structure(pdb, mod_pdb)
            total_clashes = count_clashes(structure, clash_cutoff=0.60)[0]

            chain_clashes = {}
            for chain in structure.get_chains():
                chain_id = chain.get_id()
                chain_clashes[chain_id] = count_clashes(chain, clash_cutoff=0.60)[0]

            inter_chain_clashes = total_clashes - sum(chain_clashes.values())

            item["side_chain_p_num_clashes"] = total_clashes
            item["clashes_per_chain_side_ch_packing"] = chain_clashes
            item["inter_chain_clashes_side_ch_packing"] = inter_chain_clashes
        

        except Exception as e:
            print_log(f'Clash analysis for {mod_pdb} failed for {e}', level='ERROR')

    
    if args.hdock_models:
        # pdb_parts = pdb.rsplit('_')[0]
        pdb_parts = pdb.rsplit('_',1)[0]
        original_item = pdb_dict.get(pdb_parts)
    else:
        # Use consistent key access logic
        pdb_key = item.get("entry_id", item.get("pdb"))
        original_item = pdb_dict.get(pdb_key)


    if original_item is None:
        # print(f"{pdb} doesn't have a json file in {args.test_set}")
        original_item = get_default_item(pdb_dict)
        # print("Using a default item from pdb_dict")
    else:
        # print("All good")
        pass

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


    return item  # Return the original result


def load_structure(pdb_file_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file_path)
    return structure

def count_clashes(structure, clash_cutoff=0.60):
    # Set what we count as a clash for each pair of atoms
    clash_cutoffs = {i + "_" + j: (clash_cutoff * (atom_radii[i] + atom_radii[j])) for i in atom_radii for j in atom_radii}

    # Extract atoms for which we have radii
    atoms = [x for x in structure.get_atoms() if x.element in atom_radii]
    coords = np.array([a.coord for a in atoms], dtype="d")

    # Build a KDTree using scipy
    kdt = cKDTree(coords)

    # Initialize a list to hold clashes and clash_details
    clashes = []
    clash_details = []

    # Iterate through all atoms
    for atom_1 in atoms:
        # Find atoms that could be clashing
        kdt_search = kdt.query_ball_point(atom_1.coord, max(clash_cutoffs.values()))

        # Get index and distance of potential clashes
        potential_clash = [(ix, np.linalg.norm(coords[ix] - atom_1.coord)) for ix in kdt_search]

        for ix, atom_distance in potential_clash:
            atom_2 = atoms[ix]
            if (
                atom_1.parent.id == atom_2.parent.id
                or (atom_2.name == "C" and atom_1.name == "N")
                or (atom_2.name == "N" and atom_1.name == "C")
                or (atom_2.name == "SG" and atom_1.name == "SG" and atom_distance > 1.88)
            ):
                continue

            clash_type = f"{atom_1.element}-{atom_2.element} clash"
            if atom_distance < clash_cutoffs[atom_2.element + "_" + atom_1.element]:
                clashes.append((atom_1, atom_2))
                clash_details.append({
                    'clash_type': clash_type,
                    'atoms_involved': (atom_1, atom_2),
                    'distance': atom_distance,
                    'residues_involved': (atom_1.parent.id, atom_2.parent.id)
                })

    return len(clashes) // 2, list(clash_details)




# Create the argument parser
arg_parser = argparse.ArgumentParser(description='do side chain packing and refinement')
arg_parser.add_argument('--summary_json', type=str, required=True, help='Path to the summary in json format providing H/L/antigen')
arg_parser.add_argument('--out_file', type=str, required=True, help='Path to the summary in json format providing H/L/antigen')
arg_parser.add_argument('--test_set', type=str, required=True, help='Path to the reference data')
arg_parser.add_argument('--iteration', type=str, default="0", help='Path to the reference data')
arg_parser.add_argument('--hdock_models', type=str, default="", help='Path to the reference data')
arg_parser.add_argument('--cdr_model', type=str, required=True, help='Type of model that generates CDRs',
                        choices=['DIFFAB', 'DiffAb', 'dyMEAN', 'ADesign', 'AbDesign', 'ADesigner', 'ADESIGN'])

args = arg_parser.parse_args()


def main(args):
    # Create tracking directory in the same location as summary files
    summary_dir = os.path.dirname(args.summary_json)
    
    # Create meaningful filename for tracking
    iteration = args.iteration
    model_type = args.cdr_model

    numbering = "imgt"
    with open(args.test_set, 'r') as fin:
        data = fin.read().strip().split('\n')

    # Create a dictionary of {pdb:whole dictionary corresponding to such pdb}
    pdb_dict = {}
    for item in data:
        json_obj = json.loads(item)
        pdb_id = json_obj.get("entry_id", json_obj.get("pdb"))
        pdb_dict[pdb_id] = json_obj

    with open(args.summary_json, 'r') as fin:
        summary = fin.read().strip().split('\n')
    
    print(f"Total entries in summary: {len(summary)}")
    
    # Check if the file doesn't exist or is empty
    if not os.path.exists(args.out_file) or os.stat(args.out_file).st_size == 0:
        print("Output file doesn't exist or is empty, processing all entries...")
        with open(args.out_file, 'w') as f:
            pass

        # Filter entries that have valid mod_pdb files before processing
        valid_entries = []
        for entry_str in summary:
            try:
                entry_json = json.loads(entry_str)
                mod_pdb = entry_json.get("mod_pdb")
                
                # Check if mod_pdb exists and is not empty
                if mod_pdb and os.path.exists(mod_pdb) and os.path.getsize(mod_pdb) > 0:
                    # Additional check: verify it's a valid PDB file
                    try:
                        with open(mod_pdb, 'r') as f:
                            first_lines = f.read(100)
                            if any(keyword in first_lines for keyword in ['ATOM', 'HETATM', 'HEADER']):
                                valid_entries.append(entry_str)
                            else:
                                print(f"Skipping invalid PDB file: {mod_pdb}")
                    except:
                        print(f"Error reading PDB file: {mod_pdb}")
                else:
                    print(f"Skipping entry with missing/empty mod_pdb: {mod_pdb}")
            except json.JSONDecodeError:
                print("Error parsing summary entry, skipping...")
                continue

        print(f"Valid entries to process: {len(valid_entries)}")

        process_with_args = partial(side_ch_packing, args, pdb_dict)
        skipped_tasks_count = 0 
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_with_args, k) for k in valid_entries]
            
            results = []
            completed_count = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=1500)
                    if result is not None:
                        results.append(result)
                    completed_count += 1
                    
                    # Log progress periodically
                    if completed_count % 10 == 0 or completed_count == len(valid_entries):
                        print(f"Progress: {completed_count}/{len(valid_entries)} entries processed")
                        
                except concurrent.futures.TimeoutError:
                    print("A task exceeded the time limit and was skipped.")
                    skipped_tasks_count += 1
                    completed_count += 1
                except Exception as e:
                    print(f"Error processing entry: {e}")
                    skipped_tasks_count += 1
                    completed_count += 1

        print(f"Total tasks skipped due to time limit or errors: {skipped_tasks_count}")

        with open(args.out_file, 'a') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')

    elif os.path.exists(args.out_file) and os.stat(args.out_file).st_size != 0:
        print("Output file exists and has content, checking what's already processed...")
        
        # File exists and is not empty, check what is in there to not repeat the whole process
        with open(args.out_file, 'r') as f:
            existing_data = f.read().strip().split('\n')
        
        processed_mod_pdbs = set()
        successfully_processed = set()  # Track entries that were successfully processed
        
        for entry_str in existing_data:
            if entry_str.strip():  # Skip empty lines
                try:
                    entry_json = json.loads(entry_str)
                    mod_pdb = entry_json.get("mod_pdb")
                    if mod_pdb:
                        processed_mod_pdbs.add(mod_pdb)
                        # Only consider it successfully processed if side_chain_packed is "Yes"
                        if entry_json.get('side_chain_packed') == 'Yes':
                            successfully_processed.add(mod_pdb)
                except json.JSONDecodeError:
                    continue

        print(f"Previously processed entries: {len(processed_mod_pdbs)}")
        print(f"Successfully processed entries: {len(successfully_processed)}")

        # Find entries that need processing - more thorough check
        missing_entries_to_process = []
        for entry_str in summary:
            try:
                entry_json = json.loads(entry_str)
                mod_pdb = entry_json.get("mod_pdb")
                
                # Skip if successfully processed
                if mod_pdb in successfully_processed:
                    continue
                
                # Check if the file exists and is valid
                if not mod_pdb or not os.path.exists(mod_pdb) or os.path.getsize(mod_pdb) == 0:
                    print(f"Skipping entry with missing/empty mod_pdb: {mod_pdb}")
                    continue
                
                # Additional validation for PDB file content
                try:
                    with open(mod_pdb, 'r') as f:
                        first_lines = f.read(100)
                        if not any(keyword in first_lines for keyword in ['ATOM', 'HETATM', 'HEADER']):
                            print(f"Skipping invalid PDB file: {mod_pdb}")
                            continue
                except:
                    print(f"Error reading PDB file: {mod_pdb}")
                    continue
                
                # Check if it needs reprocessing (failed or incomplete previous processing)
                if (mod_pdb not in processed_mod_pdbs or 
                    entry_json.get('side_chain_packed', 'No') != 'Yes'):
                    missing_entries_to_process.append(entry_str)
                    
            except json.JSONDecodeError:
                continue

        print(f"Entries that need (re)processing: {len(missing_entries_to_process)}")

        results = []
        if missing_entries_to_process:
            process_with_args = partial(side_ch_packing, args, pdb_dict)
            skipped_tasks_count = 0 
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_with_args, k) for k in missing_entries_to_process]
                
                completed_count = 0
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=1500)
                        if result is not None:
                            results.append(result)
                        completed_count += 1
                        
                        # Log progress periodically
                        if completed_count % 10 == 0 or completed_count == len(missing_entries_to_process):
                            print(f"Progress: {completed_count}/{len(missing_entries_to_process)} entries processed")
                            
                    except concurrent.futures.TimeoutError:
                        print("A task exceeded the time limit and was skipped.")
                        skipped_tasks_count += 1
                        completed_count += 1
                    except Exception as e:
                        print(f"Error processing entry: {e}")
                        skipped_tasks_count += 1
                        completed_count += 1

            print(f"Total tasks skipped due to time limit or errors: {skipped_tasks_count}")
        else:
            print("All entries already processed successfully. Skipping....")

        # Append new results to the file
        if results:
            with open(args.out_file, 'a') as f:
                for item in results:
                    f.write(json.dumps(item) + '\n')
                
    # Added to free memory from GPU/CPU
    gc.collect()


if __name__ == "__main__":
    main(args)

