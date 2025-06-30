from data.pdb_utils import VOCAB, AgAbComplex, AgAbComplex_mod, Protein, Peptide, AgAbComplex2
from utils.logger import print_log
from utils.relax import openmm_relax, openmm_relax_no_decorator
from define_aa_contacts_antibody_nanobody_2025 import get_cdr_residues_dict, interacting_residues, get_cdr_residues_and_interactions, clean_up_files, dedup_interactions,get_cdr_residues_and_interactions_gt_based
import concurrent
import time
from functools import partial 
from utils.renumber import renumber_pdb
import os
import argparse
import json
import copy
import gc
import cProfile
import pstats
import traceback

"""
SOURCE: https://www.blopig.com/blog/2023/05/checking-your-pdb-file-for-clashing-atoms/
with additional information about the clashes
"""

from Bio import PDB
import numpy as np
from scipy.spatial import cKDTree

parser = PDB.PDBParser(QUIET=True)
downloader = PDB.PDBList()


import warnings
from Bio import BiopythonWarning

# Suppress only the specific Biopython occupancy warnings
warnings.filterwarnings("ignore", category=BiopythonWarning, message="Missing occupancy in atom")


class IMGT:
    # heavy chain
    HFR1 = (1, 26)
    HFR2 = (39, 55)
    HFR3 = (66, 104)
    HFR4 = (118, 129)

    H1 = (27, 38)
    H2 = (56, 65)
    H3 = (105, 117)

    # light chain
    LFR1 = (1, 26)
    LFR2 = (39, 55)
    LFR3 = (66, 104)
    LFR4 = (118, 129)

    L1 = (27, 38)
    L2 = (56, 65)
    L3 = (105, 117)

    Hconserve = {
        23: ['CYS'],
        41: ['TRP'],
        104: ['CYS']
    }

    Lconserve = {
        23: ['CYS'],
        41: ['TRP'],
        104: ['CYS']
    }


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

            # Exclude clashes from atoms in the same residue, peptide bonds, and disulphide bridges (similar to your original code)
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


def extract_antibody_info(antibody, heavy_ch, light_ch, numbering):
    # classmethod extracted from AgAbComplex
    # Define numbering schemes
    _scheme = IMGT
    if numbering == 'imgt':
        _scheme = IMGT

    # get cdr/frame denotes
    h_type_mapping, l_type_mapping = {}, {}  # - for non-Fv region, 0 for framework, 1/2/3 for cdr1/2/3

    for lo, hi in [_scheme.HFR1, _scheme.HFR2, _scheme.HFR3, _scheme.HFR4]:
        for i in range(lo, hi + 1):
            h_type_mapping[i] = '0'
    for cdr, (lo, hi) in zip(['1', '2', '3'], [_scheme.H1, _scheme.H2, _scheme.H3]):
        for i in range(lo, hi + 1):
            h_type_mapping[i] = cdr
    h_conserved = _scheme.Hconserve

    for lo, hi in [_scheme.LFR1, _scheme.LFR2, _scheme.LFR3, _scheme.LFR4]:
        for i in range(lo, hi + 1):
            l_type_mapping[i] = '0'
    for cdr, (lo, hi) in zip(['1', '2', '3'], [_scheme.L1, _scheme.L2, _scheme.L3]):
        for i in range(lo, hi + 1):
            l_type_mapping[i] = cdr
    l_conserved = _scheme.Lconserve

    # get variable domain and cdr positions
    selected_peptides, cdr_pos = {}, {}

    chain_names = [heavy_ch]
    if light_ch:
        chain_names.append(light_ch)

    for c, chain_name in zip(['H', 'L'], chain_names):
        chain = antibody.get_chain(chain_name)
        # print("chain_names",chain_names)
        if chain is None:
            continue  # Skip processing if the chain is None
        # Note: possbly two chains are different segments of a same chain
        assert chain is not None, f'Chain {chain_name} not found in the antibody'
        type_mapping = h_type_mapping if c == 'H' else l_type_mapping
        conserved = h_conserved if c == 'H' else l_conserved
        res_type = ''
        for i in range(len(chain)):
            residue = chain.get_residue(i)
            residue_number = residue.get_id()[0]

            if residue_number in type_mapping:
                res_type += type_mapping[residue_number]
                if residue_number in conserved:
                    hit, symbol = False, residue.get_symbol()
                    for conserved_residue in conserved[residue_number]:
                        if symbol == VOCAB.abrv_to_symbol(conserved_residue):
                            hit = True
                            break
                        print(f"Actual residue type at position {residue_number} in chain {chain_name}: {symbol}")
                    assert hit, f'Not {conserved[residue_number]} at {residue_number} in chain {chain_name}'
            else:
                res_type += '-'
        if '0' not in res_type:
            print(heavy_ch, light_ch, antibody.pdb_id, res_type)
        start, end = res_type.index('0'), res_type.rindex('0')
        for cdr in ['1', '2', '3']:
            cdr_start, cdr_end = res_type.find(cdr), res_type.rfind(cdr)
            assert cdr_start != -1, f'cdr {c}{cdr} not found, residue type: {res_type}'
            start, end = min(start, cdr_start), max(end, cdr_end)
            cdr_pos[f'CDR-{c}{cdr}'] = (cdr_start, cdr_end)
        for cdr in ['1', '2', '3']:
            cdr = f'CDR-{c}{cdr}'
            cdr_start, cdr_end = cdr_pos[cdr]
            cdr_pos[cdr] = (cdr_start - start, cdr_end - start)
        chain = chain.get_span(start, end + 1)  # the length may exceed 130 because of inserted amino acids
        chain.set_id(chain_name)
        selected_peptides[chain_name] = chain

    return cdr_pos

def get_cdr_pos(cdr_pos, cdr):  # H/L + 1/2/3, return [begin, end] position
    # cdr_pos is a dictionary = {'CDR-H1': (24, 31), 'CDR-H2': (49, 56), 'CDR-H3': (95, 114)}
    cdr = f'CDR-{cdr}'.upper()
    if cdr in cdr_pos:
        return cdr_pos[cdr]
    else:
        return None


def clean_extended(origin_antigen_pdb, origin_antibody_pdb, template_pdb, out_pdb):
    origin_antigen_cplx = Protein.from_pdb(origin_antigen_pdb)
    origin_antibody_cplx = Protein.from_pdb(origin_antibody_pdb)
    template_cplx = Protein.from_pdb(template_pdb)
    peptides = {}

    ori_antigen_chain_to_id, ori_antibody_chain_to_id = {}, {}
    id_to_temp_chain = {}
    
    for chain_name, chain in origin_antigen_cplx:
        ori_antigen_chain_to_id[chain_name] = f'{chain.get_seq()[:5]}'
    for chain_name, chain in origin_antibody_cplx:
        ori_antibody_chain_to_id[chain_name] = f'{chain.get_seq()[:5]}'
    for chain_name, chain in template_cplx:
        id_to_temp_chain[f'{chain.get_seq()[:5]}'] = chain_name
    
    for chain_name in origin_antigen_cplx.get_chain_names():
        ori_chain = origin_antigen_cplx.get_chain(chain_name)
        temp_chain = template_cplx.get_chain(id_to_temp_chain[ori_antigen_chain_to_id[chain_name]])
        for i, residue in enumerate(ori_chain):
            if i < len(temp_chain):
                # renumber
                temp_chain.residues[i].id = residue.id
                # delete Hs
                for atom in temp_chain.residues[i].coordinate:
                    if atom[0] == 'H':
                        del temp_chain.residues[i].coordinate[atom]
            else:
                print_log(f'{origin_antigen_cplx.get_id()}, chain {chain_name} lost residues {len(ori_chain)} > {len(temp_chain)}')
                break
        temp_chain.set_id(chain_name)
        peptides[chain_name] = temp_chain
    
    for chain_name in origin_antibody_cplx.get_chain_names():
        ori_chain = origin_antibody_cplx.get_chain(chain_name)
        temp_chain = template_cplx.get_chain(id_to_temp_chain[ori_antibody_chain_to_id[chain_name]])
        for i, residue in enumerate(ori_chain):
            if i < len(temp_chain):
                # renumber
                temp_chain.residues[i].id = residue.id
                # delete Hs
                for atom in temp_chain.residues[i].coordinate:
                    if atom[0] == 'H':
                        del temp_chain.residues[i].coordinate[atom]
            else:
                print_log(f'{origin_antibody_cplx.get_id()}, chain {chain_name} lost residues {len(ori_chain)} > {len(temp_chain)}')
                break
        temp_chain.set_id(chain_name)
        peptides[chain_name] = temp_chain
    
    renumber_cplx = Protein(template_cplx.get_id(), peptides)
    renumber_cplx.to_pdb(out_pdb)





def get_unique_binding_data(track_file_2):
    """Read binding data and ensure hdock_models are unique"""
    binding_data = []
    seen_models = set()
    
    try:
        with open(track_file_2, 'r') as f:
            for line in f:
                item = json.loads(line)
                hdock_model = item.get("hdock_model")
                
                if hdock_model and hdock_model not in seen_models:
                    binding_data.append(item)
                    seen_models.add(hdock_model)
                elif hdock_model:
                    print(f"Warning: Duplicate hdock_model found and skipped: {hdock_model}")
                    
    except Exception as e:
        print(f"Error reading binding data: {e}")
        
    return binding_data



def create_unique_top_models(binding_data, n_top):
    """Create top models list ensuring no duplicates"""
    
    # First, ensure all entries have unique hdock_model values
    unique_data = {}
    for item in binding_data:
        hdock_model = item.get("hdock_model")
        if hdock_model:
            if hdock_model in unique_data:
                # Keep the one with better epitope_recall if duplicate
                existing_recall = unique_data[hdock_model].get('epitope_recall', 0.0)
                current_recall = item.get('epitope_recall', 0.0)
                if current_recall > existing_recall:
                    unique_data[hdock_model] = item
                    print(f"Replaced duplicate {hdock_model} with better epitope_recall: {current_recall} > {existing_recall}")
            else:
                unique_data[hdock_model] = item
    
    # Convert back to list
    binding_data_unique = list(unique_data.values())
    print(f"Reduced {len(binding_data)} entries to {len(binding_data_unique)} unique entries")
    
    # Filter and sort models
    filtered_list = [item for item in binding_data_unique if float(item.get("cdrh3_avg", 0.0)) != 0.0]
    filtered_list.sort(key=lambda x: x.get('epitope_recall', 0.0), reverse=True)
    
    print(f"Found {len(filtered_list)} models with non-zero cdr3_avg")
    
    # If we don't have enough non-zero models, include zero models
    if len(filtered_list) < n_top:
        print(f"Only {len(filtered_list)} models with non-zero cdr3_avg, including zero models to reach {n_top}")
        
        # Get models with zero cdr3_avg
        zero_models = [item for item in binding_data_unique if float(item.get("cdrh3_avg", 0.0)) == 0.0]
        zero_models.sort(key=lambda x: x.get('epitope_recall', 0.0), reverse=True)
        
        # Add zero models to reach the target number
        needed_models = n_top - len(filtered_list)
        filtered_list.extend(zero_models[:needed_models])
        
        # Re-sort the combined list
        filtered_list.sort(key=lambda x: (float(x.get("cdrh3_avg", 0.0)) != 0.0, x.get('epitope_recall', 0.0)), reverse=True)
    
    top_models = filtered_list[:n_top]
    
    # Final check: ensure no duplicates in top_models
    final_top_models = []
    seen_hdock_models = set()
    
    for item in top_models:
        hdock_model = item.get("hdock_model")
        if hdock_model and hdock_model not in seen_hdock_models:
            final_top_models.append(item)
            seen_hdock_models.add(hdock_model)
        elif hdock_model:
            print(f"Warning: Duplicate in top_models prevented: {hdock_model}")
    
    print(f"Final top models: {len(final_top_models)} unique entries")
    return final_top_models


def clean_existing_top_models(top_file):
    """Remove duplicates from existing top_models.json file"""
    if not os.path.exists(top_file):
        return
        
    try:
        unique_entries = {}
        
        with open(top_file, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    hdock_model = item.get("hdock_model")
                    if hdock_model:
                        if hdock_model in unique_entries:
                            print(f"Found duplicate in existing top_models.json: {hdock_model}")
                        unique_entries[hdock_model] = item
        
        # Rewrite file with unique entries only
        with open(top_file, 'w') as f:
            for item in unique_entries.values():
                f.write(json.dumps(item) + '\n')
                
        print(f"Cleaned top_models.json: {len(unique_entries)} unique entries")
        
    except Exception as e:
        print(f"Error cleaning existing top_models.json: {e}")




def detect_epitope_format(epitope):
    """
    Detect if epitope is:
    - 'single': list of antigen residues only [["A", 4, "Y"], ...]
    - 'paired': list of interaction pairs [[antigen_res, antibody_res], ...]
    """
    if not epitope:
        return 'empty'
    
    # Check first element
    first_elem = epitope[0]
    if isinstance(first_elem, (list, tuple)):
        if len(first_elem) == 3:  # [chain, pos, aa]
            return 'single'
        elif len(first_elem) == 2 and isinstance(first_elem[0], (list, tuple)):
            return 'paired'
    
    return 'unknown'



def process_epitope_data(original_item, ab_pdb):
    """
    Fixed version that properly handles single-sided (user-defined) epitopes
    """
    
    # Get original epitope
    epitope = original_item.get("epitope", [])
    if not epitope:
        print("No epitope data found in original_item")
        return [], [], 0
    
    # Check if this is user-defined epitope
    is_user_input = original_item.get("epitope_user_input", "no") == "yes"
    
    # Detect epitope format
    epitope_format = detect_epitope_format(epitope)
    print(f"Detected epitope format: {epitope_format}")
    print(f"User-defined epitope: {is_user_input}")
    
    if epitope_format == 'single' or is_user_input:
        # Handle single-sided epitope (antigen residues only)
        print("Processing single-sided/user-defined epitope")
        
        epitope_ = [tuple(item) for item in epitope]
        epitope_ = dedup_interactions(epitope_)
        
        # For single format, convert to (chain, pos) tuples
        filtered_gt_epitope = []
        for res in epitope_:
            try:
                if len(res) >= 2:
                    chain = res[0]
                    pos = str(res[1])  # Normalize to string
                    filtered_gt_epitope.append((chain, pos))
            except:
                continue
        
        # Remove duplicates
        filtered_gt_epitope = list(set(filtered_gt_epitope))
        
        print(f"Single-sided epitope: {len(filtered_gt_epitope)} antigen residues")
        print(f"Epitope residues: {sorted(filtered_gt_epitope)}")
        
        # For user-defined epitopes, no framework filtering is needed
        framework_removed = 0
        
        return filtered_gt_epitope, epitope_, framework_removed
        
    elif epitope_format == 'paired':
        # Handle paired epitope (computed interactions)
        print("Processing paired epitope")
        
        epitope_ = [tuple(item) for item in epitope]
        epitope_ = dedup_interactions(epitope_)
        
        # Get CDR dictionary for filtering
        cdr_dict = get_cdr_residues_dict(original_item, ab_pdb)
        
        if not cdr_dict:
            print("No CDR dictionary available - using all epitope interactions")
            # Fallback: extract antigen residues from all interactions
            filtered_gt_epitope = []
            for interaction in epitope_:
                try:
                    if isinstance(interaction, (list, tuple)) and len(interaction) >= 2:
                        antigen_res = interaction[0]
                        if len(antigen_res) >= 2:
                            chain = antigen_res[0]
                            pos = str(antigen_res[1])
                            filtered_gt_epitope.append((chain, pos))
                except:
                    continue
            
            filtered_gt_epitope = list(set(filtered_gt_epitope))
            return filtered_gt_epitope, epitope_, 0
        
        # Build CDR positions set
        all_cdr_positions = set()
        for chain_id, cdrs in cdr_dict.items():
            for cdr_name, positions in cdrs.items():
                for pos_info in positions:
                    try:
                        if len(pos_info) >= 2:
                            chain, pos = pos_info[0], str(pos_info[1])
                            all_cdr_positions.add((chain, pos))
                    except:
                        continue
        
        print(f"Found {len(all_cdr_positions)} CDR positions")
        
        # Filter epitope to CDR-only interactions
        filtered_epitope_interactions = []
        framework_removed = 0
        
        for interaction in epitope_:
            try:
                if isinstance(interaction, (list, tuple)) and len(interaction) >= 2:
                    antigen_res, antibody_res = interaction
                    
                    if len(antibody_res) >= 2:
                        ab_chain, ab_pos = antibody_res[0], str(antibody_res[1])
                        
                        if (ab_chain, ab_pos) in all_cdr_positions:
                            filtered_epitope_interactions.append(interaction)
                        else:
                            framework_removed += 1
            except Exception as e:
                print(f"Error processing interaction {interaction}: {e}")
                continue
        
        # Extract antigen residues from filtered interactions
        filtered_gt_epitope = []
        for interaction in filtered_epitope_interactions:
            try:
                if isinstance(interaction, (list, tuple)) and len(interaction) >= 2:
                    antigen_res = interaction[0]
                    if len(antigen_res) >= 2:
                        chain = antigen_res[0]
                        pos = str(antigen_res[1])
                        filtered_gt_epitope.append((chain, pos))
            except:
                continue
        
        # Remove duplicates
        filtered_gt_epitope = list(set(filtered_gt_epitope))
        
        print(f"Paired epitope filtered: {len(epitope_)} -> {len(filtered_epitope_interactions)} interactions")
        print(f"Framework interactions removed: {framework_removed}")
        print(f"Final antigen epitope residues: {len(filtered_gt_epitope)}")
        
        return filtered_gt_epitope, epitope_, framework_removed
    
    else:
        print(f"Unknown or empty epitope format: {epitope_format}")
        return [], [], 0


# Also update your detect_epitope_format function to handle this case better
def detect_epitope_format_improved(epitope):
    """
    Improved epitope format detection
    """
    if not epitope:
        return 'empty'
    
    # Check first entry
    first_entry = epitope[0]
    
    if isinstance(first_entry, (list, tuple)):
        if len(first_entry) == 3:
            # Format: [chain, pos, aa] - single format
            return 'single'
        elif len(first_entry) == 2:
            # Could be [chain, pos] (single) or [(antigen), (antibody)] (paired)
            first_elem = first_entry[0]
            if isinstance(first_elem, (list, tuple)):
                # [(antigen_res), (antibody_res)] - paired format
                return 'paired'
            else:
                # [chain, pos] - single format
                return 'single'
        else:
            # Unusual length, might be paired
            return 'paired'
    
    return 'single'  # Default to single


# Debug function to verify your epitope processing
def debug_epitope_processing(original_item):
    """
    Add this to debug your specific case
    """
    print("=== EPITOPE DEBUG ===")
    epitope = original_item.get("epitope", [])
    is_user_input = original_item.get("epitope_user_input", "no") == "yes"
    
    print(f"Epitope user input: {is_user_input}")
    print(f"Epitope length: {len(epitope)}")
    print(f"First few epitope entries: {epitope[:3]}")
    
    epitope_format = detect_epitope_format_improved(epitope)
    print(f"Detected format: {epitope_format}")
    
    # Process epitope
    filtered_gt_epitope, epitope_, framework_removed = process_epitope_data_fixed(original_item, ab_pdb)
    
    print(f"Processed results:")
    print(f"  Filtered GT epitope: {len(filtered_gt_epitope)} residues")
    print(f"  Framework removed: {framework_removed}")
    print(f"  Sample epitope: {filtered_gt_epitope[:5]}")
    
    return filtered_gt_epitope, epitope_, framework_removed


# from define_aa_contacts_antibody_nanobody_2025 import get_cdr_residues_dict, interacting_residues, get_cdr_residues_and_interactions, clean_up_files
def binding_residues_analysis(shared_args, hdock_model):
    start_time = time.time()

    # ag_pdb, ab_pdb, binding_rsite, item, pdb_dir_hdock, tmp_dir_for_interacting_aa = shared_args
    # ag_pdb, ab_pdb, binding_rsite, item, pdb_dir_hdock, tmp_dir_for_interacting_aa, original_gt_epitope_ = shared_args
    ag_pdb, ab_pdb, binding_rsite, item, pdb_dir_hdock, tmp_dir_for_interacting_aa, original_gt_epitope_, framework_removed = shared_args

    # print(item)

    antigen_chains = item["antigen_chains"]
    heavy_chain = item['heavy_chain']
    light_chain = item['light_chain']

    # Sanity check, make all the models to have same numbering as the ground truth pdbs
    H, L, A = heavy_chain, light_chain, antigen_chains
    mod_revised, ref_revised = False, False
    chains_list = [H] + A
    if L != "":
        chains_list = [H] + A + [L]

    try:
        mod_prot = Protein.from_pdb(hdock_model, chains_list)
    except Exception as e:
        print(f'parse {hdock_model} failed for {e}')
        return None

    try:
        ref_prot = Protein.from_pdb(ab_pdb, chains_list)
    except Exception as e:
        print(f'parse {ab_pdb} failed for {e}')
        return None

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

                # # Prepare to store revised residues
                # mod_residues, ref_residues = [], []
                # pos_map = {'-'.join(str(a) for a in res.get_id()): res for res in ref_chain.residues}

                # for res in mod_chain.residues:
                #     res_id = '-'.join(str(a) for a in res.get_id())
                #     if res_id in pos_map:
                #         mod_residues.append(res)
                #     else:
                #         print(f"Warning: Residue {res_id} in mod_chain not found in ref_chain. Skipping.")

                # ref_residues = [pos_map[res_id] for res_id in ['-'.join(str(a) for a in res.get_id()) for res in mod_residues] if res_id in pos_map]

                # # Update peptide chains within the Protein objects 
                # mod_prot.peptides[chain_name] = Peptide(chain_name, mod_residues)
                # ref_prot.peptides[chain_name] = Peptide(chain_name, ref_residues)

                # mod_revised, ref_revised = True, False  # Only mod_prot is revised
                # print(f"{mod_prot.peptides[chain_name]} chain {chain_name} length after aligned: {len(mod_prot.peptides[chain_name].residues)} == {len(ref_prot.peptides[chain_name].residues)}")

    except Exception as e:
        print(f"An exception was raised: {e}")

    if mod_revised:
        print("Entered the mod_revised mode")
        # mod_prot.to_pdb(hdock_model)

    try:
        clean_extended(ag_pdb, ab_pdb, hdock_model, hdock_model)
    except Exception as e:
        print(f"model {hdock_model}, gave this error: {e}")
        print(traceback.format_exc())

    # Step 1: Extract CDR residues using your updated function
    cdr_dict = get_cdr_residues_dict(item, hdock_model)
    
    # Get the base name of the file
    filename = os.path.basename(hdock_model)
    model_name, extension = os.path.splitext(filename)

    # Step 2: Analyze interactions using your updated pipeline
    # Define immunoglobulin chains
    immuno_chains = [heavy_chain]
    if light_chain:  # Only add light chain if it exists
        immuno_chains.append(light_chain)

    # Initialize aggregated results dictionary
    aggregated_results = {}
    
    print(f"Starting binding interface analysis")
    start_time1 = time.time()
    
    # Process each immunoglobulin chain against each antigen
    for immuno_chain in immuno_chains:
        immuno_results_dict = {} 
        
        for antigen in antigen_chains:
            print(f"Processing {immuno_chain} -> {antigen}")
            
            # Create specific temporary directory for this interaction
            tmp_specific = os.path.join(tmp_dir_for_interacting_aa, f"{model_name}_{immuno_chain}_to_{antigen}")
            os.makedirs(tmp_specific, exist_ok=True)
            tmp_pdb = os.path.join(tmp_specific, f"{model_name}_{immuno_chain}_to_{antigen}.pdb")
            
            # Prepare chains for reconstruction
            chains_to_reconstruct = []
            chains_to_reconstruct.extend(immuno_chain)
            chains_to_reconstruct.extend(antigen)
            
            # Create temporary PDB with only necessary chains if it doesn't exist
            if not os.path.exists(tmp_pdb):
                try:
                    protein = Protein.from_pdb(hdock_model, chains_to_reconstruct)
                    protein.to_pdb(tmp_pdb)
                    # Renumber according to specified scheme (assuming numbering is defined globally)
                    renumber_pdb(tmp_pdb, tmp_pdb, scheme="imgt")  # or your preferred numbering
                    
                    if not os.path.exists(tmp_pdb):
                        print(f"Failed to create temporary PDB: {tmp_pdb}")
                        continue
                        
                except Exception as e:
                    print(f"Failed to process PDB file '{hdock_model}' for interaction {immuno_chain}→{antigen}: {e}")
                    continue
            
            # Use your updated interacting_residues function
            result = interacting_residues(item, tmp_pdb, immuno_chain, antigen, tmp_specific)
            
            if result is not None:
                print(f"Found {len(result)} interactions between {immuno_chain} and {antigen}")
                immuno_results_dict[antigen] = result
            else:
                print(f"No interactions found between {immuno_chain} and {antigen}")
                immuno_results_dict[antigen] = []

        aggregated_results[immuno_chain] = immuno_results_dict
    
    end_time1 = time.time()
    print(f"Finished binding residues analysis: {end_time1 - start_time1:.2f} seconds")



    temp_item = item.copy()  # Work with a copy to avoid modifying original

    # Pass the filtered GT epitope to the function for CDR percentage calculation
    # filtered_gt_epitope is now guaranteed to be defined
    filtered_gt_epitope = binding_rsite
    temp_item = get_cdr_residues_and_interactions_gt_based(temp_item, cdr_dict, aggregated_results, filtered_gt_epitope)

    # Step 5: Calculate TWO types of epitope recall
    model_epitope_cdr = temp_item.get("epitope", [])  # CDR-only interactions
    # model_epitope_all = temp_item.get("all_interactions", [])  # All interactions
    model_epitope_all = temp_item.get("epitope_all_cdr", []) 

    # Helper function to convert interactions to simple format
    def interactions_to_simple_format(interactions):
        simple_format = []
        for interaction in interactions:
            if isinstance(interaction, (list, tuple)) and len(interaction) >= 2:
                antigen_res = interaction[0]
                if isinstance(antigen_res, (list, tuple)) and len(antigen_res) >= 2:
                    simple_format.append((antigen_res[0], str(antigen_res[1])))
        return list(set(simple_format))  # Remove duplicates

    # Convert model epitopes to simple format
    model_epitope_cdr_simple = interactions_to_simple_format(model_epitope_cdr)
    model_epitope_all_simple = interactions_to_simple_format(model_epitope_all)

    # Prepare GT epitopes
    filtered_gt_epitope_set = set(filtered_gt_epitope)  # CDR-accessible GT epitope
    original_gt_epitope_simple = interactions_to_simple_format(original_gt_epitope_)
    original_gt_epitope_set = set(original_gt_epitope_simple)  # All GT epitope

    # Calculate epitope recalls
    if len(filtered_gt_epitope_set) == 0:
        epitope_recall = 0.0
        epitope_recall_all = 0.0
        print("No CDR-filtered ground truth epitope available")
    elif len(model_epitope_cdr_simple) == 0 and len(model_epitope_all_simple) == 0:
        epitope_recall = 0.0
        epitope_recall_all = 0.0
        print("No model interactions found")
    else:
        try:
            model_epitope_cdr_set = set(model_epitope_cdr_simple)
            model_epitope_all_set = set(model_epitope_all_simple)
            
            # CDR epitope recall: CDR model vs CDR GT
            if len(model_epitope_cdr_simple) == 0:
                epitope_recall = 0.0
                overlap_cdr = set()
                print("No model CDR epitope interactions found")
            else:
                overlap_cdr = model_epitope_cdr_set.intersection(filtered_gt_epitope_set)
                epitope_recall = len(overlap_cdr) / len(filtered_gt_epitope_set)
                print(f"CDR Epitope recall: {epitope_recall:.3f} ({len(overlap_cdr)}/{len(filtered_gt_epitope_set)})")
            
            # All epitope recall: ALL model vs CDR GT (both compared to same reference)
            if len(model_epitope_all_simple) == 0:
                epitope_recall_all = 0.0
                overlap_all = set()
                print("No model interactions found")
            else:
                overlap_all = model_epitope_all_set.intersection(filtered_gt_epitope_set)
                epitope_recall_all = len(overlap_all) / len(filtered_gt_epitope_set)
                print(f"ALL Epitope recall: {epitope_recall_all:.3f} ({len(overlap_all)}/{len(filtered_gt_epitope_set)})")
            
            # Debug information
            # print(f"GT CDR epitope size: {len(filtered_gt_epitope_set)}")
            # print(f"GT original epitope size: {len(original_gt_epitope_set)}")
            # print(f"Model CDR epitope size: {len(model_epitope_cdr_set)}")
            # print(f"Model ALL epitope size: {len(model_epitope_all_set)}")
            # print(f"CDR overlapping residues: {overlap_cdr}")
            # print(f"ALL overlapping residues: {overlap_all}")
            
            # # Print the actual residues for debugging
            # print(f"Model CDR residues: {model_epitope_cdr_set}")
            # print(f"Model ALL residues: {model_epitope_all_set}")
            # print(f"GT CDR epitope residues: {filtered_gt_epitope_set}")
            
            # Only print framework_removed if it's defined
            try:
                print(f"Framework interactions removed from GT: {framework_removed}")
            except NameError:
                print("Framework interactions count not available")
                
        except Exception as e:
            print(f"Error calculating epitope recall: {e}")
            print(f"Debug info - filtered_gt_epitope: {filtered_gt_epitope}")
            epitope_recall = 0.0
            epitope_recall_all = 0.0

    # # Step 6: Extract individual CDR metrics - now you have both options
    # cdr1_involvement_all = temp_item.get("cdrh1_avg", 0.0)  # All interactions
    # cdr2_involvement_all = temp_item.get("cdrh2_avg", 0.0) 
    # cdr3_involvement_all = temp_item.get("cdrh3_avg", 0.0)
    # total_avg_cdr_involvement_all = temp_item.get("total_avg_cdrh_involvement", 0.0)

    # # GT-based metrics (recommended for evaluation)
    # cdr1_involvement_gt = temp_item.get("cdrh1_avg_gt", 0.0)  # GT epitope only
    # cdr2_involvement_gt = temp_item.get("cdrh2_avg_gt", 0.0) 
    # cdr3_involvement_gt = temp_item.get("cdrh3_avg_gt", 0.0)
    # total_avg_cdr_involvement_gt = temp_item.get("total_avg_cdrh_involvement_gt", 0.0)


    # Step 6: Extract individual CDR metrics
    # GT-based metrics (primary)
    cdr1_involvement_gt = temp_item.get("cdrh1_avg", 0.0)  # GT epitope only
    cdr2_involvement_gt = temp_item.get("cdrh2_avg", 0.0) 
    cdr3_involvement_gt = temp_item.get("cdrh3_avg", 0.0)
    total_avg_cdr_involvement_gt = temp_item.get("total_avg_cdrh_involvement", 0.0)

    # All CDR interactions (for reference)
    cdr1_involvement_all = temp_item.get("cdrh1_avg_all", 0.0)
    cdr2_involvement_all = temp_item.get("cdrh2_avg_all", 0.0) 
    cdr3_involvement_all = temp_item.get("cdrh3_avg_all", 0.0)
    total_avg_cdr_involvement_all = temp_item.get("total_avg_cdrh_involvement_all", 0.0)

    # Step 7: Clean up temporary files
    for immuno_chain in immuno_chains:
        for antigen in antigen_chains:
            tmp_dir_specific = os.path.join(tmp_dir_for_interacting_aa, f"{model_name}_{immuno_chain}_to_{antigen}")
            if os.path.exists(tmp_dir_specific):
                try:
                    clean_up_files(tmp_dir_specific, model_name)
                    if os.path.exists(tmp_dir_specific) and not os.listdir(tmp_dir_specific):
                        os.rmdir(tmp_dir_specific)
                except Exception as e:
                    print(f"Warning: Could not clean up temporary directory {tmp_dir_specific}: {e}")

    end_time = time.time()
    print(f"Process_hdock_model took: {end_time - start_time:.2f} seconds")

    # Return results with both epitope recalls
    result_dict = {
        'hdock_model': hdock_model,
        'cdrh1_avg': cdr1_involvement_gt,        # GT-based (CDR interactions with GT CDR epitope)
        'cdrh2_avg': cdr2_involvement_gt, 
        'cdrh3_avg': cdr3_involvement_gt,
        'total_avg_cdr_involvement': total_avg_cdr_involvement_gt,
        'cdrh1_avg_all': cdr1_involvement_all,      # All CDR interactions
        'cdrh2_avg_all': cdr2_involvement_all, 
        'cdrh3_avg_all': cdr3_involvement_all,
        'total_avg_cdr_involvement_all': total_avg_cdr_involvement_all,
        'epitope_recall': epitope_recall,           # CDR model vs CDR GT
        'epitope_recall_all': epitope_recall_all    # All model vs CDR GT (both use same reference)
    }

    return result_dict


def get_default_item(pdb_dict):

    return next(iter(pdb_dict.values()))



def main(args):
    # Load the test set data
    with open(args.test_set, 'r') as fin:
        data = fin.read().strip().split('\n')

    print("data", data)

    # Create dictionary of {pdb:whole dictionary corresponding to such pdb}
    pdb_dict = {}
    for item in data:
        json_obj = json.loads(item)
        pdb_dict[json_obj["pdb"]] = json_obj

    print("pdb_dict", pdb_dict)

    # Get list of directories to process
    items_in_hdock_models = os.listdir(args.hdock_models)
    dir_l = [item for item in items_in_hdock_models if os.path.isdir(os.path.join(args.hdock_models, item))]

    # Filter out tmp directories
    new_dir = []
    for dir_ in dir_l:
        if "tmp_dir" in dir_:
            continue
        new_dir.append(dir_)
    
    dir_l = new_dir[:]

    # Find directories with required number of PDB files
    dir_l_ = []  # Directories meeting criteria
    new_l = []   # Full paths to those directories
    
    for directory in new_dir:
        full_dir_path = os.path.join(args.hdock_models, directory)
        hdock_models = [file for file in os.listdir(full_dir_path) if file.startswith('model_') and file.endswith('.pdb')]
        n = int(args.top_n) + 10
        if len(hdock_models) != n:
            dir_l_.append(directory)
            new_l.append(full_dir_path)
    
    print(f"Found {len(dir_l_)} directories that need processing")
    
    processed_directories = 0
    all_top_models = {}  # Store results for each directory
    
    try:
        # Process each directory - now process ALL directories in one run
        for index, (pdb_n, full_dir_path) in enumerate(zip(dir_l_, new_l)):            
            print(f"Processing directory {index + 1}/{len(dir_l_)}: {pdb_n}")
            
            # Set up directory-specific tracking
            hdock_models = [file for file in os.listdir(full_dir_path) if file.startswith('model_') and file.endswith('.pdb')]
            
            try:
                # Prepare inputs
                print("pdb_n", pdb_n)
                pdb_parts = pdb_n.rsplit('_')[0]
                ag_pdb = os.path.join(full_dir_path, pdb_parts + "_ag.pdb")

                if not os.path.exists(ag_pdb):
                    pdb_parts = pdb_n.rsplit('_', 1)[0]
                    ag_pdb = os.path.join(full_dir_path, pdb_parts + "_ag.pdb")

                ab_pdb = os.path.join(full_dir_path, pdb_parts + "_only_nb.pdb")

                if not os.path.exists(ab_pdb):
                    ab_pdb = os.path.join(full_dir_path, pdb_parts + "_IgFold.pdb")

                    if not os.path.exists(ab_pdb):
                        pdb_parts = pdb_n.rsplit('_', 1)[0]
                        ab_pdb = os.path.join(full_dir_path, pdb_parts + "_IgFold.pdb")
                
                print("pdb_parts", pdb_parts)

                # Get metadata for this structure
                original_item = pdb_dict.get(pdb_parts)

                if original_item is None:
                    print(f"{pdb_n} doesn't have a json file in {args.test_set}")
                    original_item = get_default_item(pdb_dict)
                    print("Using a default item from pdb_dict")

                item_copy = copy.deepcopy(original_item)
                heavy_chain = item_copy["heavy_chain"]
                light_chain = item_copy["light_chain"]



                # Get hdock model paths
                hdock_models = [os.path.join(full_dir_path, file) for file in os.listdir(full_dir_path) 
                                if file.startswith('model_') and file.endswith('.pdb')]

                # Set up tracking files
                track_file = os.path.join(full_dir_path, f"{pdb_n}_ep_and_dock_information.json")
                track_file_2 = os.path.join(full_dir_path, f"{pdb_n}_binding_information.json")
                parent_dir = os.path.dirname(track_file)

                # Skip if already processed
                # top_file = os.path.join(parent_dir, "top_models.json")
                # if os.path.exists(top_file) and os.stat(top_file).st_size == args.top_n:
                #     print(f"Top models already selected for {pdb_n}, skipping...")
                #     processed_directories += 1
                #     continue  # Continue to next directory

                top_file = os.path.join(parent_dir, "top_models.json")
                if os.path.exists(top_file):
                    print("top_file",top_file)
                    try:
                        # Count the number of unique models in the file
                        with open(top_file, 'r') as f:
                            existing_models = []
                            for line in f:
                                if line.strip():  # Skip empty lines
                                    model_data = json.loads(line)
                                    existing_models.append(model_data.get("hdock_model"))
                        
                        # Check if we have exactly the required number of unique models
                        unique_models = set(filter(None, existing_models))  # Remove None values and duplicates
                        
                        if len(unique_models) == int(args.top_n):
                            print(f"Top models already selected for {pdb_n} ({len(unique_models)} models), skipping...")
                            processed_directories += 1
                            continue  # Continue to next directory
                        else:
                            print(f"Found {len(unique_models)} models in top_models.json, but need {args.top_n}. Reprocessing...")
                            
                    except (json.JSONDecodeError, KeyError, IOError) as e:
                        print(f"Error reading existing top_models.json for {pdb_n}: {e}. Reprocessing...")

                # Check if track_file exists with all zeros
                if os.path.exists(track_file):
                    with open(track_file, 'r') as f:
                        track_data = [json.loads(line) for line in f]
                        if len(track_data) >= len(hdock_models):
                            all_cdr3_avg_zero = all(float(entry.get("cdrh3_avg", 0.0)) == 0.0 for entry in track_data)
                            if all_cdr3_avg_zero:
                                print(f"All cdr3_avg values are 0.0 for {pdb_n}, skipping...")
                                continue  # Continue to next directory

                print("Processing", pdb_n)
                start_time = time.time()

                # Set up binding information
                binding_data = []
                if not os.path.exists(track_file_2):
                    with open(track_file_2, 'w') as f:
                        pass
                else:
                    with open(track_file_2, 'r') as f:
                        binding_data = [json.loads(line) for line in f]

 
                # Process epitope data with proper format detection
                filtered_gt_epitope, original_gt_epitope_, framework_removed = process_epitope_data(original_item, ab_pdb)

                if not filtered_gt_epitope:
                    print("Warning: No epitope data available after processing")
                    print(f"Original item keys: {list(original_item.keys())}")
                    if "epitope" in original_item:
                        print(f"Original epitope length: {len(original_item['epitope'])}")
                        print(f"Sample epitope data: {original_item['epitope'][:3] if original_item['epitope'] else 'Empty'}")

                print(f"Final filtered_gt_epitope: {len(filtered_gt_epitope)} residues")
                if filtered_gt_epitope:
                    print(f"Sample epitope residues: {filtered_gt_epitope[:5]}")

                binding_rsite = filtered_gt_epitope


                # Ensure filtered_gt_epitope is always defined before proceeding
                print(f"Final filtered_gt_epitope: {len(filtered_gt_epitope)} residues")

                binding_rsite = filtered_gt_epitope

                # Process antibody structure once
                try:
                    pdb = Protein.from_pdb(ab_pdb, heavy_chain)
                    item_copy["heavy_chain_seq"] = ""

                    for peptide_id, peptide in pdb.peptides.items():
                        sequence = peptide.get_seq()
                        item_copy['heavy_chain_seq'] += sequence
                    
                    # Extract dictionary of cdrHs and cdrL positions
                    cdr_pos_dict = extract_antibody_info(pdb, heavy_chain, light_chain, "imgt")

                    # Create Peptide objects per chain in Protein object
                    peptides_dict = pdb.get_peptides()

                    # Get the peptide of interest (for now only heavy chain)
                    nano_peptide = peptides_dict.get(heavy_chain)

                    # Get the sequence of cdrs given the positions
                    for i in range(1, 4):
                        cdr_name = f'H{i}'.lower()
                        cdr_pos = get_cdr_pos(cdr_pos_dict, cdr_name)
                        item_copy[f'cdr{cdr_name}_pos_mod'] = cdr_pos
                        start, end = cdr_pos 
                        end += 1
                        cdr_seq = nano_peptide.get_span(start, end).get_seq()
                        item_copy[f'cdr{cdr_name}_seq_mod'] = cdr_seq

                except Exception as e:
                    # Handle exceptions if needed
                    print(f'Something went wrong for {ab_pdb}, {e}')
                    # if something went wrong it means the IgFold and therefore all of the docked models had an issue, delete folder.
                    print(traceback.format_exc())
                    try:
                        import shutil
                        shutil.rmtree(full_dir_path)
                        print(f"Deleted corrupted directory: {full_dir_path}")
                        continue  # Continue to next directory instead of breaking
                    except Exception as e:
                        print(e)
                        continue  # Continue to next directory instead of breaking

                # Set up temporary directory
                tmp_dir_for_interacting_aa = os.path.join(args.hdock_models, f"tmp_dir_binding_computations_{pdb_n}")
                os.makedirs(tmp_dir_for_interacting_aa, exist_ok=True)
                # shared_args = ag_pdb, ab_pdb, binding_rsite, item_copy, full_dir_path, tmp_dir_for_interacting_aa
                shared_args = ag_pdb, ab_pdb, binding_rsite, item_copy, full_dir_path, tmp_dir_for_interacting_aa, original_gt_epitope_, framework_removed

                # Process models that haven't been processed yet
                if len(binding_data) != len(hdock_models):
                    complete = False
                    models_processed = len(binding_data)
                    
                    while not complete:
                        with open(track_file_2, 'r') as f:
                            data = [json.loads(line) for line in f]

                        hdock_models_already_processed = [item["hdock_model"] for item in data]
                        new_list = [model for model in hdock_models if model not in hdock_models_already_processed]

                        if not new_list:
                            print("All models have been processed.")
                            complete = True
                            continue

                        try:
                            # Process models in parallel
                            process_with_args = partial(binding_residues_analysis, shared_args)
                            skipped_tasks_count = 0
                            
                            with concurrent.futures.ProcessPoolExecutor() as executor:
                                futures = [executor.submit(process_with_args, k) for k in new_list]
                                
                                results = []
                                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                                    try:
                                        result = future.result(timeout=1000)
                                        if result is not None:
                                            results.append(result)
                                    except concurrent.futures.TimeoutError:
                                        print("A task exceeded the time limit and was skipped.")
                                        skipped_tasks_count += 1

                            # Write results to file
                            for result in results:
                                with open(track_file_2, 'a') as f:
                                    f.write(json.dumps(result) + '\n')

                        except Exception as e:
                            print(f"Error in parallel processing: {e}")
                            print(traceback.format_exc())
                        
                        # Update directory progress
                        models_processed += len(results) if 'results' in locals() else 0

                        # Check if complete
                        with open(track_file_2, 'r') as f:
                            binding_data = [json.loads(line) for line in f]

                        if len(binding_data) < len(hdock_models):
                            print("Still missing entries, continuing to process remaining models.")
                        else:
                            print("All models have been processed.")
                            complete = True

                # Read final binding data
                with open(track_file_2, 'r') as f:
                    binding_data = [json.loads(line) for line in f]

                # Process results if we have enough data
                if len(binding_data) >= len(hdock_models):
                    try:
                        # Filter and sort models
                        # filtered_list = [item for item in binding_data if float(item["cdrh3_avg"]) != 0.0]
                        # filtered_list.sort(key=lambda x: x['epitope_recall'], reverse=True)
                        # n_top = int(args.top_n)

                        # Remove duplicates first - keep the best scoring entry for each unique hdock_model
                        unique_models = {}
                        for item in binding_data:
                            hdock_path = item["hdock_model"]
                            if hdock_path not in unique_models:
                                unique_models[hdock_path] = item
                            else:
                                # Keep the one with higher epitope_recall, or higher cdrh3_avg if epitope_recall is the same
                                current = unique_models[hdock_path]
                                if (item['epitope_recall'] > current['epitope_recall'] or 
                                    (item['epitope_recall'] == current['epitope_recall'] and 
                                    float(item["cdrh3_avg"]) > float(current["cdrh3_avg"]))):
                                    unique_models[hdock_path] = item

                        # Convert back to list
                        binding_data_unique = list(unique_models.values())
                        print(f"Removed duplicates: {len(binding_data)} -> {len(binding_data_unique)} unique models")

                        # Filter and sort models
                        filtered_list = [item for item in binding_data_unique if float(item["cdrh3_avg"]) != 0.0]
                        filtered_list.sort(key=lambda x: x['epitope_recall'], reverse=True)
                        n_top = int(args.top_n)

                        # Modified logic: if we don't have enough non-zero models, include zero models
                        if len(filtered_list) < n_top:
                            print(f"Only {len(filtered_list)} models with non-zero cdr3_avg, including zero models to reach {n_top}")
                            
                            # Get models with zero cdr3_avg
                            zero_models = [item for item in binding_data if float(item["cdrh3_avg"]) == 0.0]
                            zero_models.sort(key=lambda x: x['epitope_recall'], reverse=True)
                            
                            # Add zero models to reach the target number
                            needed_models = n_top - len(filtered_list)
                            filtered_list.extend(zero_models[:needed_models])
                            
                            # Re-sort the combined list
                            filtered_list.sort(key=lambda x: (float(x["cdrh3_avg"]) != 0.0, x['epitope_recall']), reverse=True)

                        top_models = filtered_list[:n_top]

                        # Save top models
                        parent_dir = os.path.dirname(track_file_2)
                        top_file = os.path.join(parent_dir, "top_models.json")
                        
                        with open(top_file, 'w') as f:  # Use 'w' to overwrite
                            for item in top_models:
                                f.write(json.dumps(item) + '\n')

                        # Extract top model paths
                        top_model_paths = [item["hdock_model"] for item in top_models]

                        # Renumber PDB files
                        for top_model in top_model_paths:
                            renumber_pdb(top_model, top_model, "imgt")

                        # Delete non-top models if not first iteration
                        if int(args.iteration) > 1:
                            for hdock_model in hdock_models:
                                if hdock_model not in top_model_paths:
                                    try:
                                        os.remove(hdock_model)
                                    except FileNotFoundError:
                                        pass  # File already removed
                        
                        processed_directories += 1
                        print(f"Successfully processed {pdb_n}")
                        
                        # Store the top models for this directory
                        all_top_models[pdb_n] = top_models

                    except Exception as e:
                        print(f"Error processing results for {pdb_n}: {e}")
                        print(traceback.format_exc())
                        continue  # Continue to next directory

            except Exception as e:
                print(f"Error processing directory {pdb_n}: {e}")
                print(traceback.format_exc())
                continue  # Continue to next directory

        # Clean up
        gc.collect()
        
        print(f"Successfully processed {processed_directories} out of {len(dir_l_)} directories in this run")
        return all_top_models

    except Exception as e:
        print(f"Fatal error in main function: {e}")
        print(traceback.format_exc())
        return None


def parse():
    parser = argparse.ArgumentParser(description='inference of pipeline method')
    parser.add_argument('--test_set', type=str, required=True, help='Path to the summary file of dataset in json format')
    parser.add_argument('--hdock_models', type=str, required=True, help='Path to save generated PDBs from hdock')
    parser.add_argument('--top_n', type=str, required=True, help='Top n docked models to select ')
    parser.add_argument('--iteration', type=str, required=True, help='Top n docked models to select ')

    return parser.parse_args()

if __name__ == '__main__':
    main(parse())

