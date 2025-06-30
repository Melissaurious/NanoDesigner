#!/usr/bin/python
# -*- coding:utf-8 -*-
import path_setup 
import os
import sys
import json
import argparse
import logging
import traceback
import warnings
import string
import gc
import time
import copy
import subprocess
from functools import partial
import concurrent.futures
from concurrent.futures import (
    ThreadPoolExecutor, 
    ProcessPoolExecutor, 
    as_completed, 
    TimeoutError
)
import uuid
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import psutil
import os
import time
from typing import Tuple

import numpy as np
import torch
# import peptides
from scipy.spatial import cKDTree
from Bio import PDB, Align
from Bio.PDB import PDBParser, PDBList
from Bio.PDB.PDBExceptions import PDBConstructionWarning, PDBConstructionException
from Bio.Align import substitution_matrices

# Add custom paths


# Import from custom modules
from dyMEAN.data.pdb_utils import VOCAB, AgAbComplex2, Protein, Peptide as Peptide_Class
from dyMEAN.configs import CACHE_DIR, CONTACT_DIST
from dyMEAN.utils.logger import print_log
from dyMEAN.utils.renumber import renumber_pdb
from dyMEAN.evaluation.rmsd import compute_rmsd
from dyMEAN.evaluation.tm_score import tm_score
from dyMEAN.evaluation.lddt import lddt
from dyMEAN.evaluation.dockq import dockq, dockq_nano
from dyMEAN.define_aa_contacts_antibody_nanobody_2025 import get_cdr_residues_dict, interacting_residues, get_cdr_residues_and_interactions, clean_up_files, get_cdr_residues_and_interactions_gt_based

# Suppress warnings
warnings.filterwarnings('ignore', category=PDBConstructionWarning)

# Initialize parsers and downloaders
parser = PDBParser(QUIET=True)
downloader = PDBList()


import psutil
import os
import time
from typing import Tuple
import gc




def safe_load_json_lines(filepath, verbose=True):
    """
    General function to safely load JSON objects from a file (one JSON object per line).
    
    Args:
        filepath (str): Path to the JSON file
        verbose (bool): Whether to print warnings about malformed entries
    
    Returns:
        list: List of successfully parsed JSON objects (dicts, lists, strings, etc.)
              Returns empty list if file doesn't exist or can't be read
    
    Example:
        data = safe_load_json_lines('my_dataset.json')
        # Returns: [{"key": "value"}, {"another": "object"}, ...]
    """
    data = []
    malformed_count = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError as e:
                    malformed_count += 1
                    if verbose:
                        print(f"Warning: Skipping malformed JSON in {filepath}, line {line_num}: {str(e)[:50]}...")
                    continue
                    
    except IOError as e:
        if verbose:
            print(f"Error reading file {filepath}: {e}")
        return []
    
    if verbose and malformed_count > 0:
        print(f"Loaded {len(data)} valid entries from {filepath}, skipped {malformed_count} malformed lines")
    elif verbose:
        print(f"Loaded {len(data)} valid entries from {filepath}")
    
    return data


def get_optimal_workers() -> Tuple[int, str]:
    """
    Dynamically determine optimal number of workers based on system resources.
    Returns: (num_workers, reasoning)
    """
    # Get system specs
    cpu_count = os.cpu_count() or 1
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    memory_available_gb = memory.available / (1024**3)
    
    print(f"System specs: {cpu_count} CPUs, {memory_gb:.1f}GB totxal RAM, {memory_available_gb:.1f}GB available")
    
    # Conservative estimates based on your workload
    # Each worker needs ~2-4GB RAM for protein structure analysis
    memory_per_worker_gb = 3.0  # Conservative estimate
    cpu_per_worker = 1  # Your tasks are I/O intensive (file reading, external commands)
    
    # Calculate limits
    workers_by_memory = max(1, int(memory_available_gb / memory_per_worker_gb))
    workers_by_cpu = max(1, cpu_count - 1)  # Leave 1 CPU for system
    
    # Take the minimum to avoid resource exhaustion
    optimal_workers = min(workers_by_memory, workers_by_cpu, 8)  # Cap at 8 for stability
    
    # Apply additional constraints for cluster environments
    if memory_gb > 64:  # High-memory node
        reasoning = f"High-memory node: using {optimal_workers} workers (limited by CPU)"
    elif memory_gb < 16:  # Low-memory node
        optimal_workers = min(optimal_workers, 2)
        reasoning = f"Low-memory node: using {optimal_workers} workers (limited by RAM)"
    else:  # Standard node
        reasoning = f"Standard node: using {optimal_workers} workers (balanced)"
    
    return optimal_workers, reasoning


def clear_memory():
    """Clear GPU and system memory."""
    torch.cuda.empty_cache()
    gc.collect()

def add_path_to_sys(path):
    """Add a path to sys.path if not already present."""
    if path not in sys.path:
        sys.path.append(path)
    
def get_default_item(pdb_dict):
    return next(iter(pdb_dict.values()))


def get_unused_letter(used_letters):
    all_letters = set(string.ascii_uppercase)
    used_letters_set = set(used_letters)
    unused_letters = all_letters - used_letters_set
    
    if unused_letters:
        return unused_letters.pop()
    else:
        raise ValueError("All letters are used")


# TO DO! FOLDX COMPUTATIONS DOES NOT ACCEPT MORE THAN A PAIR OF CHAINS.
def merge_ag_chains_for_foldX(input_pdb, output_pdb, heavy_chain, antigen_chains):
    chains_to_merge = antigen_chains
    chains_ids_to_not_use = [heavy_chain] + antigen_chains
    new_chain_id_antigens = get_unused_letter(chains_ids_to_not_use)

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', input_pdb)

    new_structure = PDB.Structure.Structure('new_protein')
    model = PDB.Model.Model(0)
    new_structure.add(model)

    merged_chain = PDB.Chain.Chain(new_chain_id_antigens)
    residue_id_counter = 1

    for chain in structure[0]:
        if chain.id in chains_to_merge:
            for residue in chain:
                new_residue = PDB.Residue.Residue((' ', residue_id_counter, ' '), residue.resname, residue.segid)
                for atom in residue:
                    new_residue.add(atom)
                merged_chain.add(new_residue)
                residue_id_counter += 1
        else:
            model.add(chain)

    model.add(merged_chain)

    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb)

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

three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def extract_seq_info_from_pdb(pdb_file, target_chain,sequence):
    """
    This function is to extract the [(chain,position,residue) from a PDB given a specified chain
    """
    structure = PDB.PDBParser().get_structure("protein", pdb_file)
    pdb_info = []
    pdb_sequence = ""
    for model in structure:
        for chain in model:
            if chain.id == target_chain:
                for residue in chain:
                    # residue <Residue VAL het=  resseq=124 icode= >
                    if PDB.is_aa(residue):
                        chain_id = chain.id
                        residue_pos_tup = residue.id
                        #residue_pos_tup: 
                        # (' ', 111, ' ')
                        # (' ', 111, 'A')
                        # (' ', 111, 'B')
                        res_id = residue_pos_tup[1]
                        res_name = three_to_one.get(residue.get_resname())
                        pdb_sequence += res_name
                        if not residue_pos_tup[2].isalpha():
                            pdb_info.append((chain_id, res_id, res_name))

    start_index = pdb_sequence.find(sequence)
    end_index = start_index + len(sequence) - 1

    if start_index == -1:   # -1 if the sequence was not found
        return None

    seq_info =  pdb_info[start_index:end_index + 1]

    return seq_info

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


def get_mol_extcoef(input_seq):
    y_count = input_seq.count("Y")
    w_count = input_seq.count("W")
    c_count = input_seq.count("C")
    
    if c_count % 2 == 0:
        return y_count * 1490 + w_count * 5500 + (c_count / 2) * 125
    else:
        return y_count * 1490 + w_count * 5500 + ((c_count - 1) / 2) * 125

def get_mol_extcoef_cystine_bridges(input_seq):
    c_count = input_seq.count("C")
    
    if c_count % 2 == 0:
        return (c_count / 2) * 125
    else:
        return ((c_count - 1) / 2) * 125


#modification numbering: str
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



def clean_extended(origin_antigen_pdb, origin_antibody_pdb, template_pdb, out_pdb,
                   chain_mapping=None, remove_hydrogens=True, preserve_insertion_codes=True):
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


def detect_processing_mode(args, summary_data):
    """
    Detect processing mode based on args and data structure
    Returns: 'test_set' or 'pipeline'
    """
    if hasattr(args, 'iteration') and args.iteration:
        return 'pipeline'
    
    # Check data structure - test_set has pdb_data_path, pipeline has nano_source/antigen_source
    sample_entry = summary_data[0] if summary_data else {}
    if 'pdb_data_path' in sample_entry:
        return 'test_set'
    elif 'nano_source' in sample_entry or 'antigen_source' in sample_entry:
        return 'pipeline'
    else:
        # Default fallback - could also raise an error
        return 'test_set'


def evaluate_item_test_set(args, pdb_dict, item):
    """Evaluation logic for test set mode"""
    
    # SIMPLE: Construct entry_id from chain information
    H, L, A = item['heavy_chain'], item['light_chain'], item['antigen_chains']
    
    # Extract base PDB name from whatever pdb field we have
    pdb = item.get("entry_id", item.get("pdb"))
    base_pdb = pdb.split('_')[0] if '_' in pdb else pdb
    
    # Construct the entry_id that should exist in pdb_dict
    antigen_str = ''.join(A)  # ['D'] -> 'D'
    entry_id = f"{base_pdb}_{H}__{antigen_str}"
    
    # Look it up
    original_item = pdb_dict.get(entry_id)
    if original_item:
        print(f"Found match: {pdb} -> {entry_id}")
    else:
        print(f"No match for constructed entry_id: {entry_id}")
        original_item = get_default_item(pdb_dict)

    H, L, A = item['heavy_chain'], item['light_chain'], item['antigen_chains']
    mod_pdb = item["mod_pdb"]
    ref_pdb = item["ref_pdb"]


    # FIX: Use pdb_dict to get the correct reference PDB
    if original_item:
        # Try to get the reference PDB from the original item in pdb_dict
        correct_ref_pdb = original_item.get("pdb_data_path")
        
        if correct_ref_pdb and os.path.exists(correct_ref_pdb):
            ref_pdb = correct_ref_pdb
            item["ref_pdb"] = ref_pdb  # Update the item so it continues with correct ref_pdb
            print(f"Fixed reference PDB for {pdb} using pdb_dict: {ref_pdb}")
        # else:
        #     print(f"Warning: Reference PDB from pdb_dict does not exist or is None: {correct_ref_pdb}")
        #     # Fallback: try to construct the reference path
        #     fallback_ref_pdb = f"/ibex/user/rioszemm/Final_dataset_may_2025_paper/Nanobody_imgt/{pdb}.pdb"
        #     if os.path.exists(fallback_ref_pdb):
        #         ref_pdb = fallback_ref_pdb
        #         item["ref_pdb"] = ref_pdb
        #         print(f"Using constructed fallback reference PDB: {ref_pdb}")
        #     else:
        #         print(f"Continuing with original ref_pdb: {ref_pdb}")
    else:
        print(f"Warning: No original item found in pdb_dict for {pdb}")
        print(f"Continuing with original ref_pdb: {ref_pdb}")

    item["numbering"] = "imgt"
    item["gt_epitope"] = original_item["epitope"] if original_item and "epitope" in original_item else None

    if original_item and "epitope_user_input" in original_item:
        item["epitope_user_input"] = original_item["epitope_user_input"]

    if not os.path.exists(mod_pdb):
        print_log(f'{mod_pdb} not exists!', level='ERROR')
        print(f'{mod_pdb} not exists!')
        return item

    # Chain length alignment logic
    chains_list = [H] + A
    if L != "":
        chains_list = [H] + A + [L]

    try:
        mod_prot = Protein.from_pdb(mod_pdb, chains_list)
    except Exception as e:
        print(f'parse {mod_pdb} failed for {e}')
        traceback.print_exc()
        return item   # if Protein object could not be created, most likely there is something wrong with the PDB

    try:
        ref_prot = Protein.from_pdb(ref_pdb, chains_list)
    except Exception as e:
        print(f'parse {ref_pdb} failed for {e}')
        traceback.print_exc()
        return item

    # Align chains if necessary - complete alignment logic
    mod_revised, ref_revised = False, False
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
        traceback.print_exc()

    if mod_revised:
        print("Entered the mod_revised mode")
        mod_prot.to_pdb(mod_pdb)

    # Create temp directory
    parent_directory = os.path.dirname(args.summary_json)
    tmp_dir_for_interacting_aa = os.path.join(parent_directory, f"tmp_dir_binding_computations_{pdb}")
    os.makedirs(tmp_dir_for_interacting_aa, exist_ok=True)
    item["tmp_dir_for_interacting_aa"] = tmp_dir_for_interacting_aa

    return item



def evaluate_item_pipeline(args, pdb_dict, pdb_n, item):
    """Evaluation logic for pipeline mode - ALIGNED WITH TEST SET MODE"""


    mod_pdb = item["mod_pdb"]
    pdb = item.get('pdb') or item.get('entry_id') or pdb_n 
    
    # ALIGN WITH TEST SET MODE: Use same entry_id construction logic
    H, L, A = item['heavy_chain'], item['light_chain'], item['antigen_chains']
    
    # Extract base PDB name from whatever pdb field we have
    base_pdb = pdb.split('_')[0] if '_' in pdb else pdb
    
    # Construct the entry_id that should exist in pdb_dict (SAME AS TEST SET)
    antigen_str = ''.join(A)  # ['A'] -> 'A'
    entry_id = f"{base_pdb}_{H}__{antigen_str}"
    
    # Look it up (SAME AS TEST SET)
    original_item = pdb_dict.get(entry_id)
    if original_item:
        print(f"Found match: {pdb} -> {entry_id}")
    else:
        print(f"No match for constructed entry_id: {entry_id}")
        # Fallback to old logic
        pdb_parts = pdb.rsplit('_')[0]
        original_item = pdb_dict.get(pdb_parts)
        if original_item is None:
            original_item = get_default_item(pdb_dict)

 
    # USE SAME REF_PDB LOGIC AS TEST SET MODE
    ref_pdb = item.get("ref_pdb")
    
    if original_item:
        # Try to get the reference PDB from the original item in pdb_dict (SAME AS TEST SET)
        correct_ref_pdb = original_item.get("pdb_data_path")
        
        if correct_ref_pdb and os.path.exists(correct_ref_pdb):
            ref_pdb = correct_ref_pdb
            item["ref_pdb"] = ref_pdb  # Update the item so it continues with correct ref_pdb
            print(f"Using reference PDB from pdb_dict: {ref_pdb}")
        else:
            print(f"Warning: Reference PDB from pdb_dict does not exist or is None: {correct_ref_pdb}")
            # Fallback: try to construct the reference path (SAME AS TEST SET)
            fallback_ref_pdb = f"/ibex/user/rioszemm/Final_dataset_may_2025_paper/Nanobody_imgt/{base_pdb}.pdb"
            if os.path.exists(fallback_ref_pdb):
                ref_pdb = fallback_ref_pdb
                item["ref_pdb"] = ref_pdb
                print(f"Using constructed fallback reference PDB: {ref_pdb}")
    
    if not ref_pdb or not os.path.exists(ref_pdb):
        print(f"ERROR: No valid reference PDB found for {pdb}!")
        return item

    item["model"] = args.cdr_model

    # Clash computation with detailed analysis
    if not item.get("final_num_clashes"):
        try:
            pdb_id = item.get("entry_id", item.get("pdb"))
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure(pdb_id, mod_pdb)
            total_clashes = count_clashes(structure, clash_cutoff=0.60)[0]

            chain_clashes = {}
            for chain in structure.get_chains():
                chain_id = chain.get_id()
                chain_clashes[chain_id] = count_clashes(chain, clash_cutoff=0.60)[0]

            inter_chain_clashes = total_clashes - sum(chain_clashes.values())

            item["final_num_clashes"] = total_clashes
            item["final_clashes_per_chain"] = chain_clashes
            item["final_inter_chain_clashes"] = inter_chain_clashes

        except Exception as e:
            print(f"Error processing model computing number of clashes: {str(e)}")
            item["final_num_clashes"] = np.nan
            item["final_clashes_per_chain"] = np.nan
            item["final_inter_chain_clashes"] = np.nan

    item["numbering"] = "imgt"
    item["iteration"] = args.iteration
    item["entry_id"] = original_item.get("entry_id") if original_item else None
    item["gt_epitope"] = original_item["epitope"] if original_item and "epitope" in original_item else None

    if original_item and "epitope_user_input" in original_item:
        item["epitope_user_input"] = original_item["epitope_user_input"]

    if not os.path.exists(mod_pdb):
        print_log(f'{mod_pdb} not exists!', level='ERROR')
        print(f'{mod_pdb} not exists!')
        return item

    # SAME CHAIN ALIGNMENT LOGIC AS TEST SET
    chains_list = [H] + A
    if L != "":
        chains_list = [H] + A + [L]

    try:
        mod_prot = Protein.from_pdb(mod_pdb, chains_list)
    except Exception as e:
        print(f'parse {mod_pdb} failed for {e}')
        traceback.print_exc()
        return item

    try:
        ref_prot = Protein.from_pdb(ref_pdb, chains_list)
    except Exception as e:
        print(f'parse {ref_pdb} failed for {e}')
        traceback.print_exc()
        return item

    # EXACT SAME ALIGNMENT LOGIC AS TEST SET MODE
    mod_revised, ref_revised = False, False
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
                #         # Handle the case where the residue position does not match
                #         print(f"Warning: Residue {res_id} in mod_chain not found in ref_chain. Skipping.")

                # ref_residues = [pos_map[res_id] for res_id in ['-'.join(str(a) for a in res.get_id()) for res in mod_residues] if res_id in pos_map]

                # # Update peptide chains within the Protein objects 
                # mod_prot.peptides[chain_name] = Peptide_Class(chain_name, mod_residues)
                # ref_prot.peptides[chain_name] = Peptide_Class(chain_name, ref_residues)

                # mod_revised, ref_revised = True, False  # Only mod_prot is revised
                # print(f"{mod_prot.peptides[chain_name]} chain {chain_name} length after aligned: {len(mod_prot.peptides[chain_name].residues)} == {len(ref_prot.peptides[chain_name].residues)}")

    except Exception as e:
        print(f"An exception was raised during alignment: {e}")
        traceback.print_exc()

    if mod_revised:
        print("Entered the mod_revised mode")
        # mod_prot.to_pdb(mod_pdb)

    # Create unique temporary directory per model (ONLY DIFFERENCE FROM TEST SET)
    parent_directory = os.path.dirname(args.summary_json)
    
    # Extract model information for unique tmp dir
    mod_pdb_parts = mod_pdb.split(os.sep)
    model_dir = None
    
    for part in mod_pdb_parts:
        if part.startswith("model_"):
            model_dir = part
            break
    
    if model_dir is None:
        import hashlib
        unique_id = hashlib.md5(mod_pdb.encode()).hexdigest()[:8]
        model_dir = f"model_{unique_id}"
    
    tmp_dir_for_interacting_aa = os.path.join(parent_directory, f"tmp_dir_binding_computations_{base_pdb}_{model_dir}")
    os.makedirs(tmp_dir_for_interacting_aa, exist_ok=True)
    item["tmp_dir_for_interacting_aa"] = tmp_dir_for_interacting_aa

    return item


def fold_x_computations(summary):
    """FoldX computations - unified for both modes"""
    import numpy as np
    from evaluation.pred_ddg import foldx_dg, foldx_ddg

    H, L, A = summary["heavy_chain"], summary["light_chain"], summary["antigen_chains"]
    mod_pdb = summary["mod_pdb"]
    ref_pdb = summary["ref_pdb"]
    entry_id = summary.get("entry_id", summary.get("pdb"))
    tmp_dir = summary["tmp_dir_for_interacting_aa"]


    # TO DO! FOLDX COMPUTATIONS DOES NOT ACCEPT MORE THAN A PAIR OF CHAINS.
    if len(A) != 1: 
        unique_code = str(uuid.uuid4().hex)
        filename = os.path.splitext(os.path.basename(mod_pdb))[0]
        try:
            tmp_mod_pdb = os.path.join(tmp_dir, f"{filename}_{unique_code}_mod.pdb")
            merge_ag_chains_for_foldX(mod_pdb, tmp_mod_pdb, H, A)
            tmp_ref_pdb = os.path.join(tmp_dir, f"{filename}_{unique_code}_ref.pdb")
            merge_ag_chains_for_foldX(ref_pdb, tmp_ref_pdb, H, A)

            try:
                dG_affinity_mod = foldx_dg(tmp_mod_pdb, summary["heavy_chain"], summary["antigen_chains"])
                summary["FoldX_dG"] = dG_affinity_mod
            except Exception as e:
                print(f"Error computing dG_affinity for {mod_pdb}: {e}")
                summary["FoldX_dG"] = np.nan

            try:
                fold_x_ddg = foldx_ddg(tmp_ref_pdb, summary["mod_pdb"], summary["heavy_chain"], summary["antigen_chains"])
                summary["FoldX_ddG"] = fold_x_ddg
            except Exception as e:
                print(f"Error computing ddG: {e}")
                summary["FoldX_ddG"] = np.nan

            # Cleanup temp files
            try:
                os.remove(tmp_mod_pdb)
                os.remove(tmp_ref_pdb)
            except:
                pass

        except Exception as e:
            print(f"Error creating temp file for FoldX computations: {e}")
            summary["FoldX_ddG"] = np.nan
            traceback.print_exc()
    else:
        try:
            dG_affinity_mod = foldx_dg(mod_pdb, summary["heavy_chain"], summary["antigen_chains"])
            summary["FoldX_dG"] = dG_affinity_mod
        except Exception as e:
            print(f"Error computing dG_affinity for {mod_pdb}: {e}")
            summary["FoldX_dG"] = np.nan

        try:
            fold_x_ddg = foldx_ddg(ref_pdb, summary["mod_pdb"], summary["heavy_chain"], summary["antigen_chains"])
            summary["FoldX_ddG"] = fold_x_ddg
        except Exception as e:
            print(f"Error computing ddG: {e}")
            summary["FoldX_ddG"] = np.nan

    return summary



def retry_foldx_calculation(calc_func, *args, max_retries=3):
    """Simple retry logic for FoldX calculations"""
    import time
    
    for attempt in range(max_retries):
        try:
            result = calc_func(*args)
            if result is not None and not np.isnan(result):
                return result
        except Exception as e:
            print(f"FoldX attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait 1 second before retry
    
    return np.nan




def fold_x_computations_with_retry(summary):
    """Your original function with simple retry logic"""
    import numpy as np
    from dyMEAN.evaluation.pred_ddg import foldx_dg, foldx_ddg

    H, L, A = summary["heavy_chain"], summary["light_chain"], summary["antigen_chains"]
    mod_pdb = summary["mod_pdb"]
    ref_pdb = summary["ref_pdb"]

    # Initialize with NaN
    summary["FoldX_dG"] = np.nan
    summary["FoldX_ddG"] = np.nan

    # Since you only have single antigen chains:
    # Try dG calculation with retry
    dG_result = retry_foldx_calculation(foldx_dg, mod_pdb, H, A)
    if not np.isnan(dG_result):
        summary["FoldX_dG"] = float(dG_result)

    # Try ddG calculation with retry  
    ddG_result = retry_foldx_calculation(foldx_ddg, ref_pdb, mod_pdb, H, A)
    if not np.isnan(ddG_result):
        summary["FoldX_ddG"] = float(ddG_result)

    # TO DO, fold x do not accept complexes with more than one antigen.

    return summary





def safe_convert_to_hashable(ep_item):
    """Convert epitope item to hashable tuple, handling nested lists"""
    if isinstance(ep_item, list):
        # Convert nested lists to nested tuples recursively
        return tuple(tuple(sub_item) if isinstance(sub_item, list) else sub_item for sub_item in ep_item)
    else:
        return ep_item


def quick_length_alignment_fix(mod_cplx, ref_cplx):
    """
    Quick fix for small length discrepancies before RMSD calculation
    """
    try:
        # Check heavy chain lengths
        mod_heavy = mod_cplx.get_heavy_chain()
        ref_heavy = ref_cplx.get_heavy_chain()
        
        if mod_heavy and ref_heavy:
            mod_heavy_len = len(mod_heavy)
            ref_heavy_len = len(ref_heavy)
            
            if mod_heavy_len != ref_heavy_len:
                print(f"Heavy chain length mismatch: {mod_heavy_len} vs {ref_heavy_len}")
                
                # If difference is small (1-3 residues), truncate to shorter length
                if abs(mod_heavy_len - ref_heavy_len) <= 3:
                    min_len = min(mod_heavy_len, ref_heavy_len)
                    print(f"Truncating heavy chains to length {min_len}")
                    
                    # This is a conceptual approach - you'd need to implement the actual truncation
                    # based on your Protein/Chain classes
                    return True
        
        # Check light chain lengths if they exist
        mod_light = mod_cplx.get_light_chain()
        ref_light = ref_cplx.get_light_chain()
        
        if mod_light and ref_light:
            mod_light_len = len(mod_light)
            ref_light_len = len(ref_light)
            
            if mod_light_len != ref_light_len:
                print(f"Light chain length mismatch: {mod_light_len} vs {ref_light_len}")
                
                if abs(mod_light_len - ref_light_len) <= 3:
                    min_len = min(mod_light_len, ref_light_len)
                    print(f"Truncating light chains to length {min_len}")
                    return True
        
        return True
        
    except Exception as e:
        print(f"Quick alignment fix failed: {e}")
        return False




def robust_ca_extraction(chain_obj, max_length=None):
    ca_positions = []
    if not chain_obj:
        return ca_positions
    
    chain_length = len(chain_obj)
    if max_length:
        chain_length = min(chain_length, max_length)
    
    for i in range(chain_length):
        try:
            ca_pos = chain_obj.get_ca_pos(i)
            if ca_pos is not None:
                ca_positions.append(ca_pos)
        except (AttributeError, IndexError, KeyError):
            # Expected exceptions - skip this position
            continue
        except Exception as e:
            # Unexpected exception - might want to know about it
            print(f"Unexpected error at position {i}: {e}")
            continue
    
    return ca_positions


def normalize_cdr_type(item):
    """
    Normalize cdr_type to always be a list, handling various input formats
    """
    # Try different possible keys
    cdr_type = item.get('cdr_types') or item.get('cdr_type', ['H3'])
    
    # Handle different input types
    if cdr_type is None:
        return ['H3']  # Default fallback
    elif isinstance(cdr_type, str):
        return [cdr_type]  # Convert string to list
    elif isinstance(cdr_type, list):
        return cdr_type    # Already a list
    else:
        print(f"Warning: Unexpected cdr_type format: {type(cdr_type)}, using default ['H3']")
        return ['H3']




def has_metrics_data(entry):
    """Check if an entry has metrics/binding analysis data"""
    # Check for key indicators of complete processing
    # FoldX computation indicators
    has_foldx = entry.get("FoldX_dG") is not None or entry.get("FoldX_ddG") is not None
    
    # Metrics analysis indicators  
    has_epitope_recall = entry.get("epitope_recall") is not None
    has_rmsd = entry.get("RMSD(CA) aligned") is not None
    
    # Entry is considered complete if it has both FoldX and metrics analysis
    return has_foldx and has_epitope_recall and has_rmsd



def metrics_and_binding_analysis(item):
    """Main metrics computation - unified for both modes"""
    
    import traceback  # Add missing import
    
    try:  # Wrap the entire function in try-except
        tmp_dir_for_interacting_aa = item["tmp_dir_for_interacting_aa"]
        antigen_chains = item["antigen_chains"]
        mod_pdb = item["mod_pdb"]
        ref_pdb = item["ref_pdb"]
        heavy_chain = item["heavy_chain"]
        light_chain = item["light_chain"]

        pdb_id = item.get("entry_id", item.get("pdb"))
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, mod_pdb)
        tuple_result = count_clashes(structure, clash_cutoff=0.60)
        num_clashes = tuple_result[0]
        item["final_num_clashes"] = num_clashes

        # CDR sequence extraction
        try:
            pdb = Protein.from_pdb(mod_pdb, heavy_chain)
            item["heavy_chain_seq"] = ""

            for peptide_id, peptide in pdb.peptides.items():
                sequence = peptide.get_seq()
                item['heavy_chain_seq'] += sequence
            
            cdr_pos_dict = extract_antibody_info(pdb, heavy_chain, light_chain, "imgt")
            peptides_dict = pdb.get_peptides()
            nano_peptide = peptides_dict.get(heavy_chain)

            for i in range(1, 4):
                cdr_name = f'H{i}'.lower()
                cdr_pos = get_cdr_pos(cdr_pos_dict, cdr_name)
                item[f'cdr{cdr_name}_pos_mod'] = cdr_pos
                start, end = cdr_pos 
                end += 1
                cdr_seq = nano_peptide.get_span(start, end).get_seq()
                item[f'cdr{cdr_name}_seq_mod'] = cdr_seq

        except Exception as e:
            print(f'Something went wrong for {mod_pdb}, {e}')

        # Complex creation and metrics calculation
        H, L, A = item['heavy_chain'], item['light_chain'], item['antigen_chains']
        numbering = "imgt"

        try:
            mod_cplx = AgAbComplex2.from_pdb(mod_pdb, H, L, A, numbering=numbering, skip_epitope_cal=True)
            ref_cplx = AgAbComplex2.from_pdb(ref_pdb, H, L, A, numbering=numbering, skip_epitope_cal=False)
        except Exception as e:
            # print(f"Error creating AgAbComplex: {e}")
            traceback.print_exc()
            # return item
            renumber_pdb(ref_pdb,ref_pdb,numbering)


        try:
            mod_cplx = AgAbComplex2.from_pdb(mod_pdb, H, L, A, numbering=numbering, skip_epitope_cal=True)
            ref_cplx = AgAbComplex2.from_pdb(ref_pdb, H, L, A, numbering=numbering, skip_epitope_cal=False)
        except Exception as e:
            print(f"Error creating AgAbComplex: {e}")
            traceback.print_exc()
            return item


        # AAR & CAAR calculations
        # cdr_type = ['H3'] # TO DO, ADEQUATE FOR MULTIPLE CDRH DESIGN.
        # cdr_type = item.get('cdr_type', ['H3']) 

        cdr_type = normalize_cdr_type(item)

        valid_cdrs = ['H1', 'H2', 'H3', '-']
        cdr_type = [cdr for cdr in cdr_type if cdr in valid_cdrs]

        if not cdr_type:
            print("No valid CDR types found, defaulting to H3")
            cdr_type = ['H3']


        try:
            epitope = ref_cplx.get_epitope()
            
            # Process each CDR individually - no concatenation needed
            for cdr in cdr_type:
                try:
                    gt_cdr = ref_cplx.get_cdr(cdr)
                    pred_cdr = mod_cplx.get_cdr(cdr)
                    
                    if gt_cdr is None or pred_cdr is None:
                        print(f"CDR {cdr} not found in one or both complexes")
                        continue
                        
                    cur_gt_s = gt_cdr.get_seq()
                    cur_pred_s = pred_cdr.get_seq()
                    
                    # Calculate contact information for this CDR
                    cur_contact = []
                    for ab_residue in gt_cdr:
                        contact = False
                        for ag_residue, _, _ in epitope:
                            dist = ab_residue.dist_to(ag_residue)
                            if dist < CONTACT_DIST:
                                contact = True
                        cur_contact.append(int(contact))
                    
                    # Calculate CDR-specific metrics
                    if len(cur_gt_s) != len(cur_pred_s):
                        print(f"Length mismatch for CDR {cdr}: GT={len(cur_gt_s)}, Pred={len(cur_pred_s)}")
                        continue
                        
                    hit, chit = 0, 0
                    for a, b, contact in zip(cur_gt_s, cur_pred_s, cur_contact):
                        if a == b:
                            hit += 1
                            if contact == 1:
                                chit += 1
                    
                    # Store individual CDR metrics
                    item[f'AAR {cdr}'] = round(hit * 1.0 / len(cur_gt_s), 3)
                    item[f'CAAR {cdr}'] = round(chit * 1.0 / (sum(cur_contact) + 1e-10), 3)
                    
                    print(f"CDR {cdr}: AAR={item[f'AAR {cdr}']}, CAAR={item[f'CAAR {cdr}']}")
                    
                except Exception as e:
                    print(f"Error processing CDR {cdr}: {e}")
                    continue

            # REMOVE these lines - they're not needed:
            # item['Gt_hit'] = gt_s 
            # item['Pred_s_hit'] = pred_s

            # Extract CDR positions and sequences (this part can stay)
            for i in range(1, 4):
                cdr_name = f'H{i}'.lower()
                try:
                    cdr_pos, cdr = ref_cplx.get_cdr_pos(cdr_name), ref_cplx.get_cdr(cdr_name)
                    item[f'cdr{cdr_name}_pos_ref'] = cdr_pos
                    item[f'cdr{cdr_name}_seq_ref'] = cdr.get_seq()
                except:
                    pass

            for i in range(1, 4):
                cdr_name = f'H{i}'.lower()
                try:
                    cdr_pos, cdr = mod_cplx.get_cdr_pos(cdr_name), mod_cplx.get_cdr(cdr_name)
                    item[f'cdr{cdr_name}_pos_mod'] = cdr_pos
                    item[f'cdr{cdr_name}_seq_mod'] = cdr.get_seq()
                except:
                    pass

            item['heavy_chain_seq_ref'] = ref_cplx.get_heavy_chain().get_seq()
            item['heavy_chain_seq_mod'] = mod_cplx.get_heavy_chain().get_seq()

        except AttributeError as e:
            print(f"An attribute error occurred: {e}, error at AAR, CAAR computations, skipping entry")
            traceback.print_exc() 
            return item
        except Exception as e:
            print(f"An unexpected error occurred: {e}, error at AAR, CAAR computations, skipping entry")
            traceback.print_exc() 
            return item


        # 2. RMSD(CA) w/o align
        gt_x, pred_x = [], []
        try:
            for xl, c in zip([gt_x, pred_x], [ref_cplx, mod_cplx]):
                # handle the heavy chain
                heavy_chain_c = c.get_heavy_chain()
                if heavy_chain:
                    for i in range(len(heavy_chain_c)):
                        ca_pos = heavy_chain_c.get_ca_pos(i)
                        if ca_pos is not None:
                            xl.append(ca_pos)

                # handle the LIGHT chain (FIXED: was incorrectly get_heavy_chain())
                light_chain_c = c.get_light_chain()  # CORRECTED
                if light_chain_c:
                    for i in range(len(light_chain_c)):
                        ca_pos = light_chain_c.get_ca_pos(i)
                        if ca_pos is not None:
                            xl.append(ca_pos)

            assert len(gt_x) == len(pred_x), 'coordinates length conflict'
            gt_x, pred_x = np.array(gt_x), np.array(pred_x)
            
            # General RMSD for entire antibody
            item['RMSD(CA) aligned'] = round(compute_rmsd(gt_x, pred_x, aligned=False), 3)
            item['RMSD(CA)'] = round(compute_rmsd(gt_x, pred_x, aligned=True), 3)
            
        except AssertionError:
            print("AssertionError: coordinates length conflict for general RMSD")
            traceback.print_exc() 
            item['RMSD(CA) aligned'] = np.nan
            item['RMSD(CA)'] = np.nan
        except Exception as e:
            print(f"Error in general RMSD calculation: {e}")
            traceback.print_exc() 
            item['RMSD(CA) aligned'] = np.nan
            item['RMSD(CA)'] = np.nan

        # Part B: CDR-specific RMSD calculations - ENHANCED VERSION
        # Get CDR types from the normalized input
        cdr_type = normalize_cdr_type(item)
        print(f"Processing CDR-specific RMSD for: {cdr_type}")

        if cdr_type is not None:
            for cdr in cdr_type:
                try:
                    # Get CDR regions
                    gt_cdr, pred_cdr = ref_cplx.get_cdr(cdr), mod_cplx.get_cdr(cdr)
                    
                    if gt_cdr is None or pred_cdr is None:
                        print(f"CDR {cdr} not found in one or both complexes")
                        item[f'RMSD(CA) CDR{cdr}'] = np.nan
                        item[f'RMSD(CA) CDR{cdr} aligned'] = np.nan
                        continue
                    
                    # Handle potential length mismatches by using minimum length
                    min_len = min(len(gt_cdr), len(pred_cdr))
                    if len(gt_cdr) != len(pred_cdr):
                        print(f"CDR {cdr} length mismatch: GT={len(gt_cdr)}, Pred={len(pred_cdr)}, using min={min_len}")
                    
                    # Extract CA positions using your original method
                    gt_x_cdr = np.array([gt_cdr.get_ca_pos(i) for i in range(min_len)])
                    pred_x_cdr = np.array([pred_cdr.get_ca_pos(i) for i in range(min_len)])
                    
                    # Calculate aligned RMSD (with superposition)
                    try:
                        item[f'RMSD(CA) CDR{cdr}'] = round(compute_rmsd(gt_x_cdr, pred_x_cdr, aligned=True), 3)
                    except Exception as e:
                        print(f"Error calculating aligned RMSD for CDR {cdr}: {e}")
                        traceback.print_exc() 
                        item[f'RMSD(CA) CDR{cdr}'] = np.nan

                    # Calculate unaligned RMSD (without superposition)
                    try:
                        item[f'RMSD(CA) CDR{cdr} aligned'] = round(compute_rmsd(gt_x_cdr, pred_x_cdr, aligned=False), 3)
                    except Exception as e:
                        print(f"Error calculating unaligned RMSD for CDR {cdr}: {e}")
                        traceback.print_exc() 
                        item[f'RMSD(CA) CDR{cdr} aligned'] = np.nan
                        
                except Exception as e:
                    print(f"Error processing CDR {cdr}: {e}")
                    traceback.print_exc()
                    item[f'RMSD(CA) CDR{cdr}'] = np.nan
                    item[f'RMSD(CA) CDR{cdr} aligned'] = np.nan



        # 3. TMscore
        try:
            tm_score_ = tm_score(mod_cplx.antibody, ref_cplx.antibody)
            item['TMscore'] = tm_score_
        except Exception as e:
            print(f"An error occurred during TMscore calculation: {e}")
            traceback.print_exc() 
            item['TMscore'] = np.nan

        # 4. LDDT
        try:
            score, _ = lddt(mod_cplx.antibody, ref_cplx.antibody)
            item['LDDT'] = score
        except Exception as e:
            traceback.print_exc() 
            item['LDDT'] = np.nan

        # 5. DockQ
        if L: # TO DO, ADEQUATE FOR MULTIPLE CDRH DESIGN. CHECK!!!
            try:
                score = dockq(mod_cplx, ref_cplx, cdrh3_only=True) # consistent with HERN
            except Exception as e:
                traceback.print_exc() 
                score = 0
            item['DockQ'] = score
        else:
            try:
                score = dockq_nano(mod_cplx, ref_cplx, cdrh3_only=True)
            except Exception as e:
                traceback.print_exc() 
                score = 0
            item['DockQ'] = score


        # Step 4: NEW ENHANCED BINDING ANALYSIS using improved functions
        # print(f"Starting enhanced binding analysis for {mod_pdb}")
        
        # Use the new get_cdr_residues_dict function for robust CDR extraction
        cdr_dict = get_cdr_residues_dict(item, mod_pdb)
        # print(f"CDR dictionary extracted: {cdr_dict}")

        # Get the base name of the file for temporary file naming
        filename = os.path.basename(mod_pdb)
        model_name, extension = os.path.splitext(filename)

        # Define immunoglobulin chains (handle both nanobodies and antibodies)
        immuno_chains = [heavy_chain]
        if light_chain:  # Only add light chain if it exists (for antibodies)
            immuno_chains.append(light_chain)

        # Initialize aggregated results dictionary
        aggregated_results = {}
        
        # Process each immunoglobulin chain against each antigen
        for immuno_chain in immuno_chains:
            immuno_results_dict = {} 
            
            for antigen in antigen_chains:
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
                        protein = Protein.from_pdb(mod_pdb, chains_to_reconstruct)
                        protein.to_pdb(tmp_pdb)
                        # Renumber according to specified scheme
                        renumber_pdb(tmp_pdb, tmp_pdb, scheme=numbering)
                        
                        # Verify file creation
                        if not os.path.exists(tmp_pdb):
                            print(f"Failed to create temporary PDB: {tmp_pdb}")
                            continue
                            
                    except Exception as e:
                        print(f"Failed to process PDB file '{mod_pdb}' for interaction {immuno_chain}→{antigen}: {e}")
                        continue
                
                # Use the improved interacting_residues function
                result = interacting_residues(item, tmp_pdb, immuno_chain, antigen, tmp_specific)
                
                if result is not None:
                    print(f"Found {len(result)} interactions between {immuno_chain} and {antigen}")
                    immuno_results_dict[antigen] = result
                else:
                    print(f"No interactions found between {immuno_chain} and {antigen}")
                    immuno_results_dict[antigen] = []

            aggregated_results[immuno_chain] = immuno_results_dict

        # print(f"Aggregated results: {aggregated_results}")

        # Step 5: Filter ground truth epitope to CDR-only for consistency
        original_gt_epitope = item.get("gt_epitope", [])
        
        # Initialize filtered_gt_epitope early to avoid UnboundLocalError
        filtered_gt_epitope = []
        framework_removed = 0
        
        print(f"Original GT epitope has {len(original_gt_epitope)} interactions")
        
        if original_gt_epitope and cdr_dict:
            try:
                # Build CDR position lookup from cdr_dict
                all_cdr_positions = set()
                for chain_id, cdrs in cdr_dict.items():
                    for cdr_name, positions in cdrs.items():
                        for chain, pos, aa in positions:
                            # Normalize position to string for consistent comparison
                            all_cdr_positions.add((chain, str(pos)))
                
                print(f"Found CDR positions: {len(all_cdr_positions)}")
                
                # Filter GT epitope to keep only CDR interactions
                filtered_gt_epitope_interactions = []
                framework_interactions = []
                
                for interaction in original_gt_epitope:
                    if isinstance(interaction, (list, tuple)) and len(interaction) >= 2:
                        antigen_res, antibody_res = interaction
                        
                        # Extract antibody chain and position
                        if isinstance(antibody_res, (list, tuple)) and len(antibody_res) >= 3:
                            ab_chain, ab_pos, ab_aa = antibody_res
                            ab_pos_str = str(ab_pos)  # Normalize to string
                            
                            # Check if this antibody position is in CDR regions
                            if (ab_chain, ab_pos_str) in all_cdr_positions:
                                filtered_gt_epitope_interactions.append(interaction)
                            else:
                                framework_interactions.append(interaction)
                                print(f"Filtered out framework interaction: {antigen_res} <-> {antibody_res}")
                
                # Extract just the antigen residues from filtered interactions for epitope recall calculation
                filtered_gt_epitope = []
                for interaction in filtered_gt_epitope_interactions:
                    antigen_res = interaction[0]
                    if isinstance(antigen_res, (list, tuple)) and len(antigen_res) >= 2:
                        # Keep as (chain, position) tuple for comparison
                        filtered_gt_epitope.append((antigen_res[0], str(antigen_res[1])))
                
                # Remove duplicates
                filtered_gt_epitope = list(set(filtered_gt_epitope))
                
                framework_removed = len(framework_interactions)
                
                print(f"GT epitope after CDR filtering: {len(filtered_gt_epitope)} residues")
                print(f"Removed {framework_removed} framework interactions")
                print(f"CDR-only GT epitope residues: {filtered_gt_epitope}")
                
                # Update item with filtered GT epitope info
                item["gt_epitope_original"] = original_gt_epitope  # Keep original for reference
                item["gt_epitope_filtered_interactions"] = filtered_gt_epitope_interactions  # CDR-only interactions
                item["gt_framework_interactions_removed"] = framework_removed
                
            except Exception as e:
                print(f"Error during GT epitope filtering: {e}")
                print("Using original GT epitope without filtering")
                # Fallback: convert original epitope to simple format
                filtered_gt_epitope = []
                for interaction in original_gt_epitope:
                    if isinstance(interaction, (list, tuple)) and len(interaction) >= 2:
                        antigen_res = interaction[0]
                        if isinstance(antigen_res, (list, tuple)) and len(antigen_res) >= 2:
                            filtered_gt_epitope.append((antigen_res[0], str(antigen_res[1])))
                filtered_gt_epitope = list(set(filtered_gt_epitope))
                framework_removed = 0
                
        else:
            print("No GT epitope or CDR data available for filtering")
            # If no original_gt_epitope, set to empty list
            filtered_gt_epitope = []
            framework_removed = 0

        # Ensure filtered_gt_epitope is always defined before proceeding
        print(f"Final filtered_gt_epitope: {len(filtered_gt_epitope)} residues")

        # Step 6: Use the improved get_cdr_residues_and_interactions function with GT epitope
        # This function will populate all the CDR interaction data and involvement percentages
        item = get_cdr_residues_and_interactions_gt_based(item, cdr_dict, aggregated_results, filtered_gt_epitope)


        # Step 7: Calculate epitope recall using CDR-only model epitope
        model_epitope = item.get("epitope", [])  # This is now CDR-only
        model_epitope_all_cdr = item.get("epitope_all_cdr", [])
        
        if len(filtered_gt_epitope) == 0:
            epitope_recall = 0.0
            print("No CDR-filtered ground truth epitope available")
        elif len(model_epitope) == 0:
            epitope_recall = 0.0
            print("No model epitope interactions found (CDR-only)")
        else:
            try:
                # Convert model epitope to simple format for comparison
                model_epitope_simple = []
                for interaction in model_epitope:
                    if isinstance(interaction, (list, tuple)) and len(interaction) >= 2:
                        antigen_res = interaction[0]
                        if isinstance(antigen_res, (list, tuple)) and len(antigen_res) >= 2:
                            model_epitope_simple.append((antigen_res[0], str(antigen_res[1])))
                
                # Remove duplicates
                model_epitope_simple = list(set(model_epitope_simple))
                
                model_epitope_set = set(model_epitope_simple)
                gt_epitope_set = set(filtered_gt_epitope)
                
                # Calculate recall
                if len(gt_epitope_set) > 0:
                    overlap = model_epitope_set.intersection(gt_epitope_set)
                    epitope_recall = len(overlap) / len(gt_epitope_set)
                    
                    print(f"Epitope recall: {epitope_recall:.3f} ({len(overlap)}/{len(gt_epitope_set)})")
                    print(f"GT CDR epitope size: {len(gt_epitope_set)}, Model epitope size: {len(model_epitope_set)}")
                    print(f"Overlapping residues: {overlap}")
                    print(f"GT epitope residues: {gt_epitope_set}")
                    print(f"Model epitope residues: {model_epitope_set}")
                else:
                    epitope_recall = 0.0
                    print("CDR-filtered ground truth epitope is empty")
                    
            except Exception as e:
                print(f"Error calculating epitope recall: {e}")
                print(f"Debug info - filtered_gt_epitope: {filtered_gt_epitope}")
                epitope_recall = 0.0

        # Store the calculated epitope recall
        item["epitope_recall"] = epitope_recall

        # Step 8: Clean up temporary files to save disk space
        for immuno_chain in immuno_chains:
            for antigen in antigen_chains:
                tmp_dir_specific = os.path.join(tmp_dir_for_interacting_aa, f"{model_name}_{immuno_chain}_to_{antigen}")
                if os.path.exists(tmp_dir_specific):
                    try:
                        clean_up_files(tmp_dir_specific, model_name)
                        # Also try to remove the directory if empty
                        if os.path.exists(tmp_dir_specific) and not os.listdir(tmp_dir_specific):
                            os.rmdir(tmp_dir_specific)
                    except Exception as e:
                        print(f"Warning: Could not clean up temporary directory {tmp_dir_specific}: {e}")

        print(f"Enhanced binding analysis completed for {mod_pdb}")


        return item

    except Exception as e:
        print(f"Error in enhanced metrics_and_binding_analysis for {item.get('mod_pdb', 'unknown')}: {e}")
        traceback.print_exc()
        return item


def file_has_content(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

def main_test_set_mode(args, summary, pdb_dict):
    """Main processing for test set mode"""
    print("Processing in TEST SET mode")
    
    # Initialize metrics file
    metrics_file_exists = os.path.exists(args.csv_dir) and os.stat(args.csv_dir).st_size > 0
    if not metrics_file_exists:
        with open(args.csv_dir, 'w') as f:
            pass

    # Check for existing data
    existing_data = []
    if metrics_file_exists:
        with open(args.csv_dir, 'r') as f:
            existing_data = [json.loads(line) for line in f]

    processed_entry_ids = {entry['entry_id'] for entry in existing_data if 'entry_id' in entry}
    summary_to_process = [entry for entry in summary if entry.get('entry_id') not in processed_entry_ids]

    # Process evaluation
    # num_workers, reasoning = get_optimal_workers()
    process_with_args = partial(evaluate_item_test_set, args, pdb_dict)
    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_with_args, k) for k in summary_to_process]
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result(timeout=800)
                if result is not None:
                    results.append(result)
            except concurrent.futures.TimeoutError:
                print("A task exceeded the time limit and was skipped.")
            except Exception as e:
                print(f"An error occurred: {e}")

    filtered_results = [result for result in results if "tmp_dir_for_interacting_aa" in result]

    # Process metrics
    # num_workers, reasoning = get_optimal_workers()
    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(metrics_and_binding_analysis, entry) for entry in filtered_results]
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result(timeout=800)
                if result is not None:
                    results.append(result)
            except concurrent.futures.TimeoutError:
                print("A task exceeded the time limit and was skipped.")
            except Exception as e:
                print(f"An error occurred: {e}")

    # Save results
    for item in results:
        if item is not None:
            with open(args.csv_dir, "a") as f:
                f.write(json.dumps(item) + '\n')

    # Process FoldX if needed
    with open(args.csv_dir, "r") as f:
        metrics_summaries = [json.loads(line) for line in f]

    # num_workers, reasoning = get_optimal_workers()
    if metrics_summaries and 'FoldX_dG' not in metrics_summaries[-1]:
        foldx_results = []
        # with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(fold_x_computations_with_retry, entry) for entry in metrics_summaries]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=1500)
                    if result is not None:
                        foldx_results.append(result)
                except concurrent.futures.TimeoutError:
                    print("A task exceeded the time limit and was skipped.")

        with open(args.csv_dir, "w") as f:
            for result in foldx_results:
                f.write(json.dumps(result) + '\n')



def has_metrics_data(entry):
    """Check if an entry has metrics/binding analysis data"""
    # Check for key indicators of complete processing
    # FoldX computation indicators
    has_foldx = entry.get("FoldX_dG") is not None or entry.get("FoldX_ddG") is not None
    
    # Metrics analysis indicators  
    has_epitope_recall = entry.get("epitope_recall") is not None
    has_rmsd = entry.get("RMSD(CA) aligned") is not None
    
    # Entry is considered complete if it has both FoldX and metrics analysis
    return has_foldx and has_epitope_recall and has_rmsd

def main_pipeline_mode(args, summary, pdb_dict):
    """Main processing for pipeline mode with re-run capability"""
    print("Processing in PIPELINE mode")
    
    # Group by PDB
    unique_pdbs = set()
    for entry in summary:
        pdb = entry.get("pdb") or entry.get("entry_id")
        if pdb:
            unique_pdbs.add(pdb)

    # Create individual JSON files per PDB if they don't exist
    for pdb in unique_pdbs:
        file_path = os.path.join(args.csv_dir, f"{pdb}.json")
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                pass

    # Group data by PDB
    grouped_data = {}
    for entry in summary:
        model_key = entry.get("pdb") or entry.get("entry_id")
        if model_key not in grouped_data:
            grouped_data[model_key] = [entry]
        else:
            grouped_data[model_key].append(entry)

    keys_list = list(grouped_data.keys())

    for pdb_n in keys_list:
        model_list = grouped_data[pdb_n]
        file_path = os.path.join(args.csv_dir, f"{pdb_n}.json")
        
        print(f"\nProcessing PDB: {pdb_n}")
        
        # Load existing results if file has content
        existing_results = []
        entries_needing_initial_processing = []
        entries_needing_foldx = []
        entries_needing_metrics = []
        
        if os.path.exists(file_path) and os.stat(file_path).st_size > 0:
            print(f"Loading existing results for {pdb_n}")
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        existing_results.append(result)
                    except json.JSONDecodeError:
                        continue
            
            # Create lookup for existing results by identifier
            existing_lookup = {}
            for result in existing_results:
                # Use mod_pdb or entry_id as identifier
                identifier = result.get('mod_pdb') or result.get('entry_id')
                if identifier:
                    existing_lookup[identifier] = result
            
            # Categorize entries based on what processing they need
            for entry in model_list:
                identifier = entry.get('mod_pdb') or entry.get('entry_id')
                
                if identifier not in existing_lookup:
                    # Entry not processed at all
                    entries_needing_initial_processing.append(entry)
                else:
                    existing_entry = existing_lookup[identifier]
                    
                    # Check if entry is fully complete
                    if has_metrics_data(existing_entry):
                        # Entry is fully complete, will be included in final results
                        continue
                    # Check if FoldX processing is missing
                    elif existing_entry.get("FoldX_dG") is None and existing_entry.get("FoldX_ddG") is None:
                        entries_needing_foldx.append(existing_entry)
                    # Has FoldX but missing metrics
                    else:
                        entries_needing_metrics.append(existing_entry)
        else:
            # No existing results - process all entries
            entries_needing_initial_processing = model_list
        
        print(f"Entries needing initial processing: {len(entries_needing_initial_processing)}")
        print(f"Entries needing FoldX: {len(entries_needing_foldx)}")
        print(f"Entries needing metrics: {len(entries_needing_metrics)}")
        
        # Step 1: Initial processing for entries that haven't been processed
        new_results = []
        if entries_needing_initial_processing:
            print("Running initial processing...")
            process_with_args = partial(evaluate_item_pipeline, args, pdb_dict, pdb_n)
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                for i, k in enumerate(entries_needing_initial_processing):
                    future = executor.submit(process_with_args, k)
                    futures.append((i, future, k))
                
                for i, future, original_entry in futures:
                    try:
                        result = future.result(timeout=800)
                        if result is not None:
                            new_results.append(result)
                        else:
                            print(f"Initial processing task {i} returned None")
                    except concurrent.futures.TimeoutError:
                        print(f"Initial processing task {i} timed out. Entry: {original_entry.get('mod_pdb')}")
                    except Exception as e:
                        print(f"Initial processing task {i} failed: {e}")
                        print(f"Failed entry: {original_entry.get('mod_pdb') or original_entry.get('entry_id')}")
        
        # Combine new results with entries that need FoldX
        entries_for_foldx = []
        
        # Filter new results that have tmp_dir_for_interacting_aa
        filtered_new_results = [result for result in new_results if "tmp_dir_for_interacting_aa" in result]
        entries_for_foldx.extend(filtered_new_results)
        entries_for_foldx.extend(entries_needing_foldx)
        
        print(f"Total entries ready for FoldX: {len(entries_for_foldx)}")
        
        # Step 2: FoldX processing
        foldx_results = []
        if entries_for_foldx:
            print("Running FoldX computations...")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(fold_x_computations_with_retry, entry) for entry in entries_for_foldx]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=2000)
                        if result is not None:
                            foldx_results.append(result)
                    except concurrent.futures.TimeoutError:
                        print("A FoldX task exceeded the time limit and was skipped.")
                    except Exception as e:
                        print(f"FoldX task failed: {e}")
        
        # Step 3: Prepare entries for metrics analysis
        entries_for_metrics = []
        
        # Filter FoldX results with valid dG values
        for item in foldx_results:
            ddG = item.get("FoldX_dG")
            if ddG:
                try:
                    ddG_float = float(ddG)
                    entries_for_metrics.append(item)
                except:
                    pass
        
        # Add entries that already had FoldX but need metrics
        entries_for_metrics.extend(entries_needing_metrics)
        
        print(f"Total entries ready for metrics analysis: {len(entries_for_metrics)}")
        
        # Step 4: Metrics and binding analysis
        final_new_results = []
        if entries_for_metrics:
            print("Running metrics and binding analysis...")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(metrics_and_binding_analysis, entry) for entry in entries_for_metrics]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=1000)
                        if result is not None:
                            final_new_results.append(result)
                    except concurrent.futures.TimeoutError:
                        print("A metrics analysis task exceeded the time limit and was skipped.")
                    except Exception as e:
                        print(f"Metrics analysis task failed: {e}")
        
        # Step 5: Combine all results and save
        all_final_results = []
        
        # Add existing results - include ALL processed entries to prevent re-processing
        # This preserves entries that have been through initial processing, even if later stages failed
        if existing_results:
            for result in existing_results:
                # Keep all results that have been through initial processing
                if result.get("tmp_dir_for_interacting_aa") is not None:
                    all_final_results.append(result)
        
        # Add newly completed results
        all_final_results.extend(final_new_results)
        
        print(f"Total final results for {pdb_n}: {len(all_final_results)}")
        
        # Save all results
        if all_final_results:
            with open(file_path, "w") as f:
                for result in all_final_results:
                    f.write(json.dumps(result) + '\n')
            print(f"Saved {len(all_final_results)} results to {file_path}")
        else:
            print(f"No results to save for {pdb_n}")


def main(args):
    print("Starting unified metrics evaluation script")

    print("args.summary_json", args.summary_json)
    
    # # Load summary data
    # with open(args.summary_json, 'r') as fin:
    #     summary = [json.loads(line) for line in fin]

    try:
        with open(args.summary_json, 'r') as fin:
            summary = [json.loads(line) for line in fin]
        print(f"Successfully loaded {len(summary)} entries using fast method")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Fast loading failed ({e}), switching to safe method...")
        summary = safe_load_json_lines(args.summary_json)


    
    # Load test set data
    with open(args.test_set, 'r') as fin:
        data = [json.loads(line) for line in fin]

    print(f"Dataset entries: {len(data)}")
    print(f"Summary entries: {len(summary)}")

    # Create PDB dictionary
    pdb_dict = {}
    for item in data:
        pdb = item.get("entry_id", item.get("pdb"))
        pdb_dict[pdb] = item

    # Detect processing mode
    processing_mode = detect_processing_mode(args, summary)
    print(f"Detected processing mode: {processing_mode}")


    # Validate arguments based on processing mode
    if processing_mode == 'test_set':
        if not args.csv_dir:
            raise ValueError("Test set mode requires --csv_dir or --metrics_file argument")
    else:  # pipeline mode
        if not args.csv_dir:
            raise ValueError("Pipeline mode requires --csv_dir argument (directory path)")
        if not args.hdock_models:
            raise ValueError("Pipeline mode requires --hdock_models argument")

    # Route to appropriate processing function
    if processing_mode == 'test_set':
        main_test_set_mode(args, summary, pdb_dict)
    else:  # pipeline mode
        main_pipeline_mode(args, summary, pdb_dict)

    clear_memory()
    print("Processing completed")



if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Unified metrics evaluation for test sets and pipeline')
    
    # Required arguments
    parser.add_argument('--summary_json', type=str, required=True, help='Path to the summary in json format')
    parser.add_argument('--test_set', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--cdr_model', type=str, required=True, help='Type of model that generates CDRs')
    parser.add_argument('--csv_dir', '--metrics_file', type=str, required=True, help='Directory/file to save results')
    
    # Optional arguments (some may be required based on mode)
    parser.add_argument('--designed_cdr', type=str, default="multiple", help='Type of designed CDR')
    parser.add_argument('--hdock_models', type=str, default=None, help='Directory for hdock models (pipeline mode)')
    parser.add_argument('--iteration', type=str, default=None, help='Iteration number (triggers pipeline mode)')
    # parser.add_argument('--cdr_type', type=str, default=['H3'], help='Type of CDR', choices=['H3'])
    parser.add_argument('--cdr_type', choices=['H1', 'H2', 'H3', '-'], nargs='+', help='CDR types to randomize')
    
    # Add the missing arguments that are being passed to your script
    parser.add_argument('--top_n', type=int, default=None, help='Top N parameter')
    
    # Parse known args to ignore any unknown arguments
    args, unknown = parser.parse_known_args()
    
    # Warn about unknown arguments
    if unknown:
        print(f"Warning: Unknown arguments ignored: {unknown}")
    
    main(args)