#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
from time import sleep
import shutil
from shutil import rmtree
from multiprocessing import Process
from define_aa_contacts_antibody_nanobody_2025 import dedup_interactions, get_cdr_residues_dict
import random

import copy

from tqdm import tqdm
import numpy as np
import sys

from configs import CACHE_DIR
from utils.logger import print_log
from data.pdb_utils import VOCAB, AgAbComplex, AgAbComplex_mod, Protein, Peptide, AgAbComplex2
from .hdock_api import dock
from utils.renumber import renumber_pdb
from utils.logger import print_log
from utils.relax import openmm_relax, openmm_relax_no_decorator
import sys
import torch 
import time
import concurrent

import cProfile
import pstats

from Bio import PDB
import numpy as np
from scipy.spatial import cKDTree


parser = PDB.PDBParser(QUIET=True)
downloader = PDB.PDBList()

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


def main(args):
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    numbering = "imgt"  # Change to chothia on the inference script for diffab     

    # modification
    with open(args.dataset_json, 'r') as fin:
        lines = fin.read().strip().split('\n')

    print("main function")

    hdock_p = [] 

    #keep track of processing time
    parent_dir = os.path.dirname(os.path.dirname(args.hdock_models))
    os.makedirs(args.hdock_models, exist_ok=True)
    track_file = os.path.join(parent_dir, "track_file_hdock.json")
    if not os.path.exists(track_file):
        with open(track_file, 'w') as f:
            pass
    


    for line in tqdm(lines):
        item = json.loads(line)
        pdb = item['pdb']
        heavy_chain = item['heavy_chain_seq']

        # perturb CDR only at first iteration or use original
        if int(args.iteration) == 1:

            if args.initial_cdr == "randomized":
                
                ids = args.randomized                                
                id_list = list(range(1, int(ids) + 1))
                # id_list = id_list[::-1] 

                # =================================================================
                # PHASE 1: BATCH ALL IGFOLD OPERATIONS
                # =================================================================
                print(f"=== PHASE 1: Processing {len(id_list)} IgFold structures for {pdb} ===")
                
                structures_for_hdock = []
                igfold_results = []
                
                for i in id_list:
                    print(f"Processing IgFold for {pdb}_{i} ({i}/{len(id_list)})")
                    
                    # create a copy of the original item/dictionary with the information needed as it will be modified
                    item_copy = copy.deepcopy(item)  # this will be the input to the rest of the processes
                    tmp_dir_for_interacting_aa = os.path.join(args.hdock_models, f"tmp_dir_binding_computations_{pdb}_{i}")  
                    os.makedirs(tmp_dir_for_interacting_aa, exist_ok=True)

                    pdb_dir_hdock = os.path.join(args.hdock_models, pdb + f"_{i}")  
                    if os.path.exists(pdb_dir_hdock) and any(fname.startswith('model_') and fname.endswith('.pdb') for fname in os.listdir(pdb_dir_hdock)):
                        print(f"Directory for {pdb_dir_hdock} already has model files. Skipping...")
                        continue
                    else:
                        os.makedirs(pdb_dir_hdock, exist_ok=True)

                    from .igfold_api import pred, pred_nano
        
                    tmp_dir = args.hdock_models # dyMEAN_iter_original_cdr/HDOCK_iter_i

                    ag_pdb = os.path.join(tmp_dir, pdb + f"_{i}", f'{pdb}_ag.pdb') # save antigen and epitope information
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
                            continue

                    # check if there are any clashes in the antigen structure, if True, then fix it with relaxation
                    try:
                        structure = parser.get_structure("1111", ag_pdb)
                        tuple_result = count_clashes(structure, clash_cutoff=0.60)
                        num_clashes = tuple_result[0]
                        print(f"Number of clashes = {num_clashes} in {ag_pdb} pdb")

                        if num_clashes > 0:
                            print("Conducting antigen relaxation")
                            openmm_relax(ag_pdb, ag_pdb)

                    except TimeoutError as te:
                        print(f"Relaxation for {ag_pdb} timed out: {te}")
                        continue
                    except Exception as e:
                        print(f"Refinement for {ag_pdb} failed: {e}")
                        continue

                    # CDR randomization
                    heavy_chain_current = heavy_chain  # Reset for each iteration

                    

                    # Initialize placeholders 

                    item_copy["cdrh1_seq_mod"] = item_copy["cdrh1_seq"]  
                    item_copy["cdrh2_seq_mod"] = item_copy["cdrh2_seq"] 
                    item_copy["cdrh3_seq_mod"] = item_copy["cdrh3_seq"] 

                    
                    if args.cdr_type != ['-'] and '-' not in args.cdr_type:
                        for cdr_type in args.cdr_type:
                            print(f"Processing CDR type: {cdr_type}")
                            
                            cdr_pos = f"cdr{cdr_type.lower()}_pos"                            
                            cdr_start, cdr_end = item_copy[cdr_pos]
                            
                            pert_cdr = np.random.randint(low=0, high=VOCAB.get_num_amino_acid_type(), size=(cdr_end - cdr_start + 1,))
                            pert_cdr = ''.join([VOCAB.idx_to_symbol(int(i)) for i in pert_cdr])
                            
                            if cdr_type[0] == 'H':
                                l, r = heavy_chain_current[:cdr_start], heavy_chain_current[cdr_end + 1:]
                                heavy_chain_current = l + pert_cdr + r
                            else:
                                l, r = light_chain[:cdr_start], light_chain[cdr_end + 1:]
                                light_chain = l + pert_cdr + r
                            
                            # UPDATE THE CDR sequence in the item_copy
                            original_cdr_key = f"cdr{cdr_type.lower()}_seq"
                            if original_cdr_key in item:
                                original_cdr_seq = item[original_cdr_key]
                                print(f"Previous {cdr_type} sequence: {original_cdr_seq}")
                            
                            item_copy[f"cdr{cdr_type.lower()}_seq_mod"] = pert_cdr
                            print(f"New {cdr_type} sequence: {pert_cdr}")

                    # IgFold Processing
                    ab_pdb = os.path.join(tmp_dir, pdb + f"_{i}", f'{pdb}_IgFold.pdb')
                    
                    igfold_success = False
                    IgFold_clashes = 0
                    num_clashes_after = 0
                    igfold_time = 0

                    # Check if IgFold structure already exists to skip processing
                    if os.path.exists(ab_pdb):
                        print(f"IgFold structure already exists at {ab_pdb}. Skipping IgFold processing...")
                        
                        # Still need to extract CDR information for existing structure
                        try:
                            heavy_chain_seq_for_filter = ""
                            pdb_igfold = Protein.from_pdb(ab_pdb, item_copy['heavy_chain'])

                            for peptide_id, peptide in pdb_igfold.peptides.items():
                                sequence = peptide.get_seq()
                                heavy_chain_seq_for_filter += sequence

                                cdr_pos_dict = extract_antibody_info(pdb_igfold, item_copy["heavy_chain"], item_copy["light_chain"], "imgt")
                                peptides_dict = pdb_igfold.get_peptides()
                                nano_peptide = peptides_dict.get(item_copy["heavy_chain"])

                                for j in range(1, 4):
                                    cdr_name = f'H{j}'.lower()
                                    cdr_pos = get_cdr_pos(cdr_pos_dict, cdr_name)
                                    item_copy[f'cdr{cdr_name}_pos_mod'] = cdr_pos
                                    start, end = cdr_pos 
                                    end += 1
                                    cdr_seq = nano_peptide.get_span(start, end).get_seq()
                                    item_copy[f'cdr{cdr_name}_seq_mod'] = cdr_seq

                            igfold_success = True
                            print(f"Using existing IgFold structure for entry {pdb}_{i}")

                        except Exception as e:
                            print(f'Something went wrong extracting CDR info from existing {ab_pdb}, {e}')
                            continue

                    else:
                        # Original IgFold processing
                        print(f"Starting IgFold for entry {pdb}_{i}")
                        start_igfold_time = time.time()
                        
                        try:
                            heavy_chain_seq = heavy_chain_current
                            heavy_chain_id = item_copy["heavy_chain"]
                            pred_nano(heavy_chain_id, heavy_chain_seq, ab_pdb, do_refine=False)
                            end_igfold_time = time.time()
                            igfold_time = end_igfold_time - start_igfold_time

                            #renumber and revise IgFold generated a nanobody that "makes sense"
                            try:
                                renumber_pdb(ab_pdb, ab_pdb, numbering)
                            except Exception as e:
                                print("Generated nanobody could not be renumbered, skipping...")
                                continue

                            # Extract CDR information
                            heavy_chain_seq_for_filter = ""
                            pdb_igfold = Protein.from_pdb(ab_pdb, item_copy['heavy_chain'])

                            for peptide_id, peptide in pdb_igfold.peptides.items():
                                sequence = peptide.get_seq()
                                heavy_chain_seq_for_filter += sequence

                                cdr_pos_dict = extract_antibody_info(pdb_igfold, item_copy["heavy_chain"], item_copy["light_chain"], "imgt")
                                peptides_dict = pdb_igfold.get_peptides()
                                nano_peptide = peptides_dict.get(item_copy["heavy_chain"])

                                for j in range(1, 4):
                                    cdr_name = f'H{j}'.lower()
                                    cdr_pos = get_cdr_pos(cdr_pos_dict, cdr_name)
                                    item_copy[f'cdr{cdr_name}_pos_mod'] = cdr_pos
                                    start, end = cdr_pos 
                                    end += 1
                                    cdr_seq = nano_peptide.get_span(start, end).get_seq()
                                    item_copy[f'cdr{cdr_name}_seq_mod'] = cdr_seq

                            # check if produced structure contains clashes
                            structure = parser.get_structure("1111", ab_pdb)
                            tuple_result = count_clashes(structure, clash_cutoff=0.60)
                            num_clashes = tuple_result[0]
                            IgFold_clashes = tuple_result[0]
                            print(f"Number of clashes = {num_clashes} in {ab_pdb} pdb")

                            if num_clashes > 0:
                                print("Conducting antibody relaxation")
                                openmm_relax(ab_pdb, ab_pdb)

                                # check how many clashes we have after relaxing
                                structure = parser.get_structure("1111", ab_pdb)
                                tuple_result = count_clashes(structure, clash_cutoff=0.60)
                                num_clashes_after = tuple_result[0]
                                print(f"Number of clashes after relaxation = {num_clashes_after} in {ab_pdb} pdb")
                            else:
                                num_clashes_after = num_clashes

                            print(f"IgFold for entry {pdb}_{i} took {igfold_time:.2f} seconds")
                            igfold_success = True

                        except TimeoutError as te:
                            print(f"IgFold relaxation for {ab_pdb} timed out: {te}")
                            continue
                        except Exception as e:
                            print(f"IgFold processing for {ab_pdb} failed: {e}")
                            continue

                    # Store successful structures for HDock
                    if igfold_success:
                        structure_data = {
                            'index': i,
                            'pdb': pdb,
                            'item_copy': item_copy,
                            'ag_pdb': ag_pdb,
                            'ab_pdb': ab_pdb,
                            'pdb_dir_hdock': pdb_dir_hdock,
                            'igfold_time': igfold_time,
                            'igfold_clashes': IgFold_clashes,
                            'num_clashes_after': num_clashes_after
                        }
                        structures_for_hdock.append(structure_data)
                        
                        igfold_result = {
                            'pdb': f"{pdb}_{i}",
                            'success': True,
                            'igfold_time': igfold_time,
                            'initial_clashes': IgFold_clashes,
                            'final_clashes': num_clashes_after
                        }
                        igfold_results.append(igfold_result)

                # =================================================================
                # PHASE 1 COMPLETE - VALIDATION AND SUMMARY
                # =================================================================
                total_igfold_time = sum(result['igfold_time'] for result in igfold_results)
                successful_structures = len(structures_for_hdock)
                
                print(f"\n=== PHASE 1 VALIDATION for {pdb} ===")
                print(f"Expected structures: {len(id_list)}")
                print(f"Successfully processed: {successful_structures}")
                print(f"Failed structures: {len(id_list) - successful_structures}")
                
                # Validation checks
                validation_passed = True
                
                # Check 1: Success rate (informational only - we process whatever succeeded)
                success_rate = successful_structures / len(id_list) if len(id_list) > 0 else 0
                if success_rate < 1.0:
                    print(f"INFO: Success rate {success_rate:.2%} - will process {successful_structures} successful structures")
                
                # Check 2: Verify all structure files exist
                missing_files = []
                for struct_data in structures_for_hdock:
                    ag_pdb = struct_data['ag_pdb']
                    ab_pdb = struct_data['ab_pdb']
                    if not os.path.exists(ag_pdb):
                        missing_files.append(f"Missing antigen: {ag_pdb}")
                    if not os.path.exists(ab_pdb):
                        missing_files.append(f"Missing antibody: {ab_pdb}")
                
                if missing_files:
                    print(f"ERROR: Missing structure files:")
                    for missing in missing_files:
                        print(f"  - {missing}")
                    validation_passed = False
                
                # Check 3: Verify CDR information was extracted
                incomplete_cdr = []
                for struct_data in structures_for_hdock:
                    item_copy = struct_data['item_copy']
                    required_cdr_keys = ['cdrh1_seq_mod', 'cdrh2_seq_mod', 'cdrh3_seq_mod']
                    missing_cdrs = [key for key in required_cdr_keys if key not in item_copy]
                    if missing_cdrs:
                        incomplete_cdr.append(f"{struct_data['pdb']}_{struct_data['index']}: {missing_cdrs}")
                
                if incomplete_cdr:
                    print(f"WARNING: Incomplete CDR extraction:")
                    for incomplete in incomplete_cdr:
                        print(f"  - {incomplete}")
                    # This might not be critical, so don't fail validation
                
                print(f"\n=== PHASE 1 SUMMARY for {pdb} ===")
                print(f"Validation status: {'PASSED' if validation_passed else 'FAILED'}")
                print(f"Success rate: {success_rate:.2%}")
                print(f"Total IgFold time: {total_igfold_time:.2f} seconds")
                if successful_structures > 0:
                    print(f"Average time per structure: {total_igfold_time/successful_structures:.2f} seconds")
                
                # Decision point: Continue to Phase 2 or not
                if not validation_passed:
                    print(f"ERROR: Phase 1 validation failed for {pdb}. Critical errors found.")
                    continue  # Skip to next PDB entry
                
                if successful_structures == 0:
                    print(f"ERROR: No successful structures for {pdb}. Nothing to process in Phase 2.")
                    continue  # Skip to next PDB entry
                    
                # Process all successful structures, regardless of success rate
                print(f"INFO: Proceeding to Phase 2 with {successful_structures} structures")

                # =================================================================
                # PHASE 2: BATCH ALL HDOCK OPERATIONS
                # =================================================================
                print(f"\n=== PHASE 2 STARTING for {pdb} ===")
                print(f"✓ Phase 1 validation passed")
                print(f"✓ Processing {len(structures_for_hdock)} verified structures")
                print(f"✓ All IgFold operations completed successfully")
                print(f"Starting HDock batch processing...")
                
                hdock_processes = []
                hdock_results = []
                
                for struct_data in structures_for_hdock:
                    i = struct_data['index']
                    item_copy = struct_data['item_copy']
                    ag_pdb = struct_data['ag_pdb']
                    ab_pdb = struct_data['ab_pdb']
                    pdb_dir_hdock = struct_data['pdb_dir_hdock']
                    
                    print(f"Preparing HDock for {pdb}_{i}")

                    # Process epitope information
                    epitope = item_copy["epitope"]
                    epitope_format = detect_epitope_format(epitope)
                    print(f"Detected epitope format: {epitope_format}")

                    if epitope_format == 'single':
                        # Handle single-sided epitope (antigen residues only)
                        print("Processing single-sided epitope")
                        
                        epitope_ = [tuple(item) for item in epitope]
                        epitope_ = dedup_interactions(epitope_)
                        
                        print(f"Single-sided epitope: {len(epitope_)} antigen residues")
                        
                        # Extract binding_rsite directly (antigen positions)
                        binding_rsite = []
                        for res in epitope_:
                            try:
                                if len(res) >= 2:
                                    chain = res[0]
                                    pos = res[1]
                                    if isinstance(pos, str) and pos.isdigit():
                                        pos = int(pos)
                                    binding_rsite.append((chain, pos))
                            except:
                                continue
                        
                        # print("binding_rsite", binding_rsite)

                    elif epitope_format == 'paired':
                        # Handle paired epitope
                        print("Processing paired epitope")
                        
                        epitope_ = [tuple(item) for item in epitope]
                        epitope_ = dedup_interactions(epitope_)

                        cdr_dict = get_cdr_residues_dict(item_copy, ab_pdb)

                        # Filter epitope to CDR-only for consistency
                        if cdr_dict:
                            all_cdr_positions = set()
                            for chain_id, cdrs in cdr_dict.items():
                                for cdr_name, positions in cdrs.items():
                                    for _, pos, _ in positions:
                                        all_cdr_positions.add((chain_id, str(pos)))
                            
                            original_epitope = epitope_.copy()
                            filtered_epitope = []
                            framework_removed = 0
                            
                            for interaction in epitope_:
                                try:
                                    if isinstance(interaction, (list, tuple)) and len(interaction) >= 2:
                                        antigen_res, antibody_res = interaction
                                        if len(antibody_res) >= 2:
                                            ab_chain, ab_pos = antibody_res[0], str(antibody_res[1])
                                            
                                            if (ab_chain, ab_pos) in all_cdr_positions:
                                                filtered_epitope.append(interaction)
                                            else:
                                                framework_removed += 1
                                except:
                                    continue
                            
                            epitope_ = filtered_epitope
                            print(f"Epitope filtered: {len(original_epitope)} -> {len(filtered_epitope)} (removed {framework_removed} framework)")

                        # Extract binding_rsite
                        binding_rsite = []
                        for interaction in epitope_:
                            try:
                                if isinstance(interaction, (list, tuple)) and len(interaction) >= 2:
                                    antigen_res = interaction[0]
                                    if len(antigen_res) >= 2:
                                        chain = antigen_res[0]
                                        pos = antigen_res[1]
                                        if isinstance(pos, str) and pos.isdigit():
                                            pos = int(pos)
                                        binding_rsite.append((chain, pos))
                            except:
                                continue

                        # print("binding_rsite", binding_rsite)

                    else:
                        print(f"Unknown or empty epitope format: {epitope_format}")
                        binding_rsite = []

                    # Extract paratope information

                    lsite3 = extract_seq_info_from_pdb(ab_pdb, item_copy["heavy_chain"], 
                                                    item_copy.get("cdrh3_seq_mod", item_copy["cdrh3_seq"]))
                    lsite2 = extract_seq_info_from_pdb(ab_pdb, item_copy["heavy_chain"], 
                                                    item_copy.get("cdrh2_seq_mod", item_copy["cdrh2_seq"]))
                    lsite1 = extract_seq_info_from_pdb(ab_pdb, item_copy["heavy_chain"], 
                                                    item_copy.get("cdrh1_seq_mod", item_copy["cdrh1_seq"]))

                    print("modified cdrh3 seq", item_copy.get("cdrh3_seq_mod"))
                    print("original cdrh3 seq", item_copy.get("cdrh3_seq"))
                    print("Igfold structure", ab_pdb)

                    binding_lsite = []
                    if lsite3 is not None:
                        binding_lsite.extend([tup[:2] for tup in lsite3])
                    if lsite2 is not None:
                        binding_lsite.extend([tup[:2] for tup in lsite2])
                    if lsite1 is not None:
                        binding_lsite.extend([tup[:2] for tup in lsite1])

                    # print("binding_lsite", binding_lsite)

                    # Prepare tracking information
                    track_item = {}
                    track_item["heavy"] = item_copy["heavy_chain"]
                    track_item["light"] = item_copy["light_chain"]
                    track_item["antigen"] = item_copy["antigen_chains"]
                    track_item["pdb"] = f"{pdb}_{i}"
                    track_item["Hdock_n_models"] = args.n_docked_models
                    track_item["IgFold_time"] = struct_data['igfold_time']
                    track_item["IgFold_clashes"] = struct_data['igfold_clashes']
                    track_item["IgFold_clashes_after_relax"] = struct_data['num_clashes_after']
                    track_item["iteration"] = args.iteration

                    # Launch HDock process
                    print(f"Starting HDock for entry {pdb}_{i}")
                    
                    # Manage process queue
                    while len(hdock_processes) >= 4:  # Keep max 4 processes
                        removed = False
                        for p in hdock_processes:
                            if not p.is_alive():
                                p.join()
                                p.close()
                                hdock_processes.remove(p)
                                removed = True
                        if not removed:
                            sleep(10)

                    args_in = (ag_pdb, ab_pdb, pdb_dir_hdock, args.n_docked_models, binding_rsite, binding_lsite)
                    p = Process(target=dock_wrap, args=(args_in, track_file, track_item))
                    p.start()
                    hdock_processes.append(p)

                # Wait for all HDock processes to complete
                print(f"\nWaiting for all {len(hdock_processes)} HDock processes to complete...")
                completed_processes = 0
                
                while hdock_processes:
                    for p in hdock_processes[:]:  # Create a copy to iterate safely
                        if not p.is_alive():
                            p.join()
                            p.close()
                            hdock_processes.remove(p)
                            completed_processes += 1
                            print(f"HDock process completed ({completed_processes}/{len(hdock_processes) + completed_processes})")
                    
                    if hdock_processes:  # Still processes running
                        sleep(5)  # Check every 5 seconds
                
                print(f"\n=== PHASE 2 COMPLETE for {pdb} ===")
                print(f"✓ All {completed_processes} HDock processes completed successfully")
                
                # Optional: Verify HDock outputs
                hdock_output_check = []
                for struct_data in structures_for_hdock:
                    pdb_dir_hdock = struct_data['pdb_dir_hdock']
                    model_files = [f for f in os.listdir(pdb_dir_hdock) if f.startswith('model_') and f.endswith('.pdb')]
                    hdock_output_check.append({
                        'pdb': f"{pdb}_{struct_data['index']}",
                        'output_dir': pdb_dir_hdock,
                        'model_count': len(model_files),
                        'expected_models': args.n_docked_models
                    })
                
                print(f"\n=== HDock OUTPUT VERIFICATION ===")
                for check in hdock_output_check:
                    status = "✓" if check['model_count'] == check['expected_models'] else "⚠"
                    print(f"{status} {check['pdb']}: {check['model_count']}/{check['expected_models']} models")
                
                successful_hdock = sum(1 for check in hdock_output_check if check['model_count'] == check['expected_models'])
                print(f"HDock success rate: {successful_hdock}/{len(hdock_output_check)} ({successful_hdock/len(hdock_output_check)*100:.1f}%)")
                
                # =================================================================
                # FINAL SUMMARY FOR THIS PDB
                # =================================================================
                print(f"\n=== FINAL SUMMARY for {pdb} ===")
                print(f"IgFold structures: {successful_structures}/{len(id_list)} ({success_rate:.2%})")
                print(f"HDock completions: {successful_hdock}/{successful_structures}")
                print(f"Total processing time: {total_igfold_time:.2f}s (IgFold)")
                print(f"End-to-end success: {successful_hdock}/{len(id_list)} ({successful_hdock/len(id_list)*100:.1f}%)")
                print("="*60)

            elif args.initial_cdr == "original":
                pass

        else:


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

            for element in data: # FIX
                rank = element["rank"]

                parts = pdb.split('_')
                if len(parts) >= 3:  # If there are 2+ underscores, keep first two parts
                    pdb_parts = '_'.join(parts[:2])
                else:  # Otherwise, keep just the first part
                    pdb_parts = parts[0]
                new_pdb_id =  pdb_parts + f"_{rank}"

                print("new_pdb_id", new_pdb_id)
            


                # create hdock dir path
                pdb_dir_hdock = os.path.join(args.hdock_models, new_pdb_id)  

                if os.path.exists(pdb_dir_hdock) and any(fname.startswith('model_') and fname.endswith('.pdb') for fname in os.listdir(pdb_dir_hdock)):
                    print(f"Directory for {pdb_dir_hdock} already has model files. Skipping...")
                    continue
                else:
                    os.makedirs(pdb_dir_hdock, exist_ok=True)

                tmp_dir_for_interacting_aa = os.path.join(args.hdock_models, f"tmp_dir_binding_computations_{new_pdb_id}")  
                os.makedirs(tmp_dir_for_interacting_aa, exist_ok=True)

    
                tmp_dir = args.hdock_models 

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
                    print(f"Skipping entry {new_pdb_id} due to PDB loading error")
                    continue  # Skip this entry entirely instead of pass

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


                # 2.HDock
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


                # epitope has to always be the first original
                # In your second iteration (else block), the logic should be:

                # 1. BINDING_RSITE (epitope/antigen side) - ALWAYS from GT epitope (item)
                epitope = item["epitope"]  # GT epitope from original data
                epitope_format = detect_epitope_format(epitope)
                print(f"Detected epitope format: {epitope_format}")

                if epitope_format == 'single':
                    # Handle single-sided epitope (antigen residues only)
                    print("Processing single-sided epitope")
                    epitope_ = [tuple(item) for item in epitope]
                    epitope_ = dedup_interactions(epitope_)
                    
                    # Extract binding_rsite directly (antigen positions from GT)
                    binding_rsite = []
                    for res in epitope_:
                        try:
                            if len(res) >= 2:
                                chain = res[0]
                                pos = res[1]
                                if isinstance(pos, str) and pos.isdigit():
                                    pos = int(pos)
                                binding_rsite.append((chain, pos))
                        except:
                            continue

                elif epitope_format == 'paired':
                    print("Processing paired epitope")
                    epitope_ = [tuple(item) for item in epitope]
                    epitope_ = dedup_interactions(epitope_)

                    # For CDR filtering, use the CURRENT structure's CDR information (element)
                    # Create hybrid item: GT epitope + current CDR sequences
                    hybrid_item = {
                        "heavy_chain": element["heavy_chain"],
                        "light_chain": element["light_chain"],
                        "cdrh1_seq_mod": element["cdrh1_seq_mod"],
                        "cdrh2_seq_mod": element["cdrh2_seq_mod"], 
                        "cdrh3_seq_mod": element["cdrh3_seq_mod"],
                        # Add light chain CDRs if they exist
                    }
                    
                    cdr_dict = get_cdr_residues_dict(hybrid_item, ab_pdb)
                    
                    # Filter epitope using CURRENT CDRs, but extract GT antigen positions
                    if cdr_dict:
                        all_cdr_positions = set()
                        for chain_id, cdrs in cdr_dict.items():
                            for cdr_name, positions in cdrs.items():
                                for _, pos, _ in positions:
                                    all_cdr_positions.add((chain_id, str(pos)))
                        
                        # Filter GT epitope to only CDR interactions
                        original_epitope = epitope_.copy()
                        filtered_epitope = []
                        framework_removed = 0
                        
                        for interaction in epitope_:
                            try:
                                if isinstance(interaction, (list, tuple)) and len(interaction) >= 2:
                                    antigen_res, antibody_res = interaction
                                    if len(antibody_res) >= 2:
                                        ab_chain, ab_pos = antibody_res[0], str(antibody_res[1])
                                        
                                        # Check if antibody position is in CURRENT CDRs
                                        if (ab_chain, ab_pos) in all_cdr_positions:
                                            filtered_epitope.append(interaction)
                                        else:
                                            framework_removed += 1
                            except:
                                continue
                        
                        epitope_ = filtered_epitope
                        print(f"Epitope filtered: {len(original_epitope)} -> {len(filtered_epitope)} (removed {framework_removed} framework)")

                    # Extract GT antigen positions for binding_rsite
                    binding_rsite = []
                    for interaction in epitope_:
                        try:
                            if isinstance(interaction, (list, tuple)) and len(interaction) >= 2:
                                antigen_res = interaction[0]  # GT antigen residue
                                if len(antigen_res) >= 2:
                                    chain = antigen_res[0]
                                    pos = antigen_res[1]
                                    if isinstance(pos, str) and pos.isdigit():
                                        pos = int(pos)
                                    binding_rsite.append((chain, pos))
                        except:
                            continue

                print("binding_rsite (from GT epitope):", binding_rsite)

                # 2. BINDING_LSITE (paratope/antibody side) - ALWAYS from CURRENT CDRs (element)
                cdrh3_seq = element["cdrh3_seq_mod"]  # Current modified sequences
                cdrh2_seq = element["cdrh2_seq_mod"]
                cdrh1_seq = element["cdrh1_seq_mod"]

                # Extract paratope from CURRENT structure
                lsite3 = extract_seq_info_from_pdb(ab_pdb, element["heavy_chain"], cdrh3_seq)
                lsite2 = extract_seq_info_from_pdb(ab_pdb, element["heavy_chain"], cdrh2_seq)
                lsite1 = extract_seq_info_from_pdb(ab_pdb, element["heavy_chain"], cdrh1_seq)

                binding_lsite = []
                if lsite3 is not None:
                    binding_lsite.extend([tup[:2] for tup in lsite3])
                if lsite2 is not None:
                    binding_lsite.extend([tup[:2] for tup in lsite2])
                if lsite1 is not None:
                    binding_lsite.extend([tup[:2] for tup in lsite1])

                print("binding_lsite (from current CDRs):", binding_lsite)

                # inputs for dockwrapper

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
                        choices=['MEAN', 'Rosetta', 'DiffAb', 'dyMEAN', 'ADesign', 'ADesigner'])
    parser.add_argument('--dataset_json', type=str, required=True, help='Path to the summary file of dataset in json format')
    parser.add_argument('--out_dir', type=str, required=False, help='Path to save generated PDBs')
    parser.add_argument('--data_out_dir', type=str, default=None, help='Directory to save formatted data')
    parser.add_argument('--num_workers', type=str, default=4, help='Number of cores to use for speeding up')
    # parser.add_argument('--cdr_type', type=str, default='H3', help='Type of cdr to generate',
    #                     choices=['H1', 'H2', 'H3', '-'])
    parser.add_argument('--cdr_type', choices=['H1', 'H2', 'H3', '-'], nargs='+', help='CDR types to randomize')
    parser.add_argument('--iteration', type=int, help='Iteration number')
    parser.add_argument('--n_docked_models', type=int, help='Hdock models to predict per entry')
    parser.add_argument("--randomized", type=int, required=True, help="Total number of randomized nanobodies from the CDR(s)")
    parser.add_argument("--best_mutants", type=int, required=True, help="Total number of randomized nanobodies from the CDR(s)")
    parser.add_argument('--hdock_models', type=str, required=True, help='Path to save generated PDBs from hdock')
    parser.add_argument('--csv_dir', type=str, required=True, help='Path where results are')
    parser.add_argument('--csv_dir_', type=str, required=True, help='Path where results are')
    parser.add_argument('--initial_cdr', type=str, required=True, default='randomized', help='Keep original or randomized') 




    return parser.parse_args()

if __name__ == '__main__':

        main(parse())
