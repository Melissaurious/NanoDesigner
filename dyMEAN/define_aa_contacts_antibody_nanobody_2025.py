import os
import subprocess
import argparse
import json
import pandas as pd
import time
from Bio import PDB
from shutil import rmtree
import gc
import traceback
import re

# def interacting_residues(item, pdb_file, antigen, tmp_dir, iteration=1, metrics_dir="metrics"):
# OLD FUNCTION
# def interacting_residues(item, pdb_file, antigen, tmp_dir):


aa_dict = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}




def clean_up_files(directory, file_name):
    # Check if directory exists first
    if not os.path.exists(directory):
        # print(f"Directory {directory} doesn't exist, nothing to clean up")
        return
    
    try:
        list_dir = os.listdir(directory)
    except FileNotFoundError:
        # Directory was deleted between the exists check and listdir call
        return
    
    for file in list_dir:
        if file_name in file:
            full_path = os.path.join(directory, file)
            try:
                os.remove(full_path)
                # print(f"Deleted file: {full_path}")
            except FileNotFoundError:
                print(f"File not found, could not delete: {full_path}")
            except Exception as e:
                print(f"Error deleting file {full_path}: {e}")


def extract_pdb_info(pdb_file, target_chain):
    """
    Extract [(chain, position, residue)] from a PDB given a specified chain.
    Handles insertion codes (like 100A, 100B) properly.
    """
    structure = PDB.PDBParser().get_structure("protein", pdb_file)
    pdb_info = []
    for model in structure:
        for chain in model:
            if chain.id == target_chain:
                for residue in chain:
                    if PDB.is_aa(residue):
                        chain_id = chain.id
                        residue_pos_tup = residue.id
                        # Handle residue numbering with insertion codes
                        res_id = residue_pos_tup[1]
                        insertion_code = residue_pos_tup[2].strip()  # Get insertion code if exists
                        if insertion_code:
                            res_id = f"{res_id}{insertion_code}"  # Combine number and insertion code
                        
                        res_name = three_to_one.get(residue.get_resname(), 'X')  # Default to 'X' if unknown
                        pdb_info.append((chain_id, res_id, res_name))
    return pdb_info


def extract_seq_info_from_pdb(pdb_file, target_chain, sequence):
    """
    Extract [(chain, position, residue)] for a given sequence in a PDB chain.
    Handles insertion codes properly.
    """
    # First get all residues with proper numbering
    pdb_info = extract_pdb_info(pdb_file, target_chain)
    
    # Create the sequence string for matching
    pdb_sequence = ''.join([res[2] for res in pdb_info])
    
    # Find the sequence in the PDB sequence
    start_index = pdb_sequence.find(sequence)
    
    if start_index == -1:
        print(f"Sequence '{sequence}' not found in chain {target_chain}")
        return None
    
    end_index = start_index + len(sequence) - 1
    seq_info = pdb_info[start_index:end_index + 1]
    
    return seq_info


def get_cdr_residues_dict(item, pdb_file):
    """
    Extract all CDR residues from a PDB file based on sequence information.
    Uses actual chain IDs from the item dictionary.
    Prioritizes _mod sequences (from modified PDB) over original sequences.
    
    Parameters:
    - item: Dictionary with antibody information
    - pdb_file: Path to the PDB file
    
    Returns:
    - Dictionary with CDR residues organized by actual chain labels and CDR type
    """
    # Get actual chain IDs
    heavy_chain = item.get("heavy_chain")
    light_chain = item.get("light_chain")
    
    # Initialize the CDR dictionary structure with actual chain IDs
    cdr_dict = {}
    
    if heavy_chain:
        cdr_dict[heavy_chain] = {
            "CDR1": [],
            "CDR2": [],
            "CDR3": []
        }
    
    # Only add light chain structure if it exists
    if light_chain:
        cdr_dict[light_chain] = {
            "CDR1": [],
            "CDR2": [],
            "CDR3": []
        }
    
    # Helper function to get CDR sequence with priority: _mod > original > _ref
    def get_cdr_sequence(cdr_base_name):
        """Get CDR sequence with fallback priority"""
        # Priority 1: _mod sequence (from modified PDB)
        mod_key = f"{cdr_base_name}_seq_mod"
        if mod_key in item and item[mod_key]:
            return item[mod_key], mod_key
        
        # Priority 2: original sequence 
        orig_key = f"{cdr_base_name}_seq"
        if orig_key in item and item[orig_key]:
            return item[orig_key], orig_key
        
        # Priority 3: _ref sequence (from reference PDB)
        ref_key = f"{cdr_base_name}_seq_ref"
        if ref_key in item and item[ref_key]:
            return item[ref_key], ref_key
        
        return None, None
    
    # Process heavy chain CDRs
    if heavy_chain:
        # Get CDR H1 residues
        cdrh1_seq, h1_source = get_cdr_sequence("cdrh1")
        if cdrh1_seq:
            cdrh1_residues = extract_seq_info_from_pdb(pdb_file, heavy_chain, cdrh1_seq)
            if cdrh1_residues:
                cdr_dict[heavy_chain]["CDR1"] = cdrh1_residues
            else:
                print(f"CDR H1 sequence '{cdrh1_seq}' from {h1_source} not found in chain {heavy_chain}")
        
        # Get CDR H2 residues
        cdrh2_seq, h2_source = get_cdr_sequence("cdrh2")
        if cdrh2_seq:
            cdrh2_residues = extract_seq_info_from_pdb(pdb_file, heavy_chain, cdrh2_seq)
            if cdrh2_residues:
                cdr_dict[heavy_chain]["CDR2"] = cdrh2_residues
            else:
                print(f"CDR H2 sequence '{cdrh2_seq}' from {h2_source} not found in chain {heavy_chain}")
        
        # Get CDR H3 residues
        cdrh3_seq, h3_source = get_cdr_sequence("cdrh3")
        if cdrh3_seq:
            cdrh3_residues = extract_seq_info_from_pdb(pdb_file, heavy_chain, cdrh3_seq)
            if cdrh3_residues:
                cdr_dict[heavy_chain]["CDR3"] = cdrh3_residues
            else:
                print(f"CDR H3 sequence '{cdrh3_seq}' from {h3_source} not found in chain {heavy_chain}")
    
    # Process light chain CDRs if light chain exists
    if light_chain:
        # Get CDR L1 residues
        cdrl1_seq, l1_source = get_cdr_sequence("cdrl1")
        if cdrl1_seq:
            cdrl1_residues = extract_seq_info_from_pdb(pdb_file, light_chain, cdrl1_seq)
            if cdrl1_residues:
                cdr_dict[light_chain]["CDR1"] = cdrl1_residues
            else:
                print(f"CDR L1 sequence '{cdrl1_seq}' from {l1_source} not found in chain {light_chain}")
        
        # Get CDR L2 residues
        cdrl2_seq, l2_source = get_cdr_sequence("cdrl2")
        if cdrl2_seq:
            cdrl2_residues = extract_seq_info_from_pdb(pdb_file, light_chain, cdrl2_seq)
            if cdrl2_residues:
                cdr_dict[light_chain]["CDR2"] = cdrl2_residues
            else:
                print(f"CDR L2 sequence '{cdrl2_seq}' from {l2_source} not found in chain {light_chain}")
        
        # Get CDR L3 residues
        cdrl3_seq, l3_source = get_cdr_sequence("cdrl3")
        if cdrl3_seq:
            cdrl3_residues = extract_seq_info_from_pdb(pdb_file, light_chain, cdrl3_seq)
            if cdrl3_residues:
                cdr_dict[light_chain]["CDR3"] = cdrl3_residues
            else:
                print(f"CDR L3 sequence '{cdrl3_seq}' from {l3_source} not found in chain {light_chain}")
    
    print("cdr_dict", cdr_dict)
    return cdr_dict


def dedup_interactions(epitope_list):
    """
    Safely remove duplicates from epitope list, handling nested lists/tuples.
    """
    if not epitope_list:
        return []
    
    seen = set()
    result = []
    
    for item in epitope_list:
        # Convert to hashable format
        if isinstance(item, list):
            # Convert list to tuple recursively
            hashable_item = tuple(tuple(x) if isinstance(x, list) else x for x in item)
        elif isinstance(item, tuple):
            # Already hashable, but check for nested lists
            hashable_item = tuple(tuple(x) if isinstance(x, list) else x for x in item)
        else:
            # Simple item (string, number, etc.)
            hashable_item = item
        
        if hashable_item not in seen:
            seen.add(hashable_item)
            result.append(item)  # Keep original format
    
    return result



def normalize_pos(pos):
    """
    Normalize position to string while preserving insertion codes.
    Handles cases where pos might be int, str, or other types.
    """
    pos_str = str(pos)
    # Remove any whitespace that might have been introduced
    return pos_str.strip()

def get_cdr_residues_and_interactions_gt_based(item, cdr_dict, results, gt_epitope_residues=None):
    """
    Process CDR residues and interaction results to extract epitope information
    and calculate CDR involvement metrics based on GT epitope.
    CDR percentages now reflect interactions with GT epitope residues only.
    """
    # Get chain identifiers
    heavy_chain = item.get("heavy_chain")
    light_chain = item.get("light_chain")
    
    # Identify which keys in cdr_dict correspond to heavy and light chains
    heavy_chain_key = None
    light_chain_key = None
    
    if heavy_chain in cdr_dict:
        heavy_chain_key = heavy_chain
    if light_chain in cdr_dict:
        light_chain_key = light_chain
    
    if not heavy_chain_key and heavy_chain:
        for key in cdr_dict:
            if key in results and 'CDR3' in cdr_dict[key]:
                heavy_chain_key = key
                break
    
    if not light_chain_key and light_chain:
        for key in cdr_dict:
            if key != heavy_chain_key and key in results:
                light_chain_key = key
                break
    
    print(f"Identified heavy chain key: {heavy_chain_key}, light chain key: {light_chain_key}")
    
    # Convert gt_epitope_residues to set for fast lookup
    gt_epitope_set = set()
    if gt_epitope_residues:
        gt_epitope_set = set(gt_epitope_residues)
        print(f"GT epitope set for CDR calculation: {len(gt_epitope_set)} residues")
    
    # Initialize interaction lists
    epitope_model_all_cdr = []  # ALL CDR interactions (for reference)
    epitope_model_gt = []       # ONLY CDR interactions with GT epitope residues
    all_interactions = []       # All interactions for reference
    framework_interactions = [] # Framework interactions
    
    # CDR interactions with any antigen residues
    cdrh1_interactions_to_ag = []
    cdrh2_interactions_to_ag = []
    cdrh3_interactions_to_ag = []
    cdrl1_interactions_to_ag = []
    cdrl2_interactions_to_ag = []
    cdrl3_interactions_to_ag = []
    
    # NEW: CDR interactions with GT epitope residues only
    cdrh1_interactions_to_gt = []
    cdrh2_interactions_to_gt = []
    cdrh3_interactions_to_gt = []
    cdrl1_interactions_to_gt = []
    cdrl2_interactions_to_gt = []
    cdrl3_interactions_to_gt = []
    
    def normalize_pos(pos):
        """
        Normalize position to string while preserving insertion codes.
        Handles cases where pos might be int, str, or other types.
        """
        pos_str = str(pos)
        # Remove any whitespace that might have been introduced
        return pos_str.strip()
    
    # Create position sets for CDRs
    cdrh1_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(heavy_chain_key, {}).get('CDR1', [])}
    cdrh2_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(heavy_chain_key, {}).get('CDR2', [])}
    cdrh3_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(heavy_chain_key, {}).get('CDR3', [])}
    
    cdrl1_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(light_chain_key, {}).get('CDR1', [])} if light_chain_key else set()
    cdrl2_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(light_chain_key, {}).get('CDR2', [])} if light_chain_key else set()
    cdrl3_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(light_chain_key, {}).get('CDR3', [])} if light_chain_key else set()
    
    # Combine all CDR positions for quick lookup
    all_cdr_positions = cdrh1_positions | cdrh2_positions | cdrh3_positions | cdrl1_positions | cdrl2_positions | cdrl3_positions
    
    def is_gt_epitope_interaction(interaction):
        """Check if interaction involves a GT epitope residue"""
        if not gt_epitope_set:
            return False
        ag_res = interaction[0]
        if isinstance(ag_res, (list, tuple)) and len(ag_res) >= 2:
            return (ag_res[0], str(ag_res[1])) in gt_epitope_set
        return False
    
    # Process heavy chain interactions
    if heavy_chain_key and heavy_chain_key in results:
        for antigen, interactions in results[heavy_chain_key].items():
            for interaction in interactions:
                all_interactions.append(interaction)
                
                # Check if interaction has the expected format
                if not isinstance(interaction, (list, tuple)) or len(interaction) < 2:
                    continue
                    
                ag_res, ab_res = interaction
                
                # Check if ab_res has the expected format
                if not isinstance(ab_res, (list, tuple)) or len(ab_res) < 3:
                    continue
                    
                ab_chain, ab_pos, ab_aa = ab_res
                print(f"DEBUG - Unpacked ab_res: chain='{ab_chain}', pos='{ab_pos}', aa='{ab_aa}'")  # Show unpacked values
                norm_pos = normalize_pos(ab_pos)
                print(f"DEBUG - Normalized position: original='{ab_pos}', normalized='{norm_pos}'")  # Show before/after normalization

                
                # Check if this is a CDR interaction
                if norm_pos in all_cdr_positions:
                    epitope_model_all_cdr.append(interaction)  # Add to ALL CDR interactions
                    
                    # Check if it's a GT epitope interaction
                    is_gt_interaction = is_gt_epitope_interaction(interaction)
                    if is_gt_interaction:
                        epitope_model_gt.append(interaction)  # Add to GT epitope interactions
                    
                    # Assign to specific CDR (all interactions)
                    if norm_pos in cdrh1_positions:
                        cdrh1_interactions_to_ag.append(interaction)
                        if is_gt_interaction:
                            cdrh1_interactions_to_gt.append(interaction)
                    elif norm_pos in cdrh2_positions:
                        cdrh2_interactions_to_ag.append(interaction)
                        if is_gt_interaction:
                            cdrh2_interactions_to_gt.append(interaction)
                    elif norm_pos in cdrh3_positions:
                        cdrh3_interactions_to_ag.append(interaction)
                        if is_gt_interaction:
                            cdrh3_interactions_to_gt.append(interaction)
                else:
                    framework_interactions.append(interaction)
    
    # Process light chain interactions (same logic)
    if light_chain_key and light_chain_key in results:
        for antigen, interactions in results[light_chain_key].items():
            for interaction in interactions:
                all_interactions.append(interaction)
                
                if not isinstance(interaction, (list, tuple)) or len(interaction) < 2:
                    continue
                    
                ag_res, ab_res = interaction
                
                if not isinstance(ab_res, (list, tuple)) or len(ab_res) < 3:
                    continue
                    
                ab_chain, ab_pos, ab_aa = ab_res
                norm_pos = normalize_pos(ab_pos)
                
                if norm_pos in all_cdr_positions:
                    epitope_model_all_cdr.append(interaction)
                    
                    is_gt_interaction = is_gt_epitope_interaction(interaction)
                    if is_gt_interaction:
                        epitope_model_gt.append(interaction)
                    
                    if norm_pos in cdrl1_positions:
                        cdrl1_interactions_to_ag.append(interaction)
                        if is_gt_interaction:
                            cdrl1_interactions_to_gt.append(interaction)
                    elif norm_pos in cdrl2_positions:
                        cdrl2_interactions_to_ag.append(interaction)
                        if is_gt_interaction:
                            cdrl2_interactions_to_gt.append(interaction)
                    elif norm_pos in cdrl3_positions:
                        cdrl3_interactions_to_ag.append(interaction)
                        if is_gt_interaction:
                            cdrl3_interactions_to_gt.append(interaction)
                else:
                    framework_interactions.append(interaction)


    print(f"DEBUG: GT epitope set: {gt_epitope_set}")
    print(f"DEBUG: All CDR interactions found: {len(epitope_model_all_cdr)}")
    print(f"DEBUG: GT CDR interactions found: {len(epitope_model_gt)}")
    print(f"DEBUG: CDRH3 interactions to GT: {cdrh3_interactions_to_gt}")
    print(f"DEBUG: All antigen residues contacted by CDRs: {set((ag[0], str(ag[1])) for ag, ab in epitope_model_all_cdr)}")
    print(f"DEBUG: GT epitope residues contacted by CDRs: {set((ag[0], str(ag[1])) for ag, ab in epitope_model_gt)}")
        
    # Remove duplicates
    epitope_model_all_cdr = dedup_interactions(epitope_model_all_cdr)
    epitope_model_gt = dedup_interactions(epitope_model_gt)
    all_interactions = dedup_interactions(all_interactions)
    framework_interactions = dedup_interactions(framework_interactions)
    
    # Deduplicate CDR-specific lists
    cdr_lists = [cdrh1_interactions_to_ag, cdrh2_interactions_to_ag, cdrh3_interactions_to_ag,
                 cdrl1_interactions_to_ag, cdrl2_interactions_to_ag, cdrl3_interactions_to_ag]
    
    cdr_gt_lists = [cdrh1_interactions_to_gt, cdrh2_interactions_to_gt, cdrh3_interactions_to_gt,
                    cdrl1_interactions_to_gt, cdrl2_interactions_to_gt, cdrl3_interactions_to_gt]
    
    for i, cdr_list in enumerate(cdr_lists):
        cdr_lists[i] = dedup_interactions(cdr_list)
    
    for i, cdr_gt_list in enumerate(cdr_gt_lists):
        cdr_gt_lists[i] = dedup_interactions(cdr_gt_list)
    
    # Update the individual lists
    (cdrh1_interactions_to_ag, cdrh2_interactions_to_ag, cdrh3_interactions_to_ag,
     cdrl1_interactions_to_ag, cdrl2_interactions_to_ag, cdrl3_interactions_to_ag) = cdr_lists
    
    (cdrh1_interactions_to_gt, cdrh2_interactions_to_gt, cdrh3_interactions_to_gt,
     cdrl1_interactions_to_gt, cdrl2_interactions_to_gt, cdrl3_interactions_to_gt) = cdr_gt_lists
    
    # Calculate CDR lengths
    cdrh1_len = len(cdr_dict.get(heavy_chain_key, {}).get('CDR1', []))
    cdrh2_len = len(cdr_dict.get(heavy_chain_key, {}).get('CDR2', []))
    cdrh3_len = len(cdr_dict.get(heavy_chain_key, {}).get('CDR3', []))
    
    if cdrh1_len == 0:
        cdrh1_len = len(item.get("cdrh1_seq", ""))
    if cdrh2_len == 0:
        cdrh2_len = len(item.get("cdrh2_seq", ""))
    if cdrh3_len == 0:
        cdrh3_len = len(item.get("cdrh3_seq", ""))
    
    # Calculate involvement based on ALL interactions (for reference)
    cdrh1_involvement_all = (len(cdrh1_interactions_to_ag)/cdrh1_len)*100 if cdrh1_len > 0 else 0.0
    cdrh2_involvement_all = (len(cdrh2_interactions_to_ag)/cdrh2_len)*100 if cdrh2_len > 0 else 0.0
    cdrh3_involvement_all = (len(cdrh3_interactions_to_ag)/cdrh3_len)*100 if cdrh3_len > 0 else 0.0
    
    # Calculate involvement based on GT epitope interactions (NEW PRIMARY METRIC)
    cdrh1_involvement_gt = (len(cdrh1_interactions_to_gt)/cdrh1_len)*100 if cdrh1_len > 0 else 0.0
    cdrh2_involvement_gt = (len(cdrh2_interactions_to_gt)/cdrh2_len)*100 if cdrh2_len > 0 else 0.0
    cdrh3_involvement_gt = (len(cdrh3_interactions_to_gt)/cdrh3_len)*100 if cdrh3_len > 0 else 0.0
    
    total_avg_cdrh_involvement_all = float((cdrh1_involvement_all + cdrh2_involvement_all + cdrh3_involvement_all) / 3)
    total_avg_cdrh_involvement_gt = float((cdrh1_involvement_gt + cdrh2_involvement_gt + cdrh3_involvement_gt) / 3)
    
    # Light chain calculations (similar logic)
    cdrl1_len = cdrl2_len = cdrl3_len = 0
    cdrl1_involvement_all = cdrl2_involvement_all = cdrl3_involvement_all = 0.0
    cdrl1_involvement_gt = cdrl2_involvement_gt = cdrl3_involvement_gt = 0.0
    total_avg_cdrl_involvement_all = total_avg_cdrl_involvement_gt = 0.0
    
    if light_chain_key:
        cdrl1_len = len(cdr_dict.get(light_chain_key, {}).get('CDR1', []))
        cdrl2_len = len(cdr_dict.get(light_chain_key, {}).get('CDR2', []))
        cdrl3_len = len(cdr_dict.get(light_chain_key, {}).get('CDR3', []))
        
        if cdrl1_len == 0:
            cdrl1_len = len(item.get("cdrl1_seq", ""))
        if cdrl2_len == 0:
            cdrl2_len = len(item.get("cdrl2_seq", ""))
        if cdrl3_len == 0:
            cdrl3_len = len(item.get("cdrl3_seq", ""))
            
        cdrl1_involvement_all = (len(cdrl1_interactions_to_ag)/cdrl1_len)*100 if cdrl1_len > 0 else 0.0
        cdrl2_involvement_all = (len(cdrl2_interactions_to_ag)/cdrl2_len)*100 if cdrl2_len > 0 else 0.0
        cdrl3_involvement_all = (len(cdrl3_interactions_to_ag)/cdrl3_len)*100 if cdrl3_len > 0 else 0.0
        
        cdrl1_involvement_gt = (len(cdrl1_interactions_to_gt)/cdrl1_len)*100 if cdrl1_len > 0 else 0.0
        cdrl2_involvement_gt = (len(cdrl2_interactions_to_gt)/cdrl2_len)*100 if cdrl2_len > 0 else 0.0
        cdrl3_involvement_gt = (len(cdrl3_interactions_to_gt)/cdrl3_len)*100 if cdrl3_len > 0 else 0.0
        
        total_avg_cdrl_involvement_all = float((cdrl1_involvement_all + cdrl2_involvement_all + cdrl3_involvement_all) / 3)
        total_avg_cdrl_involvement_gt = float((cdrl1_involvement_gt + cdrl2_involvement_gt + cdrl3_involvement_gt) / 3)
    
    # Print summary with both metrics
    total_interactions = len(all_interactions)
    cdr_interactions_all = len(epitope_model_all_cdr)
    cdr_interactions_gt = len(epitope_model_gt)
    framework_count = len(framework_interactions)
    
    print(f"Total interactions: {total_interactions}, CDR (all): {cdr_interactions_all}, CDR (GT): {cdr_interactions_gt}, Framework: {framework_count}")
    print(f"=== CDR Involvement with ALL antigen residues ===")
    print(f"CDRH1: {len(cdrh1_interactions_to_ag)}/{cdrh1_len} = {cdrh1_involvement_all:.1f}%")
    print(f"CDRH2: {len(cdrh2_interactions_to_ag)}/{cdrh2_len} = {cdrh2_involvement_all:.1f}%")
    print(f"CDRH3: {len(cdrh3_interactions_to_ag)}/{cdrh3_len} = {cdrh3_involvement_all:.1f}%")
    
    if gt_epitope_set:
        print(f"=== CDR Involvement with GT EPITOPE residues ===")
        print(f"CDRH1: {len(cdrh1_interactions_to_gt)}/{cdrh1_len} = {cdrh1_involvement_gt:.1f}%")
        print(f"CDRH2: {len(cdrh2_interactions_to_gt)}/{cdrh2_len} = {cdrh2_involvement_gt:.1f}%")
        print(f"CDRH3: {len(cdrh3_interactions_to_gt)}/{cdrh3_len} = {cdrh3_involvement_gt:.1f}%")
    
    # Update item with both sets of metrics
    # CHANGED: Now item["epitope"] contains only GT epitope interactions
    item["epitope"] = epitope_model_gt  # GT epitope interactions only
    item["epitope_all_cdr"] = epitope_model_all_cdr  # All CDR interactions (for reference)
    item["all_interactions"] = all_interactions
    item["framework_interactions"] = framework_interactions
    
    # All antigen interactions
    item["cdrh1_interactions_to_ag"] = cdrh1_interactions_to_ag
    item["cdrh2_interactions_to_ag"] = cdrh2_interactions_to_ag
    item["cdrh3_interactions_to_ag"] = cdrh3_interactions_to_ag
    item["total_avg_cdrh_involvement_all"] = total_avg_cdrh_involvement_all
    item["cdrh3_avg_all"] = cdrh3_involvement_all
    item["cdrh2_avg_all"] = cdrh2_involvement_all
    item["cdrh1_avg_all"] = cdrh1_involvement_all
    
    # GT epitope interactions (NEW PRIMARY METRICS)
    item["cdrh1_interactions_to_gt"] = cdrh1_interactions_to_gt
    item["cdrh2_interactions_to_gt"] = cdrh2_interactions_to_gt
    item["cdrh3_interactions_to_gt"] = cdrh3_interactions_to_gt
    item["total_avg_cdrh_involvement"] = total_avg_cdrh_involvement_gt
    item["cdrh3_avg"] = cdrh3_involvement_gt
    item["cdrh2_avg"] = cdrh2_involvement_gt
    item["cdrh1_avg"] = cdrh1_involvement_gt

    if light_chain_key:
        # All antigen interactions
        item["cdrl1_interactions_to_ag"] = cdrl1_interactions_to_ag
        item["cdrl2_interactions_to_ag"] = cdrl2_interactions_to_ag
        item["cdrl3_interactions_to_ag"] = cdrl3_interactions_to_ag
        item["total_avg_cdrl_involvement_all"] = total_avg_cdrl_involvement_all
        item["cdrl3_avg_all"] = cdrl3_involvement_all
        item["cdrl2_avg_all"] = cdrl2_involvement_all
        item["cdrl1_avg_all"] = cdrl1_involvement_all
        
        # GT epitope interactions
        item["cdrl1_interactions_to_gt"] = cdrl1_interactions_to_gt
        item["cdrl2_interactions_to_gt"] = cdrl2_interactions_to_gt
        item["cdrl3_interactions_to_gt"] = cdrl3_interactions_to_gt
        item["total_avg_cdrl_involvement_gt"] = total_avg_cdrl_involvement_gt
        item["cdrl3_avg_gt"] = cdrl3_involvement_gt
        item["cdrl2_avg_gt"] = cdrl2_involvement_gt
        item["cdrl1_avg_gt"] = cdrl1_involvement_gt
    
    return item


def get_cdr_residues_and_interactions(item, cdr_dict, results):
    """
    Process CDR residues and interaction results to extract epitope information
    and calculate CDR involvement metrics. ONLY CDR interactions are included in epitope.
    """
    # Get chain identifiers
    heavy_chain = item.get("heavy_chain")
    light_chain = item.get("light_chain")
    
    # Identify which keys in cdr_dict correspond to heavy and light chains
    heavy_chain_key = None
    light_chain_key = None
    
    if heavy_chain in cdr_dict:
        heavy_chain_key = heavy_chain
    if light_chain in cdr_dict:
        light_chain_key = light_chain
    
    if not heavy_chain_key and heavy_chain:
        for key in cdr_dict:
            if key in results and 'CDR3' in cdr_dict[key]:
                heavy_chain_key = key
                break
    
    if not light_chain_key and light_chain:
        for key in cdr_dict:
            if key != heavy_chain_key and key in results:
                light_chain_key = key
                break
    
    print(f"Identified heavy chain key: {heavy_chain_key}, light chain key: {light_chain_key}")
    
    # Initialize interaction lists
    epitope_model = []  # ONLY CDR interactions
    all_interactions = []  # All interactions for reference
    framework_interactions = []  # Framework interactions
    
    cdrh1_interactions_to_ag = []
    cdrh2_interactions_to_ag = []
    cdrh3_interactions_to_ag = []
    cdrl1_interactions_to_ag = []
    cdrl2_interactions_to_ag = []
    cdrl3_interactions_to_ag = []
    
    def normalize_pos(pos):
        return str(pos)
    
    # Create position sets for CDRs
    cdrh1_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(heavy_chain_key, {}).get('CDR1', [])}
    cdrh2_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(heavy_chain_key, {}).get('CDR2', [])}
    cdrh3_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(heavy_chain_key, {}).get('CDR3', [])}
    
    cdrl1_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(light_chain_key, {}).get('CDR1', [])} if light_chain_key else set()
    cdrl2_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(light_chain_key, {}).get('CDR2', [])} if light_chain_key else set()
    cdrl3_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(light_chain_key, {}).get('CDR3', [])} if light_chain_key else set()
    
    # Combine all CDR positions for quick lookup
    all_cdr_positions = cdrh1_positions | cdrh2_positions | cdrh3_positions | cdrl1_positions | cdrl2_positions | cdrl3_positions
    
    # Process heavy chain interactions
    if heavy_chain_key and heavy_chain_key in results:
        for antigen, interactions in results[heavy_chain_key].items():
            for interaction in interactions:
                all_interactions.append(interaction)
                
                # Check if interaction has the expected format
                if not isinstance(interaction, (list, tuple)) or len(interaction) < 2:
                    continue
                    
                ag_res, ab_res = interaction
                
                # Check if ab_res has the expected format
                if not isinstance(ab_res, (list, tuple)) or len(ab_res) < 3:
                    continue
                    
                ab_chain, ab_pos, ab_aa = ab_res
                norm_pos = normalize_pos(ab_pos)
                
                # Check if this is a CDR interaction
                if norm_pos in all_cdr_positions:
                    epitope_model.append(interaction)  # ONLY add CDR interactions
                    
                    # Assign to specific CDR
                    if norm_pos in cdrh1_positions:
                        cdrh1_interactions_to_ag.append(interaction)
                    elif norm_pos in cdrh2_positions:
                        cdrh2_interactions_to_ag.append(interaction)
                    elif norm_pos in cdrh3_positions:
                        cdrh3_interactions_to_ag.append(interaction)
                else:
                    framework_interactions.append(interaction)
    
    # Process light chain interactions
    if light_chain_key and light_chain_key in results:
        for antigen, interactions in results[light_chain_key].items():
            for interaction in interactions:
                all_interactions.append(interaction)
                
                # Check if interaction has the expected format
                if not isinstance(interaction, (list, tuple)) or len(interaction) < 2:
                    continue
                    
                ag_res, ab_res = interaction
                
                # Check if ab_res has the expected format
                if not isinstance(ab_res, (list, tuple)) or len(ab_res) < 3:
                    continue
                    
                ab_chain, ab_pos, ab_aa = ab_res
                norm_pos = normalize_pos(ab_pos)
                
                if norm_pos in all_cdr_positions:
                    epitope_model.append(interaction)
                    
                    if norm_pos in cdrl1_positions:
                        cdrl1_interactions_to_ag.append(interaction)
                    elif norm_pos in cdrl2_positions:
                        cdrl2_interactions_to_ag.append(interaction)
                    elif norm_pos in cdrl3_positions:
                        cdrl3_interactions_to_ag.append(interaction)
                else:
                    framework_interactions.append(interaction)
    
    # Remove duplicates using the module-level function
    epitope_model = dedup_interactions(epitope_model)
    all_interactions = dedup_interactions(all_interactions)
    framework_interactions = dedup_interactions(framework_interactions)
    
    # Deduplicate CDR-specific lists
    cdr_lists = [cdrh1_interactions_to_ag, cdrh2_interactions_to_ag, cdrh3_interactions_to_ag,
                 cdrl1_interactions_to_ag, cdrl2_interactions_to_ag, cdrl3_interactions_to_ag]
    
    for i, cdr_list in enumerate(cdr_lists):
        cdr_lists[i] = dedup_interactions(cdr_list)
    
    # Update the individual lists
    (cdrh1_interactions_to_ag, cdrh2_interactions_to_ag, cdrh3_interactions_to_ag,
     cdrl1_interactions_to_ag, cdrl2_interactions_to_ag, cdrl3_interactions_to_ag) = cdr_lists
    
    # Calculate CDR lengths and involvement
    cdrh1_len = len(cdr_dict.get(heavy_chain_key, {}).get('CDR1', []))
    cdrh2_len = len(cdr_dict.get(heavy_chain_key, {}).get('CDR2', []))
    cdrh3_len = len(cdr_dict.get(heavy_chain_key, {}).get('CDR3', []))
    
    if cdrh1_len == 0:
        cdrh1_len = len(item.get("cdrh1_seq", ""))
    if cdrh2_len == 0:
        cdrh2_len = len(item.get("cdrh2_seq", ""))
    if cdrh3_len == 0:
        cdrh3_len = len(item.get("cdrh3_seq", ""))
    
    cdrh1_involvement = (len(cdrh1_interactions_to_ag)/cdrh1_len)*100 if cdrh1_len > 0 else 0.0
    cdrh2_involvement = (len(cdrh2_interactions_to_ag)/cdrh2_len)*100 if cdrh2_len > 0 else 0.0
    cdrh3_involvement = (len(cdrh3_interactions_to_ag)/cdrh3_len)*100 if cdrh3_len > 0 else 0.0
    
    total_avg_cdrh_involvement = float((cdrh1_involvement + cdrh2_involvement + cdrh3_involvement) / 3)
    
    # Light chain calculations
    cdrl1_len = cdrl2_len = cdrl3_len = 0
    cdrl1_involvement = cdrl2_involvement = cdrl3_involvement = 0.0
    total_avg_cdrl_involvement = 0.0
    
    if light_chain_key:
        cdrl1_len = len(cdr_dict.get(light_chain_key, {}).get('CDR1', []))
        cdrl2_len = len(cdr_dict.get(light_chain_key, {}).get('CDR2', []))
        cdrl3_len = len(cdr_dict.get(light_chain_key, {}).get('CDR3', []))
        
        if cdrl1_len == 0:
            cdrl1_len = len(item.get("cdrl1_seq", ""))
        if cdrl2_len == 0:
            cdrl2_len = len(item.get("cdrl2_seq", ""))
        if cdrl3_len == 0:
            cdrl3_len = len(item.get("cdrl3_seq", ""))
            
        cdrl1_involvement = (len(cdrl1_interactions_to_ag)/cdrl1_len)*100 if cdrl1_len > 0 else 0.0
        cdrl2_involvement = (len(cdrl2_interactions_to_ag)/cdrl2_len)*100 if cdrl2_len > 0 else 0.0
        cdrl3_involvement = (len(cdrl3_interactions_to_ag)/cdrl3_len)*100 if cdrl3_len > 0 else 0.0
        
        total_avg_cdrl_involvement = float((cdrl1_involvement + cdrl2_involvement + cdrl3_involvement) / 3)
    
    # Print summary
    total_interactions = len(all_interactions)
    cdr_interactions = len(epitope_model)
    framework_count = len(framework_interactions)
    
    print(f"Total interactions: {total_interactions}, CDR: {cdr_interactions}, Framework: {framework_count}")
    print(f"CDRH1: {len(cdrh1_interactions_to_ag)}/{cdrh1_len} = {cdrh1_involvement:.1f}%")
    print(f"CDRH2: {len(cdrh2_interactions_to_ag)}/{cdrh2_len} = {cdrh2_involvement:.1f}%")
    print(f"CDRH3: {len(cdrh3_interactions_to_ag)}/{cdrh3_len} = {cdrh3_involvement:.1f}%")
    
    # Update item
    item["epitope"] = epitope_model  # CDR-only epitope
    item["all_interactions"] = all_interactions
    item["framework_interactions"] = framework_interactions
    
    item["cdrh1_interactions_to_ag"] = cdrh1_interactions_to_ag
    item["cdrh2_interactions_to_ag"] = cdrh2_interactions_to_ag
    item["cdrh3_interactions_to_ag"] = cdrh3_interactions_to_ag
    item["total_avg_cdrh_involvement"] = total_avg_cdrh_involvement
    item["cdrh3_avg"] = cdrh3_involvement
    item["cdrh2_avg"] = cdrh2_involvement
    item["cdrh1_avg"] = cdrh1_involvement
    
    if light_chain_key:
        item["cdrl1_interactions_to_ag"] = cdrl1_interactions_to_ag
        item["cdrl2_interactions_to_ag"] = cdrl2_interactions_to_ag
        item["cdrl3_interactions_to_ag"] = cdrl3_interactions_to_ag
        item["total_avg_cdrl_involvement"] = total_avg_cdrl_involvement
        item["cdrl3_avg"] = cdrl3_involvement
        item["cdrl2_avg"] = cdrl2_involvement
        item["cdrl1_avg"] = cdrl1_involvement
    
    return item



def validate_insertion_code_handling():
    """
    Test function to validate insertion code handling across different scenarios.
    """
    test_cases = [
        ("38", "38", True),
        ("38A", "38A", True),
        ("38A", "38B", False),
        ("38", "38A", False),
        (38, "38", True),
        ("100A", "100B", False),
        ("100A", "100a", True),  # Case insensitive
        ("-1", "-1", True),
        ("-1A", "-1A", True),
    ]
    
    print("Testing insertion code comparison:")
    for pos1, pos2, expected in test_cases:
        result = compare_positions_robust(pos1, pos2)
        status = "✅" if result == expected else "❌"
        print(f"{status} {pos1} vs {pos2}: {result} (expected {expected})")


# Improved version of the position set creation in your CDR functions
def create_cdr_position_sets(cdr_dict, heavy_chain_key, light_chain_key):
    """
    Create position sets for CDRs with proper insertion code handling.
    """
    # Heavy chain CDR positions
    cdrh1_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(heavy_chain_key, {}).get('CDR1', [])}
    cdrh2_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(heavy_chain_key, {}).get('CDR2', [])}
    cdrh3_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(heavy_chain_key, {}).get('CDR3', [])}
    
    # Light chain CDR positions (if exists)
    cdrl1_positions = set()
    cdrl2_positions = set()
    cdrl3_positions = set()
    
    if light_chain_key:
        cdrl1_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(light_chain_key, {}).get('CDR1', [])}
        cdrl2_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(light_chain_key, {}).get('CDR2', [])}
        cdrl3_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(light_chain_key, {}).get('CDR3', [])}
    
    # Combine all CDR positions for quick lookup
    all_cdr_positions = (cdrh1_positions | cdrh2_positions | cdrh3_positions | 
                        cdrl1_positions | cdrl2_positions | cdrl3_positions)
    
    return {
        'cdrh1': cdrh1_positions,
        'cdrh2': cdrh2_positions, 
        'cdrh3': cdrh3_positions,
        'cdrl1': cdrl1_positions,
        'cdrl2': cdrl2_positions,
        'cdrl3': cdrl3_positions,
        'all': all_cdr_positions
    }


def debug_position_matching(interaction, cdr_positions):
    """
    Debug function to help identify position matching issues.
    """
    if not isinstance(interaction, (list, tuple)) or len(interaction) < 2:
        print(f"Invalid interaction format: {interaction}")
        return False
        
    ag_res, ab_res = interaction
    
    if not isinstance(ab_res, (list, tuple)) or len(ab_res) < 3:
        print(f"Invalid antibody residue format: {ab_res}")
        return False
        
    ab_chain, ab_pos, ab_aa = ab_res
    norm_pos = normalize_pos(ab_pos)
    
    print(f"Checking position {ab_pos} (normalized: {norm_pos}) in chain {ab_chain}")
    
    # Check which CDR this position belongs to
    for cdr_name, positions in cdr_positions.items():
        if cdr_name != 'all' and norm_pos in positions:
            print(f"  Found in {cdr_name}: {positions}")
            return True
    
    if norm_pos in cdr_positions['all']:
        print(f"  Found in combined CDR positions")
        return True
    else:
        print(f"  Not found in any CDR positions")
        print(f"  Available positions: {sorted(list(cdr_positions['all']))}")
        return False


# Additional validation for your existing functions
def validate_pdb_extraction(pdb_file, target_chain):
    """
    Validate that PDB extraction properly handles insertion codes.
    """
    print(f"Validating PDB extraction for chain {target_chain}")
    pdb_info = extract_pdb_info(pdb_file, target_chain)
    
    # Check for insertion codes
    insertion_codes_found = []
    for chain_id, res_id, res_name in pdb_info:
        if any(c.isalpha() for c in str(res_id)):  # Contains letters (insertion codes)
            insertion_codes_found.append((chain_id, res_id, res_name))
    
    if insertion_codes_found:
        print(f"Found {len(insertion_codes_found)} residues with insertion codes:")
        for chain_id, res_id, res_name in insertion_codes_found[:10]:  # Show first 10
            print(f"  {chain_id}:{res_id}:{res_name}")
        if len(insertion_codes_found) > 10:
            print(f"  ... and {len(insertion_codes_found) - 10} more")
    else:
        print("No insertion codes found in this chain")
    
    return pdb_info, insertion_codes_found


def interacting_residues(item, pdb_file, immuno_chain, antigen, tmp_dir):
    out_dir = tmp_dir



    # Get the path from environment variable
    nanobodies_project_dir = os.environ.get('NANOBODIES_PROJECT_DIR')
    if not nanobodies_project_dir:
        raise EnvironmentError("NANOBODIES_PROJECT_DIR environment variable not set")
    
    dr_sasa_path = os.path.join(nanobodies_project_dir, "dr_sasa_n", "build", "dr_sasa")


    if not os.path.exists(dr_sasa_path):
            raise FileNotFoundError(f"dr_sasa executable not found at: {dr_sasa_path}, add path manually")
    

    # command = ["./NanobodiesProject/dr_sasa_n/build/dr_sasa", "-m", "4", "-i", pdb_file]
    command = [dr_sasa_path, "-m", "4", "-i", pdb_file]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmp_dir)
    stdout, stderr = process.communicate()


    if process.returncode != 0:
        print("Command failed!")
        print("Error message:", stderr.decode())
        traceback.print_exc() 
        return None

    # Construct the output tsv file from SASA analysis
    folder_path, file_name = os.path.split(pdb_file)
    file_name = os.path.splitext(file_name)[0]

    results_tsv_file = os.path.join(out_dir, file_name + f".{immuno_chain}_vs_{antigen}.by_res.tsv")

    # If the file we need does not exist, we cannot proceed
    if not os.path.exists(results_tsv_file):
        clean_up_files(out_dir, file_name)
        return None

    # Load the TSV file into a DataFrame
    try:
        df = pd.read_csv(results_tsv_file, sep='\t', index_col=0)
    except Exception as e:
        clean_up_files(out_dir, file_name)
        return None

    non_zero_values = df[df != 0].stack()
    if non_zero_values.empty:
        clean_up_files(out_dir, file_name)
        return None

    # Define the position comparison function
    def compare_positions(pos1, pos2):  #_robust
        """
        Enhanced position comparison that handles insertion codes properly.
        Returns True if they represent the same position.
        """
        pos1_str = normalize_pos(pos1)
        pos2_str = normalize_pos(pos2)
        
        # If positions are identical strings
        if pos1_str == pos2_str:
            return True
        
        # Extract numeric and insertion code parts
        def extract_parts(pos):
            pos = str(pos).strip()
            num_part = ''
            ins_part = ''
            for c in pos:
                if c.isdigit() or c == '-':  # Handle negative numbers
                    num_part += c
                else:
                    ins_part += c
            return int(num_part) if num_part else 0, ins_part.upper()  # Normalize case
        
        try:
            num1, ins1 = extract_parts(pos1_str)
            num2, ins2 = extract_parts(pos2_str)
            
            # Compare numeric parts first, then insertion codes
            return num1 == num2 and ins1 == ins2
        except ValueError:
            # If conversion fails, fall back to string comparison
            return pos1_str == pos2_str

    # Extract row and column indices of non-zero values
    non_zero_indices = non_zero_values.index.tolist()

    # Create a set to keep track of unique positions
    unique_positions = set()

    contacts = []
    contacts_more_info = []
    contacts_complete_info = []
    
    # Iterate through each non-zero index
    for index in non_zero_indices:
        row, col = index

        # Check if row and col are strings
        if not isinstance(row, str) or not isinstance(col, str):
            continue
        
        row_aa, row_chain, row_pos = row.split('/')
        col_aa, col_chain, col_pos = col.split('/')

        # Check if position is valid - contains at least one digit
        if not any(c.isdigit() for c in row_pos) or not any(c.isdigit() for c in col_pos):
            continue

        # Check if we've already seen these positions using our compare_positions function
        position_already_seen = False
        for seen_chain, seen_pos in unique_positions:
            if ((row_chain == seen_chain and compare_positions(row_pos, seen_pos)) or 
                (col_chain == seen_chain and compare_positions(col_pos, seen_pos))):
                position_already_seen = True
                break
                
        if position_already_seen:
            continue

        if df.at[row, col] < 1.0:
            continue
    
        row_aa_code = aa_dict.get(row_aa, row_aa)
        col_aa_code = aa_dict.get(col_aa, col_aa)
        
        # Keep positions as strings throughout
        contact_tuple = [(row_chain, row_pos), (col_chain, col_pos)]
        contacts.append(contact_tuple)

        tup_info_extra = (row_chain, row_pos, row_aa_code)
        contacts_more_info.append(tup_info_extra)

        tup_info_extra = (col_chain, col_pos, col_aa_code)
        contacts_more_info.append(tup_info_extra)

        contact_tuple_more_info = [(row_chain, row_pos, row_aa_code), (col_chain, col_pos, col_aa_code)]
        contacts_complete_info.append(contact_tuple_more_info)

        # Add the positions to the set - keep as strings
        unique_positions.add((row_chain, row_pos))
        unique_positions.add((col_chain, col_pos))

    if len(contacts) == 0:
        clean_up_files(out_dir, file_name)
        return None

    clean_up_files(out_dir, file_name)

    # print(contacts_complete_info)
    
    return contacts_complete_info

