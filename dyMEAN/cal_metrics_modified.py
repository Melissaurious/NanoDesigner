import os
import json
import argparse
import sys
import string
import traceback
import concurrent 

import numpy as np
import torch
from Bio import PDB
from Bio.Align import substitution_matrices
from Bio import Align

current_working_dir = os.getcwd()
sys.path.append(current_working_dir)

from configs import CACHE_DIR, CONTACT_DIST
from utils.logger import print_log
from data.pdb_utils import VOCAB, AgAbComplex2, Protein
from evaluation.rmsd import compute_rmsd
from evaluation.tm_score import tm_score
from evaluation.lddt import lddt
from evaluation.dockq import dockq, dockq_nano


parent_dir = os.path.dirname(current_working_dir)
sys.path.append(parent_dir)
from functionalities.complex_analysis import extract_seq_info_from_pdb, extract_antibody_info, get_cdr_pos, count_clashes
from functionalities.nanobody_antibody_interacting_residues import interacting_residues

parser = PDB.PDBParser(QUIET=True)
downloader = PDB.PDBList()

def load_structure(pdb_file_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file_path)
    return structure


def get_unused_letter(used_letters):
    all_letters = set(string.ascii_uppercase)  # Get all uppercase letters
    used_letters_set = set(used_letters)  # Convert the input list to a set for faster lookup

    # Find the difference between all letters and used letters
    unused_letters = all_letters - used_letters_set

    if unused_letters:
        return unused_letters.pop()  # Return any unused letter
    else:
        raise ValueError("All letters are used")

def merge_ag_chains_for_foldX(input_pdb, output_pdb, heavy_chain, antigen_chains):
    # Create a parser object to read the PDB file
    chains_to_merge =  antigen_chains
    chains_ids_to_not_use = [heavy_chain] + antigen_chains

    # Given the IDs of the heavy and antigen chains, define a chain ID for the antigen chains different than them 
    # TO DO
    new_chain_id_antigens = get_unused_letter(chains_ids_to_not_use)

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', input_pdb)

    # Create a new structure to hold the modified model
    new_structure = PDB.Structure.Structure('new_protein')
    model = PDB.Model.Model(0)
    new_structure.add(model)

    # New chain to merge the specified chains into
    merged_chain = PDB.Chain.Chain(new_chain_id_antigens)
    residue_id_counter = 1  # Counter to ensure unique residue IDs

    for chain in structure[0]:  # Assuming only one model
        if chain.id in chains_to_merge:
            for residue in chain:
                # Create a new residue with a unique ID
                new_residue = PDB.Residue.Residue((' ', residue_id_counter, ' '), residue.resname, residue.segid)
                for atom in residue:
                    new_residue.add(atom)
                merged_chain.add(new_residue)
                residue_id_counter += 1
        else:
            model.add(chain)

    model.add(merged_chain)

    # Write the new structure to a PDB file
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb)



import uuid
def fold_x_computations(summary):

    import numpy as np
    from evaluation.pred_ddg import foldx_dg, foldx_ddg

    H,L,A = summary["heavy_chain"], summary["light_chain"], summary["antigen_chains"]
    mod_pdb = summary["mod_pdb"]
    ref_pdb = summary["ref_pdb"]

    # TO DO, FoldX cannot compute dG,ddG when more than 2 chains are involved.
    # if len(A) != 1:
    #     unique_code = str(uuid.uuid4().hex)
    #     filename = os.path.splitext(os.path.basename(mod_pdb))[0]
    #     try:
    #         tmp_mod_pdb = os.path.join(tmp_dir, f"{filename}_{unique_code}_mod.pdb")
    #         merge_ag_chains_for_foldX(mod_pdb, tmp_mod_pdb, H, A)
    #         tmp_ref_pdb = os.path.join(tmp_dir, f"{filename}_{unique_code}_ref.pdb")
    #         merge_ag_chains_for_foldX(ref_pdb, tmp_ref_pdb, H, A)

    #         try:
    #             dG_affinity_mod = foldx_dg(tmp_mod_pdb, summary["heavy_chain"], summary["antigen_chains"])
    #             summary["FoldX_dG"] = dG_affinity_mod
    #         except Exception as e:
    #             print(f"Error computing dG_affinity for {mod_pdb}: {e}")
    #             summary["FoldX_dG"] = np.nan

    #         try:

    #             fold_x_ddg = foldx_ddg(tmp_ref_pdb,summary["mod_pdb"],summary["heavy_chain"], summary["antigen_chains"])
    #             summary["FoldX_ddG"] = fold_x_ddg #float(fold_x_ddg)
    #         except Exception as e:
    #             print(f"Error computing dG_affinity for {summary['ref_pdb']}: {e}")
    #             summary["FoldX_ddG"] = np.nan
        
    #         try:
    #             os.remove(tmp_mod_pdb)
    #         except Exception as e:
    #             pass

    #         try:
    #             os.remove(tmp_ref_pdb)
    #         except Exception as e:
    #             pass
                
    #     except Exception as e:
    #         print(f"Error creating temp file for FoldX computations: {e}")
    #         summary["FoldX_ddG"] = np.nan
 
    # else:
    try:
        dG_affinity_mod = foldx_dg(mod_pdb, summary["heavy_chain"], summary["antigen_chains"])
        summary["FoldX_dG"] = dG_affinity_mod
    except Exception as e:
        print(f"Error computing dG_affinity for {mod_pdb}: {e}")
        summary["FoldX_dG"] = np.nan

    try:

        fold_x_ddg = foldx_ddg(ref_pdb,summary["mod_pdb"],summary["heavy_chain"], summary["antigen_chains"])
        summary["FoldX_ddG"] = fold_x_ddg #float(fold_x_ddg)
    except Exception as e:
        print(f"Error computing dG_affinity for {summary['ref_pdb']}: {e}")
        summary["FoldX_ddG"] = np.nan



    return summary



def metrics_and_binding_analysis(item):
    
    tmp_dir_for_interacting_aa = item["tmp_dir_for_interacting_aa"]
    antigen_chains = item["antigen_chains"]
    mod_pdb = item["mod_pdb"]
    ref_pdb = item["ref_pdb"]
    heavy_chain = item["heavy_chain"]
    light_chain = item["light_chain"]


    pdb_id = item.get("entry_id", item.get("pdb"))
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, mod_pdb)
    tuple_result = count_clashes(structure, clash_cutoff=0.63)
    num_clashes = tuple_result[0]
    item["final_num_clashes"] = num_clashes

    cdr_type = ['H3', 'H2', 'H1'] if item["cdr_type"] == 'all' else ['H3']

    try:
        pdb = Protein.from_pdb(mod_pdb, heavy_chain)
        item["heavy_chain_seq"] = ""

        for peptide_id, peptide in pdb.peptides.items():
            sequence = peptide.get_seq()
            item['heavy_chain_seq'] += sequence
        
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
            item[f'cdr{cdr_name}_pos_mod'] = cdr_pos
            start, end = cdr_pos 
            end += 1
            cdr_seq = nano_peptide.get_span(start, end).get_seq()
            item[f'cdr{cdr_name}_seq_mod'] = cdr_seq

    except Exception as e:
        # Handle exceptions if needed
        print(f'Something went wrong for {mod_pdb}, {e}')


    H, L, A = item['heavy_chain'], item['light_chain'], item['antigen_chains']
    numbering = "imgt"

    
    try:
        mod_cplx = AgAbComplex2.from_pdb(mod_pdb, H, L, A, numbering=numbering, skip_epitope_cal=True)
    except Exception as e:
        print(f"Error creating AgAbComplex for {mod_pdb}")
        print("WARNING!,Error creating AgAbComplex:", e)
        traceback.print_exc()  
        return item

    try:
        ref_cplx = AgAbComplex2.from_pdb(ref_pdb, H, L, A, numbering=numbering, skip_epitope_cal=False)
    except Exception as e:
        print(f"Error creating AgAbComplex for {ref_pdb}")
        print("WARNING!,Error creating AgAbComplex:", e)
        traceback.print_exc() 
        return item

    # 1. AAR & CAAR
    # CAAR

    try:
        epitope = ref_cplx.get_epitope()
        is_contact = []

        if cdr_type is None:  # entire antibody
            gt_s = ref_cplx.get_heavy_chain().get_seq()
            if ref_cplx.get_light_chain() is not None:
                gt_s += ref_cplx.get_light_chain().get_seq()

            pred_s = mod_cplx.get_heavy_chain().get_seq()
            if mod_cplx.get_light_chain() is not None:
                pred_s += mod_cplx.get_light_chain().get_seq()

            # contact
            for chain in [ref_cplx.get_heavy_chain(), ref_cplx.get_light_chain()]:
                if chain is not None:
                    for ab_residue in chain:
                        contact = False
                        for ag_residue, _, _ in epitope:
                            dist = ab_residue.dist_to(ag_residue)
                            if dist < CONTACT_DIST:
                                contact = True
                        is_contact.append(int(contact))
        else:
            gt_s, pred_s = '', ''
            for cdr in cdr_type:
                gt_cdr = ref_cplx.get_cdr(cdr)
                cur_gt_s = gt_cdr.get_seq()
                cur_pred_s = mod_cplx.get_cdr(cdr).get_seq()
                gt_s += cur_gt_s
                pred_s += cur_pred_s

                # contact
                cur_contact = []
                for ab_residue in gt_cdr:
                    contact = False
                    for ag_residue, _, _ in epitope:
                        dist = ab_residue.dist_to(ag_residue)
                        if dist < CONTACT_DIST:
                            contact = True
                    cur_contact.append(int(contact))
                is_contact.extend(cur_contact)

                hit, chit = 0, 0
                for a, b, contact in zip(cur_gt_s, cur_pred_s, cur_contact):
                    if a == b:
                        hit += 1
                        if contact == 1:
                            chit += 1
                item[f'AAR {cdr}'] = round(hit * 1.0 / len(cur_gt_s), 3)
                item[f'CAAR {cdr}'] = round(chit * 1.0 / (sum(cur_contact) + 1e-10), 3)

        if len(gt_s) != len(pred_s):
            print_log(f'Length conflict: {len(gt_s)} and {len(pred_s)}', level='WARN')
        hit, chit = 0, 0
        for a, b, contact in zip(gt_s, pred_s, is_contact):
            if a == b:
                hit += 1
                if contact == 1:
                    chit += 1

        item['Gt_hit'] = gt_s  # Added
        item['Pred_s_hit'] = pred_s  # Added

        for i in range(1, 4):
            cdr_name = f'H{i}'
            if cdr_name in cdr_type:  # Only process CDRs in cdr_type
                cdr_pos, cdr = ref_cplx.get_cdr_pos(cdr_name.lower()), ref_cplx.get_cdr(cdr_name)
                item[f'cdr{cdr_name.lower()}_pos_ref'] = cdr_pos
                item[f'cdr{cdr_name.lower()}_seq_ref'] = cdr.get_seq()

        for i in range(1, 4):
            cdr_name = f'H{i}'
            if cdr_name in cdr_type:  # Only process CDRs in cdr_type
                cdr_pos, cdr = mod_cplx.get_cdr_pos(cdr_name.lower()), mod_cplx.get_cdr(cdr_name)
                item[f'cdr{cdr_name.lower()}_pos_mod'] = cdr_pos
                item[f'cdr{cdr_name.lower()}_seq_mod'] = cdr.get_seq()

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

    # RMSD(CA) calculation restricted to CDRs in cdr_type
    gt_x, pred_x = [], []
    try:
        for xl, c in zip([gt_x, pred_x], [ref_cplx, mod_cplx]):
            heavy_chain = c.get_heavy_chain()
            if heavy_chain:
                for i in range(len(heavy_chain)):
                    ca_pos = heavy_chain.get_ca_pos(i)
                    if ca_pos is not None:  # Ensure valid CA position
                        xl.append(ca_pos)

            light_chain = c.get_light_chain()  # Uncommented line to include light chain
            if light_chain:
                for i in range(len(light_chain)):
                    ca_pos = light_chain.get_ca_pos(i)
                    if ca_pos is not None:  # Ensure valid CA position
                        xl.append(ca_pos)

        # Check length match and non-emptiness of arrays
        if len(gt_x) == len(pred_x) and len(gt_x) > 0:
            gt_x, pred_x = np.array(gt_x), np.array(pred_x)
            item['RMSD(CA) aligned'] = round(compute_rmsd(gt_x, pred_x, aligned=False), 3)
            item['RMSD(CA)'] = round(compute_rmsd(gt_x, pred_x, aligned=True), 3)
        else:
            raise ValueError('Mismatch in gt_x and pred_x lengths or arrays are empty')

    except AssertionError:
        traceback.print_exc()
        item['RMSD(CA) aligned'] = np.nan
        item['RMSD(CA)'] = np.nan
    except Exception as e:
        traceback.print_exc()
        item['RMSD(CA) aligned'] = np.nan
        item['RMSD(CA)'] = np.nan

    if cdr_type is not None:
        for cdr in cdr_type:
            gt_cdr, pred_cdr = ref_cplx.get_cdr(cdr), mod_cplx.get_cdr(cdr)
            
            # Only proceed if both CDRs are valid and non-empty
            if gt_cdr and pred_cdr and len(gt_cdr) == len(pred_cdr):
                gt_x = np.array([gt_cdr.get_ca_pos(i) for i in range(len(gt_cdr)) if gt_cdr.get_ca_pos(i) is not None])
                pred_x = np.array([pred_cdr.get_ca_pos(i) for i in range(len(pred_cdr)) if pred_cdr.get_ca_pos(i) is not None])

                if len(gt_x) > 0 and len(pred_x) > 0 and len(gt_x) == len(pred_x):
                    try:
                        item[f'RMSD(CA) CDR{cdr}'] = round(compute_rmsd(gt_x, pred_x, aligned=True), 3)
                    except Exception as e:
                        traceback.print_exc()
                        item[f'RMSD(CA) CDR{cdr}'] = np.nan

                    try:
                        item[f'RMSD(CA) CDR{cdr} aligned'] = round(compute_rmsd(gt_x, pred_x, aligned=False), 3)
                    except Exception as e:
                        traceback.print_exc()
                        item[f'RMSD(CA) CDR{cdr} aligned'] = np.nan
                else:
                    item[f'RMSD(CA) CDR{cdr}'] = np.nan
                    item[f'RMSD(CA) CDR{cdr} aligned'] = np.nan
            else:
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
    if L:
        try:
            score = dockq(mod_cplx, ref_cplx, cdrh3_only=True) # consistent with HERN
        except Exception as e:
            traceback.print_exc() 
            score = 0
        item['DockQ'] = score
    else:
        try:
            score = dockq_nano(mod_cplx, ref_cplx, cdrh3_only=True)
            score2 = dockq_nano(mod_cplx, ref_cplx, cdrh3_only=False)
        except Exception as e:
            traceback.print_exc() 
            score = 0
        item['DockQ'] = score
        item['DockQ_all_cdrs'] = score2

        

    # 6. Aligned Score (Added)m
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM50")

    try:
        alignments = aligner.align(gt_s, pred_s)
        for alignment in sorted(alignments)[0:1]:
            item['Aligned Score'] = alignment.score
    except Exception as e:
        print(f"An error occurred during alignment: {e}")
        traceback.print_exc() 
        # Handle the error or perform any necessary cleanup
        item['Aligned Score'] = np.nan

    try:
        mod_cplx = AgAbComplex2.from_pdb(mod_pdb, H, L, A, numbering=numbering, skip_epitope_cal=True)
        item['heavy_chain_seq_mod'] = mod_cplx.get_heavy_chain().get_seq()
        mod_seq = item['heavy_chain_seq_mod']
    except Exception as e:
        print(f"Error creating AgAbComplex for {mod_pdb}")
        print("WARNING!,Error creating mod AgAbComplex:", e)
        pass

    cdrh3_seq = item["cdrh3_seq_mod"]
    cdrh2_seq = item["cdrh2_seq_mod"]
    cdrh1_seq = item["cdrh1_seq_mod"]

    # Paratope
    lsite3 = extract_seq_info_from_pdb(mod_pdb, item["heavy_chain"], cdrh3_seq)
    lsite2 = extract_seq_info_from_pdb(mod_pdb, item["heavy_chain"], cdrh2_seq)
    lsite1 = extract_seq_info_from_pdb(mod_pdb, item["heavy_chain"], cdrh1_seq)

    binding_lsite = lsite3 + lsite2 + lsite1
    binding_lsite =[tup[:2] for tup in binding_lsite] # paratope, which comes from the current (designed) nanobody


    # Get the base name of the file
    filename = os.path.basename(mod_pdb)

    # Split the filename into name and extension
    model_name, extension = os.path.splitext(filename)

    epitope_model = []
    cdr1_interactions_to_ag = []
    cdr2_interactions_to_ag = []
    cdr3_interactions_to_ag = []

    for antigen in antigen_chains:
        # the program doesnt work with PDBs with one more than one chain in the PDB; create a temp pdb with the current pdb
        if len(antigen_chains) > 1:
            tmp_pdb = os.path.join(tmp_dir_for_interacting_aa, f"{model_name}_chain{antigen}.pdb")
            chains_to_reconstruct = []
            chains_to_reconstruct.extend(H)
            chains_to_reconstruct.extend(antigen)
            # program crashes with more than 2 chains, as we only care about cdrh involvement we will skip reconstruction of light chain in this temporal file
            # if L:
            #     chains_to_reconstruct.extend(L)
            try:
                protein = Protein.from_pdb(mod_pdb, chains_to_reconstruct)
                protein.to_pdb(tmp_pdb)
            except Exception as e:
                print(f"Failed to process PDB file '{mod_pdb}': {e}")
                continue

            result = interacting_residues(item, tmp_pdb, antigen, tmp_dir_for_interacting_aa)
            if result is not None:
                # epitope_result, cdr3_matching_res, cdr2_matching_res, cdr1_matching_res = interacting_residues(item, tmp_pdb, antigen,tmp_dir)
                epitope_result, cdr3_matching_res, cdr2_matching_res, cdr1_matching_res = result
            else:
                continue
        else:
            result = interacting_residues(item, mod_pdb, antigen, tmp_dir_for_interacting_aa)
            if result is not None:
                # epitope_result, cdr3_matching_res, cdr2_matching_res, cdr1_matching_res = interacting_residues(item, tmp_pdb, antigen,tmp_dir)
                epitope_result, cdr3_matching_res, cdr2_matching_res, cdr1_matching_res = result
            else:
                continue
        
        epitope_model.extend(epitope_result) 
        cdr3_interactions_to_ag.extend(cdr3_matching_res)
        cdr2_interactions_to_ag.extend(cdr2_matching_res)
        cdr1_interactions_to_ag.extend(cdr1_matching_res)

    cdr1_interactions_to_ag = [list(t) for t in set(tuple(i) for i in cdr1_interactions_to_ag)]
    cdr2_interactions_to_ag = [list(t) for t in set(tuple(i) for i in cdr2_interactions_to_ag)]
    cdr3_interactions_to_ag = [list(t) for t in set(tuple(i) for i in cdr3_interactions_to_ag)]
    item["cdr1_interactions_to_ag"]=cdr1_interactions_to_ag
    item["cdr2_interactions_to_ag"]=cdr2_interactions_to_ag
    item["cdr3_interactions_to_ag"]=cdr3_interactions_to_ag


    cdr3_involvement = (len(cdr3_interactions_to_ag)/len(cdrh3_seq))*100 if len(cdrh3_seq) != 0.0 else 0.0
    cdr2_involvement = (len(cdr2_interactions_to_ag)/len(cdrh2_seq))*100 if len(cdrh2_seq) != 0.0 else 0.0
    cdr1_involvement = (len(cdr1_interactions_to_ag)/len(cdrh1_seq))*100 if len(cdrh1_seq) != 0.0 else 0.0
    total_avg_cdr_involvement = float((cdr3_involvement + cdr2_involvement + cdr1_involvement) / 3)

    item["total_avg_cdr_involvement"] = total_avg_cdr_involvement
    item["cdr1_avg"]=cdr1_involvement
    item["cdr2_avg"]=cdr2_involvement
    item["cdr3_avg"]=cdr3_involvement

    epitope = item.get("gt_epitope", item.get("epitope"))
    num_aa_gt_epitope = len(epitope)

    epitope_ = [tuple(item) for item in epitope]
    epitope_ = list(set(epitope_))
    gt_epitope = [tuple(item[:2]) for item in epitope_]

    epitope_model = [tuple(element) for element in epitope_model]
    item["epitope"] = epitope_model
    epitope_model = [tuple(element[:2]) for element in epitope_model]
    epitope_model = list(set(epitope_model))

    # how much of the gt_epitope was recovered?
    epitope_model_set = set(epitope_model)
    gt_epitope_set = set(gt_epitope)

    # Find the common tuples (order may not be preserved)
    common_amino_acids_set = epitope_model_set.intersection(gt_epitope_set)
    common_amino_acids_list = list(common_amino_acids_set)

    # Calculate recall
    epitope_recall = len(common_amino_acids_list) / len(gt_epitope)

    item["epitope_recall"] = epitope_recall

    # percentage of number of aminoacids compared to eptiope
    new_epitope_num_aa = len(item.get("epitope"))

    # ratio of residues of new epitope compared to gt epitope
    # the epitope recall may be low but, a bigger surface area may had been recovered, keep track of this 
    item["new_epitope_vs_gt_epitope_aa_ratio"] = new_epitope_num_aa/num_aa_gt_epitope  

    return item




parser = argparse.ArgumentParser(description='evaluate AAR, TM-score, RMSD, LDDT')
parser.add_argument('--summary_json', type=str, required=True, help='Path to the summary in json format providing H/L/antigen')
parser.add_argument('--metrics_file', type=str, required=True, help='Path to the dataset')
parser.add_argument('--test_set', type=str, required=True, help='Path to the dataset')
parser.add_argument('--cdr_model', type=str, required=True, help='Type of model that generates CDRs')
parser.add_argument('--cdr_type', type=str, default=['H3'], help='Type of CDR',
                    choices=['H3', 'all'])

args = parser.parse_args()


def main(args):


    if not os.path.exists(args.metrics_file):
        with open(args.metrics_file, 'w') as f:
            pass

    # create temporal folder for dSASA computations
    parent_dir = os.path.dirname(os.path.dirname(args.summary_json))
    tmp_dir_dSASA = os.path.join(parent_dir, "tmp_dir_for_interacting_aa")
    os.makedirs(tmp_dir_dSASA, exist_ok=True)


    summary = []

    # Read each line as a separate JSON object
    with open(args.summary_json, 'r') as fin:
        for line_num, line in enumerate(fin, start=1):
            try:
                # Load each line as a JSON object and add the new key-value
                obj = json.loads(line.strip())
                obj["tmp_dir_for_interacting_aa"] = tmp_dir_dSASA
                summary.append(obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_num}: {e}")
                continue  # Skip this line if there's an error

        for item in summary:
            item["tmp_dir_for_interacting_aa"] = tmp_dir_dSASA


    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor: 
        futures = [executor.submit(fold_x_computations, entry) for entry in summary]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result(timeout=1500)  
                if result is not None:
                    results.append(result)
            except concurrent.futures.TimeoutError:
                print("A task exceeded the time limit and was skipped.")


    if results:
        with open(args.metrics_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + '\n')


    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(metrics_and_binding_analysis, entry) for entry in results]
        
        results_2 = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result(timeout=1000)
                if result is not None:
                    results_2.append(result)
            except concurrent.futures.TimeoutError:
                print("A task exceeded the time limit and was skipped.")
            except Exception as e:
                print(f"An error occurred: {e}")


    if results_2:
        with open(args.metrics_file, "w") as f:
            for result in results_2:
                f.write(json.dumps(result) + '\n')
    

if __name__ == "__main__":
    main(args)