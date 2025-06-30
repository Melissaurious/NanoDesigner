#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from time import time
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import numpy as np

from data.pdb_utils import AgAbComplex, AgAbComplex_mod
from evaluation.rmsd import compute_rmsd
from evaluation.tm_score import tm_score
from evaluation.lddt import lddt
from evaluation.dockq import dockq, dockq_nano
from utils.relax import openmm_relax, rosetta_sidechain_packing
from utils.logger import print_log

from configs import CONTACT_DIST
import time

#For Aligned Score
from Bio.Align import substitution_matrices
from Bio import Align

#For creating a csv file
import csv

import subprocess
import sys
sys.path.append('/home/rioszemm/NanobodiesProject/binding_ddg_predictor')


def cal_metrics(inputs, flag="Nb"):
    start_time = time.time()
    sidechain = None
    if flag == "Ab":
        if len(inputs) == 6:
            mod_pdb, ref_pdb, H, L, A, cdr_type = inputs
            # sidechain = False
        elif len(inputs) == 7:
            mod_pdb, ref_pdb, H, L, A, cdr_type, sidechain = inputs
        do_refine = False
    elif flag == "Nb":
        if len(inputs) == 5:
            mod_pdb, ref_pdb, H, A, cdr_type = inputs
            # sidechain = False
        elif len(inputs) == 6:
            mod_pdb, ref_pdb, H, A, cdr_type, sidechain = inputs
        do_refine = False

    # sidechain packing
    if sidechain:
        start_time = time.time()
        # refined_pdb = mod_pdb[:-4] + '_sidechain.pdb'
        refined_pdb = mod_pdb[:-4] + '.pdb'
        mod_pdb = rosetta_sidechain_packing(mod_pdb, refined_pdb)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Time taken for conducting side chain packing: {execution_time} seconds")

    # load complex
    if do_refine:
        refined_pdb = mod_pdb[:-4] + '_refine.pdb'
        pdb_id = os.path.split(mod_pdb)[-1]
        print(f'{pdb_id} started refining')
        start = time()
        mod_pdb = openmm_relax(mod_pdb, refined_pdb, excluded_chains=A)  # relax clashes
        print(f'{pdb_id} finished openmm relax, elapsed {round(time() - start)} s')

    if flag == "Ab":
        try:
            mod_cplx = AgAbComplex.from_pdb(mod_pdb, H, L, A, numbering='imgt', skip_epitope_cal=True)
            ref_cplx = AgAbComplex.from_pdb(ref_pdb, H, L, A, numbering='imgt', skip_epitope_cal=False)
        except Exception as e:
            print("Error creating AgAbComplex:", e)
            return None

    if flag == "Nb":
        try:
            mod_cplx = AgAbComplex_mod.from_pdb(mod_pdb, H, A, numbering='imgt', skip_epitope_cal=True)
            ref_cplx = AgAbComplex_mod.from_pdb(ref_pdb, H, A, numbering='imgt', skip_epitope_cal=False)
        except Exception as e:
            print("Error creating AgAbComplex_mod:", e)
            return None

    results = {}
    
    cdr_type = [cdr_type] if type(cdr_type) == str else cdr_type
    # results['Model ID'] = mod_cplx.get_id() #Added
    # print("mod_cplx.pdb_id",mod_cplx.pdb_id)
    # print("mod_cplx.pdb",mod_cplx.pdb)
    results['Model ID'] = mod_cplx.pdb
    results["mod_pdb"] = mod_pdb
    results["ref_pdb"] = ref_pdb
    # Split the path by '/'
    path_parts = mod_pdb.split('/')
    results["pdb"]= path_parts[-3]



    # 1. AAR & CAAR
    # CAAR
    epitope = ref_cplx.get_epitope()
    is_contact = []
    if cdr_type is None:  # entire antibody
        # gt_s = ref_cplx.get_heavy_chain().get_seq() + ref_cplx.get_light_chain().get_seq()
        # pred_s = mod_cplx.get_heavy_chain().get_seq() + mod_cplx.get_light_chain().get_seq()
        gt_s = ref_cplx.get_heavy_chain().get_seq()
        pred_s = mod_cplx.get_heavy_chain().get_seq()
        # contact
        for chain in [ref_cplx.get_heavy_chain()]:#, ref_cplx.get_light_chain()]:
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
            results[f'AAR {cdr}'] = round(hit * 1.0 / len(cur_gt_s),3)
            results[f'CAAR {cdr}'] = round(chit * 1.0 / (sum(cur_contact) + 1e-10),3)

    if len(gt_s) != len(pred_s):
        print_log(f'Length conflict: {len(gt_s)} and {len(pred_s)}', level='WARN')
    hit, chit = 0, 0
    for a, b, contact in zip(gt_s, pred_s, is_contact):
        if a == b:
            hit += 1
            if contact == 1:
                chit += 1
    results['AAR'] = round(hit * 1.0 / len(gt_s),3)
    results['CAAR'] = round(chit * 1.0 / (sum(is_contact) + 1e-10),3)


    results['Ground truth sequence'] = gt_s #Added
    results['Predicted sequence'] = pred_s #Added

    # # 2. RMSD(CA) w/o align
    if flag == "Ab":
        gt_x, pred_x = [], []
        try:
            for xl, c in zip([gt_x, pred_x], [ref_cplx, mod_cplx]):
                for chain in [c.get_heavy_chain(), c.get_light_chain()]:
                    for i in range(len(chain)):
                        xl.append(chain.get_ca_pos(i))
            assert len(gt_x) == len(pred_x), 'coordinates length conflict'
            gt_x, pred_x = np.array(gt_x), np.array(pred_x)
            results['RMSD(CA) aligned'] = round(compute_rmsd(gt_x, pred_x, aligned=False),3)
            results['RMSD(CA)'] = round(compute_rmsd(gt_x, pred_x, aligned=True),3)
        except AssertionError:
            results['RMSD(CA) aligned'] = np.nan
            results['RMSD(CA)'] = np.nan


    # 2. RMSD(CA) w/o align

    if flag == "Nb":
        gt_x, pred_x = [], []
        try:
            for xl, c in zip([gt_x, pred_x], [ref_cplx, mod_cplx]):
                for chain in [c.get_heavy_chain()]:
                    for i in range(len(chain)):
                        xl.append(chain.get_ca_pos(i))
            assert len(gt_x) == len(pred_x), 'coordinates length conflict'
            gt_x, pred_x = np.array(gt_x), np.array(pred_x)
            results['RMSD(CA) aligned'] = round(compute_rmsd(gt_x, pred_x, aligned=False),3)
            results['RMSD(CA)'] = round(compute_rmsd(gt_x, pred_x, aligned=True),3)
        except AssertionError:
            results['RMSD(CA) aligned'] = np.nan
            results['RMSD(CA)'] = np.nan

    # results['RMSD(CA)'] = compute_rmsd(gt_x, pred_x, aligned=True)
    if cdr_type is not None:
        for cdr in cdr_type:
            gt_cdr, pred_cdr = ref_cplx.get_cdr(cdr), mod_cplx.get_cdr(cdr)
            gt_x = np.array([gt_cdr.get_ca_pos(i) for i in range(len(gt_cdr))])
            pred_x = np.array([pred_cdr.get_ca_pos(i) for i in range(len(pred_cdr))])
            try:
                results[f'RMSD(CA) CDR{cdr}'] = round(compute_rmsd(gt_x, pred_x, aligned=True),3)
            except Exception as e:
                results[f'RMSD(CA) CDR{cdr}'] = np.nan

            try:
                results[f'RMSD(CA) CDR{cdr} aligned'] = round(compute_rmsd(gt_x, pred_x, aligned=False),3)
            except Exception as e:
                results[f'RMSD(CA) CDR{cdr} aligned'] = np.nan


    # 3. TMscore
    tm_score_ = tm_score(mod_cplx.antibody, ref_cplx.antibody)
    results['TMscore'] = tm_score_


    try:
        tm_score_ = tm_score(mod_cplx.antibody, ref_cplx.antibody)
        results['TMscore'] = tm_score_
    except Exception as e:
        print(f"An error occurred during TMscore calculation: {e}")
        results['TMscore'] = np.nan

    # 4. LDDT
    try:
        score, _ = lddt(mod_cplx.antibody, ref_cplx.antibody)
        results['LDDT'] = score
    except Exception as e:
        print(f"An error occurred during LDDT calculation: {e}")
        results['LDDT'] = np.nan
        
    # 5. DockQ
    if flag == "Ab":
        try:
            score = dockq(mod_cplx, ref_cplx, cdrh3_only=True) # consistent with HERN
            # score = dockq_nano(mod_cplx, ref_cplx, cdrh3_only=True)
        except Exception as e:
            print_log(f'Error in dockq: {e}, set to 0', level='ERROR')
            score = 0
        results['DockQ'] = score

    if flag == "Nb":
        try:
            # score = dockq(mod_cplx, ref_cplx, cdrh3_only=True) # consistent with HERN
            score = dockq_nano(mod_cplx, ref_cplx, cdrh3_only=True)
        except Exception as e:
            print_log(f'Error in dockq: {e}, set to 0', level='ERROR')
            score = 0
        results['DockQ'] = score

    # print(f'{mod_cplx.get_id()}: {results}')

    # 6. Aligned Score (Added)
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM50")
    alignments = aligner.align(gt_s, pred_s)

    for alignment in sorted(alignments)[0:1]:
        #print("Aligned Score = %.1f:" % alignment.score)
        results['Aligned Score'] = alignment.score

    # 7. ddG (binding-ddg-predictor)
    start_time_ddg = time.time()

    # # # Set the environment variables for the subprocess
    # env = os.environ.copy()
    # env['PYTHONPATH'] = '/home/rioszemm/NanobodiesProject/binding_ddg_predictor'

    # # # 7. ddG (Added)
    # ddg_conda_command = "conda activate ddg-predict"
    ddg_command =["python", "/home/rioszemm/NanobodiesProject/binding_ddg_predictor/scripts/predict.py", ref_pdb, mod_pdb]
    cwd = "/home/rioszemm/NanobodiesProject/binding_ddg_predictor"

    # # Run the command and capture the output
    # result = subprocess.run(ddg_conda_command, shell=True, capture_output=True, text=True, cwd=cwd)
    result = subprocess.run(ddg_command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)

    # result = subprocess.run(ddg_conda_command, shell=True, capture_output=True, text=True, cwd=cwd, env=env)
    # result = subprocess.run(ddg_command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)

    # # Print the stderr
    print(result.stderr)

    # Extract the produced value from stdout
    output_lines = result.stdout.strip().split('\n')
    print("output_lines", output_lines)


    if output_lines:
        ddG_line = output_lines[-1].strip()  # Assuming the produced value is the last line
        print("ddG_predictor_output_last_line =",ddG_line)

        if ':' in ddG_line:
            ddG_value = ddG_line.split(':')[1].strip()  # Extract the value after the colon
            print("ddG_value:", ddG_value)  # Print the value after the colon
            
            try:
                ddG = float(ddG_value)  # Convert the value to a float if possible
                # print("ddG:", ddG)  # Print the converted value
            except ValueError:
                ddG = np.nan  # Assign np.nan if the conversion fails
                print("Conversion to float failed")
        else:
            ddG = np.nan  # Assign np.nan if the line doesn't contain the expected colon
            print("Expected colon not found in ddG_line")
    else:
        ddG = np.nan  # Assign np.nan if there are no output lines
        print("No output lines")

    results['Predicted ddG'] = ddG

    if results['Ground truth sequence'] == results['Predicted sequence']:
        results['Predicted ddG'] = 0.0


    # ddg_conda_command = "conda deactivate"
    # subprocess.run(ddg_conda_command, shell=True, capture_output=True, text=True, cwd=cwd, env=env)


    end_time_ddg = time.time()
    execution_time = end_time_ddg - start_time_ddg
    print(f"Time taken for ddG calculation: {execution_time} seconds")

    # print(f'{mod_cplx.get_id()}: {results}')
    # print(f'Used reference file {ref_pdb}')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time taken for total metrics computations: {execution_time} seconds")

    return results

#results will contain the following keys: ddG, Aligned score, DockQ, LDDT, TMscore, 'RMSD(CA) aligned', 'RMSD(CA) cdrh3', RMSD(CA) CDR3 aligned, AAR, CAAR, 'Model ID', 'Ground truth sequence' and 'Predicted sequence'

def save_to_csv(csv_file, metrics):
    file_exists = os.path.exists(csv_file)

    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics)


def run_cal_metrics(inputs, csv_file, flag ="Nb"):

    metrics, ddG = cal_metrics(inputs, flag)
    if metrics == None:
        return
    save_to_csv(csv_file, metrics)


    return metrics, ddG

