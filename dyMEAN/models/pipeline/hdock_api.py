#!/usr/bin/python
# -*- coding:utf-8 -*-
import os

from configs import HDOCK_DIR, CACHE_DIR
from utils.time_sign import get_time_sign
import time

import shutil
from shutil import rmtree

HDOCK = os.path.join(HDOCK_DIR, 'hdock')
CREATEPL = os.path.join(HDOCK_DIR, 'createpl')

TMP_DIR = os.path.join(CACHE_DIR, 'hdock')
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)


def dock(pdb1, pdb2, out_folder, sample=1, binding_rsite=None, binding_lsite=None):

    start_time = time.time() 

    # working directory
    out_folder = os.path.abspath(out_folder)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    # pdb1 is the receptor (antigen), pdb2 is the ligand (antibody)
    unique_id = get_time_sign()
    tmp_file_list = []
    ori_pdb1, ori_pdb2 = pdb1, pdb2
    pdb1 = f'{unique_id}_{os.path.split(pdb1)[-1]}'
    print("path to antigen (submited to hdock)",pdb1)
    pdb2 = f'{unique_id}_{os.path.split(pdb2)[-1]}'
    print("path to antibody (submited to hdock)",pdb2)
    os.system(f'cd {out_folder}; cp {ori_pdb1} {pdb1}')
    os.system(f'cd {out_folder}; cp {ori_pdb2} {pdb2}')
    tmp_file_list.append(pdb1)
    tmp_file_list.append(pdb2)

    # binding site on antigen
    arg_rsite, rsite_name = '', f'{unique_id}_rsite.txt'
    if binding_rsite is not None:
        rsite = os.path.join(out_folder, rsite_name)
        with open(rsite, 'w') as fout:
            sites = []
            for chain_name, residue_id in binding_rsite:
                sites.append(f'{residue_id}:{chain_name}')
            fout.write(', '.join(sites))
        arg_rsite = f'-rsite {rsite_name}'
        tmp_file_list.append(rsite_name)

    # binding site on heavy chain
    arg_lsite, lsite_name = '', f'{unique_id}_lsite.txt'
    if binding_lsite is not None:
        lsite = os.path.join(out_folder, lsite_name)
        with open(lsite, 'w') as fout:
            sites = []
            for chain_name, residue_id in binding_lsite:
                sites.append(f'{residue_id}:{chain_name}')
            fout.write(', '.join(sites))
        arg_lsite = f'-lsite {lsite_name}'
        tmp_file_list.append(lsite_name)

    # dock
    dock_out = f'{unique_id}_Hdock.out'
    tmp_file_list.append(dock_out)
    p = os.popen(f'cd {out_folder}; {HDOCK} {pdb1} {pdb2} {arg_rsite} {arg_lsite} -out {dock_out}')
    p.read()
    p.close()

    p = os.popen(f'cd {out_folder}; {CREATEPL} {dock_out} top{sample}.pdb -nmax {sample} -complex -models') 
    p.read()
    p.close()

    for f in tmp_file_list:
        os.remove(os.path.join(out_folder, f))

    results = [os.path.join(out_folder, f'model_{i + 1}.pdb') for i in range(sample)]

    end_time = time.time() 
    total_time = end_time - start_time
    print(f"Hdock took: {end_time - start_time:.2f} seconds for {out_folder} results")

    return results, total_time 



import subprocess

def dock_subprocess(pdb1, pdb2, out_folder, sample=1, binding_rsite=None, binding_lsite=None):
    start_time = time.time()

    out_folder = os.path.abspath(out_folder)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    unique_id = get_time_sign()
    pdb1_path = os.path.join(out_folder, f'{unique_id}_{os.path.basename(pdb1)}')
    pdb2_path = os.path.join(out_folder, f'{unique_id}_{os.path.basename(pdb2)}')
    
    # Copy files to the working directory
    shutil.copy(pdb1, pdb1_path)
    shutil.copy(pdb2, pdb2_path)

    # Prepare binding sites if applicable
    arg_rsite, rsite_path = '', ''
    if binding_rsite is not None:
        rsite_path = os.path.join(out_folder, f'{unique_id}_rsite.txt')
        with open(rsite_path, 'w') as f:
            f.write(', '.join(f'{rid}:{chain}' for chain, rid in binding_rsite))
        arg_rsite = f'-rsite {rsite_path}'

    arg_lsite, lsite_path = '', ''
    if binding_lsite is not None:
        lsite_path = os.path.join(out_folder, f'{unique_id}_lsite.txt')
        with open(lsite_path, 'w') as f:
            f.write(', '.join(f'{rid}:{chain}' for chain, rid in binding_lsite))
        arg_lsite = f'-lsite {lsite_path}'

    # Command for docking
    
    print("Working directory:", out_folder)
    dock_out = os.path.join(out_folder, f"{unique_id}_Hdock.out")
    dock_cmd = f'{HDOCK} {pdb1_path} {pdb2_path} {arg_rsite} {arg_lsite} -out {dock_out }'
    # subprocess.run(dock_cmd, shell=True, check=True, cwd=out_folder)
    print("Executing command:", dock_cmd)
    try:
        result = subprocess.run(dock_cmd, shell=True, check=True, cwd=out_folder, text=True, capture_output=True)
        print("Command Output:", result.stdout)
        print("Command Error Output:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit status {e.returncode}")
        print("Output:", e.output)
        print("Error:", e.stderr)

    # Process the docking output to create models
    print("# Process the docking output to create models")
    createpl_cmd = f'{CREATEPL} {unique_id}_Hdock.out top{sample}.pdb -nmax {sample} -complex -models'
    subprocess.run(createpl_cmd, shell=True, check=True, cwd=out_folder)

    # Clean up temporary files
    for f in [pdb1_path, pdb2_path, rsite_path, lsite_path, f'{unique_id}_Hdock.out']:
        if os.path.exists(f):
            os.remove(f)

    end_time = time.time()
    print(f"HDOCK took: {end_time - start_time:.2f} seconds for {out_folder} results")

    # Assuming you return paths to the models
    return [os.path.join(out_folder, f'model_{i + 1}.pdb') for i in range(sample)]
