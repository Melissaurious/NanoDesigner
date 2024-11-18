#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import shutil

import torch

from utils.time_sign import get_time_sign
import numpy as np

FILE_DIR = os.path.abspath(os.path.split(__file__)[0])
MODULE_DIR = os.path.join(FILE_DIR, 'ddg')
from .ddg.utils.misc import *
from .ddg.utils.data import *
from .ddg.utils.protein import *
from configs import FOLDX_BIN, CACHE_DIR


def foldx_minimize_energy(pdb_path, out_path=None):
    filename = get_time_sign() + os.path.basename(os.path.splitext(pdb_path)[0]) + '_foldx.pdb'
    tmpfile = os.path.join(CACHE_DIR, filename)
    shutil.copyfile(pdb_path, tmpfile)
    print("CACHE_DIR", CACHE_DIR)
    p = os.popen(f'cd {CACHE_DIR}; {FOLDX_BIN} --command=Optimize --pdb={filename}')
    p.read()
    p.close()
    os.remove(tmpfile)
    filename = 'Optimized_' + filename
    tmpfile = os.path.join(CACHE_DIR, filename)
    if out_path is None:
        out_path = os.path.join(os.path.split(pdb_path)[0], filename)
    shutil.copyfile(tmpfile, out_path)
    os.remove(tmpfile)
    return out_path


def foldx_dg(pdb_path, rec_chains, lig_chains):
    filename = get_time_sign() + os.path.basename(os.path.splitext(pdb_path)[0]) + '_foldx.pdb'
    tmpfile = os.path.join(CACHE_DIR, filename)
    
    print(f"PDB Path: {pdb_path}")
    print(f"Temporary file: {tmpfile}")
    
    # Copy the PDB file to the cache directory
    try:
        shutil.copyfile(pdb_path, tmpfile)
    except Exception as e:
        print(f"Error copying file {pdb_path} to {tmpfile}: {e}")
        return np.nan
    
    rec, lig = ''.join(rec_chains), ''.join(lig_chains)

    # Run FoldX command
    command = f'cd {CACHE_DIR}; {FOLDX_BIN} --command=AnalyseComplex --pdb={filename} --analyseComplexChains={rec},{lig}'
    p = os.popen(command)
    output = p.read()
    p.close()
    

    # Find the section of interaction energies between the specified chains
    interaction_lines = output.split('\n')
    interaction_start = False
    total_energy = None
    for line in interaction_lines:
        if 'interaction between' in line and rec in line and lig in line:
            interaction_start = True
        if interaction_start and 'Total' in line:
            # Extract the total energy value
            try:
                total_energy = float(line.split('=')[-1].strip())
            except ValueError as e:
                print(f"Error parsing total energy from line '{line}': {e}")
            break
    
    # Clean up the temporary file
    try:
        os.remove(tmpfile)
    except OSError as e:
        print(f"Error deleting temporary file {tmpfile}: {e}")

    if total_energy is not None:
        return total_energy
    else:
        print("Could not find interaction energy between chains")
        return np.nan


def foldx_ddg(wt_pdb, mut_pdb, rec_chains, lig_chains):
    wt_dg = foldx_dg(wt_pdb, rec_chains, lig_chains)
    mut_dg = foldx_dg(mut_pdb, rec_chains, lig_chains)
    return wt_dg - mut_dg
