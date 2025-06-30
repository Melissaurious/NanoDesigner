#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from os.path import splitext, basename
from copy import deepcopy
import json
import argparse
import time
import numpy as np
os.environ['OPENMM_CPU_THREADS'] = '4'  # prevent openmm from using all cpus available
from simtk.openmm import app
from simtk import unit
from simtk.openmm import *

from openmm import LangevinIntegrator, Platform, CustomExternalForce #, unit
from openmm.app import PDBFile, Simulation, ForceField, HBonds, Modeller
from simtk.unit import kilocalories_per_mole, angstroms,kilojoules_per_mole
# from simtk import unit
from pdbfixer import PDBFixer
from openmm import unit
import logging
logging.getLogger('openmm').setLevel(logging.ERROR)
import concurrent
from concurrent.futures import ProcessPoolExecutor
import time

import sys
sys.path.append('/ibex/user/rioszemm/NanobodiesProject/dyMEAN')

from data.pdb_utils import Peptide
from evaluation.rmsd import kabsch
from configs import CACHE_DIR, Rosetta_DIR
# from utils.time_sign import get_time_sign
import time

import threading
from functools import wraps
import uuid
from simtk.openmm import Platform
# from openmm import Platform
print([Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())])  #['Reference', 'CPU', 'CUDA', 'OpenCL']


import warnings
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning  # Import the required warning class

# Define a filter to ignore the PDBConstructionWarning
warnings.filterwarnings('ignore', category=PDBConstructionWarning)

from Bio import PDB
import numpy as np
from scipy.spatial import cKDTree

parser = PDB.PDBParser(QUIET=True)
downloader = PDB.PDBList()


FILE_DIR = os.path.abspath(os.path.split(__file__)[0])
from concurrent.futures import ProcessPoolExecutor

import subprocess
import psutil
from functools import partial

from joblib import Parallel, delayed

from datetime import datetime
import gc
import torch

from Bio import PDB
import numpy as np
from scipy.spatial import cKDTree

parser = PDB.PDBParser(QUIET=True)
downloader = PDB.PDBList()

from concurrent.futures import TimeoutError, ProcessPoolExecutor
import traceback
import multiprocessing as mp


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



def clash_analysis_hdock(item):

    if not isinstance(item, dict):
        print(item)
        print(f"Error: expected 'item' to be a dict, got {type(item).__name__} instead.")
        return None

    try:
        path_to_model = item.get("model", item.get("hdock_model"))
        base_name = os.path.basename(path_to_model)
        model_name = base_name.split('.')[0]

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure(model_name, path_to_model)
        total_clashes = count_clashes(structure, clash_cutoff=0.60)[0]

        chain_clashes = {}
        for chain in structure.get_chains():
            chain_id = chain.get_id()
            chain_clashes[chain_id] = count_clashes(chain, clash_cutoff=0.60)[0]

        inter_chain_clashes = total_clashes - sum(chain_clashes.values())

        item["Hdock_num_clashes"] = total_clashes
        item["clashes_per_chain_hdock"] = chain_clashes
        item["inter_chain_clashes_hdock"] = inter_chain_clashes

        return item

    except Exception as e:
        print(f"Error processing model {item.get('model', 'Unknown')}: {str(e)}")
        return None  # Or you could return item with an error indicator



def clash_analysis_refined(item):

    if not isinstance(item, dict):
        print(item)
        print(f"Error: expected 'item' to be a dict, got {type(item).__name__} instead.")
        return None

    try:
        path_to_model = item.get("model", item.get("hdock_model"))
        base_name = os.path.basename(path_to_model)
        model_name = base_name.split('.')[0]

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure(model_name, path_to_model)
        total_clashes = count_clashes(structure, clash_cutoff=0.60)[0]

        chain_clashes = {}
        for chain in structure.get_chains():
            chain_id = chain.get_id()
            chain_clashes[chain_id] = count_clashes(chain, clash_cutoff=0.60)[0]

        inter_chain_clashes = total_clashes - sum(chain_clashes.values())

        item["num_clashes_after_refinement"] = total_clashes
        item["clashes_per_chain_after_refinement"] = chain_clashes
        item["inter_chain_clashes_after_refinement"] = inter_chain_clashes

        return item

    except Exception as e:
        print(f"Error processing model {item.get('model', 'Unknown')}: {str(e)}")
        return None  # Or you could return item with an error indicator


def clash_analysis(item,step="Hdock"):

    if not isinstance(item, dict):
        print(item)
        print(f"Error: expected 'item' to be a dict, got {type(item).__name__} instead.")
        return None

    try:
        path_to_model = item["model"]
        base_name = os.path.basename(path_to_model)
        model_name = base_name.split('.')[0]

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure(model_name, path_to_model)
        tuple_result = count_clashes(structure, clash_cutoff=0.60)
        num_clashes = tuple_result[0]

        if step == "Hdock":
            item["Hdock_num_clashes"] = num_clashes
            return item

        if step == "refinement":
            item["num_clashes_after_refinement"] = num_clashes
            return item


    except Exception as e:
        print(f"Error processing model {item.get('model', 'Unknown')}: {str(e)}")
        return None  # Or you could return item with an error indicator



import traceback


def openmm_relax(item, platform=None, properties=None, excluded_chains=None, inverse_exclude=False):
   """
   Updated function with winning OpenMM configuration and basic memory management
   """
   import gc
   
   # Basic memory cleanup before starting
   gc.collect()
   try:
       import torch
       torch.cuda.empty_cache()
   except ImportError:
       pass  # torch not available, skip GPU cleanup
   
   # Add debug output at the start - FIXED field name
   pdb_id = item.get("hdock_model", "unknown").split("/")[-1]
   print(f"\nStarting refinement of {pdb_id}...")
   
   # FIXED: Use consistent field name
   hdock_clash_num = item.get("Hdock_num_clashes")
   print(f"Structure {pdb_id} has {hdock_clash_num} clashes")
   
   if hdock_clash_num is not None and int(hdock_clash_num) == 0:
       item['refined'] = 'No'
       print(f"No clashes detected for {pdb_id}. Skipping refinement.")
       return item

   try:
       # Get output PDB path with multiple fallback options
       out_pdb = item.get("model", item.get("hdock_model", item.get("mod_pdb")))

       if out_pdb is None:
           out_pdb = os.path.join(CACHE_DIR, 'output.pdb')
           
       print(f"Working on {out_pdb}")

       tolerance = (2.39 * unit.kilocalories_per_mole).in_units_of(unit.kilojoules_per_mole)
    #    tolerance = (2.39 * unit.kilocalories_per_mole).in_units_of(unit.kilojoules_per_mole) / unit.nanometer
       stiffness = 10.0 * unit.kilocalories_per_mole / (unit.angstroms ** 2)

       if excluded_chains is None:
           excluded_chains = []

       fixer = PDBFixer(out_pdb)
       fixer.findMissingResidues()
       fixer.findMissingAtoms()
       fixer.addMissingAtoms()
       fixer.findNonstandardResidues()
       fixer.replaceNonstandardResidues()
       
       # WINNING FORCE FIELD
       force_field = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

       modeller = Modeller(fixer.topology, fixer.positions)
       modeller.addHydrogens(force_field)
       system = force_field.createSystem(modeller.topology, constraints=HBonds)

       force = CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
       force.addGlobalParameter("k", stiffness)
       for p in ["x0", "y0", "z0"]:
           force.addPerParticleParameter(p)

       # Add flexible atoms
       for residue in modeller.topology.residues():
           if (not inverse_exclude and residue.chain.id in excluded_chains) or \
           (inverse_exclude and residue.chain.id not in excluded_chains):
               for atom in residue.atoms():
                   system.setParticleMass(atom.index, 0)
           
           for atom in residue.atoms():
               if atom.element.name != 'hydrogen':
                   force.addParticle(atom.index, modeller.positions[atom.index])

       system.addForce(force)
       integrator = LangevinIntegrator(0, 0.01, 0.0)

       # IMPROVED PLATFORM HANDLING
       if platform is None:
           try:
               platform = Platform.getPlatformByName('CUDA')
               properties = properties or {}
               properties['CudaDeviceIndex'] = properties.get('CudaDeviceIndex', '0')
               cuda_id = properties.get('CudaDeviceIndex', '0')
           except Exception as e:
               print(f"CUDA platform selection failed: {e}")
               try:
                   platform = Platform.getPlatformByName('CPU')
                   properties = None
                   cuda_id = 'CPU'
                   print("Falling back to CPU platform.")
               except Exception:
                   platform = Platform.getPlatformByName('Reference')
                   properties = None
                   cuda_id = 'CPU'
                   print("Falling back to Reference platform.")
       else:
           print(f"Platform specified: {platform.getName()}")
           cuda_id = properties.get('CudaDeviceIndex', '0') if properties else '0'

       # ROBUST SIMULATION CREATION
       try:
           simulation = Simulation(modeller.topology, system, integrator, platform, properties)
       except Exception as e:
           print(f"Error creating simulation with platform: {e}")
           print("Falling back to default platform...")
           simulation = Simulation(modeller.topology, system, integrator)
           
       simulation.context.setPositions(modeller.positions)
       simulation.minimizeEnergy(tolerance)

       state = simulation.context.getState(getPositions=True, getEnergy=True)
       
       print(f"Writing output to {out_pdb}...")
       with open(out_pdb, 'w') as fout:
           PDBFile.writeFile(simulation.topology, state.getPositions(), fout, keepIds=True)

       # Memory cleanup after simulation
       del simulation
       del system
       del modeller
       gc.collect()
       try:
           import torch
           torch.cuda.empty_cache()
       except ImportError:
           pass

       item['refined'] = 'Yes'
       print(f"Successfully refined {pdb_id}")

   except Exception as e:
       import traceback
       print("Error processing", item)
       print(f"Error processing item {item.get('entry_id', item.get('model', 'Unknown ID'))}: {str(e)}")
       print(traceback.format_exc())
       
       # Memory cleanup on error too
       gc.collect()
       try:
           import torch
           torch.cuda.empty_cache()
       except ImportError:
           pass
           
       item['refined'] = 'No'
       item['refinement_error'] = str(e)

   return item



# Add this helper function at the top of your script
def get_processed_models_count(refinement_file, entry_name):
    """Count how many models from this entry are already in the global refinement file"""
    count = 0
    if os.path.exists(refinement_file):
        try:
            with open(refinement_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and entry_name in line:
                        count += 1
        except:
            pass
    return count

def simple_entry_lock(entry_name, base_dir):
    """Simple lock mechanism with stale lock detection"""
    lock_file = os.path.join(base_dir, f".{entry_name}.lock")
    try:
        # Try to create lock file exclusively
        with open(lock_file, 'x') as f:
            f.write(f"{os.getpid()}:{time.time()}")
        return lock_file
    except FileExistsError:
        # Check if lock is stale (older than 20 minutes)
        try:
            with open(lock_file, 'r') as f:
                content = f.read().strip()
                if ':' in content:
                    pid, timestamp = content.split(':')
                    if time.time() - float(timestamp) > 1200:  # 20 minutes
                        os.unlink(lock_file)
                        return simple_entry_lock(entry_name, base_dir)  # Retry
        except:
            # If we can't read the lock, remove it
            os.unlink(lock_file)
            return simple_entry_lock(entry_name, base_dir)  # Retry
        return None

def cleanup_lock(lock_file):
    """Clean up lock file"""
    try:
        if lock_file and os.path.exists(lock_file):
            os.unlink(lock_file)
    except:
        pass

# YOUR ORIGINAL MAIN FUNCTION WITH MINIMAL CHANGES:
def main(args):
    # if os.path.exists(args.inference_summary):
    #     print(f"Inference step already conducted, thefore this step is redundant")
    #     return

    track_file1 = os.path.join(args.hdock_models, f"refinement_hdock_models_iter{args.iteration}.json")
    if not os.path.exists(track_file1):
        with open(track_file1, 'w') as f:
            pass

    items_in_hdock_models = os.listdir(args.hdock_models)
    dir_l = [item for item in items_in_hdock_models if os.path.isdir(os.path.join(args.hdock_models, item))]

    new_dir = []
    for dir_ in dir_l:
        if "tmp_dir" in dir_ or dir_.startswith('.'):  # Skip lock files too
            continue
        new_dir.append(dir_)

    dir_l_ = []
    new_l = []

    for directory in new_dir:
        full_dir_path = os.path.join(args.hdock_models, directory)
        hdock_models = [file for file in os.listdir(full_dir_path) if file.startswith('model_') and file.endswith('.pdb')]
        dir_l_.append(directory)
        new_l.append(full_dir_path)

    for index, (pdb_n, full_dir_path) in enumerate(zip(dir_l_, new_l)):

        # ADDITION 1: Simple lock mechanism
        lock_file = simple_entry_lock(pdb_n, args.hdock_models)
        if lock_file is None:
            print(f"Entry {pdb_n} is being processed by another instance, skipping...")
            continue

        try:
            # ADDITION 2: Check if already processed globally
# ADDITION 2: Check if already processed globally
            processed_count = get_processed_models_count(track_file1, pdb_n)
            
            # Your original track file logic
            track_file = os.path.join(args.hdock_models, pdb_n, "track_file.json")
            
            # If track_file is empty, ignore any previous global processing count
            if os.path.exists(track_file) and os.stat(track_file).st_size == 0:
                processed_count = 0
                print(f"Track file is empty for {pdb_n}, resetting processed_count to 0")

            if not os.path.exists(track_file):
                with open(track_file, 'w') as f:
                    pass
                print(f"Created new track file for {pdb_n}")
            elif os.path.exists(track_file) and os.stat(track_file).st_size != 0:
                print(f"Skipping {track_file} already processed, continue...")
                continue
            else:
                print(f"Track file exists but is empty for {pdb_n}, will process...")

            pdb_parts = pdb_n.rsplit('_',1)[0]
            
            hdock_models = [os.path.join(full_dir_path, file) for file in os.listdir(full_dir_path) if file.startswith('model_') and file.endswith('.pdb')]

            list_items = []
            for model in hdock_models:
                list_items.append({"model": model, "pdb_n": pdb_n})

            top_n_file = os.path.join(args.hdock_models, pdb_n, "top_models.json")
            if os.path.exists(top_n_file):
                with open(top_n_file, 'r') as f:
                    all_items = [json.loads(line) for line in f]
                
                # ADDITION 3: Skip already processed models from top_models.json
                if processed_count > 0:
                    list_items = all_items[processed_count:]
                    print(f"Using top_models.json: skipping {processed_count} already processed, processing {len(list_items)} remaining")
                else:
                    list_items = all_items
                    print(f"Using top_models.json: processing all {len(list_items)} models")
            else:
                print(f"{top_n_file} does not exist")
                
            # Skip if no models to process
            if not list_items:
                print(f"All models for {pdb_n} already processed, skipping...")
                continue
        
            # YOUR ORIGINAL PROCESSING LOGIC (unchanged)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(clash_analysis_hdock, item) for item in list_items]

                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=300)
                        if result is not None:
                            results.append(result)
                    except TimeoutError:
                        print("A task exceeded the time limit and was skipped.")
                    except Exception as e:
                        print("An error occurred:", e)
                        traceback.print_exc()

            only_clashing = [item for item in results if int(item.get("Hdock_num_clashes")) >= 1]
            no_clashes = [item for item in results if int(item.get("Hdock_num_clashes")) == 0]

            # Process models with clashes through refinement
            result_refinement = []
            for item in only_clashing:
                output = openmm_relax(item)
                result_refinement.append(output)
            
            # Add models without clashes directly (mark as not refined)
            for item in no_clashes:
                item['refined'] = 'No'
                item['num_clashes_after_refinement'] = 0
                item['clashes_per_chain_after_refinement'] = item.get('clashes_per_chain_hdock', {})
                item['inter_chain_clashes_after_refinement'] = item.get('inter_chain_clashes_hdock', 0)
                result_refinement.append(item)

            # write in the meantime all info
            with open(track_file, 'a') as f:
                for item in result_refinement:
                    f.write(json.dumps(item) + '\n')

            torch.cuda.empty_cache()
            gc.collect()

            # write in the meantime all info
            with open(track_file, 'r') as f:
                data = [json.loads(line) for line in f]

            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(clash_analysis_refined, item) for item in data]

                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=300)
                        if result is not None:
                            results.append(result)
                    except TimeoutError:
                        print("A task exceeded the time limit and was skipped.")
                    except Exception as e:
                        print("An error occurred:", e)
                        traceback.print_exc()

            with open(track_file1,'a') as f:
                for item in results:
                    f.write(json.dumps(item) + '\n')

            with open(track_file, 'w') as f:
                for entry in results:
                    f.write(json.dumps(entry) + '\n')
                    num_clashes = entry['num_clashes_after_refinement']
            
            # Create list of remaining files 
            remaining_hdock_models = [os.path.join(full_dir_path, file) for file in os.listdir(full_dir_path) if file.startswith('model_') and file.endswith('.pdb')]

            top_n_plus_ten_file = os.path.join(args.hdock_models, pdb_n, "hdock_models.json")
            if os.path.exists(top_n_plus_ten_file):
                with open(top_n_plus_ten_file,'r') as f:
                    data = [json.loads(line) for line in f]

                json_hdock_models = [item['hdock_model'] for item in data]
                top_n_files = [file for file in json_hdock_models if file in remaining_hdock_models][:int(args.top_n)]
            else:
                top_n_files = remaining_hdock_models[:int(args.top_n)]

            torch.cuda.empty_cache()
            gc.collect()

        finally:
            # ADDITION 4: Always cleanup lock
            cleanup_lock(lock_file)


def parse():
    parser = argparse.ArgumentParser(description='inference of pipeline method')
    parser.add_argument('--hdock_models', type=str, required=True, help='Path to the summary file of dataset in json format')
    parser.add_argument('--dataset_json', type=str, required=True, help='Path to the summary file of dataset in json format')
    parser.add_argument('--iteration', type=str, required=True, help='Top n docked models to select ')
    parser.add_argument('--top_n', type=str, required=True, help='Top n docked models to select ')
    parser.add_argument('--inference_summary', type=str, required=True, help='Top n docked models to select ')
    

    return parser.parse_args()

if __name__ == '__main__':
    main(parse())