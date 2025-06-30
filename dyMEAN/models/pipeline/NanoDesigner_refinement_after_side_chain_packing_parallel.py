#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from os.path import splitext, basename
from copy import deepcopy
import json
import argparse

import numpy as np
# os.environ['OPENMM_CPU_THREADS'] = '4'  # prevent openmm from using all cpus available
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
# from .time_sign import get_time_sign


FILE_DIR = os.path.abspath(os.path.split(__file__)[0])
from concurrent.futures import ProcessPoolExecutor

import subprocess
import psutil

from joblib import Parallel, delayed

from datetime import datetime
import gc
import torch

from Bio import PDB
import numpy as np
from scipy.spatial import cKDTree

parser = PDB.PDBParser(QUIET=True)
downloader = PDB.PDBList()
import traceback


import torch  # Make sure this is at the top of your script
import gc

# Later in your code where cleanup happens:
def cleanup():
    if 'torch' in globals():  # Safety check
        torch.cuda.empty_cache()
    gc.collect()

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



def openmm_relax(item, platform=None, properties=None, excluded_chains=None, inverse_exclude=False):
    """
    Refinement function with the critical simulation initialization fixed and memory management
    """
    import gc
    
    # Basic memory cleanup before starting
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass  # torch not available, skip GPU cleanup
    
    # Add debug output at the start
    pdb_id = item.get("mod_pdb", "unknown").split("/")[-1]
    print(f"\nStarting refinement of {pdb_id}...")
    
    num_clashes = item.get("side_chain_p_num_clashes", item.get("inference_clashes"))
    print(f"Structure {pdb_id} has {num_clashes} clashes")
    
    if num_clashes is not None:
        if int(num_clashes) == 0:
            item['refined'] = 'No'
            print(f"No clashes detected for {pdb_id}. Skipping refinement.")
            return item

    try:
        out_pdb = item["mod_pdb"]
        print(f"Working on {out_pdb}")


        # tolerance = (2.39 * unit.kilocalories_per_mole).in_units_of(unit.kilojoules_per_mole) / unit.nanometer
        tolerance = (2.39 * unit.kilocalories_per_mole).in_units_of(unit.kilojoules_per_mole)
        stiffness = 10.0 * unit.kilocalories_per_mole / (unit.angstroms ** 2)

        if excluded_chains is None:
            excluded_chains = []

        fixer = PDBFixer(out_pdb)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        
        force_field = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.addHydrogens(force_field)
        system = force_field.createSystem(modeller.topology, constraints=HBonds)

        force = CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        force.addGlobalParameter("k", stiffness)
        for p in ["x0", "y0", "z0"]:
            force.addPerParticleParameter(p)

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

        if platform is None:
            try:
                platform = Platform.getPlatformByName('CUDA')
                properties = properties or {}
                properties['CudaDeviceIndex'] = properties.get('CudaDeviceIndex', '0')
            except Exception as e:
                print(f"CUDA platform selection failed: {e}")
                platform = Platform.getPlatformByName('CPU')
                properties = None
                print("Falling back to CPU platform.")
        else:
            print(f"Platform specified: {platform.getName()}")
            
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
        
        return item

    except Exception as e:
        import traceback
        print(f"Error processing item {item.get('mod_pdb', 'Unknown ID')}: {str(e)}")
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





import os
import json
import argparse
import time
import hashlib

# Add these helper functions at the top of your script
def get_item_hash(item):
    """Create a unique hash for each item based on mod_pdb path"""
    return hashlib.md5(item["mod_pdb"].encode()).hexdigest()[:8]

def simple_item_lock(item_hash, output_dir):
    """Simple lock mechanism for individual items"""
    lock_file = os.path.join(output_dir, f".{item_hash}.lock")
    try:
        # Try to create lock file exclusively
        with open(lock_file, 'x') as f:
            f.write(f"{os.getpid()}:{time.time()}")
        return lock_file
    except FileExistsError:
        # Check if lock is stale (older than 30 minutes)
        try:
            with open(lock_file, 'r') as f:
                content = f.read().strip()
                if ':' in content:
                    pid, timestamp = content.split(':')
                    if time.time() - float(timestamp) > 1800:  # 30 minutes
                        os.unlink(lock_file)
                        return simple_item_lock(item_hash, output_dir)  # Retry
        except:
            pass
        return None

def cleanup_lock(lock_file):
    """Clean up lock file"""
    try:
        if lock_file and os.path.exists(lock_file):
            os.unlink(lock_file)
    except:
        pass

def is_already_processed(mod_pdb, output_file):
    """Check if a specific mod_pdb is already in the output file"""
    if not os.path.exists(output_file) or os.stat(output_file).st_size == 0:
        return False
    
    try:
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("mod_pdb") == mod_pdb:
                        return True
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return False



def main(args):


    import gc
    import torch
    torch.cuda.empty_cache()
    gc.collect()

    output_dir = os.path.dirname(args.out_file)

    
    # Create a temporary lock directory for coordination
    lock_dir = os.path.join(output_dir, '.locks')
    os.makedirs(lock_dir, exist_ok=True)
        
    # Read input data
    print(f"Reading input data from {args.in_file}")
    with open(args.in_file, 'r') as f:
        data = [json.loads(line) for line in f]
    print(f"Successfully loaded {len(data)} items from input file")
    
    # Your existing data filtering logic
    if args.cdr_model != "dyMEAN":
        data = [item for item in data if item.get("side_chain_packed") == "Yes"]
        print(f"Filtered to {len(data)} items with side_chain_packed=Yes")
    else:
        print("Using dyMEAN mode, calculating clashes...")
        for item in data:
            try:
                pdb = item["pdb"]
                pdb_path = item["mod_pdb"]
                item["side_chain_packed"] = "No"
                structure = parser.get_structure(pdb, pdb_path)
                total_clashes = count_clashes(structure, clash_cutoff=0.60)[0]
                
                chain_clashes = {}
                for chain in structure.get_chains():
                    chain_id = chain.get_id()
                    chain_clashes[chain_id] = count_clashes(chain, clash_cutoff=0.60)[0]

                inter_chain_clashes = total_clashes - sum(chain_clashes.values())

                item["inference_clashes"] = total_clashes
                item["clashes_per_chain_inference"] = chain_clashes
                item["inter_chain_clashes_inference"] = inter_chain_clashes
                
                print(f"Structure {pdb_path}: {total_clashes} clashes detected")
            except Exception as e:
                print(f"{pdb_path} could not be refined, skipping: {e}")
    
    print(f"Total entries to refine: {len(data)}")
    
    try:
        # Create output file if it doesn't exist
        if not os.path.exists(args.out_file):
            try:
                with open(args.out_file, 'w') as f:
                    pass
                print(f"Created new output file: {args.out_file}")
            except Exception as e:
                print(f"Error creating output file: {e}")
                raise
        
        # Filter out items that are already processed or currently being processed
        items_to_process = []
        skipped_processed = 0
        skipped_locked = 0
        
        for item in data:
            mod_pdb = item["mod_pdb"]
            
            # Skip if already processed
            if is_already_processed(mod_pdb, args.out_file):
                skipped_processed += 1
                continue
            
                            # Skip if currently being processed by another instance (unless lock is stale)
            item_hash = get_item_hash(item)
            lock_file = os.path.join(lock_dir, f"{item_hash}.lock")
            if os.path.exists(lock_file):
                # Check if lock is stale (older than 20 minutes)
                try:
                    lock_age = time.time() - os.path.getmtime(lock_file)
                    if lock_age > 1200:  # 20 minutes
                        print(f"Removing stale lock for {item['mod_pdb']} (age: {lock_age/60:.1f} minutes)")
                        os.unlink(lock_file)
                    else:
                        skipped_locked += 1
                        continue
                except:
                    # If we can't check the lock, remove it
                    os.unlink(lock_file)
            
            items_to_process.append(item)
        
        print(f"Skipped {skipped_processed} already processed items")
        print(f"Skipped {skipped_locked} items currently being processed by other instances")
        print(f"Will process {len(items_to_process)} items")
        
        if not items_to_process:
            print("No items to process!")
            return
                
        # Process each entry and save results immediately
        processed_count = 0
        for i, item in enumerate(items_to_process):
            item_hash = get_item_hash(item)
            lock_file = simple_item_lock(item_hash, lock_dir)
            
            if lock_file is None:
                print(f"Item {item.get('mod_pdb', 'unknown')} is being processed by another instance, skipping...")
                continue
            
            try:
                print(f"\nProcessing item {i+1}/{len(items_to_process)}: {item.get('mod_pdb', 'unknown')}")
                
                # Double-check if it was processed while we were waiting
                if is_already_processed(item["mod_pdb"], args.out_file):
                    print(f"Item was processed by another instance while waiting, skipping...")
                    continue
                
                # Process the item with proper platform
                output = openmm_relax(item)
                
                processed_count += 1
                
                # Write the result immediately to avoid data loss
                try:
                    with open(args.out_file, 'a') as f:
                        # Write as a single operation to avoid partial writes
                        line = json.dumps(output) + '\n'
                        f.write(line)
                        f.flush()
                        # Optionally, force OS to write to disk
                        os.fsync(f.fileno())
                    print(f"Saved result for item {i+1}/{len(items_to_process)}")
                except Exception as e:
                    print(f"Error writing to output file: {e}")
                
                # Clean up GPU memory periodically
                if (i + 1) % 10 == 0:
                    import gc
                    import torch
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("Memory cleared")
                    
            except Exception as e:
                print(f"Error processing item {i+1}/{len(items_to_process)}: {e}")
                import traceback
                print(traceback.format_exc())
            finally:
                # Always cleanup lock
                cleanup_lock(lock_file)
        
        print(f"\nInstance completed. Processed {processed_count} items.")



        cleanup()
        
    except Exception as e:
        print(f"Error in main processing loop: {e}")
        import traceback
        print(traceback.format_exc())


def parse():
    parser = argparse.ArgumentParser(description='inference of pipeline method')
    parser.add_argument('--in_file', type=str, required=True, help='Path to the summary file of dataset in json format')
    parser.add_argument('--out_file', type=str, required=True, help='Path to the summary file of dataset in json format')
    parser.add_argument('--cdr_model', type=str, required=True, help='Path to the summary file of dataset in json format')

    return parser.parse_args()

if __name__ == '__main__':
    main(parse())