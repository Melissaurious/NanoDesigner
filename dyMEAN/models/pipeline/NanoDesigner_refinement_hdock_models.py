#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import json
import argparse
import time
import logging
import warnings
import traceback
import gc
import concurrent
from concurrent.futures import TimeoutError

import numpy as np
import torch
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from Bio import PDB
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from pdbfixer import PDBFixer

# OpenMM imports (importing both simtk and openmm for compatibility)
from simtk import unit
from simtk.openmm import app, Platform
from simtk.unit import kilocalories_per_mole, angstroms, kilojoules_per_mole
from openmm import LangevinIntegrator, Platform, CustomExternalForce, unit
from openmm.app import PDBFile, Simulation, ForceField, HBonds, Modeller

current_working_dir = os.getcwd()
sys.path.append(current_working_dir)

from configs import CACHE_DIR

# Configure logging and warnings
logging.getLogger('openmm').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=PDBConstructionWarning)

# Initialize PDB parser and downloader
parser = PDB.PDBParser(QUIET=True)
downloader = PDB.PDBList()

# Print available OpenMM platforms
print([Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())])  # ['Reference', 'CPU', 'CUDA', 'OpenCL']


parent_dir = os.path.dirname(current_working_dir)
sys.path.append(parent_dir)
from functionalities.complex_analysis import count_clashes



def load_structure(pdb_file_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file_path)
    return structure



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
        total_clashes = count_clashes(structure, clash_cutoff=0.63)[0]

        chain_clashes = {}
        for chain in structure.get_chains():
            chain_id = chain.get_id()
            chain_clashes[chain_id] = count_clashes(chain, clash_cutoff=0.63)[0]

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
        total_clashes = count_clashes(structure, clash_cutoff=0.63)[0]

        chain_clashes = {}
        for chain in structure.get_chains():
            chain_id = chain.get_id()
            chain_clashes[chain_id] = count_clashes(chain, clash_cutoff=0.63)[0]

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
        tuple_result = count_clashes(structure, clash_cutoff=0.63)
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



def openmm_relax_no_decorator(item, platform=None, properties=None, excluded_chains=None, inverse_exclude=False):
    
    hdock_clash_num = item.get("Hdock_num_clashes")
    
    if int(hdock_clash_num) == 0:
        item['refined'] = 'No'
        return item

    try:
        out_pdb = item.get("hdock_model", item.get("mod_pdb"))
        print(f"Model path: {out_pdb}")

        start_time = time.time()

        tolerance = (2.39 * unit.kilocalories_per_mole).in_units_of(unit.kilojoules_per_mole)
        stiffness = 10.0 * unit.kilocalories_per_mole / (unit.angstroms ** 2)

        if excluded_chains is None:
            excluded_chains = []

        if out_pdb is None:
            out_pdb = os.path.join(CACHE_DIR, 'output.pdb')

        fixer = PDBFixer(out_pdb)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()  # [OXT]
        fixer.addMissingAtoms()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        # fixer.findMissingBonds()  'PDBFixer' object has no attribute 'findMissingBonds'

                                 
        force_field = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    

        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.addHydrogens(force_field)
        system = force_field.createSystem(modeller.topology, constraints=HBonds)

        force = CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        force.addGlobalParameter("k", stiffness)
        for p in ["x0", "y0", "z0"]:
            force.addPerParticleParameter(p)

        # add flexible atoms
        for residue in modeller.topology.residues():
            if (not inverse_exclude and residue.chain.id in excluded_chains) or \
            (inverse_exclude and residue.chain.id not in excluded_chains): # antigen
                for atom in residue.atoms():
                    system.setParticleMass(atom.index, 0)
            
            for atom in residue.atoms():
                # if atom.name in ['N', 'CA', 'C', 'CB']:
                if atom.element.name != 'hydrogen':
                    force.addParticle(atom.index, modeller.positions[atom.index])

        system.addForce(force)
        integrator = LangevinIntegrator(0, 0.01, 0.0)


        if platform is None:
            try:
                platform = Platform.getPlatformByName('CUDA')
                cuda_id = properties.get('CudaDeviceIndex', '0') if properties else '0'
            except Exception:
                platform = Platform.getPlatformByName('Reference')
                cuda_id = 'CPU'
        else:
            cuda_id = properties.get('CudaDeviceIndex', '0') if properties else '0'

        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy(tolerance)

        state = simulation.context.getState(getPositions=True, getEnergy=True)
        with open(out_pdb, 'w') as fout:
            PDBFile.writeFile(simulation.topology, state.getPositions(), fout, keepIds=True)

        end_time = time.time()

        item['refined'] = 'Yes'
        item['refinement_time'] = end_time - start_time

    except Exception as e:
        print("Error processing", item)
        print(f"Error processing item {item.get('entry_id', 'Unknown ID')}: {str(e)}")
        print(traceback.format_exc())
        item['refined'] = 'No'

    print("Function complete.")
    return item



def main(args):

    if os.path.exists(args.inference_summary):
        print(f"Inference step already conducted, thefore this step is redundant")
        return

    track_file1 = os.path.join(args.hdock_models, f"refinement_hdock_models_iter{args.iteration}.json")
    if not os.path.exists(track_file1):
        with open(track_file1, 'w') as f:
            pass


    items_in_hdock_models = os.listdir(args.hdock_models)
    dir_l = [item for item in items_in_hdock_models if os.path.isdir(os.path.join(args.hdock_models, item))]

    new_dir = [] 
    for dir_ in dir_l:
        if "tmp_dir" in dir_:
            continue
        new_dir.append(dir_)

    dir_l_ = []  # List to hold directories with more than 3 .pdb files
    new_l = []  # List to hold full directory paths with more than 3 .pdb files

    for directory in new_dir:

        full_dir_path = os.path.join(args.hdock_models, directory)
        hdock_models = [file for file in os.listdir(full_dir_path) if file.startswith('model_') and file.endswith('.pdb')]
        dir_l_.append(directory)
        new_l.append(full_dir_path)

    for index, (pdb_n, full_dir_path) in enumerate(zip(dir_l_, new_l)):

        # create track file
        track_file = os.path.join(args.hdock_models, pdb_n, "track_file.json")
        print("processing entries from", full_dir_path)
        if not os.path.exists(track_file):
            with open(track_file, 'w') as f:
                pass
        elif os.path.exists(track_file) and os.stat(track_file).st_size != 0:
            print(f"Skipping {track_file} already processed, continue...")
            continue


        
        hdock_models = [os.path.join(full_dir_path, file) for file in os.listdir(full_dir_path) if file.startswith('model_') and file.endswith('.pdb')]


        list_items = [] 
        for model in hdock_models:
            list_items.append({"model": model, "pdb_n": pdb_n})


        top_n_file = track_file = os.path.join(args.hdock_models, pdb_n, "top_models.json")
        if os.path.exists(top_n_file):
            with open(top_n_file, 'r') as f:
                list_items = [json.loads(line) for line in f]
            
    
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(clash_analysis_hdock, item) for item in list_items]

            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    # Optionally, you can add a timeout to future.result()
                    result = future.result(timeout=300)  # 300 seconds timeout
                    if result is not None:
                        results.append(result)
                except TimeoutError:
                    print("A task exceeded the time limit and was skipped.")
                except Exception as e:
                    print("An error occurred:", e)
                    traceback.print_exc()  # This will print the stack trace of the exception

        only_clashing = [item for item in results if int(item.get("Hdock_num_clashes")) >= 1]
        

        result_refinement = []
        for item in only_clashing:
            output = openmm_relax_no_decorator(item)
            result_refinement.append(output)

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
                    # Optionally, you can add a timeout to future.result()
                    result = future.result(timeout=300)  # 300 seconds timeout
                    if result is not None:
                        results.append(result)
                except TimeoutError:
                    print("A task exceeded the time limit and was skipped.")
                except Exception as e:
                    print("An error occurred:", e)
                    traceback.print_exc()  # This will print the stack trace of the exception


        with open(track_file1,'a') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')

        with open(track_file, 'w') as f:
            for entry in results:
                f.write(json.dumps(entry) + '\n')
                num_clashes = entry['num_clashes_after_refinement']
                if num_clashes != 0:
                    hdock_model = entry["model"]
                    try:
                        os.remove(hdock_model)
                    except FileNotFoundError:
                        pass  # File was already removed or does not exist
        
        # Create list of remaining files 
        remaining_hdock_models = [os.path.join(full_dir_path, file) for file in os.listdir(full_dir_path) if file.startswith('model_') and file.endswith('.pdb')]

        top_n_plus_ten_file = os.path.join(args.hdock_models, pdb_n, "hdock_models.json")
        if os.path.exists(top_n_plus_ten_file):
            with open(top_n_plus_ten_file,'r') as f:
                data = [json.loads(line) for line in f]

            json_hdock_models = [item['hdock_model'] for item in data]
            # Filter to keep only the existing files
            top_n_files = [file for file in json_hdock_models if file in remaining_hdock_models][:int(args.top_n)]

        else:
            #if file does not exist, simply keep the top_n
            top_n_files = remaining_hdock_models[:int(args.top_n)]



        torch.cuda.empty_cache()
        gc.collect()



def parse():
    parser = argparse.ArgumentParser(description='inference of pipeline method')
    parser.add_argument('--hdock_models', type=str, required=True, help='Path to the summary file of dataset in json format')
    parser.add_argument('--dataset_json', type=str, required=True, help='Path to dataset in json format')
    parser.add_argument('--iteration', type=str, required=True, help='Current iteration')
    parser.add_argument('--top_n', type=str, required=True, help='Top n docked models to select ')
    parser.add_argument('--inference_summary', type=str, required=True, help='Summary file from inference stage')
    





    return parser.parse_args()

if __name__ == '__main__':
    main(parse())