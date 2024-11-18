import os
import sys
import json
import argparse
import time
import logging
import warnings
import subprocess
import gc
import traceback
from copy import deepcopy, copy
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from Bio import PDB
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from pdbfixer import PDBFixer

# OpenMM imports (importing both simtk and openmm to handle compatibility)
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

# File directory definition
FILE_DIR = os.path.abspath(os.path.split(__file__)[0])



import sys
current_working_dir = os.getcwd()
parent_dir = os.path.dirname(current_working_dir)
sys.path.append(parent_dir)
from functionalities.complex_analysis import count_clashes



parser = PDB.PDBParser(QUIET=True)
downloader = PDB.PDBList()


def openmm_relax_no_decorator(item, platform=None, properties=None, excluded_chains=None, inverse_exclude=False):

    num_clashes = item.get("side_chain_p_num_clashes", item.get("inference_clashes"))
    if num_clashes is not None:
        if int(num_clashes) == 0:
            item['refined'] = 'No'
            return item

    try:
        out_pdb = item["mod_pdb"]

        import time  # Ensure this is imported within the worker function if using ProcessPoolExecutor
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
    
        # force_field = ForceField('amber99sb.xml')
        # force_field = ForceField('charmm36.xml')  # simulations with this took longer, and still got the errors "No template found for residue 20 CYS"
        # system = forcefield.createSystem(pdb.topology,nonbondedMethod=PME,nonbondedCutoff=1*nanometer, constraints=HBonds)
        # forcefield = ForceField('amber14/protein.ff14SB.xml')
        # forcefield = ForceField('amoeba2018.xml')

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
        tot_time = end_time - start_time


        item["refinement_time_after_inf"] = tot_time
        item['refined'] = 'Yes'

    except Exception as e:
        print(f"Error processing item {item.get('mod_pdb', 'Unknown ID')}: {str(e)}")
        print(traceback.format_exc())
        item['refined'] = 'No'  # Indicate that refinement was not successful

    return item



def main(args):


    data = []
    with open(args.in_file, 'r') as f:
        try:
            data = [json.loads(line) for line in f]
        except json.JSONDecodeError as e:
            print(f"Error parsing file: {str(e)}")
            f.seek(0)  # Reset file pointer to the beginning
            for line in f:
                line = line.strip()  # Remove any leading/trailing whitespace
                if line:  # Only process non-empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as line_error:
                        print(f"Skipping malformed line: {line_error}")



    # refine only entries that were successfully packed, as they wont be considered to pass to next iteration
    print("Refining only side-chain packed complexes....")
    
    if args.cdr_model != "dyMEAN":
        data = [item for item in data if item.get("side_chain_packed") == "Yes"]
    else:
        for item in data:
            try:
                pdb = item["pdb"]
                pdb_path = item["mod_pdb"]
                # print("Skipping Side chain packing as dyMEAN is end-to-end")
                item["side_chain_packed"] = "No"
                structure = parser.get_structure(pdb, pdb_path)
                total_clashes = count_clashes(structure, clash_cutoff=0.63)[0]

                chain_clashes = {}
                for chain in structure.get_chains():
                    chain_id = chain.get_id()
                    chain_clashes[chain_id] = count_clashes(chain, clash_cutoff=0.63)[0]

                inter_chain_clashes = total_clashes - sum(chain_clashes.values())

                item["inference_clashes"] = total_clashes
                item["clashes_per_chain_inference"] = chain_clashes
                item["inter_chain_clashes_inference"] = inter_chain_clashes
            

            except Exception as e:
                print(f"{pdb_path} could not be refined, skipping the evaluation of this model")
                # item["refined"] = "No"

    print("total entries to refine", len(data))


    # Determine if the summary entries were already processed or not
    if not os.path.exists(args.out_file) or os.stat(args.out_file).st_size == 0:
        with open(args.out_file, 'w') as f:
            pass
        
        # Prepare inputs: a list of items instead of separate PDB paths
        inputs = []  # List of items
        for item in data:
            inputs.append(item)  

        start_time = time.time()
        result_refinement = []
        for item in inputs:
            output = openmm_relax_no_decorator(item)
            result_refinement.append(output)
        end_time = time.time()

        print(f"Total time to refine {len(inputs)} entries was = {end_time-start_time}")


        # save results
        with open(args.out_file, 'a') as f:
            for item in result_refinement:
                f.write(json.dumps(item) + '\n')


        torch.cuda.empty_cache()
        gc.collect()

        # Quantify clashes after inference

    elif os.path.exists(args.out_file) and os.stat(args.out_file).st_size != 0:
        with open(args.out_file, 'r') as f:
            data = f.read().strip().split('\n')


        processed_mod_pdbs = set()
        for entry in data:
            entry_json = json.loads(entry)
            processed_mod_pdbs.add(entry_json["mod_pdb"])

        # determine which entries havent been refined
        missing_entries_to_refine = [element for element in data if json.loads(element)['mod_pdb'] not in processed_mod_pdbs and json.loads(element).get('refined', 'No') != 'Yes']

         # Prepare inputs: a list of items instead of separate PDB paths
        inputs = []  # List of items
        for json_object in missing_entries_to_refine:
            item = json.loads(json_object)
            inputs.append(item)  


        start_time = time.time()
        result_refinement = []
        for item in inputs:
            output = openmm_relax_no_decorator(item)
            result_refinement.append(output)
        end_time = time.time()

        print(f"Total time to refine {len(inputs)} entries was = {end_time-start_time}")

        # save results
        with open(args.out_file, 'a') as f:
            for item in result_refinement:
                f.write(json.dumps(item) + '\n')


        torch.cuda.empty_cache()
        gc.collect()



def parse():
    parser = argparse.ArgumentParser(description='inference of pipeline method')
    parser.add_argument('--in_file', type=str, required=True, help='Path to the summary file of dataset in json format')
    parser.add_argument('--out_file', type=str, required=True, help='Path to the summary file of dataset in json format')
    parser.add_argument('--cdr_model', type=str, required=True, help='Path to the summary file of dataset in json format')


    return parser.parse_args()

if __name__ == '__main__':
    main(parse())