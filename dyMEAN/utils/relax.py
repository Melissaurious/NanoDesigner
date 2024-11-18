#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from os.path import splitext, basename
from copy import deepcopy

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

from data.pdb_utils import Peptide
from evaluation.rmsd import kabsch
from configs import CACHE_DIR, Rosetta_DIR
from utils.time_sign import get_time_sign
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
from .time_sign import get_time_sign


FILE_DIR = os.path.abspath(os.path.split(__file__)[0])

import subprocess


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

def count_clashes(structure, clash_cutoff=0.63):
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

def get_gpu_memory():
    """Returns the GPU memory usage by calling nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                                capture_output=True, text=True, check=True)
        memory_used = result.stdout.strip().split('\n')
        memory_used = [int(x) for x in memory_used]  # Convert to integers
        return memory_used
    except subprocess.CalledProcessError as e:
        print("Failed to run nvidia-smi:", e)
        return None


def _align_(mod_chain: Peptide, ref_chain: Peptide):
    mod_ca, ref_ca = [], []
    mod_atoms = []
    for residue in mod_chain:
        coord_map = residue.get_coord_map()
        mod_ca.append(coord_map['CA'])
        for atom in residue.get_atom_names():
            mod_atoms.append(coord_map[atom])
    for residue in ref_chain:
        coord_map = residue.get_coord_map()
        ref_ca.append(coord_map['CA'])
    _, Q, t = kabsch(np.array(mod_ca), np.array(ref_ca))
    mod_atoms = np.dot(mod_atoms, Q) + t
    mod_atoms = mod_atoms.tolist()
    atom_idx = 0
    residues = []
    for mod_res, ref_res in zip(mod_chain, ref_chain):
        coord_map = {}
        for atom in mod_res.get_atom_names():
            coord_map[atom] = mod_atoms[atom_idx]
            atom_idx += 1
        residue = deepcopy(ref_res)
        residue.set_coord(coord_map)
        residues.append(residue)
    return Peptide(ref_chain.get_id(), residues)



class TimeoutError(Exception):
    pass

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def run():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=run)
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                thread.join()
                raise TimeoutError(f"Timed out after {seconds} seconds")
            elif exception[0]:
                raise exception[0]

            return result[0]

        return wrapper
    return decorator

@timeout(1200)
def openmm_relax(pdb, out_pdb=None, excluded_chains=None, inverse_exclude=False):

    start_time = time.time()

    # tolerance = 2.39 * kilocalories_per_mole
    # stiffness = 10.0 * kilocalories_per_mole / (angstroms ** 2)

    tolerance = (2.39 * unit.kilocalories_per_mole).in_units_of(unit.kilojoules_per_mole)
    stiffness = 10.0 * unit.kilocalories_per_mole / (unit.angstroms ** 2)

    # tolerance = 2.39
    # stiffness = 10.0 * kilocalories_per_mole / (angstroms ** 2)

    if excluded_chains is None:
        excluded_chains = []

    if out_pdb is None:
        out_pdb = os.path.join(CACHE_DIR, 'output.pdb')

    fixer = PDBFixer(pdb)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()  # [OXT]
    fixer.addMissingAtoms()
    
    # force_field = ForceField("amber14/protein.ff14SB.xml")
    # force_field = ForceField('amber99sb.xml')
    force_field = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(force_field)
    # system = force_field.createSystem(modeller.topology)
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
    # platform = Platform.getPlatformByName('CPU')
    # simulation = Simulation(modeller.topology, system, integrator, platform)

    # Attempt to use CUDA platform, fallback to CPU if not available
    try:
        platform = Platform.getPlatformByName('CUDA')
        print("Using CUDA platform")
    except Exception:
        platform = Platform.getPlatformByName('Reference')  # Fallback to CPU
        print("Using CPU platform (Reference)")

    simulation = Simulation(modeller.topology, system, integrator, platform)


    # simulation = Simulation(modeller.topology, system, integrator)#, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(tolerance)

    state = simulation.context.getState(getPositions=True, getEnergy=True)

    with open(out_pdb, 'w') as fout:
        PDBFile.writeFile(simulation.topology, state.getPositions(), fout, keepIds=True)
    
    end_time = time.time()
    print(f"Relaxation took: {end_time - start_time:.2f} seconds")

    return out_pdb




# def openmm_relax_no_decorator(pdb, out_pdb=None, excluded_chains=None, inverse_exclude=False):
def openmm_relax_no_decorator(pdb, out_pdb=None, platform=None, properties=None, excluded_chains=None, inverse_exclude=False):

    start_time = time.time()

    tolerance = 2.39
    stiffness = 10.0 * kilocalories_per_mole / (angstroms ** 2)

    if excluded_chains is None:
        excluded_chains = []

    if out_pdb is None:
        out_pdb = os.path.join(CACHE_DIR, 'output.pdb')

    # Get initial GPU memory usage
    # initial_gpu_memory = get_gpu_memory()
    # print("Initial GPU Memory Usage (megabytes):", initial_gpu_memory)

    fixer = PDBFixer(pdb)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()  # [OXT]
    fixer.addMissingAtoms()
    
    # force_field = ForceField("amber14/protein.ff14SB.xml")
    # force_field = ForceField('amber99sb.xml')
    force_field = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(force_field)
    # system = force_field.createSystem(modeller.topology)
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

    # Retrieve the index of the GPU assigned to this job
    # gpu_indices = os.getenv("CUDA_VISIBLE_DEVICES")
    # if gpu_indices:
    #     gpu_index = gpu_indices.split(',')[0]  # Use the first visible GPU
    #     platform = Platform.getPlatformByName('CUDA')
    #     platform.setPropertyDefaultValue('CudaDeviceIndex', gpu_index)
    #     print(f"Using CUDA platform with GPU index: {gpu_index}")
    #     simulation = Simulation(modeller.topology, system, integrator, platform)
    # else:
    #     print("No GPU index available in CUDA_VISIBLE_DEVICES. Defaulting to platform configuration.")
    #     simulation = Simulation(modeller.topology, system, integrator)

    # Create the simulation using provided platform and properties
    # Set up the simulation with the provided platform and properties
    if platform is None:
        try:
            platform = Platform.getPlatformByName('CUDA')
            cuda_id = properties.get('CudaDeviceIndex', '0') if properties else '0'
        except Exception:
            platform = Platform.getPlatformByName('Reference')
            cuda_id = 'CPU'
    else:
        cuda_id = properties.get('CudaDeviceIndex', '0') if properties else '0'


    # Attempt to use CUDA platform, fallback to CPU if not available
    # try:
    #     platform = Platform.getPlatformByName('CUDA')
    #     cuda_id = platform.getIndex()
    #     print("Using CUDA platform,  ID:", cuda_id)
    # except Exception:
    #     platform = Platform.getPlatformByName('Reference')  # Fallback to CPU
    #     print("Using CPU platform (Reference)")

    # simulation = Simulation(modeller.topology, system, integrator)#, platform)
    # simulation = Simulation(modeller.topology, system, integrator, platform)
    
    simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(tolerance)

    # Get GPU memory usage after setup
    # post_setup_gpu_memory = get_gpu_memory()
    # print("GPU Memory Usage After Setup (megabytes):", post_setup_gpu_memory)

    state = simulation.context.getState(getPositions=True, getEnergy=True)

    with open(out_pdb, 'w') as fout:
        PDBFile.writeFile(simulation.topology, state.getPositions(), fout, keepIds=True)

    # Get final GPU memory usage
    # final_gpu_memory = get_gpu_memory()
    # print("Final GPU Memory Usage (megabytes):", final_gpu_memory)

    end_time = time.time()

    print(f"Relaxation took = {end_time-start_time} on GPU id = {cuda_id}" )
    print(f"Process {os.getpid()} - Relaxation took = {end_time-start_time:.2f} seconds on GPU id = {cuda_id}")


    return out_pdb


def rosetta_sidechain_packing(pdb, out_pdb=None):
    start_time = time.time()
    
    unique_id = str(uuid.uuid4())
    rosetta_exe = os.path.join(Rosetta_DIR, 'fixbb.static.linuxgccrelease')
    resfile = os.path.join(CACHE_DIR, 'resfile.txt')
    
    # Check if Rosetta executable exists
    if not os.path.exists(rosetta_exe):
        print(f"Rosetta executable not found at {rosetta_exe}. Skipping side-chain packing.")
        return None, None  # Indicate failure by returning None values

    # Create resfile if it doesn't exist
    if not os.path.exists(resfile):
        with open(resfile, 'w') as fout:
            fout.write(f'NATAA\nstart')
    
    cmd = f'{rosetta_exe} -in:file:s {pdb} -in:file:fullatom -resfile {resfile} ' + \
          f'-nstruct 1 -out:path:all {CACHE_DIR} -out:prefix {unique_id} -overwrite -mute all'
    
    # Run the command and wait for completion
    p = os.popen(cmd)
    p.read()
    p.close()
    
    filename = splitext(basename(pdb))[0]
    tmp_pdb = os.path.join(CACHE_DIR, unique_id + filename + '_0001.pdb')
    
    # Check if output PDB file was created successfully
    if not os.path.exists(tmp_pdb):
        print(f"Side-chain packing failed. Output file {tmp_pdb} not found.")
        return None, None  # Indicate failure by returning None values

    # Move the file if out_pdb is provided
    if out_pdb is not None:
        os.system(f'mv {tmp_pdb} {out_pdb}')
        tmp_pdb = out_pdb

    total_time = time.time() - start_time
    return tmp_pdb, total_time



