#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import shutil

import torch

from dyMEAN.utils.time_sign import get_time_sign
import numpy as np

FILE_DIR = os.path.abspath(os.path.split(__file__)[0])
MODULE_DIR = os.path.join(FILE_DIR, 'ddg')
# from .ddg.models.predictor import DDGPredictor
# from .ddg.utils.misc import *
# from .ddg.utils.data import *
# from .ddg.utils.protein import *

# CKPT = torch.load(os.path.join(MODULE_DIR, 'data', 'model.pt'))
# MODEL = DDGPredictor(CKPT['config'].model)
# MODEL.load_state_dict(CKPT['model'])
# DEVICE = torch.device('cuda:0')
# MODEL.to(DEVICE)
# MODEL.eval()


# def pred_ddg(wt_pdb, mut_pdb):
#     batch = load_wt_mut_pdb_pair(wt_pdb, mut_pdb)
#     batch = recursive_to(batch, DEVICE)

#     with torch.no_grad():
#         pred = MODEL(batch['wt'], batch['mut']).item()

#     return pred


from configs import FOLDX_BIN, CACHE_DIR


def foldx_minimize_energy(pdb_path, out_path=None):
    filename = get_time_sign() + os.path.basename(os.path.splitext(pdb_path)[0]) + '_foldx.pdb'
    tmpfile = os.path.join(CACHE_DIR, filename)
    shutil.copyfile(pdb_path, tmpfile)
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


# def foldx_dg(pdb_path, rec_chains, lig_chains):
#     filename = get_time_sign() + os.path.basename(os.path.splitext(pdb_path)[0]) + '_foldx.pdb'
#     tmpfile = os.path.join(CACHE_DIR, filename)
#     shutil.copyfile(pdb_path, tmpfile)
#     rec, lig = ''.join(rec_chains), ''.join(lig_chains)
#     # p = os.popen(f'cd {CACHE_DIR}; {FOLDX_BIN} --command=Optimize --pdb={filename}')
#     # p.read()
#     # p.close()
#     # os.remove(tmpfile)

#     # filename = 'Optimized_' + filename
#     # tmpfile = os.path.join(CACHE_DIR, filename)
#     p = os.popen(f'cd {CACHE_DIR}; {FOLDX_BIN} --command=AnalyseComplex --pdb={filename} --analyseComplexChains={rec},{lig}')
#     aff = float(p.read().split('\n')[-8].split(' ')[-1])
#     p.close()
#     os.remove(tmpfile)
#     return aff


# def foldx_dg(pdb_path, rec_chains, lig_chains):
#     filename = get_time_sign() + os.path.basename(os.path.splitext(pdb_path)[0]) + '_foldx.pdb'
#     tmpfile = os.path.join(CACHE_DIR, filename)
#     shutil.copyfile(pdb_path, tmpfile)
#     rec, lig = ''.join(rec_chains), ''.join(lig_chains)

#     output_file = os.path.join(CACHE_DIR, 'foldx_output.txt')  # Specify the path and filename for the output file

#     p = os.popen(f'cd {CACHE_DIR}; {FOLDX_BIN} --command=AnalyseComplex --pdb={filename} --analyseComplexChains={rec},{lig}')
#     output = p.read()
#     print(output)
#     p.close()

#     with open(output_file, 'w') as f:
#         f.write(output)

#     aff = float(output.split('\n')[-8].split(' ')[-1])
#     os.remove(tmpfile)
#     return aff


# def foldx_dg(pdb_path, rec_chains, lig_chains):
#     filename = get_time_sign() + os.path.basename(os.path.splitext(pdb_path)[0]) + '_foldx.pdb'
#     tmpfile = os.path.join(CACHE_DIR, filename)
#     shutil.copyfile(pdb_path, tmpfile)
#     rec, lig = ''.join(rec_chains), ''.join(lig_chains)

#     output_file = '/ibex/user/rioszemm/foldx_output.txt'  # Specify the path and filename for the output file

#     p = os.popen(f'cd {CACHE_DIR}; {FOLDX_BIN} --command=AnalyseComplex --pdb={filename} --analyseComplexChains={rec},{lig}')
#     output = p.read()
#     p.close()

#     with open(output_file, 'w') as f:
#         f.write(output)

#     aff = float(output.split('\n')[-8].split(' ')[-1])
#     os.remove(tmpfile)
#     return aff



# def foldx_dg(pdb_path, rec_chains, lig_chains):
#     filename = get_time_sign() + os.path.basename(os.path.splitext(pdb_path)[0]) + '_foldx.pdb'
#     tmpfile = os.path.join(CACHE_DIR, filename)
#     shutil.copyfile(pdb_path, tmpfile)
#     rec, lig = ''.join(rec_chains), ''.join(lig_chains)

#     # Run FoldX command
#     command = f'cd {CACHE_DIR}; {FOLDX_BIN} --command=AnalyseComplex --pdb={filename} --analyseComplexChains={rec},{lig}'
#     p = os.popen(command)
#     output = p.read()
#     p.close()
    
#     # Find the section of interaction energies between the specified chains
#     interaction_lines = output.split('\n')
#     interaction_start = False
#     total_energy = None
#     for line in interaction_lines:
#         if 'interaction between' in line and rec in line and lig in line:
#             interaction_start = True
#         if interaction_start and 'Total' in line:
#             # Extract the total energy value
#             try:
#                 total_energy = float(line.split('=')[-1].strip())
#             except ValueError as e:
#                 print("Error parsing total energy:", e)
#             break
    
#     # Clean up the temporary file
#     try:
#         os.remove(tmpfile)
#     except OSError as e:
#         print(f"Error deleting temporary file {tmpfile}: {e}")

#     if total_energy is not None:
#         return total_energy
#     else:
#         print("Could not find interaction energy between chains")
#         return np.nan



# def foldx_dg(pdb_path, rec_chains, lig_chains):
#     filename = get_time_sign() + os.path.basename(os.path.splitext(pdb_path)[0]) + '_foldx.pdb'
#     tmpfile = os.path.join(CACHE_DIR, filename)
    
#     # Debug: print paths
#     print(f"PDB Path: {pdb_path}")
#     print(f"Temporary file: {tmpfile}")
    
#     # Copy the PDB file to the cache directory
#     try:
#         shutil.copyfile(pdb_path, tmpfile)
#     except Exception as e:
#         print(f"Error copying file {pdb_path} to {tmpfile}: {e}")
#         return np.nan
    
#     rec, lig = ''.join(rec_chains), ''.join(lig_chains)
#     # Debug: print chains
#     # print(f"Receptor chains: {rec}")
#     # print(f"Ligand chains: {lig}")
    
#     # Run FoldX command
#     command = f'cd {CACHE_DIR}; {FOLDX_BIN} --command=AnalyseComplex --pdb={filename} --analyseComplexChains={rec},{lig}'
#     # print(f"Running command: {command}")
#     p = os.popen(command)
#     output = p.read()
#     p.close()
    
#     # Debug: print FoldX output
#     # print("FoldX output:")
#     # print(output)
    
#     # Find the section of interaction energies between the specified chains
#     interaction_lines = output.split('\n')
#     interaction_start = False
#     total_energy = None
#     for line in interaction_lines:
#         if 'interaction between' in line and rec in line and lig in line:
#             interaction_start = True
#             # print(f"Interaction start found in line: {line}")
#         if interaction_start and 'Total' in line:
#             # Extract the total energy value
#             try:
#                 total_energy = float(line.split('=')[-1].strip())
#                 # print(f"Total energy extracted: {total_energy}")
#             except ValueError as e:
#                 print(f"Error parsing total energy from line '{line}': {e}")
#             break
    
#     # Clean up the temporary file
#     try:
#         os.remove(tmpfile)
#         # print(f"Temporary file {tmpfile} deleted")
#     except OSError as e:
#         print(f"Error deleting temporary file {tmpfile}: {e}")

#     if total_energy is not None:
#         return total_energy
#     else:
#         print("Could not find interaction energy between chains")
#         return np.nan



def foldx_dg(pdb_path, rec_chains, lig_chains):
    """Calculate binding energy with FoldX, using tempfile for unique filenames."""
    import os
    import shutil
    import numpy as np
    import tempfile
    import subprocess
    
    # Define paths
    # foldx_bin = '/ibex/user/rioszemm/foldx_1Linux64_0/foldx_20251231'
    foldx_bin = FOLDX_BIN
    # rotabase_path = '/ibex/user/rioszemm/foldx_1Linux64_0/rotabase.txt'
    rotabase_path = os.path.join(os.path.dirname(foldx_bin), 'rotabase.txt')


    # Generate a temporary working directory
    temp_dir = tempfile.mkdtemp()
    # print(f"Working in temporary directory: {temp_dir}")
    
    try:
        # Create a unique filename using tempfile
        fd, temp_pdb = tempfile.mkstemp(suffix='.pdb', dir=temp_dir)
        os.close(fd)
        
        # Get just the basename for FoldX
        pdb_basename = os.path.basename(temp_pdb)
        
        # Copy the PDB file to the unique temp file
        shutil.copy(pdb_path, temp_pdb)
        # print(f"Copied PDB to {temp_pdb}")
        
        # Copy rotabase.txt to the temp directory
        temp_rotabase = os.path.join(temp_dir, "rotabase.txt")
        shutil.copy(rotabase_path, temp_rotabase)
        # print(f"Copied rotabase.txt to {temp_rotabase}")
        
        # Change to the temp directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        # Format chain strings
        rec = ''.join(rec_chains) if isinstance(rec_chains, list) else rec_chains
        lig = ''.join(lig_chains) if isinstance(lig_chains, list) else lig_chains
        
        # Run FoldX command
        command = f"{foldx_bin} --command=AnalyseComplex --pdb={pdb_basename} --analyseComplexChains={rec},{lig}"
        print(f"Running command: {command}")
        
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        stdout_str = stdout.decode('utf-8')
        stderr_str = stderr.decode('utf-8')
        
        if stderr_str:
            print(f"FoldX stderr: {stderr_str}")
            
        if process.returncode != 0:
            print(f"FoldX failed with return code {process.returncode}")
            return np.nan
            
        # First look for interaction file
        interaction_file = f"Interaction_{pdb_basename.replace('.pdb', '')}.fxout"
        if os.path.exists(interaction_file):
            print(f"Found interaction file: {interaction_file}")
            with open(interaction_file, 'r') as f:
                content = f.read()
                
            # Parse the file to find interaction energy
            for line in content.split('\n'):
                if not line.startswith('#') and line.strip():
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            energy = float(parts[1])
                            print(f"Found energy: {energy}")
                            return energy
                        except:
                            pass
        
        # If no interaction file or energy found, parse stdout
        interaction_start = False
        for line in stdout_str.split('\n'):
            if 'interaction between' in line and rec in line and lig in line:
                interaction_start = True
                
            if interaction_start and 'Total' in line:
                try:
                    energy = float(line.split('=')[-1].strip())
                    print(f"Found energy from stdout: {energy}")
                    return energy
                except:
                    pass
                    
        print("Could not find interaction energy")
        return np.nan
        
    except Exception as e:
        print(f"Error: {e}")
        return np.nan
        
    finally:
        # Return to original directory
        if 'original_dir' in locals():
            os.chdir(original_dir)
            
        # Clean up temp dir
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up {temp_dir}")
        except:
            pass




# def foldx_ddg(wt_pdb, mut_pdb, rec_chains, lig_chains):
#     wt_dg = foldx_dg(wt_pdb, rec_chains, lig_chains)
#     mut_dg = foldx_dg(mut_pdb, rec_chains, lig_chains)
#     return mut_dg - wt_dg






def foldx_ddg(wt_pdb, mut_pdb, rec_chains, lig_chains):
    """Calculate ddG with improved handling of NaN values."""
    import numpy as np
    
    # Get individual dG values
    print(f"Calculating dG for reference structure: {wt_pdb}")
    wt_dg = foldx_dg(wt_pdb, rec_chains, lig_chains)
    print(f"Reference dG: {wt_dg}")
    
    print(f"Calculating dG for model structure: {mut_pdb}")
    mut_dg = foldx_dg(mut_pdb, rec_chains, lig_chains)
    print(f"Model dG: {mut_dg}")
    
    # Handle different scenarios
    if np.isnan(wt_dg) and np.isnan(mut_dg):
        print("Both structures have no interface - returning 0.0")
        return 0.0
    elif np.isnan(wt_dg) and not np.isnan(mut_dg):
        print("Reference has no interface but model does - binding created")
        return mut_dg  # Return the model dG as ddG
    elif not np.isnan(wt_dg) and np.isnan(mut_dg):
        print("Model has no interface but reference does - binding lost")
        return -wt_dg  # Return negative of reference dG as ddG
    else:
        # Normal case - both have interfaces
        ddg = mut_dg - wt_dg
        print(f"Both have interfaces - ddG = {ddg}")
        return ddg









# def foldx_dg_debug(pdb_path, rec_chains, lig_chains, debug=True):
#     """Enhanced version of foldx_dg with detailed debugging."""
#     import os
#     import shutil
#     import numpy as np
#     import subprocess
#     import tempfile
#     from pathlib import Path
    
#     # Make sure the CACHE_DIR exists
#     if not os.path.exists(CACHE_DIR):
#         os.makedirs(CACHE_DIR)
    
#     # Generate a unique filename
#     filename = get_time_sign() + os.path.basename(os.path.splitext(pdb_path)[0]) + '_foldx.pdb'
#     tmpfile = os.path.join(CACHE_DIR, filename)
    
#     if debug:
#         print(f"\n{'='*50}")
#         print(f"FOLDX_DG DEBUG - START")
#         print(f"{'='*50}")
#         print(f"PDB Path: {pdb_path}")
#         print(f"PDB exists: {os.path.exists(pdb_path)}")
#         if os.path.exists(pdb_path):
#             print(f"PDB file size: {os.path.getsize(pdb_path)} bytes")
#         print(f"Receptor chains: {rec_chains}")
#         print(f"Ligand chains: {lig_chains}")
#         print(f"Temporary file: {tmpfile}")
#         print(f"CACHE_DIR exists: {os.path.exists(CACHE_DIR)}")
#         print(f"FOLDX_BIN: {FOLDX_BIN}")
#         print(f"FOLDX_BIN exists: {os.path.exists(FOLDX_BIN)}")
    
#     # Check if chains exist in PDB
#     if debug:
#         try:
#             with open(pdb_path, 'r') as f:
#                 pdb_content = f.read()
            
#             # Extract unique chain IDs from ATOM records
#             chains = set()
#             for line in pdb_content.split('\n'):
#                 if line.startswith('ATOM  ') or line.startswith('HETATM'):
#                     chain_id = line[21:22].strip()
#                     if chain_id:
#                         chains.add(chain_id)
            
#             print(f"Chains found in PDB: {sorted(chains)}")
#             for chain in rec_chains:
#                 if chain not in chains:
#                     print(f"WARNING: Receptor chain '{chain}' not found in PDB!")
#             for chain in lig_chains:
#                 if chain not in chains:
#                     print(f"WARNING: Ligand chain '{chain}' not found in PDB!")
#         except Exception as e:
#             print(f"Error checking chains in PDB: {e}")
    
#     # Copy the PDB file to the cache directory
#     try:
#         shutil.copyfile(pdb_path, tmpfile)
#         if debug:
#             print(f"Successfully copied PDB to temporary file")
#     except Exception as e:
#         print(f"Error copying file {pdb_path} to {tmpfile}: {e}")
#         return np.nan
    
#     rec, lig = ''.join(rec_chains), ''.join(lig_chains)
    
#     # Run FoldX command
#     command = f'cd {CACHE_DIR}; {FOLDX_BIN} --command=AnalyseComplex --pdb={filename} --analyseComplexChains={rec},{lig}'
#     if debug:
#         print(f"Running FoldX command: {command}")
    
#     try:
#         # Use subprocess instead of os.popen for better error handling
#         process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         stdout, stderr = process.communicate()
#         output = stdout.decode('utf-8')
#         error_output = stderr.decode('utf-8')
        
#         if debug:
#             print(f"FoldX return code: {process.returncode}")
#             if error_output:
#                 print(f"FoldX stderr output:\n{error_output}")
#             print(f"FoldX stdout output:\n{output[:500]}...")  # Print first 500 chars
            
#             # Check for output files
#             interaction_file = os.path.join(CACHE_DIR, f"Interaction_{filename.replace('.pdb', '')}.fxout")
#             print(f"Interaction file exists: {os.path.exists(interaction_file)}")
#             if os.path.exists(interaction_file):
#                 with open(interaction_file, 'r') as f:
#                     interaction_content = f.read()
#                 print(f"Interaction file content:\n{interaction_content}")
#     except Exception as e:
#         print(f"Error executing FoldX command: {e}")
#         return np.nan
    
#     # Find the section of interaction energies between the specified chains
#     interaction_lines = output.split('\n')
#     interaction_start = False
#     total_energy = None
#     for line in interaction_lines:
#         if 'interaction between' in line and rec in line and lig in line:
#             interaction_start = True
#             if debug:
#                 print(f"Interaction start found in line: {line}")
#         if interaction_start and 'Total' in line:
#             # Extract the total energy value
#             try:
#                 total_energy = float(line.split('=')[-1].strip())
#                 if debug:
#                     print(f"Total energy extracted: {total_energy}")
#             except ValueError as e:
#                 print(f"Error parsing total energy from line '{line}': {e}")
#             break
    
#     # Also look for interaction energy in the FoldX output file
#     if total_energy is None:
#         interaction_file = os.path.join(CACHE_DIR, f"Interaction_{filename.replace('.pdb', '')}.fxout")
#         if os.path.exists(interaction_file):
#             try:
#                 with open(interaction_file, 'r') as f:
#                     lines = f.readlines()
#                 for line in lines:
#                     if line.strip() and not line.startswith('#'):
#                         parts = line.split()
#                         if len(parts) > 1 and parts[0].startswith(f"{rec},{lig}"):
#                             try:
#                                 total_energy = float(parts[1])
#                                 if debug:
#                                     print(f"Total energy extracted from interaction file: {total_energy}")
#                                 break
#                             except (ValueError, IndexError):
#                                 continue
#             except Exception as e:
#                 print(f"Error reading interaction file: {e}")
    
#     # Clean up the temporary file
#     try:
#         os.remove(tmpfile)
#         if debug:
#             print(f"Temporary file {tmpfile} deleted")
#     except OSError as e:
#         print(f"Error deleting temporary file {tmpfile}: {e}")

#     if total_energy is not None:
#         if debug:
#             print(f"Returning total energy: {total_energy}")
#             print(f"{'='*50}")
#             print(f"FOLDX_DG DEBUG - END")
#             print(f"{'='*50}\n")
#         return total_energy
#     else:
#         print("Could not find interaction energy between chains")
#         if debug:
#             print(f"{'='*50}")
#             print(f"FOLDX_DG DEBUG - END")
#             print(f"{'='*50}\n")
#         return np.nan


# def foldx_ddg_debug(wt_pdb, mut_pdb, rec_chains, lig_chains, debug=True):
#     """Enhanced version of foldx_ddg with detailed debugging."""
#     if debug:
#         print(f"\n{'='*50}")
#         print(f"FOLDX_DDG DEBUG - START")
#         print(f"{'='*50}")
#         print(f"Wild-type PDB: {wt_pdb}")
#         print(f"Mutant PDB: {mut_pdb}")
    
#     wt_dg = foldx_dg_debug(wt_pdb, rec_chains, lig_chains, debug)
#     mut_dg = foldx_dg_debug(mut_pdb, rec_chains, lig_chains, debug)
    
#     if debug:
#         print(f"Wild-type dG: {wt_dg}")
#         print(f"Mutant dG: {mut_dg}")
        
#     ddg = mut_dg - wt_dg
    
#     if debug:
#         print(f"Calculated ddG: {ddg}")
#         print(f"{'='*50}")
#         print(f"FOLDX_DDG DEBUG - END")
#         print(f"{'='*50}\n")
    
#     return ddg