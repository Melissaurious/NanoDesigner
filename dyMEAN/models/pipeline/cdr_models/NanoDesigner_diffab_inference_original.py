#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
from tqdm import tqdm
from shutil import rmtree
import logging
import sys
# sys.path.append('/ibex/user/rioszemm/NanobodiesProject/diffab')
from diffab.diffab.tools.runner.design_for_pdb import design_for_pdb
logging.disable('INFO')
import glob
import sys
import time
from functools import wraps
import subprocess
import threading
import psutil

def get_default_item(pdb_dict):
    # Returns the first item in the pdb_dict. 
    # You can modify this function if you need to select a specific item instead of the first one.
    return next(iter(pdb_dict.values()))

def load_config(config_path):
    """Load config from YAML file using EasyDict"""
    import yaml
    import os
    from easydict import EasyDict
    
    with open(config_path, 'r', encoding="utf-8") as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name

def get_target_samples(config, iteration):
    """Extract target samples from config based on iteration"""
    if int(iteration) == 1:
        if hasattr(config.sampling, 'num_samples_iter_1'):
            num_samples = config.sampling.num_samples_iter_1
        else:
            num_samples = 3
    else:
        if hasattr(config.sampling, 'num_samples_iter_x'):
            num_samples = config.sampling.num_samples_iter_x
        else:
            num_samples = 10
    
    return num_samples

class Arg:
    def __init__(self, pdb, heavy, light, antigen, config, out_root, summary_dir, pdb_code, model, cdr_type, iteration, ref_pdb, dymean_code_dir):
        self.pdb_path = pdb
        self.heavy = heavy
        self.light = light
        self.antigen = antigen
        self.no_renumber = True
        self.config = config
        self.out_root = out_root
        self.tag = ''
        self.seed = 0
        self.device = 'cuda'
        self.batch_size = 16
        self.summary_dir = summary_dir
        self.pdb_code = pdb_code
        self.model = model
        self.cdr_type = cdr_type
        self.iteration = iteration
        self.ref_pdb = ref_pdb
        self.dymean_code_dir = dymean_code_dir

def get_processed_models_from_summary(summary_file):
    """Extract processed models from summary file"""
    processed_models = set()
    
    if not os.path.exists(summary_file) or os.path.getsize(summary_file) == 0:
        return processed_models
    
    try:
        with open(summary_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        # Assuming summary contains JSON entries with model information
                        # Adjust this parsing based on your actual summary file format
                        entry = json.loads(line)
                        # Extract model identifier - adjust key names based on your summary format
                        if 'model_path' in entry:
                            model_path = entry['model_path']
                            model_name = os.path.basename(model_path).split('.')[0]
                            processed_models.add(model_path)
                        elif 'model_name' in entry:
                            processed_models.add(entry['model_name'])
                    except json.JSONDecodeError:
                        # If line is not JSON, treat as plain text model path/name
                        processed_models.add(line.strip())
    except Exception as e:
        print(f"Warning: Could not read summary file {summary_file}: {e}")
    
    return processed_models

def get_models_from_output_dirs(out_dir, pdb_model):
    """Get models that have output directories (processed or partially processed)"""
    models_with_dirs = set()
    pdb_output_dir = os.path.join(out_dir, pdb_model)
    
    if os.path.exists(pdb_output_dir):
        for item in os.listdir(pdb_output_dir):
            item_path = os.path.join(pdb_output_dir, item)
            if os.path.isdir(item_path):
                models_with_dirs.add(item)
    
    return models_with_dirs

def identify_missing_and_incomplete_models(hdock_models, out_dir, pdb_model, target_samples, summary_file=None):
    """Identify models that are missing or incomplete"""
    # Get models that have been processed according to summary
    processed_from_summary = get_processed_models_from_summary(summary_file) if summary_file else set()
    
    # Get models that have output directories
    models_with_dirs = get_models_from_output_dirs(out_dir, pdb_model)
    
    missing_models = []
    incomplete_models = []
    complete_models = []
    
    for hdock_model in hdock_models:
        model_name = os.path.basename(hdock_model).split(".")[0]
        model_output_dir = os.path.join(out_dir, pdb_model, model_name)
        
        # Check if model has output directory
        if not os.path.exists(model_output_dir):
            missing_models.append({
                'model_path': hdock_model,
                'model_name': model_name,
                'status': 'missing_directory',
                'designs_found': 0,
                'designs_needed': target_samples
            })
            continue
        
        # Check completion status
        is_complete, num_designs = check_model_completion(model_output_dir, target_samples)
        
        model_info = {
            'model_path': hdock_model,
            'model_name': model_name,
            'output_dir': model_output_dir,
            'designs_found': num_designs,
            'designs_needed': target_samples
        }
        
        if is_complete:
            model_info['status'] = 'complete'
            complete_models.append(model_info)
        else:
            model_info['status'] = 'incomplete'
            incomplete_models.append(model_info)
    
    return missing_models, incomplete_models, complete_models

def print_processing_summary(missing_models, incomplete_models, complete_models, pdb_model):
    """Print a summary of model processing status"""
    total_models = len(missing_models) + len(incomplete_models) + len(complete_models)
    
    print(f"\n=== Processing Summary for {pdb_model} ===")
    print(f"Total models: {total_models}")
    print(f"Complete models: {len(complete_models)}")
    print(f"Incomplete models: {len(incomplete_models)}")
    print(f"Missing models: {len(missing_models)}")
    
    if incomplete_models:
        print(f"\nIncomplete models (need more designs):")
        for model in incomplete_models:
            print(f"  - {model['model_name']}: {model['designs_found']}/{model['designs_needed']} designs")
    
    if missing_models:
        print(f"\nMissing models (no output directory):")
        for model in missing_models:
            print(f"  - {model['model_name']}: {model['status']}")

def count_total_models(args, pdb_dict):
    """Count the total number of unique models to be processed across all subfolders"""
    total_models = 0
    
    # Get all subdirectories in hdock_models
    if not os.path.exists(args.hdock_models):
        print(f"Error: HDock models directory does not exist: {args.hdock_models}")
        return 0
        
    items_in_hdock_models = os.listdir(args.hdock_models)
    dir_list = [item for item in items_in_hdock_models if os.path.isdir(os.path.join(args.hdock_models, item))]
    
    # Filter out temporary directories
    valid_dirs = [dir_ for dir_ in dir_list if "tmp_dir" not in dir_]
    
    print(f"Found {len(valid_dirs)} valid subdirectories in {args.hdock_models}")
    
    for pdb_folder in valid_dirs:
        pdb_folder_path = os.path.join(args.hdock_models, pdb_folder)
        if os.path.isdir(pdb_folder_path) and "tmp_dir_binding_computations" not in pdb_folder_path:
            file_pdbs_to_Design = os.path.join(pdb_folder_path, 'top_models.json')
            
            if os.path.exists(file_pdbs_to_Design):
                with open(file_pdbs_to_Design, 'r') as f:
                    lines = f.readlines()
                    
                # Get unique hdock_model entries only
                unique_hdock_models = set()
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            if "hdock_model" in item:
                                unique_hdock_models.add(item["hdock_model"])
                        except json.JSONDecodeError:
                            continue
                            
                folder_models = len(unique_hdock_models)
                total_models += folder_models
                print(f"  {pdb_folder}: {folder_models} unique models")
            else:
                hdock_models = glob.glob(os.path.join(pdb_folder_path, 'model_*.pdb'))
                folder_models = len(hdock_models)
                total_models += folder_models
                print(f"  {pdb_folder}: {folder_models} models (glob pattern)")
    
    return total_models

def get_item_by_entry_id(pdb_dict, entry_id):
    """Get item from pdb_dict by entry_id, similar to the second version"""
    return pdb_dict.get(entry_id, None)

def check_model_completion(output_dir, target_samples):
    """Check if a model has the required number of design files"""
    if not os.path.exists(output_dir):
        return False, 0
    
    # Get all PDB files in the output directory
    output_files = glob.glob(os.path.join(output_dir, "*.pdb"))
    
    # Filter out reference files from count
    design_files = [f for f in output_files if not any(ref in os.path.basename(f).lower() 
                                                      for ref in ['ref', 'reference', 'temp'])]
    
    num_designs = len(design_files)
    is_complete = num_designs >= target_samples
    
    return is_complete, num_designs

def process_hdock_model(hdock_model, heavy, light, antigen, args, summary_dir, pdb_model, ref_pdb, target_samples):
    """Process a single HDock model with enhanced functionality"""
    
    model_name = os.path.basename(hdock_model).split(".")[0]
    pdb_code = pdb_model
    output_dir = os.path.join(args.out_dir, pdb_model, model_name)
    
    # Check if already successfully processed with target number of samples
    is_complete, num_designs = check_model_completion(output_dir, target_samples)
    
    if is_complete:
        print(f"Skipping {output_dir} - already has {num_designs}/{target_samples} design files (complete).")
        return True
    elif num_designs > 0:
        print(f"Incomplete {output_dir} - has {num_designs}/{target_samples} design files. Reprocessing...")
    else:
        print(f"Processing {output_dir} - no design files found.")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    start_time = time.time()
    
    # Renumbering process
    sys.path.append(args.dymean_code_dir)
    from utils.renumber import renumber_pdb

    try:
        renumber_pdb(hdock_model, hdock_model, scheme="chothia")
        renumber_success = True
    except Exception as e:
        print(f"An error occurred during the renumbering process: {str(e)}")
        renumber_success = False
    
    # Create enhanced args for design_for_pdb based on the original version
    from easydict import EasyDict
    
    # Create a temporary reference file in chothia numbering for design input
    temp_ref_path = os.path.join(output_dir, f"{pdb_code}_temp_ref.pdb")
    try:
        renumber_pdb(ref_pdb, temp_ref_path, scheme="chothia")
    except:
        # If renumbering fails, use original
        import shutil
        shutil.copy(ref_pdb, temp_ref_path)
    
    # Use the original Arg class that matches the original design_for_pdb function
    design_args = Arg(
        pdb=hdock_model,          # pdb_path parameter
        heavy=heavy,              # heavy chain
        light=light,              # light chain  
        antigen=antigen,          # antigen chains
        config=args.config,       # config file path
        out_root=output_dir,      # output directory
        summary_dir=args.summary_dir,  # summary file path
        pdb_code=pdb_code,        # pdb code
        model=model_name,         # model name
        cdr_type=args.cdr_type,   # CDR type
        iteration=args.iteration, # iteration number
        ref_pdb=ref_pdb,          # reference pdb
        dymean_code_dir=args.dymean_code_dir
    )
    
    # Design process
    try:
        design_for_pdb(design_args)
        design_success = True
    except Exception as e:
        print(f"Design failed for {model_name}: {str(e)}")
        design_success = False

    # Clean up temporary reference file
    if os.path.exists(temp_ref_path):
        try:
            os.remove(temp_ref_path)
        except:
            pass
    
    # Check if design was successful with target number of samples
    is_complete, num_designs = check_model_completion(output_dir, target_samples)
    
    execution_time = time.time() - start_time
    final_success = renumber_success and design_success and is_complete
    
    print(f"Processed {model_name}: renumber={renumber_success}, design={design_success}, "
          f"design_files={num_designs}/{target_samples}, complete={is_complete}, time={execution_time:.2f}s")
    
    return final_success

def process_pdb_folder(pdb_folder, args, pdb_dict, summary_dir, target_samples):
    """Process a single PDB folder with enhanced missing model detection"""
    
    pdb_folder_path = os.path.join(args.hdock_models, pdb_folder)
    if not os.path.isdir(pdb_folder_path) or "tmp_dir_binding_computations" in pdb_folder_path:
        return 0
    
    pdb_model = os.path.basename(pdb_folder_path)
    print(f"\nProcessing PDB folder: {pdb_model}")
    
    # Get data from dictionary
    pdb_parts = pdb_model.rsplit('_')[0]
    original_item = pdb_dict.get(pdb_parts)
    if original_item is None:
        original_item = get_default_item(pdb_dict)
    
    heavy, light, antigen = original_item['heavy_chain'], original_item['light_chain'], original_item['antigen_chains']
    # ref_pdb = original_item.get("pdb_data_path")
    ref_pdb = original_item.get("pdb_data_path") or original_item.get("nano_source") # this is just to keep track, modify later in the test case scenario
    
    # Get unique models to design from top_models.json
    hdock_models = []
    file_pdbs_to_Design = os.path.join(pdb_folder_path, 'top_models.json')
    if os.path.exists(file_pdbs_to_Design):
        with open(file_pdbs_to_Design, 'r') as f:
            lines = f.readlines()
        
        # Parse each line as JSON and get unique hdock_model entries
        unique_hdock_models = set()
        total_entries = 0
        for line in lines:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    total_entries += 1
                    if "hdock_model" in item:
                        unique_hdock_models.add(item["hdock_model"])
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line in {file_pdbs_to_Design}: {line[:50]}...")
                    continue
        
        hdock_models = list(unique_hdock_models)
        print(f"Found {total_entries} total entries, {len(hdock_models)} unique hdock_models in top_models.json")
        
        # Debug: print first few models found
        if hdock_models:
            print(f"Sample models found: {hdock_models[:3]}")
    else:
        hdock_models = glob.glob(os.path.join(pdb_folder_path, 'model_*.pdb'))
        print(f"No top_models.json found, using glob pattern: {len(hdock_models)} models")
    
    if not hdock_models:
        print(f"No valid HDock models found in {pdb_folder_path}. Skipping this folder.")
        return 0
    
    # Identify missing, incomplete, and complete models
    missing_models, incomplete_models, complete_models = identify_missing_and_incomplete_models(
        hdock_models, args.out_dir, pdb_model, target_samples, summary_dir
    )
    
    # Print processing summary
    print_processing_summary(missing_models, incomplete_models, complete_models, pdb_model)
    
    # Process only missing and incomplete models
    models_to_process = missing_models + incomplete_models
    
    if not models_to_process:
        print(f"All models for {pdb_model} are complete. Nothing to process.")
        return len(complete_models)
    
    print(f"\nProcessing {len(models_to_process)} models that need work...")
    
    processed_count = 0
    for model_info in models_to_process:
        hdock_model = model_info['model_path']
        if process_hdock_model(hdock_model, heavy, light, antigen, args, summary_dir, pdb_model, ref_pdb, target_samples):
            processed_count += 1
    
    total_successful = len(complete_models) + processed_count
    print(f"Completed processing {pdb_model}: {total_successful}/{len(hdock_models)} models successful")
            
    return processed_count

def generate_missing_models_report(args, pdb_dict, target_samples):
    """Generate a comprehensive report of missing and incomplete models"""
    report = {
        'summary': {
            'total_pdb_folders': 0,
            'total_models': 0,
            'complete_models': 0,
            'incomplete_models': 0,
            'missing_models': 0
        },
        'details': []
    }
    
    # Get all subdirectories in hdock_models
    if not os.path.exists(args.hdock_models):
        print(f"Error: HDock models directory does not exist: {args.hdock_models}")
        return report
        
    items_in_hdock_models = os.listdir(args.hdock_models)
    dir_list = [item for item in items_in_hdock_models if os.path.isdir(os.path.join(args.hdock_models, item))]
    
    # Filter out temporary directories
    valid_dirs = [dir_ for dir_ in dir_list if "tmp_dir" not in dir_]
    
    for pdb_folder in valid_dirs:
        pdb_folder_path = os.path.join(args.hdock_models, pdb_folder)
        if not os.path.isdir(pdb_folder_path) or "tmp_dir_binding_computations" in pdb_folder_path:
            continue
            
        pdb_model = os.path.basename(pdb_folder_path)
        report['summary']['total_pdb_folders'] += 1
        
        # Get HDock models
        hdock_models = []
        file_pdbs_to_Design = os.path.join(pdb_folder_path, 'top_models.json')
        if os.path.exists(file_pdbs_to_Design):
            with open(file_pdbs_to_Design, 'r') as f:
                lines = f.readlines()
            
            unique_hdock_models = set()
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if "hdock_model" in item:
                            unique_hdock_models.add(item["hdock_model"])
                    except json.JSONDecodeError:
                        continue
            hdock_models = list(unique_hdock_models)
        else:
            hdock_models = glob.glob(os.path.join(pdb_folder_path, 'model_*.pdb'))
        
        if not hdock_models:
            continue
            
        # Analyze models
        missing_models, incomplete_models, complete_models = identify_missing_and_incomplete_models(
            hdock_models, args.out_dir, pdb_model, target_samples, args.summary_dir
        )
        
        folder_report = {
            'pdb_folder': pdb_model,
            'total_models': len(hdock_models),
            'complete_models': len(complete_models),
            'incomplete_models': len(incomplete_models),
            'missing_models': len(missing_models),
            'missing_details': missing_models,
            'incomplete_details': incomplete_models
        }
        
        report['details'].append(folder_report)
        report['summary']['total_models'] += len(hdock_models)
        report['summary']['complete_models'] += len(complete_models)
        report['summary']['incomplete_models'] += len(incomplete_models)
        report['summary']['missing_models'] += len(missing_models)
    
    return report

def main(args):
    # Load config the same way as design_for_pdb
    config, config_name = load_config(args.config)
    target_samples = get_target_samples(config, args.iteration)
    
    print(f"Loaded config: {config_name}")
    print(f"Target samples per model: {target_samples}")

    # Load dataset
    with open(args.dataset, 'r') as fin:
        lines = fin.read().strip().split('\n')

    with open(args.dataset, 'r') as fin:
        data = [json.loads(line) for line in fin]

    pdb_dict = {}
    for item in data:
        json_obj = item
        pdb_dict[json_obj["pdb"]] = json_obj

    # Create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    summary_dir = args.summary_dir
    if not os.path.exists(summary_dir) or os.path.getsize(summary_dir) == 0:
        print(f"Creating new summary file: {summary_dir}")
        with open(summary_dir, 'w') as f:
            pass
    else:
        print(f"Summary file exists with content, resuming: {summary_dir}")
    
    # Generate comprehensive missing models report
    print("\n=== Generating Missing Models Report ===")
    report = generate_missing_models_report(args, pdb_dict, target_samples)
    
    print(f"\n=== Overall Summary ===")
    print(f"Total PDB folders: {report['summary']['total_pdb_folders']}")
    print(f"Total models: {report['summary']['total_models']}")
    print(f"Complete models: {report['summary']['complete_models']}")
    print(f"Incomplete models: {report['summary']['incomplete_models']}")
    print(f"Missing models: {report['summary']['missing_models']}")
    print(f"Completion rate: {report['summary']['complete_models']}/{report['summary']['total_models']} ({100*report['summary']['complete_models']/max(1,report['summary']['total_models']):.1f}%)")
    
    # Save detailed report
    report_file = os.path.join(os.path.dirname(args.summary_dir), 'missing_models_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Detailed report saved to: {report_file}")
        
    # Count total models to process
    total_models = count_total_models(args, pdb_dict)
    print(f"Total designs expected: {total_models * target_samples}")
    
    # Get all subdirectories and process them
    items_in_hdock_models = os.listdir(args.hdock_models)
    dir_list = [item for item in items_in_hdock_models if os.path.isdir(os.path.join(args.hdock_models, item))]
    valid_dirs = [dir_ for dir_ in dir_list if "tmp_dir" not in dir_]
    
    print(f"\n=== Processing {len(valid_dirs)} PDB folders ===")
    
    # Process all folders (only missing and incomplete models)
    total_processed = 0
    for pdb_folder in tqdm(valid_dirs, desc="Processing PDB folders"):
        processed = process_pdb_folder(pdb_folder, args, pdb_dict, summary_dir, target_samples)
        total_processed += processed

    print(f"\n=== Final Summary ===")
    print(f"Total models processed in this run: {total_processed}")

def parse():
    parser = argparse.ArgumentParser(description='generation by diffab with missing model detection')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--summary_dir', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--config', type=str, required=True, help='config to the diffab model')
    parser.add_argument('--hdock_models', type=str, default=None, help='Hdock directory')
    parser.add_argument('--cdr_type', choices=['H1', 'H2', 'H3', '-'], nargs='+', help='CDR types to randomize')
    parser.add_argument('--iteration', type=int, default=None, help='Iteration number')
    parser.add_argument('--gpu', type=int, default=None, help='GPU')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('--diffab_code_dir', type=str, default=None, help='Directory to DiffAb code')
    parser.add_argument('--dymean_code_dir', type=str, default=None, help='Directory to dyMEAN code')
    
    return parser.parse_args()

if __name__ == '__main__':
    main(parse())