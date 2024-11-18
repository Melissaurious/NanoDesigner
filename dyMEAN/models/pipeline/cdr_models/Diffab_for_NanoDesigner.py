#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
from tqdm import tqdm
from shutil import rmtree
import logging

logging.disable('INFO')
import glob

import yaml
from easydict import EasyDict

def load_config(config_path):
    with open(config_path, 'r', encoding="utf-8") as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name



# from diffab.tools.renumber.run import renumber

def get_default_item(pdb_dict):
    # Returns the first item in the pdb_dict. 
    # You can modify this function if you need to select a specific item instead of the first one.
    return next(iter(pdb_dict.values()))



class Arg:
    def __init__(self, pdb, heavy, light, antigen, config, out_root, summary_dir, pdb_code,model, cdr_type,iteration,ref_pdb, dymean_code_dir):
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
        self.dymean_code_dir =dymean_code_dir


def main(args):
    import sys
    current_working_dir = os.getcwd()
    print(args.diffab_code_dir)
    sys.path.append(args.diffab_code_dir)
    from diffab.tools.runner.design_for_pdb import design_for_pdb
    config, config_name = load_config(args.config)

    # Determine the required number of samples based on the iteration
    if int(args.iteration) == 1:
        num_samples = config.sampling.num_samples_iter_1
    else:
        num_samples = config.sampling.num_samples_iter_x

    # Load dataset
    with open(args.dataset, 'r') as fin:
        data = [json.loads(line) for line in fin]

    pdb_dict = {item["pdb"]: item for item in data}

    # Ensure output directory exists
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Prepare summary file
    summary_dir = os.path.join(args.out_dir, f"summary_iter_{args.iteration}.json")
    if not os.path.exists(summary_dir):
        with open(summary_dir, 'w') as f:
            pass

    # Process each line in dataset
    for line in tqdm(data):
        item = line
        heavy, light, antigen = item['heavy_chain'], item['light_chain'], item['antigen_chains']
        
        # List valid directories in hdock_models
        items_in_hdock_models = os.listdir(args.hdock_models)
        dir_l = [d for d in items_in_hdock_models if os.path.isdir(os.path.join(args.hdock_models, d))]
        
        new_dir = [d for d in dir_l if "tmp_dir" not in d]

        for pdb_folder in new_dir:
            pdb_folder_path = os.path.join(args.hdock_models, pdb_folder) 
            if os.path.isdir(pdb_folder_path) and "tmp_dir_binding_computations" not in pdb_folder_path:
                pdb_model = os.path.basename(pdb_folder_path) 
                print(pdb_model) 

                # Get original item and necessary chains
                pdb_parts = pdb_model.rsplit('_')[0]
                original_item = pdb_dict.get(pdb_parts, get_default_item(pdb_dict))

                heavy, light, antigen = original_item['heavy_chain'], original_item['light_chain'], original_item['antigen_chains']
                ref_pdb = original_item.get("pdb_data_path")

                # Get hdock models
                hdock_models = set()
                file_pdbs_to_Design = os.path.join(pdb_folder_path, 'top_models.json')
                if os.path.exists(file_pdbs_to_Design):
                    with open(file_pdbs_to_Design, 'r') as f:
                        data = [json.loads(line) for line in f]
                    hdock_models.update(item["hdock_model"] for item in data)
                else:
                    hdock_models = glob.glob(os.path.join(pdb_folder_path, 'model_*.pdb'))

                hdock_models = list(hdock_models)


                # Skip if no valid models are found
                if not hdock_models:
                    print(f"No valid HDock models found in {pdb_folder_path}. Skipping this folder.")
                    continue

                print("Models to design:", len(hdock_models))

                for hdock_model in hdock_models:
                    pdb_code = pdb_model
                    model = hdock_model.split("/")[-1].split(".")[0]
                    output_dir = os.path.join(args.out_dir, pdb_model, model)
                    print(output_dir)

                    # Check if output directory already has the required number of `.pdb` files
                    if os.path.exists(output_dir):
                        pdb_files = [file for file in os.listdir(output_dir) if file.endswith(".pdb")]
                        if len(pdb_files) >= num_samples:
                            print(f"Skipping {output_dir} as it already contains {num_samples} PDB files.")
                            continue  # Skip to the next model if sufficient files are present
                    else:
                        # Create output directory if it doesn't exist
                        os.makedirs(output_dir)
                        print(f"Created directory: {output_dir}")

                    # Proceed with renumbering and calling `design_for_pdb`
                    sys.path.append(current_working_dir)
                    from utils.renumber import renumber_pdb

                    try:
                        renumber_pdb(hdock_model, hdock_model, scheme="chothia")
                    except Exception as e:
                        print(f"An error occurred during the renumbering process: {str(e)}")
                        continue  # Skip to the next model if renumbering fails
                    
                    sys.path.append(args.diffab_code_dir)
                    design_for_pdb(Arg(hdock_model, heavy, light, antigen, args.config, output_dir, summary_dir, pdb_code, model, args.cdr_type, args.iteration, ref_pdb,args.dymean_code_dir))

def parse():
    parser = argparse.ArgumentParser(description='generation by diffab')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    # parser.add_argument('--summary_dir', type=str, required=True, help='Path to the dataset')

    parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--config', type=str, required=True, help='config to the diffab model')
    parser.add_argument('--hdock_models', type=str, default=None, help='Hdock directory')
    parser.add_argument('--cdr_type', type=str, default='H3', help='Type of CDR',
                        choices=['H3'])
    parser.add_argument('--iteration', type=int, default=None, help='Hdock directory')
    parser.add_argument('--diffab_code_dir', type=str, default=None, help='Directory to DiffAb code')
    parser.add_argument('--dymean_code_dir', type=str, default=None, help='Directory to dyMEAN code')



    return parser.parse_args()


if __name__ == '__main__':
    main(parse())



