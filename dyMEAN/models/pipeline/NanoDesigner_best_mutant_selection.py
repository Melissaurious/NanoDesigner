import argparse
import os
import json
import numpy as np
import math

def file_has_content(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


def safe_float(value):
    """Convert value to float, returning inf for NaN or invalid values"""
    try:
        f_val = float(value)
        return f_val if not np.isnan(f_val) else float('inf')
    except (ValueError, TypeError):
        return float('inf')


def is_valid_numeric(value):
    """Check if value is a valid (non-NaN) number"""
    try:
        f_val = float(value)
        return not np.isnan(f_val)
    except (ValueError, TypeError):
        return False


def has_valid_json_files(csv_dir, pdb):
    """Check if there are any non-empty JSON files for the given PDB"""
    if not os.path.exists(csv_dir):
        return False
    
    json_files = [file for file in os.listdir(csv_dir) if file.endswith('.json') and pdb in file]
    
    for file in json_files:
        full_path = os.path.join(csv_dir, file)
        if os.path.getsize(full_path) > 0:
            return True
    
    return False

def passes_strict_filter(entry,energy_function):
    dG = entry.get(energy_function)
    if dG is None:
        return False

    cdr3_avg = entry.get('cdrh3_avg')
    if cdr3_avg is None:
        return False

    epitope_recall = entry.get('epitope_recall')
    if epitope_recall is None:
        return False

    try:
        dG_float = float(dG)
        if math.isnan(dG_float):
            return False
    except ValueError:
        return False

    clashes = int(entry['final_num_clashes'])
    side_chain_packed = entry.get('side_chain_packed')

    return (clashes == 0 and side_chain_packed == "Yes" and 
            float(cdr3_avg) != 0.0 and float(epitope_recall) != 0.0)

def passes_hierarchical_filter(entry,energy_function):
    """Hierarchical binding check for fallback cases"""
    # Basic requirements (same as strict)
    dG = entry.get(energy_function)
    if dG is None:
        return False

    try:
        dG_float = float(dG)
        if math.isnan(dG_float):
            return False
    except ValueError:
        return False

    clashes = int(entry['final_num_clashes'])
    side_chain_packed = entry.get('side_chain_packed')
    
    if not (clashes == 0 and side_chain_packed == "Yes"):
        return False

    # Hierarchical binding checks (in order of preference)
    cdr3_avg = entry.get('cdrh3_avg', 0)
    if cdr3_avg > 0:
        return True
    
    total_avg_cdrh_involvement = entry.get('total_avg_cdrh_involvement', 0)
    if total_avg_cdrh_involvement > 0:
        return True
    
    cdrh3_avg_all = entry.get('cdrh3_avg_all', 0)
    if cdrh3_avg_all > 0:
        return True
    
    total_avg_cdrh_involvement_all = entry.get('total_avg_cdrh_involvement_all', 0)
    if total_avg_cdrh_involvement_all > 0:
        return True
    
    return False


def main(args):
    with open(args.dataset_json, 'r') as fin:
        lines = fin.read().strip().split('\n')

    if args.objective == "denovo":
        energy_function = "FoldX_dG"
    elif args.objective == "optimization":
        energy_function = "FoldX_ddG"
    else:
        print("No valid argument")
        return

    print("iteration", args.iteration)
    print("args.csv_dir", args.csv_dir)

    for line in lines:
        item = json.loads(line)
        pdb = item['pdb']

        desired_path = os.path.dirname(os.path.dirname(args.hdock_models))

        # Check if there are any valid JSON files before proceeding
        if not has_valid_json_files(args.csv_dir, pdb):
            print(f"No valid JSON files found for PDB {pdb} in {args.csv_dir}. Skipping...")
            continue

        # Create a file to start storing the best models per iteration
        best_candidates_file = os.path.join(desired_path, f"best_candidates_iter_{args.iteration}.json")
        if not os.path.exists(best_candidates_file):
            print(f"creating {best_candidates_file} file")
            with open(best_candidates_file, "w") as f:
                pass 
        elif os.stat(best_candidates_file).st_size != 0:
            continue

        # Create a file to start storing the best models for all process
        best_candidates_aggregated = os.path.join(desired_path, f"best_candidates.json")
        if not os.path.exists(best_candidates_aggregated):
            print("creating best_candidates.json file")
            with open(best_candidates_aggregated, "w") as f:
                pass 

        # keep only json files for the current pdb 
        # keep only json files for the current pdb 
        json_files = [file for file in os.listdir(args.csv_dir) if file.endswith('.json') and pdb in file]

        pool_best_candidates = []
        all_data = []

        # for each of the n_models from pdb, choose the best one 
        for file in json_files:
            full_path_json = os.path.join(args.csv_dir, file) 
            if os.path.getsize(full_path_json) > 0:  # if it is not empty
                with open(full_path_json, 'r') as f:
                    data = [json.loads(line) for line in f]
                all_data.extend(data)  # Keep track of all data for deletion purposes
            else:
                print(f"{full_path_json}, is empty, repeat the process. Skipping for now...")
                continue

            strict_candidates = [entry for entry in data if passes_strict_filter(entry, energy_function)]

            if len(strict_candidates) >= int(args.top_n):
                # Use strict candidates
                top_current_json = sorted(strict_candidates, key=lambda x: float(x[energy_function]))
                top_models = top_current_json[:int(args.top_n)]
                print(f"File {file}: Found {len(strict_candidates)} strict candidates, using top {len(top_models)}")
            else:
                # Keep strict candidates + add hierarchical candidates to fill gaps
                hierarchical_candidates = [entry for entry in data if passes_hierarchical_filter(entry,energy_function) and not passes_strict_filter(entry,energy_function)]
                
                remaining_needed = int(args.top_n) - len(strict_candidates)
                
                # Combine both pools
                combined_pool = strict_candidates + hierarchical_candidates
                
                # Sort by energy function and take top_n
                top_current_json = sorted(combined_pool, key=lambda x: float(x[energy_function]))
                top_models = top_current_json[:int(args.top_n)]
                
                print(f"File {file}: Found {len(strict_candidates)} strict + {len(hierarchical_candidates)} hierarchical candidates")
                print(f"Using {len([m for m in top_models if passes_strict_filter(m,energy_function)])} strict + {len([m for m in top_models if not passes_strict_filter(m,energy_function)])} hierarchical = {len(top_models)} total")
            
            pool_best_candidates.extend(top_models)
                        
            # if top_current_json:
            #     top_current_json = sorted(top_current_json, key=lambda x: float(x[energy_function]))
            #     top_models = top_current_json[:int(args.top_n)]
            #     pool_best_candidates.extend(top_models)
            # else:
            #     print("No entries passed the filters for file:", file)

        # Remove duplicates from pool_best_candidates based on mod_pdb
        seen_mod_pdbs = set()
        unique_pool_candidates = []
        for candidate in pool_best_candidates:
            path = candidate["mod_pdb"]
            if path not in seen_mod_pdbs:
                seen_mod_pdbs.add(path)
                unique_pool_candidates.append(candidate)
        
        pool_best_candidates = unique_pool_candidates

        # If no valid candidates found, skip creating files
        if not pool_best_candidates:
            print(f"No valid candidates found for PDB {pdb}. Skipping file creation...")
            # Remove empty files if they were created
            if os.path.exists(best_candidates_file) and os.path.getsize(best_candidates_file) == 0:
                os.remove(best_candidates_file)
            continue

        # DELETE EXCESS OF FILES, THE ONES NOT IN TOP n
        if args.iteration != 1:
            top_model_paths = set(model["mod_pdb"] for model in pool_best_candidates)
            all_model_paths = set(entry["mod_pdb"] for entry in all_data)
            models_to_delete_paths = all_model_paths - top_model_paths

            for file_path in models_to_delete_paths:
                try:
                    os.remove(file_path)
                except OSError as e:
                    pass

        if args.iteration == 1:
            lineage_best_candidates = {}

            for candidate in pool_best_candidates:
                path = candidate["mod_pdb"]
                parts = path.split('/')
                lineage = parts[-3]  # Extract lineage from the third-to-last part of the path

                if lineage not in lineage_best_candidates:
                    lineage_best_candidates[lineage] = candidate
                else:
                    if float(candidate[energy_function]) < float(lineage_best_candidates[lineage][energy_function]):
                        lineage_best_candidates[lineage] = candidate

            best_candidates_per_lineage = list(lineage_best_candidates.values())
            best_candidates_per_lineage = sorted(best_candidates_per_lineage, key=lambda x: float(x[energy_function]))
            best_perLineage_top_n = best_candidates_per_lineage[:args.top_n]

            # Add ranks to the best candidates per lineage
            for index, candidate in enumerate(best_perLineage_top_n, start=1):
                candidate["rank"] = index

            with open(best_candidates_file, 'w') as f:
                for candidate in best_perLineage_top_n:
                    f.write(json.dumps(candidate) + '\n')

            with open(best_candidates_aggregated, 'w') as f:
                best_one_current_iter = best_perLineage_top_n[0]
                f.write(json.dumps(best_one_current_iter) + '\n')
        else:
            prev_top_models = []
            prev_iteration = int(args.iteration) - 1
            best_candidates_file_prev = os.path.join(desired_path, f"best_candidates_iter_{prev_iteration}.json")

            if os.path.exists(best_candidates_file_prev):
                with open(best_candidates_file_prev, 'r') as file:
                    prev_iteration_data = [json.loads(line) for line in file]
                prev_top_models = [data for data in prev_iteration_data if energy_function in data]

            combined_candidates = prev_top_models + pool_best_candidates

            # Remove duplicates from combined candidates
            seen_mod_pdbs = set()
            unique_combined_candidates = []
            for candidate in combined_candidates:
                path = candidate["mod_pdb"]
                if path not in seen_mod_pdbs:
                    seen_mod_pdbs.add(path)
                    unique_combined_candidates.append(candidate)

            top_models_final = sorted(unique_combined_candidates, key=lambda x: float(x[energy_function]))[:args.top_n]

            for entry in top_models_final:
                print("Iteration:", entry.get("iteration", "unknown"))

            for index, top_model in enumerate(top_models_final, start=1):
                top_model["rank"] = index

            with open(best_candidates_file, 'w') as f:
                for candidate in top_models_final:
                    f.write(json.dumps(candidate) + '\n')

            if top_models_final:
                best_one_current_iter = top_models_final[0]
                with open(best_candidates_aggregated, "a") as f:
                    f.write(json.dumps(best_one_current_iter) + '\n')


def parse():
    parser = argparse.ArgumentParser(description='inference of pipeline method')
    parser.add_argument('--iteration', type=int, required=True, help='Iteration number')
    parser.add_argument('--csv_dir', type=str, required=True, help='Path where results are')
    parser.add_argument('--hdock_models', type=str, required=True, help='Path to save generated PDBs from hdock')
    parser.add_argument('--top_n', type=int, default=15, help='Number of top mutants to be selected')
    parser.add_argument('--dataset_json', type=str, required=True, help='Path to the summary file of dataset in json format')
    parser.add_argument('--objective', type=str, required=True, help='Path to the summary file of dataset in json format')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse())