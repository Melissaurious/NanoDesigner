import argparse
import os
import json
import numpy as np


def file_has_content(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


def main(args):

    with open(args.dataset_json, 'r') as fin:
        lines = fin.read().strip().split('\n')

    if args.objective == "dg":
        energy_function = "FoldX_dG"
    else:
        energy_function = "FoldX_ddG"


    for line in lines:
        item = json.loads(line)
        pdb = item['pdb']

        desired_path = os.path.dirname(os.path.dirname(args.hdock_models))

        # Create a file to start storing the best models per iteration
        best_candidates_file = os.path.join(desired_path, f"best_candidates_iter_{args.iteration}.json")
        if not os.path.exists(best_candidates_file):
            print(f"creating {best_candidates_file} file")
            with open(best_candidates_file,"w") as f:
                pass 
        elif os.stat(best_candidates_file).st_size != 0:
            continue

        # Create a file to start storing the best models for all process
        best_candidates_aggregated = os.path.join(desired_path, f"best_candidates.json")
        if not os.path.exists(best_candidates_aggregated):
            print("creating best_candidates.json file")
            with open(best_candidates_aggregated,"w") as f:
                pass 

        # keep only json files for the current pdb 
        json_files = [file for file in os.listdir(args.csv_dir) if file.endswith('.json') and pdb in file]

        pool_best_candidates = []
        # for each of the n_models from pdb, choose the best one 
        all_data = []
        for file in json_files:
            full_path_json = os.path.join(args.csv_dir,file) 
            # print("current json full path", full_path_json)
            if os.path.getsize(full_path_json) > 0:  # if it is not empty
                with open(full_path_json, 'r') as f:
                    data = [json.loads(line) for line in f]
            else:
                print(f"{full_path_json}, is empty, repeat the process. Skipping for now...")
                continue

            # select the top model from the current json file
            top_current_json = []
            for entry in data:

                dG = entry.get(energy_function)
                if dG is None:
                    print("dG is None, skipping entry")
                    continue

                cdr3_avg = entry.get('cdr3_avg')
                if cdr3_avg is None:
                    continue

                epitope_recall = entry.get('epitope_recall')
                if cdr3_avg is None:
                    continue

                try:
                    dG_float = float(dG)
                except ValueError:
                    # If conversion to float fails, execute another action
                    print("dG cannot be converted to a float, not valid, skip")
                    continue


                clashes = int(entry['final_num_clashes'])
                side_chain_packed = entry.get('side_chain_packed')
                # epitope_recall = entry.get('epitope_recall', 0.0)
                # instability_index = entry.get('instability_index', 50.0)
                    

                # if (clashes == 0 and side_chain_packed == "Yes" and
                #         float(cdr3_avg) != 0.0 and float(epitope_recall) != 0.0):# and float(instability_index) <= 40.0):
                #     top_current_json.append(entry)

                if (clashes == 0 and side_chain_packed == "Yes" and float(cdr3_avg) != 0.0):
                    top_current_json.append(entry)
                        
            top_model_paths = set()
            if top_current_json:
                top_current_json = sorted(top_current_json, key=lambda x: float(x[energy_function]))
                top_models = top_current_json[:int(args.top_n)]
                pool_best_candidates.extend(top_models)
            else:
                print("No entries passed the filters.")


            #     else:
            top_model_paths = set()
            if top_current_json:
                top_current_json = sorted(top_current_json, key=lambda x: float(x[energy_function]))

                top_models = top_current_json[:int(args.top_n)]
                pool_best_candidates.extend(top_models)
                for model in top_models:
                    top_model_paths.add(model["mod_pdb"])
            else:
                print("No entries passed the filters, moving to next file.")


            # DELETE EXCESS OF FILES, THE ONES NOT IN TOP n
            top_model_paths = set(model["mod_pdb"] for model in pool_best_candidates)
            all_model_paths = set(entry["mod_pdb"] for entry in all_data)
            models_to_delete_paths = all_model_paths - top_model_paths

            if args.iteration != 1:
                for file_path in models_to_delete_paths:
                    try:
                        os.remove(file_path)
                    except OSError as e:
                        pass


        if args.iteration == 1:
            lineage_best_candidates = {}
            seen_mod_pdbs = set()

            for candidate in pool_best_candidates:
                path = candidate["mod_pdb"]
                if path in seen_mod_pdbs:
                    continue
                seen_mod_pdbs.add(path)
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

            with open(best_candidates_file_prev, 'r') as file:
                prev_iteration_data = [json.loads(line) for line in file]

            prev_top_models = [data for data in prev_iteration_data if energy_function in data]

            combined_candidates = prev_top_models + pool_best_candidates

            seen_mod_pdbs = set()
            unique_combined_candidates = []
            for candidate in combined_candidates:
                path = candidate["mod_pdb"]
                if path in seen_mod_pdbs:
                    continue
                seen_mod_pdbs.add(path)
                unique_combined_candidates.append(candidate)

            top_models_final = sorted(combined_candidates, key=lambda x: float(x[energy_function]))[:args.top_n]

            top_final_top_n = top_models_final[:args.top_n]

            # top_models_final = prev_top_models[:args.top_n]

            for entry in top_final_top_n:
                print(entry["iteration"])

            for index, top_model in enumerate(top_final_top_n, start=1):
                top_model["rank"] = index


            with open(best_candidates_file, 'w') as f:
                for candidate in top_final_top_n:
                    f.write(json.dumps(candidate) + '\n')

            if top_models_final:
                best_one_current_iter = top_final_top_n[0]
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

