#!/usr/bin/env python3
"""
Script to create fold splits of nanobody/antibody datasets with clustering based on CDRH3 or antigen sequences.
Supports multiple clustering targets and thresholds.
"""

import argparse
import json
import os
import shutil
import numpy as np
from collections import defaultdict


def load_file(fpath):
    """Load JSON data from file, handling both single JSON array and line-delimited JSON."""
    with open(fpath, 'r') as fin:
        try:
            # Try loading as a single JSON array first
            data = json.load(fin)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            # If that fails, try line-delimited JSON
            fin.seek(0)
            lines = fin.read().strip().split('\n')
            return [json.loads(s) for s in lines if s.strip()]


def merge_datasets(file_paths):
    """Merge multiple dataset files, removing duplicates based on entry_id."""
    all_data = []
    unique_entries = {}
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            print(f"Loading data from {file_path}")
            data = load_file(file_path)
            all_data.extend(data)
        else:
            print(f"Warning: File {file_path} not found, skipping...")
    
    # Remove duplicates based on entry_id
    for entry in all_data:
        entry_id = entry.get("entry_id")
        if entry_id is not None and entry_id not in unique_entries:
            unique_entries[entry_id] = entry
    
    merged_data = list(unique_entries.values())
    print(f"Merged {len(all_data)} total entries into {len(merged_data)} unique entries")
    return merged_data


def exec_mmseq(cmd):
    """Execute MMseqs2 command."""
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text


def filter_flag(items, code):
    """Filter items based on the provided code."""
    res = []
    for item in items:
        satisfy = True
        for permit, key in zip(code, ['heavy_chain', 'light_chain', 'antigen_chains', 'resolution']):
            if permit == '*':
                continue
            if key == 'resolution':
                satisfy = float(item[key]) < 4.0 if permit == '0' else float(item[key]) >= 4.0
            else:
                satisfy = len(item[key]) == 0 if permit == '0' else len(item[key]) > 0
            if not satisfy:
                break
        res.append(satisfy)
    return res


def cluster_items(cluster_seq, seq_similarity_threshold, cdr_list, items):
    """Cluster items based on sequence similarity."""
    if cluster_seq is None:
        clu2idx = {item['pdb']: [i] for i, item in enumerate(items)}
        return clu2idx
    
    tmp_dir = './tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    fasta = os.path.join(tmp_dir, 'seq.fasta')
    
    with open(fasta, 'w') as fout:
        for item in items:
            pdb = item['pdb']
            seq = ""
            if cluster_seq == 'cdr':
                for cdr in cdr_list:
                    seq += item[f'cdr{cdr.lower()}_seq']
            elif cluster_seq == 'antigen':
                for ab in item['antigen_seqs']:
                    seq += ab
            fout.write(f'>{pdb}\n{seq}\n')
    
    db = os.path.join(tmp_dir, 'DB')
    cmd = f'mmseqs createdb {fasta} {db}'
    exec_mmseq(cmd)
    
    db_clustered = os.path.join(tmp_dir, 'DB_clu')
    cmd = f'mmseqs cluster {db} {db_clustered} {tmp_dir} --min-seq-id {seq_similarity_threshold}'
    res = exec_mmseq(cmd)
    
    tsv = os.path.join(tmp_dir, 'DB_clu.tsv')
    cmd = f'mmseqs createtsv {db} {db} {db_clustered} {tsv}'
    exec_mmseq(cmd)

    # Parse the clustering results
    pdb2clu, clu2idx = {}, defaultdict(list)
    with open(tsv, 'r') as fin:
        entries = fin.read().strip().split('\n')
    for entry in entries:
        cluster, pdb = entry.split('\t')
        pdb2clu[pdb] = cluster
    
    for i, item in enumerate(items):
        cluster = pdb2clu[item['pdb']]
        clu2idx[cluster].append(i)
    
    shutil.rmtree(tmp_dir)  # Clean up temporary files
    return clu2idx


def setup_output_directory(out_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir


def split_data(clu2idx, items, data_dir, valid_ratio, test_ratio, k_fold, seed):
    """Split data into train/valid/test sets with optional k-fold cross-validation."""
    np.random.seed(seed)
    fnames = ['train', 'valid', 'test']
    
    clusters = list(clu2idx.keys())
    np.random.shuffle(clusters)
    
    if k_fold == -1:
        # Standard split without k-fold
        valid_len = int(len(clusters) * valid_ratio)
        test_len = int(len(clusters) * test_ratio)
        splits = [clusters[:-valid_len - test_len], clusters[-valid_len - test_len:-test_len], clusters[-test_len:]]
        
        for fname, split in zip(fnames, splits):
            fpath = os.path.join(data_dir, f"{fname}.json")
            with open(fpath, 'w') as fout:
                for cluster in split:
                    for idx in clu2idx[cluster]:
                        items[idx]['cluster'] = cluster
                        fout.write(json.dumps(items[idx]) + '\n')
            print(f"Saved {len(split)} clusters to {fpath}")
    
    else:
        # Implement k-fold splitting
        print(f"Performing {k_fold}-fold cross-validation split")
        fold_size = len(clusters) // k_fold
        leftover = len(clusters) % k_fold
        
        for k in range(k_fold):
            fold_dir = os.path.join(data_dir, f'fold_{k}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # Determine test clusters for this fold
            start = k * fold_size + min(k, leftover)
            end = start + fold_size + (1 if k < leftover else 0)
            test_clusters = clusters[start:end]
            
            # Remaining clusters go to train and validation
            train_val_clusters = clusters[:start] + clusters[end:]
            valid_len = int(len(train_val_clusters) * valid_ratio)
            valid_clusters = train_val_clusters[-valid_len:]
            train_clusters = train_val_clusters[:-valid_len]
            
            for fname, fold_clusters in zip(fnames, [train_clusters, valid_clusters, test_clusters]):
                fpath = os.path.join(fold_dir, f"{fname}.json")
                with open(fpath, 'w') as fout:
                    for cluster in fold_clusters:
                        for idx in clu2idx[cluster]:
                            items[idx]['cluster'] = cluster
                            fout.write(json.dumps(items[idx]) + '\n')
                print(f"Fold {k}: Saved {len(fold_clusters)} clusters to {fpath}")


def parse():
    parser = argparse.ArgumentParser(description='Create fold splits of nanobody/antibody datasets')    
    # Input files
    parser.add_argument('--data_files', nargs='+', required=True,
                        help='Path(s) to input JSON file(s). Multiple files will be merged.')
    
    # Immunological molecule type
    parser.add_argument('--immuno_molecule', nargs='+', 
                        choices=['Antibody', 'Nanobody'], required=True,
                        help='Immunological molecule type(s): Antibody, Nanobody, or both. '
                             'When both are specified, files will be merged.')
    
    # Clustering parameters
    parser.add_argument('--cluster_targets', nargs='+', choices=['CDRH3', 'Ag'], required=True,
                        help='Clustering target(s): CDRH3 and/or Ag')
    

    
    # Molecule type options
    parser.add_argument('--molecule_types', nargs='+', 
                        choices=['Antibody', 'Nanobody', 'Nanobody_Antibody'],
                        help='Molecule types to process. If not specified, will be auto-detected from filenames. '
                             'When multiple files are provided, all three types will be generated by default.')
    
    # Output directory
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Base output directory for all generated files')
    
    # Split parameters
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Ratio of validation set (default: 0.1)')
    
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Ratio of test set (default: 0.1)')
    
    parser.add_argument('--k_fold', type=int, default=10,
                        help='K-fold cross-validation. Use -1 for standard split (default: 10)')
    
    # Other parameters
    parser.add_argument('--filter', type=str, default='1*1',
                        help='Filter string for heavy/light/antigen chains (default: "1*1")')
    
    parser.add_argument('--cdr', nargs='+', default=['H3'],
                        choices=['H1', 'H2', 'H3', 'L1', 'L2', 'L3'],
                        help='CDR regions to use for CDRH3 clustering (default: ["H3"])')
    
    parser.add_argument('--seed', type=int, default=2022,
                        help='Random seed (default: 2022)')
    
    return parser.parse_args()






def main(args):

    CDRH3_THRESHOLDS = [0.4, 0.3, 0.2]  # 40%, 30%, 20%
    AG_THRESHOLDS = [0.95, 0.8, 0.6]     # 95%, 80%, 60%
        
    # Load and merge data
    items = merge_datasets(args.data_files)
    
    # Determine molecule types to process based on immuno_molecule argument
    if len(args.immuno_molecule) == 1:
        # Single molecule type
        molecule_types = args.immuno_molecule
        print(f"Processing single molecule type: {molecule_types[0]}")
    else:
        # Both Antibody and Nanobody specified - create merged type
        molecule_types = ['Nanobody_Antibody']
        print(f"Processing merged molecule types: Antibody + Nanobody = Nanobody_Antibody")
    
    # Filter items
    flags = filter_flag(items, args.filter)
    items = [items[i] for i, flag in enumerate(flags) if flag]
    print(f'Valid entries after filtering: {len(items)}')
    
    # Process each combination of molecule type, clustering target and threshold
    for molecule_type in molecule_types:
        for cluster_target in args.cluster_targets:
            if cluster_target == 'CDRH3':
                cluster_seq = 'cdr'
                thresholds = CDRH3_THRESHOLDS
                seq_type = 'CDRH3'
            else:  # cluster_target == 'Ag'
                cluster_seq = 'antigen'
                thresholds = AG_THRESHOLDS
                seq_type = 'Ag'
            
            for threshold in thresholds:
                print(f"\nProcessing {molecule_type} - {seq_type} clustering with threshold {threshold}")
                
                # Create output directory for this combination
                threshold_str = int(threshold * 100)
                output_dir = os.path.join(args.out_dir, f'{molecule_type}_clustered_{seq_type}_{threshold_str}')
                setup_output_directory(output_dir)
                
                # Perform clustering
                clu2idx = cluster_items(cluster_seq, threshold, args.cdr, items)
                print(f"Created {len(clu2idx)} clusters")
                
                # Split data
                split_data(clu2idx, items, output_dir, args.valid_ratio, args.test_ratio, 
                          args.k_fold, args.seed)
                
                print(f"Results saved to {output_dir}")



if __name__ == '__main__':
    args = parse()
    main(args)