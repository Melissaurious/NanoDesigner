import argparse
import csv
import gc
import json
import logging
import math
import multiprocessing
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
# from concurrent.futures import ThreadPoolExecutor, as_completed, concurrent
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count, Manager
from shutil import rmtree
import shutil
import zipfile
import requests
import numpy as np
import pandas as pd
from Bio import PDB
import psutil
from pathlib import Path
from collections import defaultdict

# Local imports
sys.path.append("/ibex/user/rioszemm/NanobodiesProject/dyMEAN")
from configs import IMGT, Chothia
from data.pdb_utils import Protein, AgAbComplex2, Peptide
from define_aa_contacts_antibody_nanobody_2025 import (
    clean_up_files,
    dedup_interactions,
    extract_pdb_info,
    extract_seq_info_from_pdb,
    get_cdr_residues_and_interactions,
    get_cdr_residues_and_interactions_gt_based,
    get_cdr_residues_dict,
    interacting_residues
)
from utils.network import url_get
from utils.renumber import renumber_pdb



# Global configuration dictionary to store all paths and settings
CONFIG = {}

def download_file(url, output_path):
    """Download a file from a URL if it doesn't already exist."""
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return False
    
    print(f"Downloading {url} to {output_path}")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded: {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def unzip_file(zip_path, extract_to):
    """Unzip a file to the specified directory."""
    
    # Check if zip file exists
    if not os.path.exists(zip_path):
        print(f"Zip file not found: {zip_path}")
        return False
    
    # Create extraction directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)
    
    print(f"Unzipping {zip_path} to {extract_to}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files in zip
            file_list = zip_ref.namelist()
            print(f"Zip contains {len(file_list)} files/folders")
            
            # Extract all files
            zip_ref.extractall(extract_to)
            
        print(f"Successfully unzipped to: {extract_to}")
        
        # List what was extracted (first few items)
        extracted_items = os.listdir(extract_to)
        print(f"Extracted items: {extracted_items[:5]}{'...' if len(extracted_items) > 5 else ''}")
        
        return True
        
    except Exception as e:
        print(f"Error unzipping {zip_path}: {e}")
        return False

def setup_data_directory(output_folder, tsv_file_arg=None):
    """Download SabDab data if not present."""
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Define paths
    structures_zip = os.path.join(output_folder, "all_structures.zip")
    structures_dir = os.path.join(output_folder, "all_structures")
    
    # Download structure data zip file
    structures_url = "https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/"
    
    print("Setting up data directory...")
    
    # Download the zip file if it doesn't exist
    zip_downloaded = download_file(structures_url, structures_zip)
    
    # Extract if we downloaded the file OR if the structures directory doesn't exist
    if zip_downloaded or not os.path.exists(structures_dir):
        print(f"Extracting structures to {output_folder}")
        success = unzip_file(structures_zip, output_folder)
        if not success:
            print("Failed to extract structures zip file")
            return None, None
    
    # Verify the all_structures directory was created
    if not os.path.exists(structures_dir):
        print(f"Error: all_structures directory not found after extraction: {structures_dir}")
        return None, None
    
    # Determine TSV file path
    if tsv_file_arg and os.path.exists(tsv_file_arg):
        # Use provided TSV file
        tsv_path = tsv_file_arg
        print(f"Using provided TSV file: {tsv_path}")
    else:
        # Download TSV summary to the all_structures folder
        tsv_path = os.path.join(structures_dir, "tsv_summary.tsv")
        tsv_url = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/"
        
        print(f"Downloading TSV summary to: {tsv_path}")
        tsv_downloaded = download_file(tsv_url, tsv_path)
        
        if not tsv_downloaded and not os.path.exists(tsv_path):
            print("Failed to download TSV file and no existing file found")
            return None, None
    
    print(f"Data setup complete:")
    print(f"  - TSV file: {tsv_path}")
    print(f"  - Structures directory: {structures_dir}")
    
    return tsv_path, structures_dir


def get_system_info():
    """Get system information for optimal worker allocation"""
    cpu_count = mp.cpu_count()
    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Recommended workers: {get_optimal_workers()}")
    print("=" * 30)
    
    return {
        'cpu_count': cpu_count,
        'logical_cores': logical_cores,
        'physical_cores': physical_cores,
        'memory_gb': memory_gb
    }

def get_optimal_workers(task_type='io_bound'):
    """Calculate optimal number of workers based on task type"""
    cpu_count = mp.cpu_count()
    
    if task_type == 'io_bound':
        # For I/O bound tasks (file operations, network requests)
        return min(cpu_count * 2, 32)  # Cap at 32 to avoid overhead
    elif task_type == 'cpu_bound':
        # For CPU bound tasks (calculations, processing)
        return max(1, cpu_count - 1)  # Leave one core free
    else:
        return cpu_count



def count_entries_in_file(file_path):
    """Count the number of entries in a JSON file (one JSON object per line)"""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return 0
    
    count = 0
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        json.loads(line)
                        count += 1
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error counting entries in {file_path}: {e}")
        return 0
    
    return count

def get_processed_ids(file_path, id_field='pdb_id'):
    """Get set of already processed IDs from a JSON file"""
    processed_ids = set()
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return processed_ids
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if id_field in item:
                            processed_ids.add(item[id_field])
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error reading processed IDs from {file_path}: {e}")
    
    return processed_ids

def filter_unprocessed_entries(entries, processed_ids, id_field='pdb_id'):
    """Filter out entries that have already been processed"""
    unprocessed = []
    for entry in entries:
        entry_id = entry.get(id_field)
        if entry_id and entry_id not in processed_ids:
            unprocessed.append(entry)
    return unprocessed




def run_nanobody_antibody_pipeline(tsv_file_path, output_folder, selected_types=None):
    """
    Run the full pipeline for processing antibody and nanobody data.
    
    Args:
        tsv_file_path: Path to the input TSV file.
        output_folder: Folder to save output files.
        selected_types: List of types to process (default: ["Antibody", "Nanobody"]).
    
    Returns:
        Path to the processed TSV file.
    """
    # Access global variables instead of defining local ones
    global output_tsv_file
    
    if selected_types is None:
        selected_types = ["Antibody", "Nanobody"]
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 1: Process the TSV file to identify valid antibody and nanobody entries
    print(f"Step 1: Processing TSV file {tsv_file_path}")
    stats = process_antibody_nanobody_data(tsv_file_path, output_folder, selected_types)
    
    # Use the global output_tsv_file instead of defining a local valid_tsv_file
    print("valid_tsv_file", CONFIG.get('output_tsv_file'))
    
    return output_tsv_file


def process_antibody_nanobody_data(input_tsv_path, output_folder, selected_types=None):
    """
    Comprehensive function to process antibody and nanobody data from a TSV file.
    
    Args:
        input_tsv_path: Path to the input TSV file.
        output_folder: Folder to save output files.
        selected_types: List of types to process (default: ["Antibody", "Nanobody"]).
    
    Returns:
        Dictionary with statistics about processed entries.
    """
    if selected_types is None:
        selected_types = ["Antibody", "Nanobody"]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Output files
    # output_tsv_file = os.path.join(output_folder, f"valid_tsv_lines.tsv")
    output_tsv_file = CONFIG.get('output_tsv_file')
    stats_file = os.path.join(output_folder, "processing_stats.json")
    
    # Process the TSV file
    stats = create_enhanced_tsv(input_tsv_path, output_tsv_file, selected_types)
    
    # Save statistics to JSON file
    with open(stats_file, 'w') as f:
        json.dump(stats, f)
    
    # Print summary statistics
    print(f"Processing completed. Results saved to {output_tsv_file}")
    print(f"Statistics saved to {stats_file}")
    print(f"Total entries processed: {stats['total_entries']}")
    print(f"Valid entries: {stats['valid_entries']}")
    
    for entry_type in selected_types:
        print(f"\n{entry_type} Statistics:")
        total = stats[entry_type]['total']
        if total > 0:
            passed = stats[entry_type]['passed']
            failed = stats[entry_type]['failed']
            print(f"  Total: {total}")
            print(f"  Passed: {passed} ({passed/total*100:.1f}%)")
            print(f"  Failed: {failed} ({failed/total*100:.1f}%)")
            
            if failed > 0:
                print("  Top rejection reasons:")
                sorted_reasons = sorted(
                    stats[entry_type]["rejection_reasons"].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                for reason, count in sorted_reasons[:5]:  # Show top 5 reasons
                    print(f"    - {reason}: {count} ({count/failed*100:.1f}%)")
        else:
            print(f"  No {entry_type} entries found")
    
    return stats


def clean_value(value):
    """Clean a value by stripping and handling NA values."""
    if not value or str(value).strip().upper() == "NA":
        return ""
    return str(value).strip()

def clean_chain_id(chain_id):
    """Clean and validate a chain identifier."""
    if not chain_id or str(chain_id).upper() == "NA":
        return ""
    return str(chain_id).strip()



def determine_entry_type(entry):
    """
    Determine if an entry is an Antibody or a Nanobody based on chains.
    
    Args:
        entry: Dictionary with entry data.
        
    Returns:
        "Antibody", "Nanobody", or None if invalid.
    """
    heavy_chain = clean_chain_id(entry.get('heavy_chain', ''))
    light_chain = clean_chain_id(entry.get('light_chain', ''))
    
    if not heavy_chain:
        return None  # Invalid entry - no heavy chain
        
    if light_chain:
        return "Antibody"  # Has both heavy and light chain = Antibody
    else:
        return "Nanobody"  # Has heavy chain but no light chain = Nanobody
    


def get_rejection_reasons(entry, entry_type):
    """
    Determine all reasons why an entry might be rejected based on its type.
    
    Args:
        entry: Dictionary containing entry data.
        entry_type: "Antibody" or "Nanobody".
    
    Returns:
        List of rejection reasons, or ["Passed"] if entry passes validation.
    """
    if not entry or not isinstance(entry, dict):
        return ["Invalid entry format"]
    
    reasons = []
    
    # Check PDB ID
    pdb_id = entry.get('pdb', '')
    if not pdb_id or pdb_id == "NA":
        reasons.append("Missing PDB ID")
    
    # Check chains based on type
    heavy_chain = clean_chain_id(entry.get('heavy_chain', ''))
    light_chain = clean_chain_id(entry.get('light_chain', ''))
    
    if not heavy_chain:
        reasons.append("Missing heavy chain")
    
    if entry_type == "Antibody":
        # Antibodies need both heavy and light chains
        if not light_chain:
            reasons.append("Missing light chain (required for Antibody)")
    elif entry_type == "Nanobody":
        # Nanobodies need heavy chain but should not have light chain
        if light_chain:
            reasons.append("Has light chain (should not for Nanobody)")
    
    # Check antigen chains
    antigen_chains = entry.get('antigen_chains', [])
    valid_antigen_chains = [
        chain for chain in antigen_chains 
        if chain and chain.upper() != "NA"
    ]
    
    if not valid_antigen_chains:
        reasons.append("Missing valid antigen chains")
    
    # Check antigen type
    antigen_types = entry.get('antigen_type', [])
    has_valid_antigen_type = False
    
    for atype in antigen_types:
        if atype and atype.upper() != "NA":
            if 'protein' in atype.lower() or 'peptide' in atype.lower():
                has_valid_antigen_type = True
                break
    
    if not has_valid_antigen_type:
        reasons.append("No valid protein/peptide antigen type")
    
    # Check resolution
    try:
        resolution = float(entry.get('resolution', '999'))
        if resolution > CONFIG.get('MAX_RESOLUTION'):
            # reasons.append(f"Resolution too high ({resolution} > {MAX_RESOLUTION})")
            reasons.append(f"Resolution too high")
    except (ValueError, TypeError):
        reasons.append("Invalid resolution value")
    
    return reasons if reasons else ["Passed"]


def create_enhanced_tsv(input_tsv_path, output_tsv_path, selected_types=None):
    """
    Create enhanced TSV file with separate handling for Antibodies and Nanobodies.
    Only saves entries that pass validation.
    
    Args:
        input_tsv_path: Path to the input TSV file.
        output_tsv_path: Path to save the output TSV file.
        selected_types: List of types to process (default: ["Antibody", "Nanobody"]).
    
    Returns:
        Dictionary with processing statistics.
    """
    if selected_types is None:
        selected_types = ["Antibody", "Nanobody"]
    
    # Columns to keep in the output
    columns_to_keep = [
        'pdb', 'heavy_chain', 'light_chain', 'antigen_chains', 
        'antigen_type', 'resolution'
    ]
    
    # Statistics counters
    stats = {
        "total_entries": 0,
        "valid_entries": 0,
        "invalid_format": 0
    }
    
    for entry_type in selected_types:
        stats[entry_type] = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "rejection_reasons": defaultdict(int)
        }
    
    with open(input_tsv_path, 'r') as infile, open(output_tsv_path, 'w') as outfile:
        # Read header
        header = infile.readline().strip().split('\t')
        
        # Create output header with additional columns
        output_header = columns_to_keep + ['status', 'rejection_reasons', 'type']
        outfile.write('\t'.join(output_header) + '\n')
        
        # Process each line
        for line_num, line in enumerate(infile, 2):  # Start from 2 for line number (after header)
            stats["total_entries"] += 1
            
            fields = line.strip().split('\t')
            if len(fields) < 17:  # Skip malformed lines
                stats["invalid_format"] += 1
                continue
            
            # Extract entry information
            entry = {
                'pdb': clean_value(fields[0]),
                'heavy_chain': clean_value(fields[1]),
                'light_chain': clean_value(fields[2]),
                'antigen_chains': [clean_value(chain) for chain in fields[4].split(' | ')],
                'antigen_type': [clean_value(type) for type in fields[5].split('|')],
                'resolution': clean_value(fields[16])
            }
            
            # Filter out empty values
            entry['antigen_chains'] = [chain for chain in entry['antigen_chains'] if chain]
            entry['antigen_type'] = [atype for atype in entry['antigen_type'] if atype]
            
            # Determine entry type (Antibody or Nanobody)
            entry_type = determine_entry_type(entry)
            
            # Skip invalid entries or types not selected for processing
            if entry_type is None or entry_type not in selected_types:
                continue
                
            stats[entry_type]["total"] += 1
            
            # Get rejection reasons for this type
            reasons = get_rejection_reasons(entry, entry_type)
            if reasons == ["Passed"]:
                status = 'Passed'
                rejection_reasons = ''
                stats[entry_type]["passed"] += 1
                stats["valid_entries"] += 1
                
                # Format output fields for PASSED entries only
                output_fields = []
                for col in columns_to_keep:
                    if col == 'antigen_chains':
                        output_fields.append(' | '.join(entry[col]))
                    elif col == 'antigen_type':
                        output_fields.append('|'.join(entry[col]))
                    else:
                        output_fields.append(entry[col])
                
                # Add status, reasons, and type
                output_fields.extend([status, rejection_reasons, entry_type])
                
                # Write to output file (ONLY PASSED ENTRIES)
                outfile.write('\t'.join(output_fields) + '\n')
                
            else:
                status = 'Failed'
                rejection_reasons = ','.join(reasons)
                stats[entry_type]["failed"] += 1
                
                # Track rejection reasons
                for reason in reasons:
                    stats[entry_type]["rejection_reasons"][reason] += 1
                
                # DO NOT WRITE FAILED ENTRIES TO OUTPUT FILE
    
    # Convert defaultdicts to regular dicts for JSON serialization
    for entry_type in selected_types:
        stats[entry_type]["rejection_reasons"] = dict(stats[entry_type]["rejection_reasons"])
    
    return stats


def load_filtered_entries(tsv_file, entry_type):
    """
    Load filtered entries of a specific type from the processed TSV file.
    
    Args:
        tsv_file: Path to the processed TSV file.
        entry_type: Type of entries to load ("Antibody" or "Nanobody").
    
    Returns:
        List of entries as dictionaries.
    """
    filtered_entries = []
    
    with open(tsv_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # Only process entries of the selected type and with 'Passed' status
            if row['type'] == entry_type and row['status'] == 'Passed':
                # Skip entries with invalid or missing antigen chains
                if not row['antigen_chains'] or row['antigen_chains'] == 'NA':
                    continue
                    
                # Parse antigen chains
                antigen_chains = [chain.strip() for chain in row['antigen_chains'].split('|') if chain.strip()]
                if not antigen_chains:
                    continue

                # Handle light chain (None for Nanobodies)
                light_chain = None if entry_type == "Nanobody" or not row['light_chain'] or row['light_chain'] == 'NA' else row['light_chain']
                
                # Create entry dictionary
                entry = {
                    'pdb': row['pdb'],
                    'heavy_chain': row['heavy_chain'],
                    'light_chain': light_chain,
                    'antigen_chains': antigen_chains,
                    'antigen_type': row['antigen_type'].split('|'),
                    'resolution': row['resolution']
                }
                
                filtered_entries.append(entry)
    
    print(f"Loaded {len(filtered_entries)} filtered {entry_type} entries")
    return filtered_entries


def fetch_from_sabdab(identifier, numbering_scheme, save_path, tries=5):
    # example identifier: 3ogo
    # example numbering_scheme: 'imgt' or 'chothia'

    print(f"fetching {identifier} from sabdab")

    identifier = identifier.lower()
    url = f'https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/{identifier}/?scheme={numbering_scheme}'

    text = url_get(url, tries)
    if text is None:
        return None

    with open(save_path, 'w') as file:
        file.write(text.text)

    data = {
        'pdb': save_path
    }
    return data


def process_entry(entry_data, pdb_counter):
    """Function to process each entry with robust validation."""
    try:
        # Handle both JSON string and dictionary inputs
        if isinstance(entry_data, str):
            item = json.loads(entry_data)
        else:
            item = entry_data

        # print("item",item)
        pdb_id = item['pdb']
        # current_raw_path = os.path.join(raw_dir, f"{pdb_id}.pdb")
        current_raw_path = os.path.join(CONFIG.get('raw_dir'), f"{pdb_id}.pdb")

        print("current_raw_path", current_raw_path)
        
        H = item["heavy_chain"]
        L = item["light_chain"] if item["light_chain"] else ""
        Ag = item["antigen_chains"]
        numbering=CONFIG.get('numbering')
        
        print("numbering", numbering)

        # Convert null light_chain to empty string
        if item["light_chain"] is None:
            item["light_chain"] = ""


        # Validate that we have at least one antigen chain
        if not Ag:
            print(f"Skipping {pdb_id} - No antigen chains found")
            return None
            
        ordered_ag = ''.join(sorted(Ag))  # Sort antigen chains alphabetically
        item["entry_id"] = f"{pdb_id}_{H}_{L}_{ordered_ag}"  # Create entry_id based on formatted strings

        # # Increment counter for the specific pdb_id
        # pdb_counter[pdb_id] += 1
        # pdb_suffix = pdb_counter[pdb_id]  # Unique suffix for this pdb_id instance
    

        pdb_counter['counts'][pdb_id] += 1
        pdb_suffix = pdb_counter['counts'][pdb_id] 

        # Fetch the PDB file from SabDab if it does not exist locally
        if not os.path.exists(current_raw_path):
            data = fetch_from_sabdab(pdb_id, numbering, current_raw_path)
            if data is None:
                print(f"Failed to fetch {pdb_id} from SabDab")
                return None

        # Proceed only if the PDB file is available and valid
        if not os.path.exists(current_raw_path) or os.stat(current_raw_path).st_size == 0:
            print(f"PDB file for {pdb_id} is missing or empty")
            return None
            
        numbering_schemes = ["imgt", "chothia"]
        alternative_scheme = next(scheme for scheme in numbering_schemes if scheme != numbering)

        # Create a temporary directory for potential renumbering
        tmp_dir = os.path.join(CONFIG.get('output_folder_specific'), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_pdb = os.path.join(tmp_dir, f"{pdb_id}_tmp.pdb")

        # First try with the set numbering scheme
        # If it fails, there may be the case that by error the raw dir entries were
        # renumbered to the other numbering scheme
        try:
            # Try first with the requested numbering scheme
            try:
                # Create the AgAbComplex2 object for CDR extraction
                cplx = AgAbComplex2.from_pdb(
                    current_raw_path, 
                    item['heavy_chain'], 
                    item['light_chain'],  
                    item['antigen_chains'], 
                    numbering=numbering, 
                    skip_epitope_cal=True
                )
                print(f"Successfully created complex for {pdb_id} with {numbering} numbering")
            except Exception as e1:
                print(f"Failed to create complex for {pdb_id} with {numbering} numbering: {e1}")
                
                # Try with alternative numbering scheme
                print(f"Attempting with alternative numbering scheme ({alternative_scheme}) for {pdb_id}")
                
                # Copy the file to temporary location
                shutil.copy(current_raw_path, tmp_pdb)
                
                try:
                    # Try with alternative scheme first
                    cplx = AgAbComplex2.from_pdb(
                        tmp_pdb, 
                        item['heavy_chain'], 
                        item['light_chain'],  
                        item['antigen_chains'], 
                        numbering=alternative_scheme, 
                        skip_epitope_cal=True
                    )
                    
                    print(f"Successfully created complex for {pdb_id} with {alternative_scheme} numbering")
                    
                    # If successful with alternative scheme, renumber to the expected scheme
                    print(f"Renumbering {pdb_id} to match expected {numbering} scheme")
                    renumber_pdb(tmp_pdb, tmp_pdb, scheme=numbering)
                    
                    # Try again with the correct numbering after renumbering
                    cplx = AgAbComplex2.from_pdb(
                        tmp_pdb, 
                        item['heavy_chain'], 
                        item['light_chain'],  
                        item['antigen_chains'], 
                        numbering=numbering, 
                        skip_epitope_cal=True
                    )
                    
                    # If successful, update the original file
                    print(f"Successfully renumbered {pdb_id} to {numbering}")
                    shutil.copy(tmp_pdb, current_raw_path)
                    
                except Exception as e2:
                    print(f"Failed with both numbering schemes for {pdb_id}: {e2}")
                    return None
            
            # Check heavy chain
            heavy_chain = cplx.get_heavy_chain()
            if not heavy_chain:
                print(f"Skipping {pdb_id} - Failed to extract heavy chain")
                return None
                
            # Check light chain (for Antibodies)
            if item['light_chain'] and not cplx.get_light_chain():
                print(f"Skipping {pdb_id} - Failed to extract light chain")
                return None
                
            # Check antigen chains
            antigen_chains = cplx.get_antigen()
            if not antigen_chains:
                print(f"Skipping {pdb_id} - Failed to extract antigen chains")
                return None
            
            # Extract sequences
            item['heavy_chain_seq'] = heavy_chain.get_seq()
            item['light_chain_seq'] = cplx.get_light_chain().get_seq() if cplx.get_light_chain() else ""
            
            # Extract antigen sequences
            antigen_seqs = []
            for _, chain in antigen_chains:
                if chain:
                    antigen_seqs.append(chain.get_seq())
            
            # Skip if no antigen sequences found
            if not antigen_seqs:
                print(f"Skipping {pdb_id} - No antigen sequences extracted")
                return None
                
            item['antigen_seqs'] = antigen_seqs

            # Extract CDR positions and sequences
            # Track if any essential CDR extraction fails
            extraction_failed = False
            
            for c in ['H', 'L']:
                # Skip light chain processing if no light chain present (Nanobody case)
                if c == 'L' and not item['light_chain']:
                    # Initialize empty values for light chain CDRs
                    for i in range(1, 4):
                        cdr_name = f'{c}{i}'.lower()
                        item[f'cdr{cdr_name}_pos'] = []
                        item[f'cdr{cdr_name}_seq'] = ""
                    continue
                
                for i in range(1, 4):
                    cdr_name = f'{c}{i}'.lower()
                    try:
                        cdr = cplx.get_cdr(cdr_name)
                        if cdr:
                            # Save CDR positions
                            cdr_positions = cplx.get_cdr_pos(cdr_name)
                            item[f'cdr{cdr_name}_pos'] = cdr_positions
                            item[f'cdr{cdr_name}_seq'] = cdr.get_seq()
                        else:
                            item[f'cdr{cdr_name}_pos'] = []
                            item[f'cdr{cdr_name}_seq'] = ""
                            # For heavy chain CDRs, missing any is a showstopper
                            if c == 'H':
                                print(f"Skipping {pdb_id} - Missing CDR {cdr_name}")
                                extraction_failed = True
                                break
                    except Exception as e:
                        print(f"Error extracting CDR {cdr_name} for {pdb_id}: {e}")
                        item[f'cdr{cdr_name}_pos'] = []
                        item[f'cdr{cdr_name}_seq'] = ""
                        # Any exception for heavy chain CDRs is a showstopper
                        if c == 'H':
                            extraction_failed = True
                            break
                
                # Break out of CDR loop if there's a problem with heavy chain CDRs
                if extraction_failed:
                    break
            
            # If essential CDR extraction failed, skip this entry
            if extraction_failed:
                return None

            # Add numbering scheme to the item
            item["numbering"] = numbering

            # Construct the pdb_data_path with the unique suffix and add it to the item
            pdb_data_filename = f"{pdb_id}_{pdb_suffix}.pdb"
            item["pdb_data_path"] = os.path.join(CONFIG.get('output_folder_specific'), pdb_data_filename)

            # Save the reconstructed PDB to output_folder_specific
            try:
                if not os.path.exists(item["pdb_data_path"]):
                    chains_to_keep = [item['heavy_chain']] + item['antigen_chains']
                    if item['light_chain']:  # Only add light chain if it exists
                        chains_to_keep.append(item['light_chain'])
                    
                    # Use the potentially renumbered PDB if it exists
                    source_pdb = tmp_pdb if os.path.exists(tmp_pdb) else current_raw_path
                    protein = Protein.from_pdb(source_pdb, chains_to_keep)
                    protein.to_pdb(item["pdb_data_path"])
                    
                    # Verify the output file was created successfully
                    if not os.path.exists(item["pdb_data_path"]) or os.stat(item["pdb_data_path"]).st_size == 0:
                        print(f"Failed to save reconstructed PDB for {pdb_id}")
                        return None
            except Exception as e:
                print(f"Error saving reconstructed PDB for {pdb_id}: {e}")
                return None

            # Clean up temporary file if it exists
            if os.path.exists(tmp_pdb):
                try:
                    os.remove(tmp_pdb)
                except:
                    pass

            # Return the item to be written to the output JSON file
            return item
            
        except Exception as e:
            print(f"Error creating complex for {pdb_id}: {e}")
            
            # Clean up temporary file if it exists
            if os.path.exists(tmp_pdb):
                try:
                    os.remove(tmp_pdb)
                except:
                    pass
            
            return None

    except Exception as e:
        if 'pdb_id' in locals():
            print(f"Error processing {pdb_id}: {e}")
        else:
            print(f"Error processing entry: {e}")
        return None



def parallel_process_and_write(entries, process_func, output_file, max_workers=10):
    """Process entries and write results immediately to avoid memory buildup."""
    write_lock = threading.Lock()
    processed_count = 0

    # Initialize thread-safe counter
    pdb_counter = {
        'counts': defaultdict(int),
        'lock': threading.Lock()
    }
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_entry = {executor.submit(process_func, entry, pdb_counter): entry for entry in entries}
        
        # Process results as they complete
        for future in as_completed(future_to_entry):
            try:
                result = future.result()
                if result:
                    # Write immediately with thread safety
                    with write_lock:
                        with open(output_file, 'a') as f:
                            f.write(json.dumps(result) + '\n')
                        processed_count += 1
                        if processed_count % 100 == 0:  # Progress indicator
                            print(f"Processed {processed_count} entries...")
            except Exception as e:
                entry = future_to_entry[future]
                pdb_id = entry.get('pdb', 'unknown') if isinstance(entry, dict) else 'unknown'
                print(f"Error processing {pdb_id}: {e}")
    
    return processed_count



def get_cdr_residues_and_interactions_vhh_only(item, cdr_dict, results):
    """
    Process CDR residues and interaction results to extract epitope information
    and calculate CDR involvement metrics. ONLY HEAVY CHAIN CDR interactions are included in epitope.
    Designed specifically for VHH/nanobody analysis.
    """
    # Get chain identifiers
    heavy_chain = item.get("heavy_chain")
    light_chain = item.get("light_chain")
    
    # Identify heavy chain key in cdr_dict and results
    heavy_chain_key = None
    if heavy_chain in cdr_dict:
        heavy_chain_key = heavy_chain
    elif heavy_chain:
        for key in cdr_dict:
            if key in results and 'CDR3' in cdr_dict[key]:
                heavy_chain_key = key
                break
    
    print(f"VHH analysis - Heavy chain key: {heavy_chain_key}")
    
    # Initialize interaction lists - VHH focused
    epitope_model = []  # ONLY HEAVY CHAIN CDR interactions
    all_interactions = []  # All interactions for reference
    framework_interactions = []  # Framework interactions
    
    cdrh1_interactions_to_ag = []
    cdrh2_interactions_to_ag = []
    cdrh3_interactions_to_ag = []
    
    def normalize_pos(pos):
        return str(pos)
    
    # Create position sets for HEAVY CHAIN CDRs only
    cdrh1_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(heavy_chain_key, {}).get('CDR1', [])}
    cdrh2_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(heavy_chain_key, {}).get('CDR2', [])}
    cdrh3_positions = {normalize_pos(pos) for _, pos, _ in cdr_dict.get(heavy_chain_key, {}).get('CDR3', [])}
    
    # Combine heavy chain CDR positions for quick lookup
    heavy_cdr_positions = cdrh1_positions | cdrh2_positions | cdrh3_positions
    
    # Process ONLY heavy chain interactions
    if heavy_chain_key and heavy_chain_key in results:
        for antigen, interactions in results[heavy_chain_key].items():
            for interaction in interactions:
                all_interactions.append(interaction)
                
                # Check if interaction has the expected format
                if not isinstance(interaction, (list, tuple)) or len(interaction) < 2:
                    continue
                    
                ag_res, ab_res = interaction
                
                # Check if ab_res has the expected format
                if not isinstance(ab_res, (list, tuple)) or len(ab_res) < 3:
                    continue
                    
                ab_chain, ab_pos, ab_aa = ab_res
                norm_pos = normalize_pos(ab_pos)
                
                # Check if this is a HEAVY CHAIN CDR interaction
                if norm_pos in heavy_cdr_positions:
                    epitope_model.append(interaction)  # ONLY add heavy chain CDR interactions
                    
                    # Assign to specific heavy chain CDR
                    if norm_pos in cdrh1_positions:
                        cdrh1_interactions_to_ag.append(interaction)
                    elif norm_pos in cdrh2_positions:
                        cdrh2_interactions_to_ag.append(interaction)
                    elif norm_pos in cdrh3_positions:
                        cdrh3_interactions_to_ag.append(interaction)
                else:
                    framework_interactions.append(interaction)
    
    # Also process light chain interactions for completeness, but DON'T include in epitope
    light_chain_key = None
    if light_chain in cdr_dict:
        light_chain_key = light_chain
    elif light_chain:
        for key in cdr_dict:
            if key != heavy_chain_key and key in results:
                light_chain_key = key
                break
    
    if light_chain_key and light_chain_key in results:
        print(f"Processing light chain {light_chain_key} for reference (not included in VHH epitope)")
        for antigen, interactions in results[light_chain_key].items():
            for interaction in interactions:
                all_interactions.append(interaction)
                # All light chain interactions go to framework for VHH analysis
                framework_interactions.append(interaction)
    
    # Remove duplicates
    epitope_model = dedup_interactions(epitope_model)
    all_interactions = dedup_interactions(all_interactions)
    framework_interactions = dedup_interactions(framework_interactions)
    
    # Deduplicate heavy chain CDR-specific lists
    cdrh1_interactions_to_ag = dedup_interactions(cdrh1_interactions_to_ag)
    cdrh2_interactions_to_ag = dedup_interactions(cdrh2_interactions_to_ag)
    cdrh3_interactions_to_ag = dedup_interactions(cdrh3_interactions_to_ag)
    
    # Calculate CDR lengths and involvement - HEAVY CHAIN ONLY
    cdrh1_len = len(cdr_dict.get(heavy_chain_key, {}).get('CDR1', []))
    cdrh2_len = len(cdr_dict.get(heavy_chain_key, {}).get('CDR2', []))
    cdrh3_len = len(cdr_dict.get(heavy_chain_key, {}).get('CDR3', []))
    
    if cdrh1_len == 0:
        cdrh1_len = len(item.get("cdrh1_seq", ""))
    if cdrh2_len == 0:
        cdrh2_len = len(item.get("cdrh2_seq", ""))
    if cdrh3_len == 0:
        cdrh3_len = len(item.get("cdrh3_seq", ""))
    
    cdrh1_involvement = (len(cdrh1_interactions_to_ag)/cdrh1_len)*100 if cdrh1_len > 0 else 0.0
    cdrh2_involvement = (len(cdrh2_interactions_to_ag)/cdrh2_len)*100 if cdrh2_len > 0 else 0.0
    cdrh3_involvement = (len(cdrh3_interactions_to_ag)/cdrh3_len)*100 if cdrh3_len > 0 else 0.0
    
    total_avg_cdrh_involvement = float((cdrh1_involvement + cdrh2_involvement + cdrh3_involvement) / 3)
    
    # Print summary - VHH focused
    total_interactions = len(all_interactions)
    vhh_epitope_interactions = len(epitope_model)
    framework_count = len(framework_interactions)
    
    print(f"VHH Analysis Summary:")
    print(f"Total interactions: {total_interactions}")
    print(f"VHH epitope (CDRH only): {vhh_epitope_interactions}")
    print(f"Framework + Light chain: {framework_count}")
    print(f"CDRH1: {len(cdrh1_interactions_to_ag)}/{cdrh1_len} = {cdrh1_involvement:.1f}%")
    print(f"CDRH2: {len(cdrh2_interactions_to_ag)}/{cdrh2_len} = {cdrh2_involvement:.1f}%")
    print(f"CDRH3: {len(cdrh3_interactions_to_ag)}/{cdrh3_len} = {cdrh3_involvement:.1f}%")
    
    # Update item - VHH specific metrics
    item["epitope"] = epitope_model  # VHH epitope (CDRH-only)
    item["all_interactions"] = all_interactions
    item["framework_interactions"] = framework_interactions
    
    # Heavy chain CDR metrics
    item["cdrh1_interactions_to_ag"] = cdrh1_interactions_to_ag
    item["cdrh2_interactions_to_ag"] = cdrh2_interactions_to_ag
    item["cdrh3_interactions_to_ag"] = cdrh3_interactions_to_ag
    item["total_avg_cdrh_involvement"] = total_avg_cdrh_involvement
    item["cdrh3_avg"] = cdrh3_involvement
    item["cdrh2_avg"] = cdrh2_involvement
    item["cdrh1_avg"] = cdrh1_involvement
    
    # Clear any light chain metrics to avoid confusion
    item["total_avg_cdrl_involvement"] = 0.0
    item["cdrl1_avg"] = 0.0
    item["cdrl2_avg"] = 0.0
    item["cdrl3_avg"] = 0.0
    
    return item


def process_single_interaction_vhh(item, tmp_dir_for_interacting_aa):
    """
    Modified version of your original function for VHH analysis.
    Only processes heavy chain and epitope contains only CDRH interactions.
    """
    try:
        pdb_file = item["pdb_data_path"]
        heavy_chain = item["heavy_chain"]
        light_chain = item["light_chain"]
        antigen_chains = item["antigen_chains"]
        
        # Get the CDR residues dictionary
        cdr_dict = get_cdr_residues_dict(item, pdb_file)
        
        # For VHH: Remove all light chain CDR data from dictionary
        if light_chain and light_chain in cdr_dict:
            # Remove light chain CDRs completely
            del cdr_dict[light_chain]
        
        # Get the base name of the file
        filename = os.path.basename(pdb_file)
        # Split the filename into name and extension
        model_name, extension = os.path.splitext(filename)
        
        # For VHH analysis: only process heavy chain
        immuno_chains = [heavy_chain]
        # Skip light chain processing entirely for VHH
        # if light_chain:  # Process light chain for completeness but exclude from epitope
        #     immuno_chains.append(light_chain)
        
        print(f"VHH mode: Processing {item.get('pdb', 'unknown')}")
        
        aggregated_results = {}
        
        for immuno_chain in immuno_chains:
            immuno_results_dict = {}
            
            for antigen in antigen_chains:
                # Create temporary directory for this specific interaction
                tmp_ = os.path.join(tmp_dir_for_interacting_aa, f"{model_name}_{immuno_chain}_to_{antigen}")
                os.makedirs(tmp_, exist_ok=True)
                tmp_pdb = os.path.join(tmp_, f"{model_name}_{immuno_chain}_to_{antigen}.pdb")
                
                # Fixed: properly construct chains list
                chains_to_reconstruct = [immuno_chain]  # immuno_chain is already a string
                chains_to_reconstruct.extend(antigen_chains)  # antigen_chains is a list of strings
                
                if not os.path.exists(tmp_pdb):
                    try:
                        # Reconstruct PDB with only the chains we need
                        protein = Protein.from_pdb(pdb_file, chains_to_reconstruct)
                        protein.to_pdb(tmp_pdb)
                        renumber_pdb(tmp_pdb, tmp_pdb, scheme=item["numbering"])
                        # Verify file was created
                        assert os.path.exists(tmp_pdb), f"Temporary PDB file not created: {tmp_pdb}"
                    except Exception as e:
                        print(f"Failed to process PDB file '{pdb_file}' for interaction {immuno_chain}→{antigen}: {e}")
                        continue
                
                # Process interaction between immunoglobulin chain and antigen
                result = interacting_residues(item, tmp_pdb, immuno_chain, antigen, tmp_)
                print(f"Result from {immuno_chain} to {antigen}: {result}")
                
                # Only store result if it's not None
                if result is not None:
                    immuno_results_dict[antigen] = result  # Store by antigen key
                else:
                    print(f"No interactions found for {immuno_chain} to {antigen}")
                    immuno_results_dict[antigen] = []  # Store empty list instead of None
            
            # Only add to aggregated results if we have some results
            if immuno_results_dict:
                aggregated_results[immuno_chain] = immuno_results_dict
        
        # Get final data with VHH-specific CDR residues and interactions
        print("Aggregated results:", aggregated_results)
        
        if aggregated_results:
            # Use VHH-specific function: only CDRH interactions in epitope
            new_item = get_cdr_residues_and_interactions_vhh_only(item, cdr_dict, aggregated_results)
            
            # Additional cleanup: remove any remaining light chain data
            if light_chain:
                # Remove light chain CDR data if it somehow got included
                keys_to_remove = [key for key in new_item.keys() if 'cdrl' in key.lower()]
                for key in keys_to_remove:
                    del new_item[key]
                
                # Also remove light chain from any interaction lists
                for interaction_key in ['all_interactions', 'framework_interactions', 'epitope']:
                    if interaction_key in new_item and isinstance(new_item[interaction_key], list):
                        new_item[interaction_key] = [
                            interaction for interaction in new_item[interaction_key] 
                            if not (isinstance(interaction, dict) and 
                                   interaction.get('immuno_chain') == light_chain)
                        ]
        else:
            print("No interactions found, returning original item with empty epitope")
            new_item = item.copy()
            new_item["epitope"] = []
            new_item["all_interactions"] = []
            new_item["framework_interactions"] = []
        
        print(f"Final VHH processed item: epitope has {len(new_item.get('epitope', []))} CDRH interactions")
        
        # Clean up temporary files to save space
        for immuno_chain in immuno_chains:
            for antigen in antigen_chains:
                tmp_dir = os.path.join(tmp_dir_for_interacting_aa, f"{model_name}_{immuno_chain}_to_{antigen}")
                if os.path.exists(tmp_dir):
                    try:
                        clean_up_files(tmp_dir, model_name)
                        # Also remove the temporary directory if it's empty
                        if os.path.exists(tmp_dir) and not os.listdir(tmp_dir):
                            os.rmdir(tmp_dir)
                    except Exception as e:
                        print(f"Warning: Could not clean up {tmp_dir}: {e}")
        
        return new_item
        
    except Exception as e:
        print(f"Error processing VHH item {item.get('pdb', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None



def process_and_save_worker(args):
    """
    Worker function for processing a single item and saving the result.
    This function is at module level so it can be pickled by multiprocessing.
    """
    item, tmp_dir_for_interacting_aa, output_file, checkpoint_file, file_lock = args
    
    try:
        result = process_single_interaction_vhh(item, tmp_dir_for_interacting_aa)
        if result is not None:
            # Acquire lock for file operations to avoid conflicts
            with file_lock:
                with open(output_file, 'a') as f:
                    f.write(json.dumps(result) + '\n')
                    f.flush()  # Ensure it's written immediately
                
                # Update checkpoint file with processed ID
                if 'pdb' in result:
                    with open(checkpoint_file, 'a') as f:
                        f.write(result['pdb'] + '\n')
                        f.flush()
                
                print(f"Successfully processed and saved: {result.get('pdb', 'unknown')}")
                
        return result is not None
    except Exception as e:
        print(f"Error processing item {item.get('pdb', 'unknown')}: {e}")
        traceback.print_exc()
        return False

def process_entries_parallel(cdr_json_file, tmp_dir_for_interacting_aa, output_file, num_processes=None):
    """
    Process all entries in the JSON file in parallel to extract interaction information.
    Results are written to the output file as they are produced.
    
    Args:
        cdr_json_file: Path to the JSON file containing CDR data
        tmp_dir_for_interacting_aa: Directory to store temporary files
        output_file: Path to output JSON file
        num_processes: Number of processes to use (defaults to CPU count - 1)
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(tmp_dir_for_interacting_aa, exist_ok=True)

    # Create checkpoint file
    checkpoint_file = output_file + ".checkpoint"
    processed_pdb_ids = set()

    # Try to load previously processed IDs
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_pdb_ids = set(f.read().strip().split('\n'))
        print(f"Loaded {len(processed_pdb_ids)} previously processed entries")
    
    # Set number of processes
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    
    print(f"Starting parallel processing with {num_processes} processes")
    
    # Load entries from file
    try:
        with open(cdr_json_file, 'r') as f:
            entries_text = f.read().strip().split('\n')
        
        # Parse JSON objects
        entries = []
        for entry_text in entries_text:
            try:
                entry = json.loads(entry_text)
                # Skip entries that have already been processed
                if 'pdb' in entry and entry['pdb'] in processed_pdb_ids:
                    print(f"Skipping already processed entry: {entry['pdb']}")
                    continue
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
        
        print(f"Loaded {len(entries)} entries to process from {cdr_json_file}")
        
        # Create a lock for file operations to prevent race conditions
        manager = Manager()
        file_lock = manager.Lock()
        
        # Create or clear output file if it doesn't exist or if we're starting fresh
        if not os.path.exists(output_file) or len(processed_pdb_ids) == 0:
            with open(output_file, 'w') as f:
                pass  # Just create/clear the file
        
        # Prepare arguments for each worker
        worker_args = [
            (item, tmp_dir_for_interacting_aa, output_file, checkpoint_file, file_lock)
            for item in entries
        ]
        
        # Process entries in parallel with a pool of workers
        successful_count = 0
        with Pool(num_processes) as pool:
            for success in pool.imap_unordered(process_and_save_worker, worker_args):
                if success:
                    successful_count += 1
                    print(f"Progress: {successful_count}/{len(entries)} entries processed successfully")
        
        print(f"Processing complete. {successful_count}/{len(entries)} entries processed successfully.")
        print(f"Results saved to {output_file}")
        
        return successful_count
    
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        traceback.print_exc()
        return 0




# def normalize_epitope_by_residues(epitope):
def normalize_epitope(epitope):
    """
    Normalize epitope by position and amino acid only, ignoring chain names.
    This catches functionally identical epitopes with different chain naming.
    """
    if not epitope or not isinstance(epitope, list):
        return None
    
    normalized = []
    for contact_pair in epitope:
        if isinstance(contact_pair, list) and len(contact_pair) == 2:
            pair_normalized = []
            for contact in contact_pair:
                if isinstance(contact, list) and len(contact) == 3:
                    # Only use position and amino acid, ignore chain name
                    chain, position, aa = contact[0], contact[1], contact[2]
                    pair_normalized.append((str(position), str(aa)))  # Skip chain
            
            if len(pair_normalized) == 2:
                pair_normalized.sort()
                normalized.append(tuple(pair_normalized))
    
    normalized.sort()


def process_json_file(input_path, output_path):
    # Keep track of seen entries for global uniqueness
    seen_pdb_paths = set()
    seen_entry_ids = set()
    
    # Keep track of epitopes per PDB for epitope uniqueness within same PDB
    pdb_epitopes = defaultdict(set)
    
    unique_entries = []
    total_entries = 0
    skipped_global = 0
    skipped_epitope = 0
    
    # Read the input file line by line (assuming each line is a separate JSON object)
    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                total_entries += 1
                
                # Extract key identifiers
                pdb_path = entry.get("pdb_data_path")
                entry_id = entry.get("entry_id")
                pdb_id = entry.get("pdb")
                epitope = entry.get("epitope")
                
                # First check: Global uniqueness (existing logic)
                if pdb_path in seen_pdb_paths or entry_id in seen_entry_ids:
                    skipped_global += 1
                    continue
                
                # Second check: Epitope uniqueness within same PDB
                if pdb_id and epitope:
                    epitope_signature = normalize_epitope(epitope)
                    if epitope_signature:
                        if epitope_signature in pdb_epitopes[pdb_id]:
                            skipped_epitope += 1
                            print(f"Skipping entry {entry_id} (PDB: {pdb_id}) - duplicate epitope")
                            continue
                        else:
                            pdb_epitopes[pdb_id].add(epitope_signature)
                
                # Entry passed all uniqueness checks
                unique_entries.append(entry)
                
                # Track global identifiers
                if pdb_path:
                    seen_pdb_paths.add(pdb_path)
                if entry_id:
                    seen_entry_ids.add(entry_id)
                        
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line {line_num}: {e}")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
    
    # Write the unique entries to the output file
    with open(output_path, 'w') as f:
        for entry in unique_entries:
            f.write(json.dumps(entry) + '\n')
    
    # Print detailed statistics
    print(f"\n=== Deduplication Results ===")
    print(f"Total entries processed: {total_entries}")
    print(f"Skipped due to global duplicates (pdb_path/entry_id): {skipped_global}")
    print(f"Skipped due to duplicate epitopes within same PDB: {skipped_epitope}")
    print(f"Unique entries kept: {len(unique_entries)}")
    print(f"Output saved to: {output_path}")
    
    # Print PDB epitope statistics
    pdb_with_multiple_epitopes = {pdb: len(epitopes) for pdb, epitopes in pdb_epitopes.items() if len(epitopes) > 1}
    if pdb_with_multiple_epitopes:
        print(f"\nPDBs with multiple unique epitopes:")
        for pdb, count in sorted(pdb_with_multiple_epitopes.items()):
            print(f"  {pdb}: {count} unique epitopes")



def parse():
    parser = argparse.ArgumentParser(description='Structure analysis pipeline for antibodies and nanobodies')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder path where all results will be saved')
    parser.add_argument('--type', type=str, default="Nanobody", choices=['Antibody', 'Nanobody'], help='Type of structure to analyze')
    parser.add_argument('--tsv_file', type=str, default=None, help='Path to TSV summary file (auto-detected if not provided)')
    parser.add_argument('--numbering', type=str, default='imgt', choices=['imgt', 'chothia'], help='Numbering scheme to use')
    parser.add_argument('--max_resolution', type=float, default=4.0, help='Maximum resolution threshold for structure filtering')
    parser.add_argument('--raw_structures_dir', type=str, default=None,
                    help='Directory for raw structure files (default: output_folder/all_structures)') 
    parser.add_argument('--force-skip-cdr', 
                    action='store_true', 
                    help='Force skip CDR processing step (use when files are corrupted)')
       
    return parser.parse_args()




def main(args):
    """Main execution function with checkpoint/resume functionality"""

    global CONFIG  
    CONFIG = setup_paths(args)
    
    # Setup data directory and download if needed
    print("\n=== Data Setup Phase ===")
    
    # If paths not provided, download the data
    if not args.raw_structures_dir or not args.tsv_file:
        print("Downloading SabDab data...")
        default_tsv_path, default_structures_dir = setup_data_directory(args.output_folder)
        
        if default_tsv_path is None or default_structures_dir is None:
            print("Error: Failed to setup data directory")
            return
        
        if not args.tsv_file:
            args.tsv_file = default_tsv_path
        if not args.raw_structures_dir:
            args.raw_structures_dir = default_structures_dir
    
    # Verify files exist
    if not os.path.exists(args.tsv_file):
        print(f"Error: TSV file not found: {args.tsv_file}")
        return
    
    if not os.path.exists(args.raw_structures_dir):
        print(f"Error: Structures directory not found: {args.raw_structures_dir}")
        return
    
    print(f"✓ Using TSV file: {args.tsv_file}")
    print(f"✓ Using structures directory: {args.raw_structures_dir}")
    
    # Count files in structures directory
    try:
        structure_files = [f for f in os.listdir(args.raw_structures_dir) 
                          if f.endswith(('.pdb', '.cif', '.ent'))]
        print(f"✓ Found {len(structure_files)} structure files")
    except Exception as e:
        print(f"Warning: Could not count structure files: {e}")
    
    # Continue with rest of main function...
    print("\n=== Setup Complete ===")


    
    print(f"\n=== Pipeline Configuration ===")
    print(f"Processing type: {args.type}")
    print(f"Numbering scheme: {args.numbering}")
    print(f"Max resolution: {args.max_resolution}")
    num_workers = get_optimal_workers()
    # print(f"TSV file: {CONFIG.get("tsv_file_path")}")
    print(f"TSV file: {CONFIG.get('tsv_file_path')}")
    print(f"Raw structures dir: {args.raw_structures_dir}")
    print(f"Output folder: {args.output_folder}")
    print("=" * 35)


    # STEP 1: Check if TSV processing is needed

    print(f"\n=== Step 1: Checking TSV Processing Status ===")
    if os.path.exists(CONFIG.get('output_tsv_file')) and os.path.getsize(CONFIG.get('output_tsv_file')) > 0:
        print(f"Found existing processed TSV: {CONFIG.get('output_tsv_file')}")
        valid_tsv = CONFIG.get('output_tsv_file')
        skip_tsv_processing = True
    else:
        print("No existing processed TSV found. Will process from scratch.")
        skip_tsv_processing = False
    
    if not skip_tsv_processing:
        print("Running TSV processing pipeline...")
        types = [args.type]  # Process only the selected type
        valid_tsv = run_nanobody_antibody_pipeline(CONFIG.get('tsv_file_path'), args.output_folder, types)
    else:
        print("Skipping TSV processing (already completed)")
    
    # Load filtered entries for progress checking
    filtered_entries = load_filtered_entries(valid_tsv, args.type)
    total_entries = len(filtered_entries)
    print(f"Total entries to process: {total_entries}")
    
    if not filtered_entries:
        print(f"No valid {args.type} entries found in the TSV file.")
        print("Please check the TSV file format and content.")
        return



    # STEP 2: CDR Processing with skip logic
    print(f"\n=== Step 2: Checking CDR Processing Status ===")

    existing_cdr_count = count_entries_in_file(CONFIG.get('cdr_json_file'))
    print(f"Existing CDR entries: {existing_cdr_count}/{total_entries}")


    # Check for force skip argument
    force_skip_cdr = getattr(args, 'force_skip_cdr', False)

    if force_skip_cdr:
        print("FORCE SKIP enabled for CDR processing - marking as complete")
        skip_cdr_processing = True
    elif existing_cdr_count >= total_entries:
        print("CDR processing already completed. Skipping...")
        skip_cdr_processing = True
    else:
        skip_cdr_processing = False

    # Only process if not skipping
    if not skip_cdr_processing:
        if existing_cdr_count == 0:
            unprocessed_cdr_entries = filtered_entries
            print(f"No existing CDR data. Processing all {len(unprocessed_cdr_entries)} entries...")
            os.makedirs(os.path.dirname(CONFIG.get('cdr_json_file')), exist_ok=True)
        else:
            print(f"Found {existing_cdr_count} existing entries. Identifying missing entries...")
            processed_cdr_ids = get_processed_ids(CONFIG.get('cdr_json_file'), 'pdb')
            unprocessed_cdr_entries = filter_unprocessed_entries(filtered_entries, processed_cdr_ids, 'pdb')
            print(f"Need to process: {len(unprocessed_cdr_entries)} remaining entries")

        if unprocessed_cdr_entries:
            print(f"Starting CDR processing for {len(unprocessed_cdr_entries)} entries...")
            
            if not os.path.exists(CONFIG.get('cdr_json_file')):
                print(f"Creating new CDR file: {CONFIG.get('cdr_json_file')}")
                os.makedirs(os.path.dirname(CONFIG.get('cdr_json_file')), exist_ok=True)
                with open(CONFIG.get('cdr_json_file'), 'w') as f:
                    pass
            
            try:
                processed_count = parallel_process_and_write(
                    unprocessed_cdr_entries, 
                    process_entry, 
                    CONFIG.get('cdr_json_file'), 
                    num_workers
                )
                print(f"CDR processing completed: {processed_count} entries successfully processed")
                final_count = count_entries_in_file(CONFIG.get('cdr_json_file'))
                print(f"Final CDR file contains {final_count} total entries")
            except Exception as e:
                print(f"Error during CDR processing: {e}")
    else:
        print("Skipping CDR processing (already completed)")

    print(f"=== Step 2 Complete ===\n")


    # STEP 3: Binding Processing with skip logic
    print(f"\n=== Step 3: Checking Binding Processing Status ===")
    existing_binding_count = count_entries_in_file(CONFIG.get('binding_json_file'))
    expected_binding_count = count_entries_in_file(CONFIG.get('cdr_json_file'))  # Should match CDR count
    print(f"Existing binding entries: {existing_binding_count}/{expected_binding_count}")

    if existing_binding_count >= expected_binding_count and expected_binding_count > 0:
        print("Binding processing already completed. Skipping...")
        skip_binding_processing = True
    else:
        skip_binding_processing = False

    # Only process if not skipping
    if not skip_binding_processing:
        print(f"Processing binding information...")
        binding_results = process_entries_parallel(
            CONFIG.get('cdr_json_file'), 
            str(CONFIG.get('tmp_dir')), 
            str(CONFIG.get('binding_json_file')), 
            CONFIG.get('workers')
        )
        print(f"Binding processing completed")
    else:
        print("Skipping binding processing (already completed)")




    # STEP 4: CDRH3 Filtering with skip logic
    print(f"\n=== Step 4: Checking CDRH3 Filtering Status ===")
    existing_cdrh3_count = count_entries_in_file(CONFIG.get('cdrh3_json_file'))

    expected_cdrh3_count = 0
    if os.path.exists(CONFIG.get('binding_json_file')):
        with open(CONFIG.get('binding_json_file'), 'r') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'cdrh3_avg' in item and item['cdrh3_avg'] != 0:
                        expected_cdrh3_count += 1
                except json.JSONDecodeError:
                    continue

    print(f"Existing CDRH3 entries: {existing_cdrh3_count}/{expected_cdrh3_count}")

    if existing_cdrh3_count >= expected_cdrh3_count and expected_cdrh3_count > 0:
        print("CDRH3 filtering already completed. Skipping...")
        skip_cdrh3_filtering = True
    else:
        skip_cdrh3_filtering = False

    # Only process if not skipping
    if not skip_cdrh3_filtering:
        print("Filtering entries with non-zero CDRH3...")
        
        filtered_count = 0
        with open(CONFIG.get('cdrh3_json_file'), 'w') as out_f:
            pass  # Create/clear file
        
        with open(CONFIG.get('binding_json_file'), 'r') as in_f:
            for line in in_f:
                try:
                    item = json.loads(line.strip())
                    if 'cdrh3_avg' in item and item['cdrh3_avg'] != 0:
                        with open(CONFIG.get('cdrh3_json_file'), 'a') as out_f:
                            out_f.write(json.dumps(item) + '\n')
                        filtered_count += 1
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue
        
        print(f"Filtered {filtered_count} entries with non-zero cdrh3_avg")
    else:
        print("Skipping CDRH3 filtering (already completed)")


    # STEP 5: Duplicate Removal with skip logic
    print(f"\n=== Step 5: Checking Duplicate Removal Status ===")
    existing_unique_count = count_entries_in_file(CONFIG.get('cdrh3_json_file_unique'))
    source_count = count_entries_in_file(CONFIG.get('cdrh3_json_file'))

    print(f"Existing unique entries: {existing_unique_count} (source: {source_count})")

    if (existing_unique_count > 0 and 
        os.path.exists(CONFIG.get('cdrh3_json_file_unique')) and 
        os.path.exists(CONFIG.get('cdrh3_json_file')) and
        os.path.getmtime(CONFIG.get('cdrh3_json_file_unique')) >= os.path.getmtime(CONFIG.get('cdrh3_json_file'))):
        print("Duplicate removal already completed and up-to-date. Skipping...")
        skip_duplicate_removal = True
    else:
        skip_duplicate_removal = False

    # Only process if not skipping
    if not skip_duplicate_removal:
        print("Removing duplicates...")
        process_json_file(str(CONFIG.get('cdrh3_json_file')), str(CONFIG.get('cdrh3_json_file_unique')))
        final_unique_count = count_entries_in_file(CONFIG.get('cdrh3_json_file_unique'))
        print(f"Removed {source_count - final_unique_count} duplicates, {final_unique_count} unique entries remain")
    else:
        print("Skipping duplicate removal (already completed)")

    print(f"\n=== Pipeline Completed Successfully ===")

    print(f"All results saved in: {CONFIG.get('output_folder')}")

    
    # Display final summary with processing status
    print(f"\n=== Final Summary ===")
    print("Processing Status:")
    print(f"  TSV Processing: {'Skipped (already done)' if skip_tsv_processing else 'Completed'}")
    print(f"  CDR Processing: {'Skipped (already done)' if skip_cdr_processing else 'Completed'}")
    print(f"  Binding Processing: {'Skipped (already done)' if skip_binding_processing else 'Completed'}")
    print(f"  CDRH3 Filtering: {'Skipped (already done)' if skip_cdrh3_filtering else 'Completed'}")
    print(f"  Duplicate Removal: {'Skipped (already done)' if skip_duplicate_removal else 'Completed'}")
    
    print("\nGenerated files:")
    output_files = []
    for file_path in [CONFIG.get('output_tsv_file'), CONFIG.get('cdr_json_file'), CONFIG.get('binding_json_file'), 
                    CONFIG.get('cdrh3_json_file'), CONFIG.get('cdrh3_json_file_unique')]:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            count = count_entries_in_file(file_path) if file_path.endswith('.json') else "N/A"
            output_files.append(f"  {os.path.basename(file_path)}: {size} bytes ({count} entries)")
    
    if output_files:
        print("\n".join(output_files))
    else:
        print("No output files were generated. Please check the input data and pipeline configuration.")
    
    print("=" * 25)







def setup_paths(args):
    """Create a paths dictionary with all the required paths"""
    paths = {}
    
    # Basic paths
    paths['output_folder'] = args.output_folder
    paths['tsv_file_path'] = args.tsv_file if args.tsv_file else f"{args.output_folder}/sabdab_summary_all-5.tsv"
    paths['nb_or_ab'] = args.type
    paths['numbering'] = args.numbering
    paths['MAX_RESOLUTION'] = args.max_resolution
    
    # Create output folder
    os.makedirs(paths['output_folder'], exist_ok=True)
    
    # Step 1 paths
    paths['output_tsv_file'] = os.path.join(paths['output_folder'], f"valid_tsv_lines_{args.type}.tsv")
    paths['txt_file'] = os.path.join(paths['output_folder'], "tsv_stats.txt")
    
    # Step 2 paths - Use output_folder/al_structures if raw_structures_dir not provided
    base_raw_dir = args.raw_structures_dir if args.raw_structures_dir is not None \
                  else os.path.join(args.output_folder, "al_structures")
    paths['raw_dir'] = os.path.join(base_raw_dir, paths['numbering'])
    
    paths['cdr_json_file'] = os.path.join(paths['output_folder'], f"{paths['nb_or_ab']}_{paths['numbering']}_cdr_data.json")
    paths['output_folder_specific'] = os.path.join(paths['output_folder'], f"{paths['nb_or_ab']}_{paths['numbering']}")
    os.makedirs(paths['output_folder_specific'], exist_ok=True)
    
    # Create the raw_dir if it doesn't exist
    os.makedirs(paths['raw_dir'], exist_ok=True)
    
    # Step 4 paths
    paths['binding_json_file'] = os.path.join(paths['output_folder'], f"{paths['nb_or_ab']}_with_binding_info_{paths['numbering']}.json")
    paths['tmp_dir'] = os.path.join(paths['output_folder'], "tmp_dir_SASA_computations")
    
    # Step 5 paths
    paths['cdrh3_json_file'] = os.path.join(paths['output_folder'], f"CDRH3_interacting_{paths['nb_or_ab']}_{paths['numbering']}.json")
    paths['cdrh3_json_file_unique'] = os.path.join(paths['output_folder'], f"CDRH3_interacting_{paths['nb_or_ab']}_{paths['numbering']}_unique.json")
    
    # Add workers info
    paths['workers'] = get_optimal_workers() #4  # or call get_optimal_workers() if available
    
    return paths


if __name__ == '__main__':
    args = parse()
    main(args)

