#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
from tqdm import tqdm

import numpy as np
import sys 
sys.path.append('/ibex/user/rioszemm/NanobodiesProject/dyMEAN')
from data.pdb_utils import VOCAB, AgAbComplex2
from evaluation.rmsd import compute_rmsd, kabsch
from utils.singleton import singleton
from utils.logger import print_log


@singleton
class ConserveTemplateGenerator:
    def __init__(self, json_path=None):
        if json_path is None:
            folder = os.path.split(__file__)[0]
            json_path = os.path.join(folder, 'template.json')
        with open(json_path, 'r') as fin:
            self.template_map = json.load(fin)
    
    def _chain_template(self, cplx: AgAbComplex2, poses, n_channel, heavy=True):
        chain = cplx.get_heavy_chain() if heavy else cplx.get_light_chain()
        if chain is None:
            return [], []  # Return empty lists if the chain is None
        chain_name = 'H' if heavy else 'L'
        hit_map = { pos: False for pos in poses }
        X, hit_index = [], []
        for i, residue in enumerate(chain):
            pos, _ = residue.get_id()
            pos = str(pos)
            if pos in hit_map:
                coord = self.template_map[chain_name][pos]  # N, CA, C, O
                ca, num_sc = coord[1], n_channel - len(coord)
                coord.extend([ca for _ in range(num_sc)])
                hit_index.append(i)
                coord = np.array(coord)
            else:
                coord = [[0, 0, 0] for _ in range(n_channel)]
            X.append(coord)
        if hit_index:
            # uniform distribution between residues and extension at two ends
            for left_i, right_i in zip(hit_index[:-1], hit_index[1:]):
                left, right = X[left_i], X[right_i]
                span, index_span = right - left, right_i - left_i
                span = span / index_span
                for i in range(left_i + 1, right_i):
                    X[i] = X[i - 1] + span
            # start and end
            if hit_index[0] != 0:
                left_i = hit_index[0]
                span = X[left_i] - X[left_i + 1]
                for i in reversed(range(0, left_i)):
                    X[i] = X[i + 1] + span
            if hit_index[-1] != len(X) - 1:
                right_i = hit_index[-1]
                span = X[right_i] - X[right_i - 1]
                for i in range(right_i + 1, len(X)):
                    X[i] = X[i - 1] + span
        return X, hit_index


    def construct_template(self, cplx: AgAbComplex2, n_channel=VOCAB.MAX_ATOM_NUMBER, align=True):
        hc, hc_hit = self._chain_template(cplx, self.template_map['H'], n_channel, heavy=True)

        # MODIFICATION TO HANDLE CASES WHERE THERE IS NO LIGHT CHAIN
        lc, lc_hit = [], []

        # Check if 'L' exists in template_map before trying to access it
        if 'L' in self.template_map and cplx.get_light_chain() is not None:
            lc, lc_hit = self._chain_template(cplx, self.template_map['L'], n_channel, heavy=False)

        # Check if light chain data is empty
        if not lc:
            template = np.array(hc)  # Use only heavy chain data if light chain is None # [N, n_channel, 3]
        else:
            template = np.array(hc + lc)  # Combine heavy and light chain data

        if align and template.size > 0:  # Proceed with alignment only if template is not empty
            true_X_bb, temp_X_bb = [], []
            chains = [cplx.get_heavy_chain()]
            temps, hits = [hc], [hc_hit]
            if cplx.get_light_chain() is not None and lc:  # Include light chain only if it's not None and lc exists
                chains.append(cplx.get_light_chain())
                temps.append(lc)
                hits.append(lc_hit)
        
            # align (will be dropped in the future)
            true_X_bb, temp_X_bb = [], []
            # Don't hardcode chains here - use the chains list built above
            for chain, temp, hit in zip(chains, temps, hits):
                if chain is None:  # Skip if chain is None
                    continue
                for i, residue_temp in zip(hit, temp):
                    residue = chain.get_residue(i)
                    bb = residue.get_backbone_coord_map()
                    for ai, atom in enumerate(VOCAB.backbone_atoms):
                        if atom not in bb:
                            continue
                        true_X_bb.append(bb[atom])
                        temp_X_bb.append(residue_temp[ai])
            
            if true_X_bb and temp_X_bb:  # Only perform alignment if there are points to align
                true_X_bb, temp_X_bb = np.array(true_X_bb), np.array(temp_X_bb)
                _, Q, t = kabsch(temp_X_bb, true_X_bb)
                template = np.dot(template, Q) + t

        return template


    # def construct_template(self, cplx: AgAbComplex2, n_channel=VOCAB.MAX_ATOM_NUMBER, align=True):
    #     hc, hc_hit = self._chain_template(cplx, self.template_map['H'], n_channel, heavy=True)

    #     # MODIFICATION TO HANDLE CASES WHERE THERE IS NO LIGHT CHAIN
    #     lc, lc_hit = [], []

    #     # Check if light chain is present in the complex and process it if it exists
    #     if cplx.get_light_chain() is not None:
    #         lc, lc_hit = self._chain_template(cplx, self.template_map['L'], n_channel, heavy=False)

    #     # Check if light chain data is empty
    #     if not lc:
    #         template = np.array(hc)  # Use only heavy chain data if light chain is None # [N, n_channel, 3]
    #     else:
    #         template = np.array(hc + lc)  # Combine heavy and light chain data

    #     if align and template.size > 0:  # Proceed with alignment only if template is not empty
    #         true_X_bb, temp_X_bb = [], []
    #         chains = [cplx.get_heavy_chain()]
    #         temps, hits = [hc], [hc_hit]
    #         if cplx.get_light_chain() is not None:  # Include light chain only if it's not None
    #             chains.append(cplx.get_light_chain())
    #             temps.append(lc)
    #             hits.append(lc_hit)
        
    #         # align (will be dropped in the future)
    #         true_X_bb, temp_X_bb = [], []
    #         chains = [cplx.get_heavy_chain(), cplx.get_light_chain()]
    #         temps, hits = [hc, lc], [hc_hit, lc_hit]
    #         for chain, temp, hit in zip(chains, temps, hits):
    #             for i, residue_temp in zip(hit, temp):
    #                 residue = chain.get_residue(i)
    #                 bb = residue.get_backbone_coord_map()
    #                 for ai, atom in enumerate(VOCAB.backbone_atoms):
    #                     if atom not in bb:
    #                         continue
    #                     true_X_bb.append(bb[atom])
    #                     temp_X_bb.append(residue_temp[ai])
    #         true_X_bb, temp_X_bb = np.array(true_X_bb), np.array(temp_X_bb)
    #         _, Q, t = kabsch(temp_X_bb, true_X_bb)
    #         template = np.dot(template, Q) + t

    #     return template


def parse():
    parser = argparse.ArgumentParser(description='framework template statistics')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--out', type=str, required=True, help='Path to save template (json)')
    parser.add_argument('--conserve_th', type=float, default=0.95, help='Threshold of hit ratio for deciding whether the residue is conserved')
    return parser.parse_args()


def main(args, dataset):


    if os.path.exists(args.out):
        print(f"Template file {args.out} already exists. Skipping processing.")
        print_log(f"Template file {args.out} already exists. Skipping processing.")
        return

        
    conserve_pos = {'H': [], 'L': []}
    pos2type = {'H': {}, 'L': {}}
    chain_names = ['H', 'L']
    for cplx in tqdm(dataset.data):
        for chain in chain_names:
            # Check if the chain exists in the complex before processing
            ag_ab_chain = cplx.get_heavy_chain() if chain == 'H' else cplx.get_light_chain()
            if ag_ab_chain is None:
                continue
            for seg in ['1', '2', '3', '4']:
                fr = cplx.get_framework(chain + 'FR' + seg)
                if fr is None:
                    continue
                for residue in fr:
                    pos, insert_code = residue.get_id()
                    if pos not in pos2type[chain]:
                        pos2type[chain][pos] = {}
                    if insert_code.strip() != '':
                        pos2type[chain][pos]['insert_code'] = 1
                        continue
                    symbol = residue.get_symbol()
                    if symbol not in pos2type[chain][pos]:
                        pos2type[chain][pos][symbol] = 0
                    pos2type[chain][pos][symbol] += 1
    th_num = len(dataset) / 2
    for chain in chain_names:
        for pos in sorted(list(pos2type[chain].keys())):
            if 'insert_code' in pos2type[chain][pos]:
                continue
            # normalize
            total_num = sum(list(pos2type[chain][pos].values()))
            for symbol in pos2type[chain][pos]:
                ratio = pos2type[chain][pos][symbol] / total_num
                if ratio > args.conserve_th and total_num > th_num:  # exclude some rarely seen positions
                    print_log(f'{chain}, {pos}, {pos2type[chain][pos]}')
                    conserve_pos[chain].append(pos)
    print_log(conserve_pos)
    print_log(f'number of conserved residues: {len(conserve_pos["H"]) + len(conserve_pos["L"])}')

    # form vague templates
    templates, masks = [], []
    for cplx in tqdm(dataset.data):
        template, mask = [], []
        for chain in chain_names:
            poses = conserve_pos[chain]
            # chain = cplx.get_heavy_chain() if chain == 'H' else cplx.get_light_chain()
            ag_ab_chain = cplx.get_heavy_chain() if chain == 'H' else cplx.get_light_chain()

            # Skip processing if the light chain is None
            if ag_ab_chain is None and chain == 'L':
                continue

            hit_map = {pos: False for pos in poses}
            skip = False
            for residue in ag_ab_chain: #chain:
                symbol = residue.get_symbol()
                pos, insert_code = residue.get_id()
                if pos not in hit_map:
                    continue
                # hit
                hit_map[pos] = True
                if insert_code.strip() != '':
                    print_log(f'insert code {insert_code}, pos {pos}')
                    skip = False
                    break
                residue_template = []
                for atom in VOCAB.backbone_atoms:
                    bb = residue.get_backbone_coord_map()
                    if atom not in bb:
                        skip = True
                        break
                    residue_template.append(bb[atom])
                template.append(residue_template)
                if skip:
                    break
            if skip:
                break
            mask.extend([hit_map[pos] for pos in poses])
        if skip:
            continue
        mask = np.array(mask)
        full_template = []
        i = 0
        for m in mask:
            if m:
                full_template.append(template[i])
                i += 1
            else:
                full_template.append([[0, 0, 0] for _ in range(4)])
        template = np.array(full_template)
        templates.append(template)
        masks.append(mask)

    # align
    # find the most complete one
    max_hit, max_hit_idx = 0, -1
    for i, mask in enumerate(masks):
        hit_cnt = mask.sum()
        if hit_cnt > max_hit:
            max_hit, max_hit_idx = hit_cnt, i
    ref, ref_mask = templates[max_hit_idx], masks[max_hit_idx]
    print_log(f'max hit number: {max_hit}')


    # print_log(f'max hit number: {max_hit}')

    # Group templates by their mask sizes
    size_to_templates = {}
    size_to_masks = {}

    for i, (template, mask) in enumerate(zip(templates, masks)):
        mask_size = len(mask)
        if mask_size not in size_to_templates:
            size_to_templates[mask_size] = []
            size_to_masks[mask_size] = []
        size_to_templates[mask_size].append(template)
        size_to_masks[mask_size].append(mask)

    # Process each group separately
    aligned_templates = []
    aligned_masks = []

    for mask_size, group_templates in size_to_templates.items():
        group_masks = size_to_masks[mask_size]
        
        # Find the most complete template in this group
        group_max_hit, group_max_hit_idx = 0, -1
        for i, mask in enumerate(group_masks):
            hit_cnt = mask.sum()
            if hit_cnt > group_max_hit:
                group_max_hit, group_max_hit_idx = hit_cnt, i
        
        group_ref = group_templates[group_max_hit_idx]
        group_ref_mask = group_masks[group_max_hit_idx]
        
        print_log(f'Group size {mask_size}, max hit number: {group_max_hit}')
        
        # Align templates within this group
        for i, template in enumerate(group_templates):
            align_mask = np.logical_and(group_masks[i], group_ref_mask)
            if not align_mask.any():
                print_log(f"Warning: No common positions for template {i} in group {mask_size}")
                continue
                
            ref_temp = group_ref[align_mask]
            cur_temp = template[align_mask]
            _, Q, t = kabsch(cur_temp.reshape(-1, 3), ref_temp.reshape(-1, 3))
            aligned_template = np.dot(template, Q) + t
            group_templates[i] = aligned_template
        
        # Calculate average template for this group
        group_final_template = np.sum(group_templates, axis=0) / np.sum(group_masks, axis=0).reshape(-1, 1, 1)
        
        # Calculate RMSDs for this group
        group_rmsds = []
        for template, mask in zip(group_templates, group_masks):
            template, ref = template[mask], group_final_template[mask]
            group_rmsds.append(compute_rmsd(template.reshape(-1, 3), ref.reshape(-1, 3), aligned=False))
        
        if group_rmsds:
            print_log(f'Group {mask_size} RMSD: max {max(group_rmsds)}, min {min(group_rmsds)}, mean {np.mean(group_rmsds)}')
        
        # Store the final template and masks for this group
        aligned_templates.append((mask_size, group_final_template))
        aligned_masks.append((mask_size, group_masks[0]))  # Use the first mask as reference

    # Now process the results for each chain separately
    template_json = {'H': {}, 'L': {}}

    # Find the positions for each chain
    h_poses = conserve_pos['H']
    l_poses = conserve_pos['L']

    # First, find which group has antibodies (both chains)
    antibody_size = None
    for size, _ in aligned_templates:
        if size == len(h_poses) + len(l_poses):
            antibody_size = size
            break

    # Process heavy chain positions
    if h_poses:
        # Find the group with heavy chain only templates
        h_chain_size = len(h_poses)
        h_template = None
        
        # First try to use the antibody template for heavy chain if available
        if antibody_size is not None:
            for size, template in aligned_templates:
                if size == antibody_size:
                    h_template = template[:len(h_poses)]
                    break
        
        # If no antibody template or couldn't use it, find a heavy-chain only template
        if h_template is None:
            for size, template in aligned_templates:
                if size == h_chain_size:
                    h_template = template
                    break
        
        # Store the heavy chain template positions
        if h_template is not None:
            for i, pos in enumerate(h_poses):
                template_json['H'][pos] = h_template[i].tolist()

    # Process light chain positions
    if l_poses:
        # Find the group with antibody templates
        l_template = None
        
        if antibody_size is not None:
            for size, template in aligned_templates:
                if size == antibody_size:
                    l_template = template[len(h_poses):]
                    break
        
        # Store the light chain template positions
        if l_template is not None:
            for i, pos in enumerate(l_poses):
                template_json['L'][pos] = l_template[i].tolist()

    # Ensure both 'H' and 'L' keys exist in the template, even if empty
    if 'L' not in template_json or not template_json['L']:
        print_log("Light chain template is empty. Adding dummy 'L' entry for compatibility.")
        # If we have no L chain positions but there are nanobodies in the dataset,
        # we still need a dummy L template to avoid shape mismatches during training
        template_json['L'] = {'0': [[0.0, 0.0, 0.0] for _ in range(4)]}

    with open(args.out, 'w') as fout:
        json.dump(template_json, fout)



if __name__ == '__main__':
    from data.dataset import E2EDataset2
    args = parse()
    dataset = E2EDataset2(args.dataset)
    main(args, dataset)



    
    # for i, template in enumerate(templates):
    #     align_mask = np.logical_and(masks[i], ref_mask)
    #     ref_temp = ref[align_mask]
    #     cur_temp = template[align_mask]
    #     _, Q, t = kabsch(cur_temp.reshape(-1, 3), ref_temp.reshape(-1, 3))
    #     # aligned_template = np.dot(template - template.reshape(-1, 3).mean(axis=0), Q) + t
    #     aligned_template = np.dot(template, Q) + t
    #     templates[i] = aligned_template
        
    # final_template = np.sum(templates, axis=0) / np.sum(masks, axis=0).reshape(-1, 1, 1)
    # rmsds = []
    # for template, mask in zip(templates, masks):
    #     template, ref = template[mask], final_template[mask]
    #     rmsds.append(compute_rmsd(template.reshape(-1, 3), ref.reshape(-1, 3), aligned=False))
    # print_log(f'rmsd: max {max(rmsds)}, min {min(rmsds)}, mean {np.mean(rmsds)}')

    # # save templates
    # template_json = { 'H': {}, 'L': {} }
    # i = 0
    # for chain in chain_names:
    #     # Skip saving templates for light chain if it was None
    #     if len(conserve_pos[chain]) == 0 and chain == 'L':
    #         continue
    #     for pos in conserve_pos[chain]:
    #         template_json[chain][pos] = final_template[i].tolist()
    #         i += 1
    # assert i == len(final_template)


    # if 'L' not in template_json or not template_json['L']:
    #     print_log("Light chain template is empty. Adding dummy 'L' entry for compatibility.")
    #     # If we have no L chain positions but there are nanobodies in the dataset,
    #     # we still need a dummy L template to avoid shape mismatches during training
    #     template_json['L'] = {'0': [[0.0, 0.0, 0.0] for _ in range(4)]}

    # with open(args.out, 'w') as fout:
    #     json.dump(template_json, fout)



    # with open(args.out, 'w') as fout:
    #     json.dump(template_json, fout)