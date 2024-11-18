#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import sys
current_working_dir = os.getcwd()
sys.path.append(current_working_dir)

from data.dataset import E2EDataset2
from data.pdb_utils import VOCAB, Residue, Peptide, Protein, AgAbComplex2
from data.framework_templates import ConserveTemplateGenerator 

from utils.logger import print_log
from utils.random_seed import setup_seed


class IMGT:
    # heavy chain
    HFR1 = (1, 26)
    HFR2 = (39, 55)
    HFR3 = (66, 104)
    HFR4 = (118, 129)

    H1 = (27, 38)
    H2 = (56, 65)
    H3 = (105, 117)

    # light chain
    LFR1 = (1, 26)
    LFR2 = (39, 55)
    LFR3 = (66, 104)
    LFR4 = (118, 129)

    L1 = (27, 38)
    L2 = (56, 65)
    L3 = (105, 117)

    Hconserve = {
        23: ['CYS'],
        41: ['TRP'],
        104: ['CYS']
    }

    Lconserve = {
        23: ['CYS'],
        41: ['TRP'],
        104: ['CYS']
    }


#modification numbering: str
def extract_antibody_info(antibody, heavy_ch, light_ch, numbering):
    # classmethod extracted from AgAbComplex
    # Define numbering schemes
    _scheme = IMGT
    if numbering == 'imgt':
        _scheme = IMGT

    # get cdr/frame denotes
    h_type_mapping, l_type_mapping = {}, {}  # - for non-Fv region, 0 for framework, 1/2/3 for cdr1/2/3

    for lo, hi in [_scheme.HFR1, _scheme.HFR2, _scheme.HFR3, _scheme.HFR4]:
        for i in range(lo, hi + 1):
            h_type_mapping[i] = '0'
    for cdr, (lo, hi) in zip(['1', '2', '3'], [_scheme.H1, _scheme.H2, _scheme.H3]):
        for i in range(lo, hi + 1):
            h_type_mapping[i] = cdr
    h_conserved = _scheme.Hconserve

    for lo, hi in [_scheme.LFR1, _scheme.LFR2, _scheme.LFR3, _scheme.LFR4]:
        for i in range(lo, hi + 1):
            l_type_mapping[i] = '0'
    for cdr, (lo, hi) in zip(['1', '2', '3'], [_scheme.L1, _scheme.L2, _scheme.L3]):
        for i in range(lo, hi + 1):
            l_type_mapping[i] = cdr
    l_conserved = _scheme.Lconserve

    # get variable domain and cdr positions
    selected_peptides, cdr_pos = {}, {}

    chain_names = [heavy_ch]
    if light_ch:
        chain_names.append(light_ch)

    for c, chain_name in zip(['H', 'L'], chain_names):
        chain = antibody.get_chain(chain_name)
        # print("chain_names",chain_names)
        if chain is None:
            continue  # Skip processing if the chain is None
        # Note: possbly two chains are different segments of a same chain
        assert chain is not None, f'Chain {chain_name} not found in the antibody'
        type_mapping = h_type_mapping if c == 'H' else l_type_mapping
        conserved = h_conserved if c == 'H' else l_conserved
        res_type = ''
        for i in range(len(chain)):
            residue = chain.get_residue(i)
            residue_number = residue.get_id()[0]

            if residue_number in type_mapping:
                res_type += type_mapping[residue_number]
                if residue_number in conserved:
                    hit, symbol = False, residue.get_symbol()
                    for conserved_residue in conserved[residue_number]:
                        if symbol == VOCAB.abrv_to_symbol(conserved_residue):
                            hit = True
                            break
                        print(f"Actual residue type at position {residue_number} in chain {chain_name}: {symbol}")
                    assert hit, f'Not {conserved[residue_number]} at {residue_number} in chain {chain_name}'
            else:
                res_type += '-'
        if '0' not in res_type:
            print(heavy_ch, light_ch, antibody.pdb_id, res_type)
        start, end = res_type.index('0'), res_type.rindex('0')
        for cdr in ['1', '2', '3']:
            cdr_start, cdr_end = res_type.find(cdr), res_type.rfind(cdr)
            assert cdr_start != -1, f'cdr {c}{cdr} not found, residue type: {res_type}'
            start, end = min(start, cdr_start), max(end, cdr_end)
            cdr_pos[f'CDR-{c}{cdr}'] = (cdr_start, cdr_end)
        for cdr in ['1', '2', '3']:
            cdr = f'CDR-{c}{cdr}'
            cdr_start, cdr_end = cdr_pos[cdr]
            cdr_pos[cdr] = (cdr_start - start, cdr_end - start)
        chain = chain.get_span(start, end + 1)  # the length may exceed 130 because of inserted amino acids
        chain.set_id(chain_name)
        selected_peptides[chain_name] = chain

    return cdr_pos

def get_cdr_pos(cdr_pos, cdr):  # H/L + 1/2/3, return [begin, end] position
    # cdr_pos is a dictionary = {'CDR-H1': (24, 31), 'CDR-H2': (49, 56), 'CDR-H3': (95, 114)}
    cdr = f'CDR-{cdr}'.upper()
    if cdr in cdr_pos:
        return cdr_pos[cdr]
    else:
        return None




def to_cplx(ori_cplx, ab_x, ab_s) -> AgAbComplex2:
    heavy_chain, light_chain = [], []
    chain = None
    for residue, residue_x in zip(ab_s, ab_x):
        residue = VOCAB.idx_to_symbol(residue)
        if residue == VOCAB.BOA:
            continue
        elif residue == VOCAB.BOH:
            chain = heavy_chain
            continue
        elif residue == VOCAB.BOL:
            chain = light_chain
            continue
        if chain is None:  # still in antigen region
            continue
        coord, atoms = {}, VOCAB.backbone_atoms + VOCAB.get_sidechain_info(residue)

        for atom, x in zip(atoms, residue_x):
            coord[atom] = x
        chain.append(Residue(
            residue, coord, _id=(len(chain), ' ')
        ))
    heavy_chain = Peptide(ori_cplx.heavy_chain, heavy_chain)
    # light_chain = Peptide(ori_cplx.light_chain, light_chain)
    for res, ori_res in zip(heavy_chain, ori_cplx.get_heavy_chain()):
        res.id = ori_res.id
    # for res, ori_res in zip(light_chain, ori_cplx.get_light_chain()):
    #     res.id = ori_res.id

    peptides = {
        ori_cplx.heavy_chain: heavy_chain,
        # ori_cplx.light_chain: light_chain
    }
    antibody = Protein(ori_cplx.pdb_id, peptides)
    cplx = AgAbComplex2(
        ori_cplx.antigen, antibody, ori_cplx.heavy_chain,
        skip_epitope_cal=True,
        skip_validity_check=True
    )
    cplx.cdr_pos = ori_cplx.cdr_pos
    return cplx



def generate(test_loader,test_set, save_dir, model, device, cdr_type=None):
    idx = 0
    summary_items = []
    for batch in tqdm(test_loader):
        with torch.no_grad():
            # move data
            for k in batch:
                if hasattr(batch[k], 'to'):
                    batch[k] = batch[k].to(device)
            # generate
            del batch['xloss_mask']
            X, S, pmets = model.sample(**batch)

            X, S, pmets = X.tolist(), S.tolist(), pmets.tolist()
            X_list, S_list = [], []
            cur_bid = -1
            if 'bid' in batch:
                batch_id = batch['bid']
            else:
                lengths = batch['lengths']
                batch_id = torch.zeros_like(batch['S'])  # [N]
                batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
                batch_id.cumsum_(dim=0)  # [N], item idx in the batch
            for i, bid in enumerate(batch_id):
                if bid != cur_bid:
                    cur_bid = bid
                    X_list.append([])
                    S_list.append([])
                X_list[-1].append(X[i])
                S_list[-1].append(S[i])
                
        for i, (x, s) in enumerate(zip(X_list, S_list)):
            ori_cplx = test_set.data[idx]
            # print(ori_cplx)
            cplx = to_cplx(ori_cplx, x, s)
            pdb_id = cplx.get_id().split('(')[0]
            mod_pdb = os.path.join(save_dir, pdb_id + '.pdb')
            cplx.to_pdb(mod_pdb)
            # ref_pdb = os.path.join(save_dir, pdb_id + '_ref.pdb')
            # ori_cplx.to_pdb(ref_pdb)

            mod_pdb = mod_pdb
            heavy_chain = cplx.heavy_chain
            light_chain = ""
            item = {}
            pdb = Protein.from_pdb(mod_pdb, heavy_chain)
            item["heavy_chain_seq"] = ""

            for peptide_id, peptide in pdb.peptides.items():
                sequence = peptide.get_seq()
                item['heavy_chain_seq'] += sequence
            
            cdr_pos_dict = extract_antibody_info(pdb, heavy_chain, light_chain, "imgt")
            peptides_dict = pdb.get_peptides()
            nano_peptide = peptides_dict.get(heavy_chain)
            for i in range(1, 4):
                cdr_name = f'H{i}'.lower()
                cdr_pos= get_cdr_pos(cdr_pos_dict,cdr_name)
                item[f'cdr{cdr_name}_pos_mod'] = cdr_pos
                start, end = cdr_pos 
                end += 1
                cdr_seq = nano_peptide.get_span(start, end).get_seq()
                item[f'cdr{cdr_name}_seq_mod'] = cdr_seq


            summary_items.append({
                'mod_pdb': mod_pdb,
                'ref_pdb': ori_cplx.pdb_path,
                'heavy_chain': cplx.heavy_chain,
                'cdrh3_seq_mod': item.get("cdrh3_seq_mod"),
                'light_chain': "",
                'antigen_chains': cplx.antigen.get_chain_names(),
                # 'cdr_type': cdr_type,
                'cdr_model': "dyMEAN",
                'pdb': pdb_id,
                'pmetric': pmets[i]
            })
            idx += 1

    # write done the summary
    summary_file = os.path.join(save_dir, 'summary.json')
    with open(summary_file, 'w') as fout:
        fout.writelines(list(map(lambda item: json.dumps(item) + '\n', summary_items)))
    print_log(f'Summary of generated complexes written to {summary_file}')



def main(args):

    setup_seed(2022)
    template_generator = ConserveTemplateGenerator(args.template)

    # load model
    model = torch.load(args.ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')


    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    # model_type
    print_log(f'Model type: {type(model)}')

    # cdr type
    cdr_type = model.cdr_type
    print_log(f'CDR type: {cdr_type}')
    print_log(f'Paratope definition: {model.paratope}')


    # load test set
    test_set = E2EDataset2(args.test_set, template_generator=template_generator,cdr="H3", paratope="H3")
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             collate_fn=E2EDataset2.collate_fn)
    
    # create save dir
    if args.save_dir is None:
        save_dir = '.'.join(args.ckpt.split('.')[:-1]) + '_results'
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # generate(test_loader,ref_pdb,save_dir)
    generate(test_loader,test_set, save_dir, model, device)#, cdr_type)


def parse():
    parser = argparse.ArgumentParser(description='Generate antibodies given epitopes')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--test_set', type=str, default=None,required=True)
    parser.add_argument('--save_dir', type=str, default=None,required=True)
    parser.add_argument('--template', type=str, default=None,required=True)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())