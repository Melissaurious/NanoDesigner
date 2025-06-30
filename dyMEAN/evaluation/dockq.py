#!/usr/bin/python
# -*- coding:utf-8 -*-
import uuid
import os
import re

from dyMEAN.data.pdb_utils import AgAbComplex, Protein, AgAbComplex_mod, AgAbComplex2
from dyMEAN.utils.time_sign import get_time_sign
from configs import DOCKQ_DIR, CACHE_DIR


def dockq(mod_cplx: AgAbComplex2, ref_cplx: AgAbComplex2, cdrh3_only=False):
    H, L = ref_cplx.heavy_chain, ref_cplx.light_chain
    prefix = get_time_sign(suffix=ref_cplx.get_id().replace('(', '').replace(')', ''))
    mod_pdb = os.path.join(CACHE_DIR, prefix + '_dockq_mod.pdb')
    ref_pdb = os.path.join(CACHE_DIR, prefix + '_dockq_ref.pdb')
    if cdrh3_only:
        mod_cdr, ref_cdr = mod_cplx.get_cdr(), ref_cplx.get_cdr()
        mod_peptides, ref_peptides = mod_cplx.antigen.peptides, ref_cplx.antigen.peptides
        mod_peptides[H], ref_peptides[H] = mod_cdr, ref_cdr  # union cdr and antigen chains
        # print("ref_cdr",ref_cdr)
        # print("mod_cdr",mod_cdr)
        pdb = mod_cplx.get_id()
        mod_cplx, ref_cplx = Protein(pdb, mod_peptides), Protein(pdb, ref_peptides)
        # print("mod_cplx",mod_cplx)
        # print("ref_cplx",ref_cplx)
        mod_cplx.to_pdb(mod_pdb)
        ref_cplx.to_pdb(ref_pdb)
        p = os.popen(f'{os.path.join(DOCKQ_DIR, "DockQ.py")} {mod_pdb} {ref_pdb} -native_chain1 {H} -no_needle')
    else:
        mod_cplx.to_pdb(mod_pdb)
        ref_cplx.to_pdb(ref_pdb)
        p = os.popen(f'{os.path.join(DOCKQ_DIR, "DockQ.py")} {mod_pdb} {ref_pdb} -native_chain1 {H} {L} -no_needle')
    text = p.read()
    print("text",text)
    p.close()
    res = re.search(r'DockQ\s+([0-1]\.[0-9]+)', text)
    print("res",res)
    print("res.group(1)", res.group(1))
    score = float(res.group(1))
    # print("score",score)
    os.remove(mod_pdb)
    os.remove(ref_pdb)
    return score


# def dockq_nano(mod_cplx: AgAbComplex_mod, ref_cplx: AgAbComplex_mod, cdrh3_only=False):
#     H= ref_cplx.heavy_chain
#     prefix = get_time_sign(suffix=ref_cplx.get_id().replace('(', '').replace(')', ''))
#     mod_pdb = os.path.join(CACHE_DIR, prefix + '_dockq_mod.pdb')
#     ref_pdb = os.path.join(CACHE_DIR, prefix + '_dockq_ref.pdb')
#     if cdrh3_only:
#         mod_cdr, ref_cdr = mod_cplx.get_cdr(), ref_cplx.get_cdr()
#         mod_peptides, ref_peptides = mod_cplx.antigen.peptides, ref_cplx.antigen.peptides
#         mod_peptides[H], ref_peptides[H] = mod_cdr, ref_cdr  # union cdr and antigen chains
#         pdb = mod_cplx.get_id()
#         mod_cplx, ref_cplx = Protein(pdb, mod_peptides), Protein(pdb, ref_peptides)
#         mod_cplx.to_pdb(mod_pdb)
#         ref_cplx.to_pdb(ref_pdb)
#         p = os.popen(f'{os.path.join(DOCKQ_DIR, "DockQ.py")} {mod_pdb} {ref_pdb} -native_chain1 {H} -no_needle')
#     else:
#         mod_cplx.to_pdb(mod_pdb)
#         ref_cplx.to_pdb(ref_pdb)
#         p = os.popen(f'{os.path.join(DOCKQ_DIR, "DockQ.py")} {mod_pdb} {ref_pdb} -native_chain1 {H} {L} -no_needle')
#     text = p.read()
#     p.close()
#     res = re.search(r'DockQ\s+([0-1]\.[0-9]+)', text)
#     score = float(res.group(1))
#     os.remove(mod_pdb)
#     os.remove(ref_pdb)
#     return score


# for paralelizing this process, require to have unique files
def dockq_nano(mod_cplx, ref_cplx, cdrh3_only=False):
    H, L = ref_cplx.heavy_chain, ref_cplx.light_chain
    unique_id = uuid.uuid4().hex  # Unique identifier for each file
    mod_pdb = os.path.join(CACHE_DIR, f"{unique_id}_dockq_mod.pdb")
    ref_pdb = os.path.join(CACHE_DIR, f"{unique_id}_dockq_ref.pdb")

    if cdrh3_only:
        mod_cdr, ref_cdr = mod_cplx.get_cdr(), ref_cplx.get_cdr()
        mod_peptides, ref_peptides = mod_cplx.antigen.peptides, ref_cplx.antigen.peptides
        mod_peptides[H], ref_peptides[H] = mod_cdr, ref_cdr  # union cdr and antigen chains

        pdb = mod_cplx.get_id()
        mod_cplx, ref_cplx = Protein(pdb, mod_peptides), Protein(pdb, ref_peptides)

        try:
            mod_cplx.to_pdb(mod_pdb)
            ref_cplx.to_pdb(ref_pdb)
        except Exception as e:
            # print(f"Error in creating PDB files: {e}")
            return 0
        p = os.popen(f'{os.path.join(DOCKQ_DIR, "DockQ.py")} {mod_pdb} {ref_pdb} -native_chain1 {H} -no_needle')
    else:
        try:
            mod_cplx.to_pdb(mod_pdb)
            ref_cplx.to_pdb(ref_pdb)
        except Exception as e:
            # print(f"Error in creating PDB files: {e}")
            return 0
        p = os.popen(f'{os.path.join(DOCKQ_DIR, "DockQ.py")} {mod_pdb} {ref_pdb} -native_chain1 {H} {L} -no_needle')
    
    try:
        text = p.read()
        p.close()
        # print("DockQ Output:", text)
        res = re.search(r'DockQ\s+([0-1]\.\d+)', text)
        if res:
            score = float(res.group(1))
        else:
            # print("DockQ did not return a valid score.")
            return 0
    except Exception as e:
        # print(f"Error during DockQ computation: {e}")
        return 0
    finally:
        if os.path.exists(mod_pdb):
            os.remove(mod_pdb)
        if os.path.exists(ref_pdb):
            os.remove(ref_pdb)

    return score
