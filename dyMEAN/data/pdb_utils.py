#!/usr/bin/python
# -*- coding:utf-8 -*-
from copy import copy, deepcopy
import math
import os
# from typing import Dict, List, Tuple
from typing import Dict, List, Optional, Tuple
import requests

import numpy as np
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure as BStructure
from Bio.PDB.Model import Model as BModel
from Bio.PDB.Chain import Chain as BChain
from Bio.PDB.Residue import Residue as BResidue
from Bio.PDB.Atom import Atom as BAtom

from dyMEAN.configs import IMGT, Chothia


class AminoAcid:
    def __init__(self, symbol: str, abrv: str, sidechain: List[str], idx=0):
        self.symbol = symbol
        self.abrv = abrv
        self.idx = idx
        self.sidechain = sidechain

    def __str__(self):
        return f'{self.idx} {self.symbol} {self.abrv} {self.sidechain}'


class AminoAcidVocab:

    MAX_ATOM_NUMBER = 14   # 4 backbone atoms + up to 10 sidechain atoms

    def __init__(self):
        self.backbone_atoms = ['N', 'CA', 'C', 'O']
        self.PAD, self.MASK = '#', '*'
        self.BOA, self.BOH, self.BOL = '&', '+', '-' # begin of antigen, heavy chain, light chain
        specials = [# special added
                (self.PAD, 'PAD'), (self.MASK, 'MASK'), # mask for masked / unknown residue
                (self.BOA, '<X>'), (self.BOH, '<H>'), (self.BOL, '<L>')
            ]
        aas = [
                ('G', 'GLY'), ('A', 'ALA'), ('V', 'VAL'), ('L', 'LEU'),
                ('I', 'ILE'), ('F', 'PHE'), ('W', 'TRP'), ('Y', 'TYR'),
                ('D', 'ASP'), ('H', 'HIS'), ('N', 'ASN'), ('E', 'GLU'),
                ('K', 'LYS'), ('Q', 'GLN'), ('M', 'MET'), ('R', 'ARG'),
                ('S', 'SER'), ('T', 'THR'), ('C', 'CYS'), ('P', 'PRO') # 20 aa
                # ('U', 'SEC') # 21 aa for eukaryote
            ]

        # max number of sidechain atoms: 10        
        self.atom_pad, self.atom_mask = 'p', 'm'
        self.atom_pos_mask, self.atom_pos_bb, self.atom_pos_pad = 'm', 'b', 'p'
        sidechain_map = {
            'G': [],   # -H
            'A': ['CB'],  # -CH3
            'V': ['CB', 'CG1', 'CG2'],  # -CH-(CH3)2
            'L': ['CB', 'CG', 'CD1', 'CD2'],  # -CH2-CH(CH3)2
            'I': ['CB', 'CG1', 'CG2', 'CD1'], # -CH(CH3)-CH2-CH3
            'F': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],  # -CH2-C6H5
            'W': ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],  # -CH2-C8NH6
            'Y': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],  # -CH2-C6H4-OH
            'D': ['CB', 'CG', 'OD1', 'OD2'],  # -CH2-COOH
            'H': ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],  # -CH2-C3H3N2
            'N': ['CB', 'CG', 'OD1', 'ND2'],  # -CH2-CONH2
            'E': ['CB', 'CG', 'CD', 'OE1', 'OE2'],  # -(CH2)2-COOH
            'K': ['CB', 'CG', 'CD', 'CE', 'NZ'],  # -(CH2)4-NH2
            'Q': ['CB', 'CG', 'CD', 'OE1', 'NE2'],  # -(CH2)-CONH2
            'M': ['CB', 'CG', 'SD', 'CE'],  # -(CH2)2-S-CH3
            'R': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],  # -(CH2)3-NHC(NH)NH2
            'S': ['CB', 'OG'],  # -CH2-OH
            'T': ['CB', 'OG1', 'CG2'],  # -CH(CH3)-OH
            'C': ['CB', 'SG'],  # -CH2-SH
            'P': ['CB', 'CG', 'CD'],  # -C3H6
        }

        self.chi_angles_atoms = {
            "ALA": [],
            # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
            "ARG": [
                ["N", "CA", "CB", "CG"],
                ["CA", "CB", "CG", "CD"],
                ["CB", "CG", "CD", "NE"],
                ["CG", "CD", "NE", "CZ"],
            ],
            "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
            "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
            "CYS": [["N", "CA", "CB", "SG"]],
            "GLN": [
                ["N", "CA", "CB", "CG"],
                ["CA", "CB", "CG", "CD"],
                ["CB", "CG", "CD", "OE1"],
            ],
            "GLU": [
                ["N", "CA", "CB", "CG"],
                ["CA", "CB", "CG", "CD"],
                ["CB", "CG", "CD", "OE1"],
            ],
            "GLY": [],
            "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
            "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
            "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
            "LYS": [
                ["N", "CA", "CB", "CG"],
                ["CA", "CB", "CG", "CD"],
                ["CB", "CG", "CD", "CE"],
                ["CG", "CD", "CE", "NZ"],
            ],
            "MET": [
                ["N", "CA", "CB", "CG"],
                ["CA", "CB", "CG", "SD"],
                ["CB", "CG", "SD", "CE"],
            ],
            "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
            "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
            "SER": [["N", "CA", "CB", "OG"]],
            "THR": [["N", "CA", "CB", "OG1"]],
            "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
            "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
            "VAL": [["N", "CA", "CB", "CG1"]],
        }

        self.sidechain_bonds = {
            "ALA": { "CA": ["CB"] },
            "GLY": {},
            "VAL": {
                "CA": ["CB"],
                "CB": ["CG1", "CG2"]
            },
            "LEU": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD2", "CD1"]
            },
            "ILE": {
                "CA": ["CB"],
                "CB": ["CG1", "CG2"],
                "CG1": ["CD1"]
            },
            "MET": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["SD"],
                "SD": ["CE"],
            },
            "PHE": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD1", "CD2"],
                "CD1": ["CE1"],
                "CD2": ["CE2"],
                "CE1": ["CZ"]
            },
            "TRP": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD1", "CD2"],
                "CD1": ["NE1"],
                "CD2": ["CE2", "CE3"],
                "CE2": ["CZ2"],
                "CZ2": ["CH2"],
                "CE3": ["CZ3"]
            },
            "PRO": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD"]
            },
            "SER": {
                "CA": ["CB"],
                "CB": ["OG"]
            },
            "THR": {
                "CA": ["CB"],
                "CB": ["OG1", "CG2"]
            },
            "CYS": {
                "CA": ["CB"],
                "CB": ["SG"]
            },
            "TYR": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD1", "CD2"],
                "CD1": ["CE1"],
                "CD2": ["CE2"],
                "CE1": ["CZ"],
                "CZ": ["OH"]
            },
            "ASN": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["OD1", "ND2"]
            },
            "GLN": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD"],
                "CD": ["OE1", "NE2"]
            },
            "ASP": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["OD1", "OD2"]
            },
            "GLU": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD"],
                "CD": ["OE1", "OE2"]
            },
            "LYS": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD"],
                "CD": ["CE"],
                "CE": ["NZ"]
            },
            "ARG": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD"],
                "CD": ["NE"],
                "NE": ["CZ"],
                "CZ": ["NH1", "NH2"]
            },
            "HIS": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["ND1", "CD2"],
                "ND1": ["CE1"],
                "CD2": ["NE2"]
            }
        }
        

        _all = aas + specials
        self.amino_acids = [AminoAcid(symbol, abrv, sidechain_map.get(symbol, [])) for symbol, abrv in _all]
        self.symbol2idx, self.abrv2idx = {}, {}
        for i, aa in enumerate(self.amino_acids):
            self.symbol2idx[aa.symbol] = i
            self.abrv2idx[aa.abrv] = i
            aa.idx = i
        self.special_mask = [0 for _ in aas] + [1 for _ in specials]

        # atom level vocab
        self.idx2atom = [self.atom_pad, self.atom_mask, 'C', 'N', 'O', 'S']
        self.idx2atom_pos = [self.atom_pos_pad, self.atom_pos_mask, self.atom_pos_bb, 'B', 'G', 'D', 'E', 'Z', 'H']
        self.atom2idx, self.atom_pos2idx = {}, {}
        for i, atom in enumerate(self.idx2atom):
            self.atom2idx[atom] = i
        for i, atom_pos in enumerate(self.idx2atom_pos):
            self.atom_pos2idx[atom_pos] = i
    
    def abrv_to_symbol(self, abrv):
        idx = self.abrv_to_idx(abrv)
        return None if idx is None else self.amino_acids[idx].symbol

    def symbol_to_abrv(self, symbol):
        idx = self.symbol_to_idx(symbol)
        return None if idx is None else self.amino_acids[idx].abrv

    def abrv_to_idx(self, abrv):
        abrv = abrv.upper()
        return self.abrv2idx.get(abrv, None)

    def symbol_to_idx(self, symbol):
        symbol = symbol.upper()
        return self.symbol2idx.get(symbol, None)
    
    def idx_to_symbol(self, idx):
        return self.amino_acids[idx].symbol

    def idx_to_abrv(self, idx):
        return self.amino_acids[idx].abrv

    def get_pad_idx(self):
        return self.symbol_to_idx(self.PAD)

    def get_mask_idx(self):
        return self.symbol_to_idx(self.MASK)
    
    def get_special_mask(self):
        return copy(self.special_mask)

    def get_atom_type_mat(self):
        atom_pad = self.get_atom_pad_idx()
        mat = []
        for i, aa in enumerate(self.amino_acids):
            atoms = [atom_pad for _ in range(self.MAX_ATOM_NUMBER)]
            if aa.symbol == self.PAD:
                pass
            elif self.special_mask[i] == 1:  # specials
                atom_mask = self.get_atom_mask_idx()
                atoms = [atom_mask for _ in range(self.MAX_ATOM_NUMBER)]
            else:
                for aidx, atom in enumerate(self.backbone_atoms + aa.sidechain):
                    atoms[aidx] = self.atom_to_idx(atom[0])
            mat.append(atoms)
        return mat

    def get_atom_pos_mat(self):
        atom_pos_pad = self.get_atom_pos_pad_idx()
        mat = []
        for i, aa in enumerate(self.amino_acids):
            aps = [atom_pos_pad for _ in range(self.MAX_ATOM_NUMBER)]
            if aa.symbol == self.PAD:
                pass
            elif self.special_mask[i] == 1:
                atom_pos_mask = self.get_atom_pos_mask_idx()
                aps = [atom_pos_mask for _ in range(self.MAX_ATOM_NUMBER)]
            else:
                aidx = 0
                for _ in self.backbone_atoms:
                    aps[aidx] = self.atom_pos_to_idx(self.atom_pos_bb)
                    aidx += 1
                for atom in aa.sidechain:
                    aps[aidx] = self.atom_pos_to_idx(atom[1])
                    aidx += 1
            mat.append(aps)
        return mat

    def get_sidechain_info(self, symbol):
        idx = self.symbol_to_idx(symbol)
        return copy(self.amino_acids[idx].sidechain)
    
    def get_sidechain_geometry(self, symbol):
        abrv = self.symbol_to_abrv(symbol)
        chi_angles_atoms = copy(self.chi_angles_atoms[abrv])
        sidechain_bonds = self.sidechain_bonds[abrv]
        return (chi_angles_atoms, sidechain_bonds)
    
    def get_atom_pad_idx(self):
        return self.atom2idx[self.atom_pad]
    
    def get_atom_mask_idx(self):
        return self.atom2idx[self.atom_mask]
    
    def get_atom_pos_pad_idx(self):
        return self.atom_pos2idx[self.atom_pos_pad]

    def get_atom_pos_mask_idx(self):
        return self.atom_pos2idx[self.atom_pos_mask]
    
    def idx_to_atom(self, idx):
        return self.idx2atom[idx]

    def atom_to_idx(self, atom):
        return self.atom2idx[atom]

    def idx_to_atom_pos(self, idx):
        return self.idx2atom_pos[idx]
    
    def atom_pos_to_idx(self, atom_pos):
        return self.atom_pos2idx[atom_pos]

    def get_num_atom_type(self):
        return len(self.idx2atom)
    
    def get_num_atom_pos(self):
        return len(self.idx2atom_pos)

    def get_num_amino_acid_type(self):
        return len(self.special_mask) - sum(self.special_mask)

    def __len__(self):
        return len(self.symbol2idx)


VOCAB = AminoAcidVocab()


def format_aa_abrv(abrv):  # special cases
    if abrv == 'MSE':
        return 'MET' # substitue MSE with MET
    return abrv


class Residue:
    def __init__(self, symbol: str, coordinate: Dict, _id: Tuple):
        self.symbol = symbol
        self.coordinate = coordinate
        self.sidechain = VOCAB.get_sidechain_info(symbol)
        self.id = _id  # (residue_number, insert_code)

    def get_symbol(self):
        return self.symbol

    def get_coord(self, atom_name):
        return copy(self.coordinate[atom_name])

    def get_coord_map(self) -> Dict[str, List]:
        return deepcopy(self.coordinate)

    def get_backbone_coord_map(self) -> Dict[str, List]:
        coord = { atom: self.coordinate[atom] for atom in self.coordinate if atom in VOCAB.backbone_atoms }
        return coord

    def get_sidechain_coord_map(self) -> Dict[str, List]:
        coord = {}
        for atom in self.sidechain:
            if atom in self.coordinate:
                coord[atom] = self.coordinate[atom]
        return coord

    def get_atom_names(self):
        return list(self.coordinate.keys())

    def get_id(self):
        return self.id

    def set_symbol(self, symbol):
        assert VOCAB.symbol_to_abrv(symbol) is not None, f'{symbol} is not an amino acid'
        self.symbol = symbol

    def set_coord(self, coord):
        self.coordinate = deepcopy(coord)

    def dist_to(self, residue):  # measured by nearest atoms
        xa = np.array(list(self.get_coord_map().values()))
        xb = np.array(list(residue.get_coord_map().values()))
        if len(xa) == 0 or len(xb) == 0:
            return math.nan
        dist = np.linalg.norm(xa[:, None, :] - xb[None, :, :], axis=-1)
        return np.min(dist)

    def to_bio(self):
        _id = (' ', self.id[0], self.id[1])
        residue = BResidue(_id, VOCAB.symbol_to_abrv(self.symbol), '    ')
        atom_map = self.coordinate
        for i, atom in enumerate(atom_map):
            fullname = ' ' + atom
            while len(fullname) < 4:
                fullname += ' '
            bio_atom = BAtom(
                name=atom,
                coord=np.array(atom_map[atom], dtype=np.float32),
                bfactor=0,
                occupancy=1.0,
                altloc=' ',
                fullname=fullname,
                serial_number=i,
                element=atom[0]  # not considering symbols with 2 chars (e.g. FE, MG)
            )
            residue.add(bio_atom)
        return residue

    def __iter__(self):
        return iter([(atom_name, self.coordinate[atom_name]) for atom_name in self.coordinate])


class Peptide:
    def __init__(self, _id, residues: List[Residue]):
        self.residues = residues
        self.seq = ''
        self.id = _id
        for residue in residues:
            self.seq += residue.get_symbol()

    def set_id(self, _id):
        self.id = _id

    def get_id(self):
        return self.id

    def get_seq(self):
        return self.seq

    def get_span(self, i, j):  # [i, j)
        i, j = max(i, 0), min(j, len(self.seq))
        if j <= i:
            return None
        else:
            residues = deepcopy(self.residues[i:j])
            return Peptide(self.id, residues)

    def get_residue(self, i):
        return deepcopy(self.residues[i])
    
    # def get_ca_pos(self, i):
    #     return copy(self.residues[i].get_coord('CA'))

    def get_ca_pos(self, i):
        try:
            return copy(self.residues[i].get_coord('CA'))
        except KeyError:
            return None

    def get_cb_pos(self, i):
        return copy(self.residues[i].get_coord('CB'))

    def set_residue_coord(self, i, coord):
        self.residues[i].set_coord(coord)

    def set_residue_translation(self, i, vec):
        coord = self.residues[i].get_coord_map()
        for atom in coord:
            ori_vec = coord[atom]
            coord[atom] = [a + b for a, b in zip(ori_vec, vec)]
        self.set_residue_coord(i, coord)

    def set_residue_symbol(self, i, symbol):
        self.residues[i].set_symbol(symbol)
        self.seq = self.seq[:i] + symbol + self.seq[i+1:]

    def set_residue(self, i, symbol, coord):
        self.set_residue_symbol(i, symbol)
        self.set_residue_coord(i, coord)

    def to_bio(self):
        chain = BChain(id=self.id)
        for residue in self.residues:
            chain.add(residue.to_bio())
        return chain

    def __iter__(self):
        return iter(self.residues)

    def __len__(self):
        return len(self.seq)

    def __str__(self):
        return self.seq


class Protein:
    def __init__(self, pdb_id, peptides):
        self.pdb_id = pdb_id
        self.peptides = peptides

    # @classmethod
    # def from_pdb(cls, pdb_path):
    #     parser = PDBParser(QUIET=True)
    #     structure = parser.get_structure('anonym', pdb_path)
    #     pdb_id = structure.header['idcode'].upper().strip()
    #     if pdb_id == '':
    #         # deduce from file name
    #         pdb_id = os.path.split(pdb_path)[1].split('.')[0] + '(filename)'

    #     peptides = {}
    #     for chain in structure.get_chains():
    #         _id = chain.get_id()
    #         residues = []
    #         has_non_residue = False
    #         for residue in chain:
    #             abrv = residue.get_resname()
    #             hetero_flag, res_number, insert_code = residue.get_id()
    #             if hetero_flag != ' ':
    #                 continue   # residue from glucose or water
    #             symbol = VOCAB.abrv_to_symbol(abrv)
    #             if symbol is None:
    #                 has_non_residue = True
    #                 # print(f'has non residue: {abrv}')
    #                 break
    #             # filter Hs because not all data include them
    #             atoms = { atom.get_id(): atom.get_coord() for atom in residue if atom.element != 'H' }
    #             residues.append(Residue(
    #                 symbol, atoms, (res_number, insert_code)
    #             ))
    #         if has_non_residue or len(residues) == 0:  # not a peptide
    #             continue
    #         peptides[_id] = Peptide(_id, residues)
    #     return cls(pdb_id, peptides)
    

    @classmethod
    def from_pdb(cls, pdb_path, chains_to_reconstruct=None):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('anonym', pdb_path)
        pdb_id = structure.header['idcode'].upper().strip()
        if pdb_id == '':
            # deduce from file name
            pdb_id = os.path.split(pdb_path)[1].split('.')[0] + '(filename)'

        peptides = {}
        for chain in structure.get_chains():
            if chains_to_reconstruct and chain.id not in chains_to_reconstruct:
                continue  # Skip chains not in the specified list

            _id = chain.get_id()
            residues = []
            has_non_residue = False
            for residue in chain:
                abrv = residue.get_resname()
                hetero_flag, res_number, insert_code = residue.get_id()
                if hetero_flag != ' ':
                    continue   # residue from glucose or water
                symbol = VOCAB.abrv_to_symbol(abrv)
                if symbol is None:
                    has_non_residue = True
                    # print(f'has non residue: {abrv}')
                    break
                # filter Hs because not all data include them
                atoms = {atom.get_id(): atom.get_coord() for atom in residue if atom.element != 'H'}
                residues.append(Residue(
                    symbol, atoms, (res_number, insert_code)
                ))
            if has_non_residue or len(residues) == 0:  # not a peptide
                continue
            peptides[_id] = Peptide(_id, residues)
        return cls(pdb_id, peptides)


    @classmethod
    def from_pdb2(cls, pdb_path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('anonym', pdb_path)
        pdb_id = structure.header['idcode'].upper().strip()
        if pdb_id == '':
            pdb_id = os.path.split(pdb_path)[1].split('.')[0] + '(filename)'

        peptides = {}
        for chain in structure.get_chains():
            _id = chain.get_id()
            residues = []
            has_non_residue = False
            for residue in chain:
                abrv = residue.get_resname()
                hetero_flag, res_number, insert_code = residue.get_id()
                if hetero_flag != ' ':
                    continue  # skip non-amino acid residues
                symbol = VOCAB.abrv_to_symbol(abrv)
                if symbol is None:
                    has_non_residue = True
                    break
                atoms = { atom.get_id(): atom.get_coord() for atom in residue }
                residues.append(Residue(symbol, atoms, (res_number, insert_code)))
            if has_non_residue or len(residues) == 0:  # not a peptide
                continue
            peptides[_id] = Peptide(_id, residues)
        return cls(pdb_id, peptides)


    # MELISSA METHOD
    def rename_chains(self, chain_mapping):
        """
        Rename the chains of the protein according to the provided mapping.

        Parameters:
        chain_mapping (dict): A dictionary mapping old chain names to new chain names.
        """
        new_peptides = {}
        for old_chain_name, peptide in self.peptides.items():
            if old_chain_name in chain_mapping:
                new_chain_name = chain_mapping[old_chain_name]
                peptide.set_id(new_chain_name)  # Assuming Peptide class has a set_id method
                new_peptides[new_chain_name] = peptide
            else:
                new_peptides[old_chain_name] = peptide
        self.peptides = new_peptides

    # def get_peptides(self) -> Peptide:
    #     return self.antibody.get_chain(self.heavy_chain)

    # def get_peptide_by_chain(self, chain_id):
    #     if chain_id in self.peptides:
    #         return self.peptides[chain_id]
    #     else:
    #         return None  # Chain ID not found in peptides dictionary

    def get_peptides(self) -> Dict[str, Peptide]:
        return self.peptides


    def get_id(self):
        return self.pdb_id

    def num_chains(self):
        return len(self.peptides)

    def get_chain(self, name):
        if name in self.peptides:
            return deepcopy(self.peptides[name])
        else:
            return None

    def get_chain_names(self):
        return list(self.peptides.keys())

    def to_bio(self):
        structure = BStructure(id=self.pdb_id)
        model = BModel(id=0)
        for name in self.peptides:
            model.add(self.peptides[name].to_bio())
        structure.add(model)
        return structure

    def to_pdb(self, path, atoms=None):
        if atoms is None:
            bio_structure = self.to_bio()
        else:
            prot = deepcopy(self)
            for _, chain in prot:
                for residue in chain:
                    coordinate = {}
                    for atom in atoms:
                        if atom in residue.coordinate:
                            coordinate[atom] = residue.coordinate[atom]
                    residue.coordinate = coordinate
            bio_structure = prot.to_bio()
        io = PDBIO()
        io.set_structure(bio_structure)
        io.save(path)

    # added
    def get_cdr_pos(self, cdr='H3'):  # H/L + 1/2/3, return [begin, end] position
        cdr = f'CDR-{cdr}'.upper()
        if cdr in self.cdr_pos:
            return self.cdr_pos[cdr]
        else:
            return None

    def get_cdr(self, cdr='H3'):
        cdr = cdr.upper()
        pos = self.get_cdr_pos(cdr)
        if pos is None:
            return None
        chain = self.get_heavy_chain() if 'H' in cdr else self.get_light_chain()
        return chain.get_span(pos[0], pos[1] + 1)


    # # modification

    def find_sequence_position(self, sequence):
        results = {}  # {chain_ID:[pos1,pos2]}
        if type(sequence) == list:
            for i in range(len(sequence)):
                seq = sequence[i]
                chain_where_found = None
                already_found_seq = False
                for chain_name, peptide in self.peptides.items():
                    if seq in peptide.seq:
                        position = peptide.seq.index(seq)
                        print(f"The sequence '{seq}' is found in chain '{chain_name}' at position {position} to {position + len(seq)} ")
                        position = peptide.seq.index(seq)
                        pos_span = [position, position + len(seq)]
                        chain_name = str(chain_name)
                        chain_where_found = chain_name
                        # print("chain_where_found", chain_where_found)
                        if already_found_seq == False:
                            if chain_name not in results:
                                results[chain_name] = [pos_span]
                                already_found_seq = True
                            else:
                                results[chain_name].append(pos_span)
                                already_found_seq = True
                        else:
                            print(f"Sequence '{seq}' is already found in chain '{chain_where_found}', keep a unique epitope per pdb")
                    else:
                        print(f"sequence {seq} was not found in chain {chain_name}")

                already_found_seq == False
        else:
            already_found_seq = False
            chain_where_found = None
            for chain_name, peptide in self.peptides.items():
                if sequence in peptide.seq:
                    position = peptide.seq.index(sequence)
                    print(f"The sequence '{sequence}' is found in chain '{chain_name}' at position {position} to {position + len(sequence)} ")
                    if already_found_seq == False:
                        results[chain_name] = [position, position + len(sequence)]
                        already_found_seq = True
                        chain_where_found = chain_name
                    else:
                        print(f"Sequence '{sequence}' is already found in chain '{chain_where_found}'")
                    break
            else:
                print(f"Sequence '{sequence}' was not found in any antigen chain")
                return None
            
            if len(results) == 0:
                return None
            
        return results



    def __iter__(self):
        return iter([(c, self.peptides[c]) for c in self.peptides])

    def __eq__(self, other):
        if not isinstance(other, Protein):
            raise TypeError('Cannot compare other type to Protein')
        for key in self.peptides:
            if key in other.peptides and self.peptides[key].seq == other.peptides[key].seq:
                continue
            else:
                return False
        return True

    def __str__(self):
        res = self.pdb_id + '\n'
        for seq_name in self.peptides:
            res += f'\t{seq_name}: {self.peptides[seq_name]}\n'
        return res


class AgAbComplex_mod:

    num_interface_residues = 48  # from PNAS (view as epitope)

    def __init__(self, antigen: Protein, antibody: Protein, heavy_chain: str,
                numbering: str='imgt', skip_epitope_cal=False, skip_validity_check=False) -> None:
        self.heavy_chain = heavy_chain
        self.numbering = numbering
        
        #modification for E2EDataset class
        self.pdb = None


        self.antigen = antigen
        if skip_validity_check:
            self.antibody, self.cdr_pos = antibody, None
        else:
            self.antibody, self.cdr_pos = self._extract_antibody_info(antibody, numbering)
        self.pdb_id = antigen.get_id()

        if skip_epitope_cal:
            self.epitope = None
        else:
            self.epitope = self._cal_epitope()

    @classmethod
    def from_pdb2(cls, pdb_path: str, heavy_chain: str, antigen_chains: List[str],
                numbering: str='imgt', skip_epitope_cal=False, skip_validity_check=False):
        protein = Protein.from_pdb(pdb_path)

        pdb_id = protein.get_id()
        #modification
        pdb = os.path.basename(os.path.dirname(pdb_path))
        ab_peptides = {
            heavy_chain: protein.get_chain(heavy_chain)
            #nanobody change
            # ,light_chain: protein.get_chain(light_chain)
        }

        ag_peptides = {chain: protein.get_chain(chain) for chain in antigen_chains if protein.get_chain(chain) is not None}
        chain_ids = list(protein.peptides.keys()) #added to get all the ids of the chains present in the protein object (Cplx)
        antigen_chains2 = [chain_id for chain_id in chain_ids if chain_id != heavy_chain]

        if list(ag_peptides.keys()) != antigen_chains2:
            print("list(ag_peptides.keys())", list(ag_peptides.keys()))
            print("antigen_chains2", antigen_chains2)
            print(f"Specified antigen chains does not match with chains ID extracted from pdb, error in json file or change of ID while renumbering")
            ag_peptides = {chain: protein.get_chain(chain) for chain in antigen_chains2 if protein.get_chain(chain) is not None} 
            for chain in antigen_chains2:
                assert chain in ag_peptides, f'Antigen chain {chain} has something wrong!'   

            antigen = Protein(pdb_id, ag_peptides)
            antibody = Protein(pdb_id, ab_peptides)

            # modification
            instance = cls(antigen, antibody, heavy_chain, numbering, skip_epitope_cal, skip_validity_check)
            instance.pdb= pdb

            return instance, antigen_chains2    
        else:
            for chain in antigen_chains:
                assert chain in ag_peptides, f'Antigen chain {chain} has something wrong!'

            antigen = Protein(pdb_id, ag_peptides)
            antibody = Protein(pdb_id, ab_peptides)

            # modification
            instance = cls(antigen, antibody, heavy_chain, numbering, skip_epitope_cal, skip_validity_check)
            instance.pdb= pdb

            return instance, None
        
    @classmethod
    def from_pdb(cls, pdb_path: str, heavy_chain: str, antigen_chains: List[str],
                numbering: str='imgt', skip_epitope_cal=False, skip_validity_check=False):
        protein = Protein.from_pdb(pdb_path)

        pdb_id = protein.get_id()
        #modification
        pdb = os.path.basename(os.path.dirname(pdb_path))
        ab_peptides = {
            heavy_chain: protein.get_chain(heavy_chain)
            #nanobody change
            # ,light_chain: protein.get_chain(light_chain)
        }

        ag_peptides = {chain: protein.get_chain(chain) for chain in antigen_chains if protein.get_chain(chain) is not None}
        for chain in antigen_chains:
            assert chain in ag_peptides, f'Antigen chain {chain} has something wrong!'

        antigen = Protein(pdb_id, ag_peptides)
        antibody = Protein(pdb_id, ab_peptides)

        # modification
        instance = cls(antigen, antibody, heavy_chain, numbering, skip_epitope_cal, skip_validity_check)
        instance.pdb= pdb

        return instance
    



        # return cls(antigen, antibody, heavy_chain, numbering, skip_epitope_cal, skip_validity_check)

    #modification numbering: str
    def _extract_antibody_info(self, antibody: Protein, numbering: str='imgt'):
        # calculating cdr pos according to number scheme (type_mapping and conserved residues)
        # numbering = numbering.lower()
        # if numbering == 'imgt':
        #     _scheme = IMGT
        # elif numbering == 'chothia':
        #     _scheme = Chothia
        # else:
        #     raise NotImplementedError(f'Numbering scheme {numbering} not implemented')
        numbering = numbering
        if numbering == "imgt":
            _scheme =IMGT

        # get cdr/frame denotes
        h_type_mapping = {}  # - for non-Fv region, 0 for framework, 1/2/3 for cdr1/2/3

        for lo, hi in [_scheme.HFR1, _scheme.HFR2, _scheme.HFR3, _scheme.HFR4]:
            for i in range(lo, hi + 1):
                h_type_mapping[i] = '0'
        for cdr, (lo, hi) in zip(['1', '2', '3'], [_scheme.H1, _scheme.H2, _scheme.H3]):
            for i in range(lo, hi + 1):
                h_type_mapping[i] = cdr
        h_conserved = _scheme.Hconserve

        # get variable domain and cdr positions
        selected_peptides, cdr_pos = {}, {}
        #nanobody change
        # for c, chain_name in zip(['H', 'L'], [self.heavy_chain, self.light_chain]):
        for c, chain_name in zip(['H'], [self.heavy_chain]):
            chain = antibody.get_chain(chain_name)
            assert chain is not None, f'Chain {chain_name} not found in the antibody'
            type_mapping = h_type_mapping
            conserved = h_conserved
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
                        assert hit, f'Not {conserved[residue_number]} at {residue_number}'
                else:
                    res_type += '-'
            if '0' not in res_type:
                print(self.heavy_chain, antibody.pdb_id, res_type)
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

        antibody = Protein(antibody.get_id(), selected_peptides)

        return antibody, cdr_pos


    def _cal_epitope(self):
        ag_rids, ag_xs, ab_xs = [], [], []
        ag_mask, ab_mask = [], []
        cdrh3 = self.get_cdr('H3')
        for _type, protein in zip(['ag', 'ab'], [self.antigen, [('A', cdrh3)]]):
            is_ag = _type == 'ag'
            rids = []
            if is_ag: 
                xs, masks = ag_xs, ag_mask
            else:
                xs, masks = ab_xs, ab_mask
            for chain_name, chain in protein:
                for i, residue in enumerate(chain):
                    bb_coord = residue.get_backbone_coord_map()
                    sc_coord = residue.get_sidechain_coord_map()
                    coord = {}
                    coord.update(bb_coord)
                    coord.update(sc_coord)
                    num_pad = VOCAB.MAX_ATOM_NUMBER - len(coord)
                    x = [coord[key] for key in coord] + [[0, 0, 0] for _ in range(num_pad)]
                    mask = [1 for _ in coord] + [0 for _ in range(num_pad)]
                    rids.append((chain_name, i))
                    xs.append(x)
                    masks.append(mask)
            if is_ag:
                ag_rids = rids
        assert len(ag_xs) != 0, 'No antigen structure!'
        # calculate distance
        ag_xs, ab_xs = np.array(ag_xs), np.array(ab_xs)  # [Nag/ab, M, 3], M == MAX_ATOM_NUM
        ag_mask, ab_mask = np.array(ag_mask).astype('bool'), np.array(ab_mask).astype('bool')  # [Nag/ab, M]
        dist = np.linalg.norm(ag_xs[:, None] - ab_xs[None, :], axis=-1)  # [Nag, Nab, M]
        dist = dist + np.logical_not(ag_mask[:, None] * ab_mask[None, :]) * 1e6  # [Nag, Nab, M]
        min_dists = np.min(np.min(dist, axis=-1), axis=-1)  # [ag_len]
        topk = min(len(min_dists), self.num_interface_residues)
        ind = np.argpartition(-min_dists, -topk)[-topk:]
        epitope = []
        for idx in ind:
            chain_name, i = ag_rids[idx]
            residue = self.antigen.peptides[chain_name].get_residue(i)
            epitope.append((residue, chain_name, i))
        return epitope

    def get_id(self) -> str:
        return self.antibody.pdb_id

    def get_antigen(self) -> Protein:
        return deepcopy(self.antigen)

    def get_epitope(self, cdrh3_pos=None) -> List[Tuple[Residue, str, int]]:
        if cdrh3_pos is not None:
            backup = self.cdr_pos
            self.cdr_pos = {'CDR-H3': [cdrh3_pos[0], cdrh3_pos[1]]}
            epitope = self._cal_epitope()
            self.cdr_pos = backup
            return deepcopy(epitope)
        if self.epitope is None:
            self.epitope = self._cal_epitope()
        return deepcopy(self.epitope)

    def get_interacting_residues(self, dist_cutoff=30) -> Tuple[List[int], List[int]]:
        ag_rids, ag_xs, ab_xs = [], [], []
        for chain_name in self.antigen.get_chain_names():
            chain = self.antigen.get_chain(chain_name)
            for i in range(len(chain)):
                try:
                    x = chain.get_ca_pos(i)
                except KeyError:  # CA position is missing
                    continue
                ag_rids.append((chain_name, i))
                ag_xs.append(x)
        for chain_name in self.antibody.get_chain_names():
            chain = self.antibody.get_chain(chain_name)
            for i in range(len(chain)):
                try:
                    x = chain.get_ca_pos(i)
                except KeyError:
                    continue
                ab_xs.append(x)
        assert len(ag_xs) != 0, 'No antigen structure!'
        # calculate distance
        ag_xs, ab_xs = np.array(ag_xs), np.array(ab_xs)
        dist = np.linalg.norm(ag_xs[:, None, :] - ab_xs[None, :, :], axis=-1)
        min_dists = np.min(dist, axis=1)  # [ag_len]
        topk = min(len(min_dists), self.num_interface_residues)
        ind = np.argpartition(-min_dists, -topk)[-topk:]
        epitope = []
        for idx in ind:
            chain_name, i = ag_rids[idx]
            residue = self.antigen.peptides[chain_name].get_residue(i)
            epitope.append((residue, chain_name, i))
        return

    
    def get_interacting_residues_2(self, dist_cutoff=6.6) -> Tuple[List[int], List[int]]:
        ag_rids, ag_xs, ab_xs = [], [], []
        paratope = []  # Add a paratope list
        for chain_name in self.antigen.get_chain_names():
            chain = self.antigen.get_chain(chain_name)
            for i in range(len(chain)):
                try:
                    x = chain.get_ca_pos(i)
                except KeyError:  # CA position is missing
                    continue
                ag_rids.append((chain_name, i))
                ag_xs.append(x)
        for chain_name in self.antibody.get_chain_names():
            chain = self.antibody.get_chain(chain_name)
            for i in range(len(chain)):
                try:
                    x = chain.get_ca_pos(i)
                except KeyError:
                    continue
                ab_xs.append(x)
                # Check if residue is in contact with antigen chains
                if any(np.linalg.norm(x - ag_x) < dist_cutoff for ag_x in ag_xs):
                    paratope.append((chain_name, i))
        assert len(ag_xs) != 0, 'No antigen structure!'
        # calculate distance
        ag_xs, ab_xs = np.array(ag_xs), np.array(ab_xs)
        dist = np.linalg.norm(ag_xs[:, None, :] - ab_xs[None, :, :], axis=-1)
        min_dists = np.min(dist, axis=1)  # [ag_len]
        topk = min(len(min_dists), self.num_interface_residues)
        ind = np.argpartition(-min_dists, -topk)[-topk:]
        epitope = []
        for idx in ind:
            chain_name, i = ag_rids[idx]
            residue = self.antigen.peptides[chain_name].get_residue(i)
            epitope.append((residue, chain_name, i))
        return epitope, paratope  # Return both epitope and paratope


    def get_heavy_chain(self) -> Peptide:
        return self.antibody.get_chain(self.heavy_chain)

    def get_light_chain(self) -> Peptide:
        return self.antibody.get_chain(self.light_chain)

    def get_framework(self, fr):  # H/L + FR + 1/2/3/4
        seg_id = int(fr[-1])
        chain = self.get_heavy_chain() if fr[0] == 'H' else self.get_light_chain()
        begin, end = -1, -1
        if seg_id == 1:
            begin, end = 0, self.get_cdr_pos(fr[0] + str(seg_id))[0]
        elif seg_id == 4:
            begin, end = self.get_cdr_pos(fr[0] + '3')[-1] + 1, len(chain)
        else:
            begin = self.get_cdr_pos(fr[0] + str(seg_id - 1))[-1] + 1
            end = self.get_cdr_pos(fr[0] + str(seg_id))[0]
        return chain.get_span(begin, end)

    def get_cdr_pos(self, cdr='H3'):  # H/L + 1/2/3, return [begin, end] position
        cdr = f'CDR-{cdr}'.upper()
        if cdr in self.cdr_pos:
            return self.cdr_pos[cdr]
        else:
            return None

    def get_cdr(self, cdr='H3'):
        cdr = cdr.upper()
        pos = self.get_cdr_pos(cdr)
        if pos is None:
            return None
        chain = self.get_heavy_chain() if 'H' in cdr else self.get_light_chain()
        return chain.get_span(pos[0], pos[1] + 1)

    def to_pdb(self, path, atoms=None):
        peptides = {}
        for name in self.antigen.get_chain_names():
            peptides[name] = self.antigen.get_chain(name)
        for name in self.antibody.get_chain_names():
            peptides[name] = self.antibody.get_chain(name)
        protein = Protein(self.get_id(), peptides)
        protein.to_pdb(path, atoms)
    


    def __str__(self):
        pdb_info = f'PDB ID: {self.pdb_id}'
        antibody_info = f'Antibody HeavyChain=-{self.heavy_chain} with lenght ({len(self.get_heavy_chain())})' #+ \
                        #f'L-{self.light_chain} ({len(self.get_light_chain())})'
        antigen_info = f'Antigen Chains: {[(ag, len(self.antigen.get_chain(ag))) for ag in self.antigen.get_chain_names()]}'
        cdr_info = f'CDRs: \n'
        for name in self.cdr_pos:
            chain = self.get_heavy_chain() if 'H' in name else self.get_light_chain()
            start, end = self.cdr_pos[name]
            cdr_info += f'\t{name}: [{start}, {end}], {chain.seq[start:end + 1]}\n'
        epitope_info = f'Epitope: \n'
        residue_map = {}
        for _, chain_name, i in self.get_epitope():
            if chain_name not in residue_map:
                residue_map[chain_name] = []
            residue_map[chain_name].append(i)
        for chain_name in residue_map:
            epitope_info += f'\t{chain_name}: {sorted(residue_map[chain_name])}\n'

        sep = '\n' + '=' * 20 + '\n'
        return sep + pdb_info + '\n' + antibody_info + '\n' + cdr_info + '\n' + antigen_info + '\n' + epitope_info + sep



class AgAbComplex2:

    num_interface_residues = 48  # from PNAS (view as epitope)

    def __init__(self, antigen: Protein, antibody: Protein, heavy_chain: str,
                 light_chain: Optional[str] = None,
                numbering: str='imgt', skip_epitope_cal=False, skip_validity_check=False) -> None:

        self.heavy_chain = heavy_chain
        self.light_chain = light_chain
        self.numbering = numbering
        self.antigen = antigen
        
        #modification for E2EDataset class
        self.pdb = None

        if skip_validity_check:
            self.antibody, self.cdr_pos = antibody, None
        else:
            self.antibody, self.cdr_pos = self._extract_antibody_info(antibody, numbering)
            
        self.pdb_id = antigen.get_id()

        if skip_epitope_cal:
            self.epitope = None
        else:
            self.epitope = self._cal_epitope()

    # PENDING TO MODIFY CAUSE NOT SURE WHERE I USED THIS
    @classmethod
    def from_pdb2(cls, pdb_path: str, heavy_chain: str, antigen_chains: List[str],
                numbering: str='imgt', skip_epitope_cal=False, skip_validity_check=False):
        protein = Protein.from_pdb(pdb_path)

        pdb_id = protein.get_id()
        #modification
        pdb = os.path.basename(os.path.dirname(pdb_path))
        ab_peptides = {
            heavy_chain: protein.get_chain(heavy_chain)
            #nanobody change
            # ,light_chain: protein.get_chain(light_chain)
        }

        ag_peptides = {chain: protein.get_chain(chain) for chain in antigen_chains if protein.get_chain(chain) is not None}
        chain_ids = list(protein.peptides.keys()) #added to get all the ids of the chains present in the protein object (Cplx)
        antigen_chains2 = [chain_id for chain_id in chain_ids if chain_id != heavy_chain]

        if list(ag_peptides.keys()) != antigen_chains2:
            print("list(ag_peptides.keys())", list(ag_peptides.keys()))
            print("antigen_chains2", antigen_chains2)
            print(f"Specified antigen chains does not match with chains ID extracted from pdb, error in json file or change of ID while renumbering")
            ag_peptides = {chain: protein.get_chain(chain) for chain in antigen_chains2 if protein.get_chain(chain) is not None} 
            for chain in antigen_chains2:
                assert chain in ag_peptides, f'Antigen chain {chain} has something wrong!'   

            antigen = Protein(pdb_id, ag_peptides)
            antibody = Protein(pdb_id, ab_peptides)

            # modification
            instance = cls(antigen, antibody, heavy_chain, numbering, skip_epitope_cal, skip_validity_check)
            instance.pdb= pdb

            return instance, antigen_chains2    
        else:
            for chain in antigen_chains:
                assert chain in ag_peptides, f'Antigen chain {chain} has something wrong!'

            antigen = Protein(pdb_id, ag_peptides)
            antibody = Protein(pdb_id, ab_peptides)

            # modification
            instance = cls(antigen, antibody, heavy_chain, numbering, skip_epitope_cal, skip_validity_check)
            instance.pdb= pdb

            return instance, None
        
    # this uses Protein.from_pdb2 so it doesnt get rid of Hs atoms
    @classmethod
    def from_pdb3(cls, pdb_path: str, heavy_chain: str, light_chain: Optional[str], antigen_chains: List[str],
                numbering: str='imgt', skip_epitope_cal=False, skip_validity_check=False):
        protein = Protein.from_pdb2(pdb_path)

        pdb_id = protein.get_id()
        #modification
        pdb = os.path.basename(os.path.dirname(pdb_path))
        ab_peptides = {
            heavy_chain: protein.get_chain(heavy_chain)
        }
        light_chain_data = protein.get_chain(light_chain)
        if light_chain_data is not None:
            ab_peptides[light_chain] = light_chain_data

        ag_peptides = {chain: protein.get_chain(chain) for chain in antigen_chains if protein.get_chain(chain) is not None}
        for chain in antigen_chains:
            assert chain in ag_peptides, f'Antigen chain {chain} has something wrong!'

        antigen = Protein(pdb_id, ag_peptides)
        antibody = Protein(pdb_id, ab_peptides)

        # modification
        instance = cls(antigen, antibody, heavy_chain, numbering, skip_epitope_cal, skip_validity_check)
        instance.pdb= pdb

        return instance

    
    # @classmethod
    # def from_pdb(cls, pdb_path: str, heavy_chain: str, light_chain: Optional[str], antigen_chains: List[str],
    #             numbering: str='imgt', skip_epitope_cal=False, skip_validity_check=False):
    #     protein = Protein.from_pdb(pdb_path)
    #     pdb_id = protein.get_id()
    #     print("pdb_id", pdb_id)
    #     #modification
    #     pdb = os.path.basename(os.path.dirname(pdb_path))

    #     # Constructing the dictionary for antibody peptides
    #     ab_peptides = {heavy_chain: protein.get_chain(heavy_chain)}
    #     if light_chain is not None:
    #         ab_peptides[light_chain] = protein.get_chain(light_chain)

    #     # Constructing the dictionary for antigen peptides
    #     ag_peptides = {chain: protein.get_chain(chain) for chain in antigen_chains if protein.get_chain(chain) is not None}
    #     for chain in antigen_chains:
    #         assert chain in ag_peptides, f'Antigen chain {chain} has something wrong!'

    #     antigen = Protein(pdb_id, ag_peptides)
    #     antibody = Protein(pdb_id, ab_peptides)

    #     instance =  cls(antigen, antibody, heavy_chain, light_chain, numbering, skip_epitope_cal, skip_validity_check)
    #     instance.pdb = pdb

    #     return instance
    @classmethod
    def from_pdb(cls, pdb_path: str, heavy_chain: str, light_chain: Optional[str], antigen_chains: List[str],
                numbering: str='imgt', skip_epitope_cal=False, skip_validity_check=False):

        # print(heavy_chain)
        # print(light_chain)
        # print(antigen_chains)

        protein = Protein.from_pdb(pdb_path)
        pdb_id = protein.get_id()
        # print("pdb_id", pdb_id)
        #modification
        pdb = os.path.basename(os.path.dirname(pdb_path))
        # print("antigen_chains",antigen_chains)

        # Extract additional ID from the file path
        additional_id = os.path.basename(pdb_path).split('.')[0]


        # Constructing the dictionary for antibody peptides
        ab_peptides = {heavy_chain: protein.get_chain(heavy_chain)}
        if light_chain is not None:
            ab_peptides[light_chain] = protein.get_chain(light_chain)

        # print("ab_peptides",ab_peptides)

        # Constructing the dictionary for antigen peptides
        ag_peptides = {chain: protein.get_chain(chain) for chain in antigen_chains if protein.get_chain(chain) is not None}
        # print("ag_peptides",ag_peptides)

        for chain in antigen_chains:
            assert chain in ag_peptides, f'Antigen chain {chain} has something wrong!'


        antigen = Protein(pdb_id, ag_peptides)
        antibody = Protein(pdb_id, ab_peptides)

        instance =  cls(antigen, antibody, heavy_chain, light_chain, numbering, skip_epitope_cal, skip_validity_check)
        instance.pdb = pdb

        instance.additional_id = additional_id
        instance.pdb_path = pdb_path
        
        return instance


        # return cls(antigen, antibody, heavy_chain, numbering, skip_epitope_cal, skip_validity_check)

    #modification numbering: str
    def _extract_antibody_info(self, antibody: Protein, numbering: str):
        # Define numbering schemes
        _scheme = IMGT
        if numbering == 'imgt':
            _scheme = IMGT
        elif numbering == 'chothia':
            _scheme = Chothia

        # print("_scheme",_scheme)

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
    
        chain_names = [self.heavy_chain]
        if self.light_chain is not None:
            chain_names.append(self.light_chain)

        for c, chain_name in zip(['H', 'L'], chain_names):
            chain = antibody.get_chain(chain_name)
            # print("chain_names",chain_names)
            if chain is None:
                continue  # Skip processing if the chain is None
            # Note: possbly two chains are different segments of a same chain
            assert chain is not None, f'Chain {chain_name} not found in the antibody'
            type_mapping = h_type_mapping if c == 'H' else l_type_mapping
            conserved = h_conserved if c == 'H' else l_conserved
            # print("h_type_mapping", h_type_mapping)
            #h_type_mapping {1: '0', 2: '0', 3: '0', 4: '0', 5: '0', 6: '0', 7: '0', 8: '0', 9: '0', 10: '0', 11: '0', 12: '0', 13: '0', 14: '0', 15: '0', 16: '0', 17: '0', 18: '0', 19: '0', 20: '0', 21: '0', 22: '0', 23: '0', 24: '0', 25: '0', 26: '0', 39: '0', 40: '0', 41: '0', 42: '0', 43: '0', 44: '0', 45: '0', 46: '0', 47: '0', 48: '0', 49: '0', 50: '0', 51: '0', 52: '0', 53: '0', 54: '0', 55: '0', 66: '0', 67: '0', 68: '0', 69: '0', 70: '0', 71: '0', 72: '0', 73: '0', 74: '0', 75: '0', 76: '0', 77: '0', 78: '0', 79: '0', 80: '0', 81: '0', 82: '0', 83: '0', 84: '0', 85: '0', 86: '0', 87: '0', 88: '0', 89: '0', 90: '0', 91: '0', 92: '0', 93: '0', 94: '0', 95: '0', 96: '0', 97: '0', 98: '0', 99: '0', 100: '0', 101: '0', 102: '0', 103: '0', 104: '0', 118: '0', 119: '0', 120: '0', 121: '0', 122: '0', 123: '0', 124: '0', 125: '0', 126: '0', 127: '0', 128: '0', 129: '0', 27: '1', 28: '1', 29: '1', 30: '1', 31: '1', 32: '1', 33: '1', 34: '1', 35: '1', 36: '1', 37: '1', 38: '1', 56: '2', 57: '2', 58: '2', 59: '2', 60: '2', 61: '2', 62: '2', 63: '2', 64: '2', 65: '2', 105: '3', 106: '3', 107: '3', 108: '3', 109: '3', 110: '3', 111: '3', 112: '3', 113: '3', 114: '3', 115: '3', 116: '3', 117: '3'}
            res_type = ''
            for i in range(len(chain)):
                residue = chain.get_residue(i)
                residue_number = residue.get_id()[0]
                # print()

                # print("residue", residue) # and object
                # print("residue_number = residue.get_id()[0]",residue_number )  # 1 ....to 23 and then 
                # print(f"Residue {residue.get_symbol()} at pos {residue_number}, chain {chain}")
                if residue_number in type_mapping:
                    res_type += type_mapping[residue_number]
                    if residue_number in conserved:
                        hit, symbol = False, residue.get_symbol()
                        # print("symbol",symbol)
                        for conserved_residue in conserved[residue_number]:
                            # print("VOCAB.abrv_to_symbol(conserved_residue)", VOCAB.abrv_to_symbol(conserved_residue))
                            # print("residue_number", residue_number)
                            # print("conserved_residue", conserved_residue)
                            # symbol C
                            # VOCAB.abrv_to_symbol(conserved_residue) C
                            # residue_number 23
                            # conserved_residue CYS
                            if symbol == VOCAB.abrv_to_symbol(conserved_residue):
                                hit = True
                                break
                            print(f"Actual residue type at position {residue_number} in chain {chain_name}: {symbol}")
                        assert hit, f'Not {conserved[residue_number]} at {residue_number} in chain {chain_name}'
                else:
                    res_type += '-'
            if '0' not in res_type:
                print(self.heavy_chain, self.light_chain, antibody.pdb_id, res_type)
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

        antibody = Protein(antibody.get_id(), selected_peptides)

        return antibody, cdr_pos

    def _cal_epitope(self):
        ag_rids, ag_xs, ab_xs = [], [], []
        ag_mask, ab_mask = [], []
        cdrh3 = self.get_cdr('H3')
        for _type, protein in zip(['ag', 'ab'], [self.antigen, [('A', cdrh3)]]):
            is_ag = _type == 'ag'
            rids = []
            if is_ag: 
                xs, masks = ag_xs, ag_mask
            else:
                xs, masks = ab_xs, ab_mask
            for chain_name, chain in protein:
                # print("chain_name", chain_name)
                if chain is None:
                    print(f"{chain} is None, skipping")
                    continue  # Skip if the chain is None
                for i, residue in enumerate(chain):
                    bb_coord = residue.get_backbone_coord_map()
                    sc_coord = residue.get_sidechain_coord_map()
                    coord = {}
                    coord.update(bb_coord)
                    coord.update(sc_coord)
                    num_pad = VOCAB.MAX_ATOM_NUMBER - len(coord)
                    x = [coord[key] for key in coord] + [[0, 0, 0] for _ in range(num_pad)]
                    mask = [1 for _ in coord] + [0 for _ in range(num_pad)]
                    rids.append((chain_name, i))
                    xs.append(x)
                    masks.append(mask)
            if is_ag:
                ag_rids = rids
        assert len(ag_xs) != 0, 'No antigen structure!'
        # calculate distance
        ag_xs, ab_xs = np.array(ag_xs), np.array(ab_xs)  # [Nag/ab, M, 3], M == MAX_ATOM_NUM
        ag_mask, ab_mask = np.array(ag_mask).astype('bool'), np.array(ab_mask).astype('bool')  # [Nag/ab, M]
        dist = np.linalg.norm(ag_xs[:, None] - ab_xs[None, :], axis=-1)  # [Nag, Nab, M]
        dist = dist + np.logical_not(ag_mask[:, None] * ab_mask[None, :]) * 1e6  # [Nag, Nab, M]
        min_dists = np.min(np.min(dist, axis=-1), axis=-1)  # [ag_len]
        topk = min(len(min_dists), self.num_interface_residues)
        ind = np.argpartition(-min_dists, -topk)[-topk:]
        epitope = []
        for idx in ind:
            chain_name, i = ag_rids[idx]
            residue = self.antigen.peptides[chain_name].get_residue(i)
            epitope.append((residue, chain_name, i))
        return epitope

    def get_id(self) -> str:
        return self.antibody.pdb_id

    def get_antigen(self) -> Protein:
        return deepcopy(self.antigen)

    def get_epitope(self, cdrh3_pos=None) -> List[Tuple[Residue, str, int]]:
        if cdrh3_pos is not None:
            backup = self.cdr_pos
            self.cdr_pos = {'CDR-H3': [cdrh3_pos[0], cdrh3_pos[1]]}
            epitope = self._cal_epitope()
            self.cdr_pos = backup
            return deepcopy(epitope)
        if self.epitope is None:
            self.epitope = self._cal_epitope()
        return deepcopy(self.epitope)

    def get_interacting_residues(self, dist_cutoff=5) -> Tuple[List[int], List[int]]:
        ag_rids, ag_xs, ab_xs = [], [], []
        for chain_name in self.antigen.get_chain_names():
            chain = self.antigen.get_chain(chain_name)
            for i in range(len(chain)):
                try:
                    x = chain.get_ca_pos(i)
                except KeyError:  # CA position is missing
                    continue
                ag_rids.append((chain_name, i))
                ag_xs.append(x)
        for chain_name in self.antibody.get_chain_names():
            chain = self.antibody.get_chain(chain_name)
            for i in range(len(chain)):
                try:
                    x = chain.get_ca_pos(i)
                except KeyError:
                    continue
                ab_xs.append(x)
        assert len(ag_xs) != 0, 'No antigen structure!'
        # calculate distance
        ag_xs, ab_xs = np.array(ag_xs), np.array(ab_xs)
        dist = np.linalg.norm(ag_xs[:, None, :] - ab_xs[None, :, :], axis=-1)
        min_dists = np.min(dist, axis=1)  # [ag_len]
        topk = min(len(min_dists), self.num_interface_residues)
        ind = np.argpartition(-min_dists, -topk)[-topk:]
        epitope = []
        for idx in ind:
            chain_name, i = ag_rids[idx]
            residue = self.antigen.peptides[chain_name].get_residue(i)
            epitope.append((residue, chain_name, i))
        return

    
    def get_interacting_residues_2(self, dist_cutoff=6.6) -> Tuple[List[int], List[int]]:
        ag_rids, ag_xs, ab_xs = [], [], []
        paratope = []  # Add a paratope list
        
        # Process antigen chains
        for chain_name in self.antigen.get_chain_names():
            chain = self.antigen.get_chain(chain_name)
            if chain:  # This checks if chain is not None or empty
                for i in range(len(chain)):
                    try:
                        x = chain.get_ca_pos(i)
                    except KeyError:  # CA position is missing
                        continue
                    ag_rids.append((chain_name, i))
                    ag_xs.append(x)

        # Process antibody chains, checking if the heavy chain is not None or empty
        for chain_name in [self.heavy_chain, self.light_chain]:
            if chain_name:  # Checks if chain_name is not None or empty
                chain = self.antibody.get_chain(chain_name)
                if chain:  # This checks if chain is not None or empty
                    for i in range(len(chain)):
                        try:
                            x = chain.get_ca_pos(i)
                        except KeyError:  # CA position is missing
                            continue
                        ab_xs.append(x)
                        # Check if residue is in contact with antigen chains
                        if any(np.linalg.norm(x - ag_x) < dist_cutoff for ag_x in ag_xs):
                            paratope.append((chain_name, i))

        # Ensure antigen structures are present before calculating distances
        if not ag_xs:
            raise ValueError('No antigen structure!')
        # calculate distance
        ag_xs, ab_xs = np.array(ag_xs), np.array(ab_xs)
        dist = np.linalg.norm(ag_xs[:, None, :] - ab_xs[None, :, :], axis=-1)
        min_dists = np.min(dist, axis=1)  # [ag_len]
        topk = min(len(min_dists), self.num_interface_residues)
        ind = np.argpartition(-min_dists, -topk)[-topk:]
        epitope = []
        for idx in ind:
            chain_name, i = ag_rids[idx]
            residue = self.antigen.peptides[chain_name].get_residue(i)
            epitope.append((residue, chain_name, i))
        return epitope, paratope  # Return both epitope and paratope


    def get_heavy_chain(self) -> Peptide:
        return self.antibody.get_chain(self.heavy_chain)

    def get_light_chain(self) -> Peptide:
        return self.antibody.get_chain(self.light_chain)

    def get_framework(self, fr):  # H/L + FR + 1/2/3/4
        seg_id = int(fr[-1])
        chain = self.get_heavy_chain() if fr[0] == 'H' else self.get_light_chain()
        if chain is None:
            return None
        begin, end = -1, -1
        if seg_id == 1:
            begin, end = 0, self.get_cdr_pos(fr[0] + str(seg_id))[0]
        elif seg_id == 4:
            begin, end = self.get_cdr_pos(fr[0] + '3')[-1] + 1, len(chain)
        else:
            begin = self.get_cdr_pos(fr[0] + str(seg_id - 1))[-1] + 1
            end = self.get_cdr_pos(fr[0] + str(seg_id))[0]
        return chain.get_span(begin, end)

    def get_cdr_pos(self, cdr='H3'):  # H/L + 1/2/3, return [begin, end] position
        cdr = f'CDR-{cdr}'.upper()
        if cdr in self.cdr_pos:
            return self.cdr_pos[cdr]
        else:
            return None

    def get_cdr(self, cdr='H3'):
        cdr = cdr.upper()
        pos = self.get_cdr_pos(cdr)
        if pos is None:
            return None
        chain = self.get_heavy_chain() if 'H' in cdr else self.get_light_chain()
        if chain is None:
            return None
        return chain.get_span(pos[0], pos[1] + 1)

    def to_pdb(self, path, atoms=None):
        peptides = {}
        for name in self.antigen.get_chain_names():
            peptides[name] = self.antigen.get_chain(name)
        for name in self.antibody.get_chain_names():
            peptides[name] = self.antibody.get_chain(name)
        protein = Protein(self.get_id(), peptides)
        protein.to_pdb(path, atoms)
    

    def __str__(self):
        pdb_info = f'PDB ID: {self.pdb_id}'
        antibody_info = f'Antibody H-{self.heavy_chain} ({len(self.get_heavy_chain())})'
        if self.light_chain:
            antibody_info += f', L-{self.light_chain} ({len(self.get_light_chain())})'
        else:
            antibody_info += ', L-None (0)'

        antigen_info = f'Antigen Chains: {[(ag, len(self.antigen.get_chain(ag))) for ag in self.antigen.get_chain_names()]}'
        cdr_info = f'CDRs: \n'
        for name in self.cdr_pos:
            chain = self.get_heavy_chain() if 'H' in name else self.get_light_chain()
            start, end = self.cdr_pos[name]
            cdr_info += f'\t{name}: [{start}, {end}], {chain.seq[start:end + 1]}\n'
        epitope_info = f'Epitope: \n'
        residue_map = {}
        for _, chain_name, i in self.get_epitope():
            if chain_name not in residue_map:
                residue_map[chain_name] = []
            residue_map[chain_name].append(i)
        for chain_name in residue_map:
            epitope_info += f'\t{chain_name}: {sorted(residue_map[chain_name])}\n'

        sep = '\n' + '=' * 20 + '\n'
        return sep + pdb_info + '\n' + antibody_info + '\n' + cdr_info + '\n' + antigen_info + '\n' + epitope_info + sep


class AgAbComplex:

    num_interface_residues = 48  # from PNAS (view as epitope)

    def __init__(self, antigen: Protein, antibody: Protein, heavy_chain: str, light_chain: str,
                 numbering: str='imgt', skip_epitope_cal=False, skip_validity_check=False) -> None:
        self.heavy_chain = heavy_chain
        self.light_chain = light_chain
        self.numbering = numbering

        self.antigen = antigen
        if skip_validity_check:
            self.antibody, self.cdr_pos = antibody, None
        else:
            self.antibody, self.cdr_pos = self._extract_antibody_info(antibody, numbering)
        self.pdb_id = antigen.get_id()

        if skip_epitope_cal:
            self.epitope = None
        else:
            self.epitope = self._cal_epitope()
    
    @classmethod
    def from_pdb(cls, pdb_path: str, heavy_chain: str, light_chain: str, antigen_chains: List[str],
                 numbering: str='imgt', skip_epitope_cal=False, skip_validity_check=False):
        protein = Protein.from_pdb(pdb_path)
        pdb_id = protein.get_id()
        ab_peptides = {
            heavy_chain: protein.get_chain(heavy_chain)
            ,light_chain: protein.get_chain(light_chain)
        }
        ag_peptides = { chain: protein.get_chain(chain) for chain in antigen_chains if protein.get_chain(chain) is not None }
        for chain in antigen_chains:
            assert chain in ag_peptides, f'Antigen chain {chain} has something wrong!'

        antigen = Protein(pdb_id, ag_peptides)
        antibody = Protein(pdb_id, ab_peptides)

        return cls(antigen, antibody, heavy_chain, light_chain, numbering, skip_epitope_cal, skip_validity_check)

    def _extract_antibody_info(self, antibody: Protein, numbering: str):
        # calculating cdr pos according to number scheme (type_mapping and conserved residues)
        numbering = numbering.lower()
        if numbering == 'imgt':
            _scheme = IMGT
        elif numbering.lower() == 'chothia':
            _scheme = Chothia
            # for i in list(range(1, 27)) + list(range(39, 56)) + list(range(66, 105)) + list(range(118, 130)):
            #     type_mapping[i] = '0'
            # for i in range(27, 39):     # cdr1
            #     type_mapping[i] = '1'
            # for i in range(56, 66):     # cdr2
            #     type_mapping[i] = '2'
            # for i in range(105, 118):   # cdr3
            #     type_mapping[i] = '3'
            # conserved = {
            #     23: ['CYS'],
            #     41: ['TRP'],
            #     104: ['CYS'],
            #     # 118: ['PHE', 'TRP']
            # }
        else:
            raise NotImplementedError(f'Numbering scheme {numbering} not implemented')

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


        for c, chain_name in zip(['H', 'L'], [self.heavy_chain, self.light_chain]):
            chain = antibody.get_chain(chain_name)
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
                        assert hit, f'Not {conserved[residue_number]} at {residue_number}'
                else:
                    res_type += '-'
            if '0' not in res_type:
                print(self.heavy_chain, self.light_chain, antibody.pdb_id, res_type)
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

        antibody = Protein(antibody.get_id(), selected_peptides)

        return antibody, cdr_pos

    def _cal_epitope(self):
        ag_rids, ag_xs, ab_xs = [], [], []
        ag_mask, ab_mask = [], []
        cdrh3 = self.get_cdr('H3')
        for _type, protein in zip(['ag', 'ab'], [self.antigen, [('A', cdrh3)]]):
            is_ag = _type == 'ag'
            rids = []
            if is_ag: 
                xs, masks = ag_xs, ag_mask
            else:
                xs, masks = ab_xs, ab_mask
            for chain_name, chain in protein:
                for i, residue in enumerate(chain):
                    bb_coord = residue.get_backbone_coord_map()
                    sc_coord = residue.get_sidechain_coord_map()
                    coord = {}
                    coord.update(bb_coord)
                    coord.update(sc_coord)
                    num_pad = VOCAB.MAX_ATOM_NUMBER - len(coord)
                    x = [coord[key] for key in coord] + [[0, 0, 0] for _ in range(num_pad)]
                    mask = [1 for _ in coord] + [0 for _ in range(num_pad)]
                    rids.append((chain_name, i))
                    xs.append(x)
                    masks.append(mask)
            if is_ag:
                ag_rids = rids
        assert len(ag_xs) != 0, 'No antigen structure!'
        # calculate distance
        ag_xs, ab_xs = np.array(ag_xs), np.array(ab_xs)  # [Nag/ab, M, 3], M == MAX_ATOM_NUM
        ag_mask, ab_mask = np.array(ag_mask).astype('bool'), np.array(ab_mask).astype('bool')  # [Nag/ab, M]
        dist = np.linalg.norm(ag_xs[:, None] - ab_xs[None, :], axis=-1)  # [Nag, Nab, M]
        dist = dist + np.logical_not(ag_mask[:, None] * ab_mask[None, :]) * 1e6  # [Nag, Nab, M]
        min_dists = np.min(np.min(dist, axis=-1), axis=-1)  # [ag_len]
        topk = min(len(min_dists), self.num_interface_residues)
        ind = np.argpartition(-min_dists, -topk)[-topk:]
        epitope = []
        for idx in ind:
            chain_name, i = ag_rids[idx]
            residue = self.antigen.peptides[chain_name].get_residue(i)
            epitope.append((residue, chain_name, i))
        return epitope


    
    def get_id(self) -> str:
        return self.antibody.pdb_id

    def get_antigen(self) -> Protein:
        return deepcopy(self.antigen)

    def get_epitope(self, cdrh3_pos=None) -> List[Tuple[Residue, str, int]]:
        if cdrh3_pos is not None:
            backup = self.cdr_pos
            self.cdr_pos = {'CDR-H3': [cdrh3_pos[0], cdrh3_pos[1]]}
            epitope = self._cal_epitope()
            self.cdr_pos = backup
            return deepcopy(epitope)
        if self.epitope is None:
            self.epitope = self._cal_epitope()
        return deepcopy(self.epitope)

    def get_interacting_residues(self, dist_cutoff=5) -> Tuple[List[int], List[int]]:
        ag_rids, ag_xs, ab_xs = [], [], []
        for chain_name in self.antigen.get_chain_names():
            chain = self.antigen.get_chain(chain_name)
            for i in range(len(chain)):
                try:
                    x = chain.get_ca_pos(i)
                except KeyError:  # CA position is missing
                    continue
                ag_rids.append((chain_name, i))
                ag_xs.append(x)
        for chain_name in self.antibody.get_chain_names():
            chain = self.antibody.get_chain(chain_name)
            for i in range(len(chain)):
                try:
                    x = chain.get_ca_pos(i)
                except KeyError:
                    continue
                ab_xs.append(x)
        assert len(ag_xs) != 0, 'No antigen structure!'
        # calculate distance
        ag_xs, ab_xs = np.array(ag_xs), np.array(ab_xs)
        dist = np.linalg.norm(ag_xs[:, None, :] - ab_xs[None, :, :], axis=-1)
        min_dists = np.min(dist, axis=1)  # [ag_len]
        topk = min(len(min_dists), self.num_interface_residues)
        ind = np.argpartition(-min_dists, -topk)[-topk:]
        epitope = []
        for idx in ind:
            chain_name, i = ag_rids[idx]
            residue = self.antigen.peptides[chain_name].get_residue(i)
            epitope.append((residue, chain_name, i))
        return
    

    def get_heavy_chain(self) -> Peptide:
        return self.antibody.get_chain(self.heavy_chain)

    def get_light_chain(self) -> Peptide:
        return self.antibody.get_chain(self.light_chain)

    def get_framework(self, fr):  # H/L + FR + 1/2/3/4
        seg_id = int(fr[-1])
        chain = self.get_heavy_chain() if fr[0] == 'H' else self.get_light_chain()
        begin, end = -1, -1
        if seg_id == 1:
            begin, end = 0, self.get_cdr_pos(fr[0] + str(seg_id))[0]
        elif seg_id == 4:
            begin, end = self.get_cdr_pos(fr[0] + '3')[-1] + 1, len(chain)
        else:
            begin = self.get_cdr_pos(fr[0] + str(seg_id - 1))[-1] + 1
            end = self.get_cdr_pos(fr[0] + str(seg_id))[0]
        return chain.get_span(begin, end)

    def get_cdr_pos(self, cdr='H3'):  # H/L + 1/2/3, return [begin, end] position
        cdr = f'CDR-{cdr}'.upper()
        if cdr in self.cdr_pos:
            return self.cdr_pos[cdr]
        else:
            return None

    def get_cdr(self, cdr='H3'):
        cdr = cdr.upper()
        pos = self.get_cdr_pos(cdr)
        if pos is None:
            return None
        chain = self.get_heavy_chain() if 'H' in cdr else self.get_light_chain()
        return chain.get_span(pos[0], pos[1] + 1)

    def to_pdb(self, path, atoms=None):
        peptides = {}
        for name in self.antigen.get_chain_names():
            peptides[name] = self.antigen.get_chain(name)
        for name in self.antibody.get_chain_names():
            peptides[name] = self.antibody.get_chain(name)
        protein = Protein(self.get_id(), peptides)
        protein.to_pdb(path, atoms)
    
    def __str__(self):
        pdb_info = f'PDB ID: {self.pdb_id}'
        antibody_info = f'Antibody H-{self.heavy_chain} ({len(self.get_heavy_chain())}), ' + \
                        f'L-{self.light_chain} ({len(self.get_light_chain())})'
        antigen_info = f'Antigen Chains: {[(ag, len(self.antigen.get_chain(ag))) for ag in self.antigen.get_chain_names()]}'
        cdr_info = f'CDRs: \n'
        for name in self.cdr_pos:
            chain = self.get_heavy_chain() if 'H' in name else self.get_light_chain()
            start, end = self.cdr_pos[name]
            cdr_info += f'\t{name}: [{start}, {end}], {chain.seq[start:end + 1]}\n'
        epitope_info = f'Epitope: \n'
        residue_map = {}
        for _, chain_name, i in self.get_epitope():
            if chain_name not in residue_map:
                residue_map[chain_name] = []
            residue_map[chain_name].append(i)
        for chain_name in residue_map:
            epitope_info += f'\t{chain_name}: {sorted(residue_map[chain_name])}\n'

        sep = '\n' + '=' * 20 + '\n'
        return sep + pdb_info + '\n' + antibody_info + '\n' + cdr_info + '\n' + antigen_info + '\n' + epitope_info + sep


def merge_to_one_chain(protein: Protein):
    residues = []
    chain_order = sorted(protein.get_chain_names())
    for chain_name in chain_order:
        chain = protein.get_chain(chain_name)
        for _, residue in enumerate(chain.residues):
            residue.id = (len(residues), ' ')
            residues.append(residue)
    return Protein(protein.get_id(), {'A': Peptide('A', residues)})


def fetch_from_pdb(identifier):
    # example identifier: 1FBI
    url = 'https://data.rcsb.org/rest/v1/core/entry/' + identifier
    res = requests.get(url)
    if res.status_code != 200:
        return None
    url = f'https://files.rcsb.org/download/{identifier}.pdb'
    text = requests.get(url)
    data = res.json()
    data['pdb'] = text.text
    return data


if __name__ == '__main__':
    import sys
    # e.g python -m data.pdb_utils 7m7e.pdb C D A,B
    pdb_path, h, l, ag = sys.argv[1:]
    cplx = AgAbComplex.from_pdb(pdb_path, h, l, ag.split(','), numbering='imgt')
    print(cplx)
