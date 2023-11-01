import torch,os
from torch.utils.data import TensorDataset,random_split
import pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os, random
import numpy as np
from dgllife.data import MoleculeCSVDataset
from functools import partial
from dgllife.utils import smiles_to_bigraph, RandomSplitter# ConsecutiveSplitter,
from dgllife.utils import MolToBigraph
from dgllife.data import UnlabeledSMILES
# Obtain the features of atoms and bonds
def load_coeff(mol):
    mol = Chem.MolToSmiles(mol, canonical=True)    
    df = pd.read_csv('train_set_5631.csv')
    sms=[Chem.MolToSmiles(Chem.MolFromSmiles(sm), canonical=True) for sm in df['smiles'].tolist()]
    idx=sms.index(mol)    
    coeff = torch.load('orb_coeff.t')
    return coeff[idx]


def featurize_atoms(mol):  
    feats = []
#    coo = load_coeff(mol)
    for atom in mol.GetAtoms():
        hy = [int(atom.GetHybridization()==y) for y in [Chem.rdchem.HybridizationType.SP,
              Chem.rdchem.HybridizationType.SP2,Chem.rdchem.HybridizationType.SP3]]
        feats.append([atom.GetAtomicNum(), atom.GetExplicitValence(), atom.GetImplicitValence(),
                      atom.GetTotalNumHs(),atom.GetDegree(), int(atom.GetIsAromatic())]+hy)#+[coo[atom.GetIdx()]]
#, atom.GetFormalCharge(), atom.GetNumRadicalElectrons(), int(atom.IsInRing())          
    return {'h': torch.tensor(feats).float()}        


def featurize_edges(mol, add_self_loop=True):   
    feats = []
#    coo = load_coeff(mol)
    num_atoms = mol.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(num_atoms):
            e_ij = mol.GetBondBetweenAtoms(i,j)
            if e_ij is None:
                bond_type = None
            else:
                bond_type = e_ij.GetBondType()
                

                feats.append([int(bond_type == x)for x in (Chem.rdchem.BondType.SINGLE,
                              Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC)])#+[coo[i] * coo[j]]
            if i == j:
                feats.append([0,0,0,0])    #+[coo[i] * coo[j]]       
    return {'e': torch.tensor(feats).float()}



if __name__ == "__main__":
    torch.manual_seed(1024)
    random.seed(1024)
    np.random.seed(1024)
    df = pd.read_csv('5000reorg.csv')
    smiles = df[df.columns[0]].tolist()
    sms=[Chem.MolToSmiles(Chem.MolFromSmiles(sm), canonical=True) for sm in df['smiles'].tolist()]
   
    mol_to_g = MolToBigraph(add_self_loop=True,
                        node_featurizer=featurize_atoms,
                        edge_featurizer=None)
    dataset = UnlabeledSMILES(sms, mol_to_graph=mol_to_g)
    torch.save(dataset, "graph_interference.bin")

    print(dataset[0])
