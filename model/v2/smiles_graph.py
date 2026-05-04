import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data

def get_atom_features(atom, normalize=True):
    """
    Extract atom features with optional normalization.
    
    Features:
        - Atomic number (1-118)
        - Degree (0-4)
        - Formal charge (-2 to 2)
        - Hybridization (0-3)
        - Is aromatic (0-1)
    """
    features = [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic())
    ]
    
    if normalize:
        # Normalize to [0, 1] range
        features[0] /= 118.0  # Atomic number max
        features[1] /= 4.0    # Degree max
        features[2] = (features[2] + 2.0) / 4.0  # Charge range [-2, 2] -> [0, 1]
        features[3] /= 3.0    # Hybridization max
    
    return features

def get_bond_features(bond, normalize=True):
    """
    Extract bond features with optional normalization.
    
    Features:
        - Bond type (0-3: single, double, triple, aromatic)
        - Is conjugated (0-1)
        - Is in ring (0-1)
    """
    features = [
        int(bond.GetBondType()),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ]
    
    if normalize:
        features[0] /= 3.0  # Bond type max
    
    return features

def smiles_to_graph(smiles, normalize=True):
    """
    Converts SMILES to a PyTorch Geometric Data object.
    
    Args:
        smiles: SMILES string
        normalize: Whether to normalize features to [0, 1]
        
    Returns:
        Data object with normalized features, or None if invalid SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Add hydrogens for more complete structural information
    mol = Chem.AddHs(mol)
    
    # Node features
    x = []
    for atom in mol.GetAtoms():
        x.append(get_atom_features(atom, normalize=normalize))
    x = torch.tensor(x, dtype=torch.float32)
    
    # Edge features and indices
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        edge_indices += [[i, j], [j, i]]
        edge_feature = get_bond_features(bond, normalize=normalize)
        edge_attrs += [edge_feature, edge_feature]
    
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        # Isolated atoms
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float32)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
