import torch
from rdkit import Chem
from torch_geometric.data import Data

def get_atom_features(atom):
    # Basic features: atomic number, degree, formal charge, hybridization, is_aromatic
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic())
    ]

def get_bond_features(bond):
    # Basic bond features: bond type
    return [
        int(bond.GetBondType()),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ]

def smiles_to_graph(smiles):
    """Converts SMILES to a PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node features
    x = []
    for atom in mol.GetAtoms():
        x.append(get_atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)
    
    # Edge features and indices
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        edge_indices += [[i, j], [j, i]]
        edge_feature = get_bond_features(bond)
        edge_attrs += [edge_feature, edge_feature]
        
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
        
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
