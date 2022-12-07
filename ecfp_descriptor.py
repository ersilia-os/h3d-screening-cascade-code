import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rd

nBits = 2048 #Use a fingerprint with a length of 2048 descriptors
radius = 3  #Construct fingerprint using groups of atoms up to 3 bonds apart

#Set maximum value for any single bit
def clip(v):
    if v > 255:
        v = 255
    return v


class Ecfp(object):
    def __init__(self):
        self.name = "ecfp"
        self.radius = radius
        self.nbits = nBits
        
    #Convert SMILES representation to rdkit's internal 'Mol' data structure
    def _smi2mol(self, smiles):
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        return mols

    #Convert 'Mol' structure to the Morgan Fingerprint vector representation
    def calc(self, smiles = None, mols = None):
        fingerprints = []
        if mols is None:
            mols = self._smi2mol(smiles)
        for mol in mols:
            counts = list(rd.GetHashedMorganFingerprint(mol, radius=self.radius, nBits=self.nbits))
            counts = [clip(x) for x in counts]
            fingerprints += [counts]
        return np.array(fingerprints)
