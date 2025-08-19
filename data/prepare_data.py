import os
from typing import List
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
from sklearn.model_selection import train_test_split

# Only initialize once
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smis2torch_fp(smiles: List[str]):
    # torch uses optimized dot products for float32 but not for int
    # or bool
    fps = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        fps[i, :] = FPGEN.GetFingerprintAsNumPy(mol)

    return torch.from_numpy(fps)


def load_and_prepare(csv_path, cache_path, smiles_col='SMILES', test_size=0.1, seed=42):
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from: {cache_path}")
        cached_data = torch.load(cache_path)
        return cached_data['X_train'], cached_data['X_val']
    else:
        print("Cache not found. Preparing dataset...")

        # Load CSV and generate fingerprints
        df = pd.read_csv(csv_path)
        fps = smis2torch_fp(df[smiles_col].tolist())

        # Split the dataset
        X_train, X_val = train_test_split(fps, test_size=test_size, random_state=seed)

        # Save to cache
        torch.save({'X_train': X_train, 'X_val': X_val}, cache_path)
        print(f"Dataset cached at: {cache_path}")

        return X_train, X_val
