import argparse
import sys
import os
import json
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, AllChem

from tqdm import tqdm
import wandb

from mapchiral.mapchiral import encode

# 1. Global Constants & Seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    print(f"[INFO] GPU Information: {torch.cuda.get_device_name(device_id)}")
else:
    print("[WARNING] CUDA is not available. Using CPU.")

FEATURE_EXTRACTORS = [
    "molformer",
    "fingerprint",
    "compressed_fp_5",
    "compressed_fp_10",
    "molformer+fingerprint",
    "molformer+compressed_fp_5",
    "molformer+compressed_fp_10",
    "stereo_fingerprint",
    "molformer+stereo_fingerprint"
]

# FEATURE_EXTRACTORS = [
#     "mapc",
#     "molformer+mapc"
# ]

PREDICTION_HEADS = ["mlp_1", "mlp_2", "mlp_3", "random_forest"]

COMPRESSED_FP_PATHS = {
    "5": "/data/users/pjajoria/model_checkpoints/autoencoder/compression_5/ae_final.pth",
    "10": "/data/users/pjajoria/model_checkpoints/autoencoder/compression_10/ae_final.pth",
}

N_SPLITS = 5

NUM_EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 64

# 2. Data Prep Helpers
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smis2torch_fp(smiles: List[str]) -> torch.Tensor:
    fps = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        fps[i] = FPGEN.GetFingerprintAsNumPy(mol)
    return torch.from_numpy(fps)


def stereo_fingerprint(smiles: List[str]) -> np.ndarray:
    fps = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)
        arr = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bv, arr := np.zeros((2048,), dtype=np.float32))
        fps[i] = arr
    return fps


def mapc(smiles: List[str]) -> np.ndarray:
    "https://github.com/reymond-group/mapchiral"
    "MinHashed Atom-Pair Fingerprint Chiral"
    fps = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        fingerprint = encode(mol, max_radius=2, n_permutations=2048, mapping=False)
        fps[i] = fingerprint.astype(np.float32) / (2 ** 32 - 1)
    return fps


# 3. Lazy Model Loaders
_molformer = None
_molformer_tok = None


def get_molformer():
    global _molformer, _molformer_tok
    if _molformer is None:
        from transformers import AutoModel, AutoTokenizer
        _molformer_tok = AutoTokenizer.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
        )
        _molformer = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            deterministic_eval=True,
            trust_remote_code=True
        )
        # **Move the model to GPU/CPU and set eval mode once**
        _molformer = _molformer.to(DEVICE)
        _molformer.eval()
    return _molformer, _molformer_tok


_compressed_models: Dict[str, torch.nn.Module] = {}


def get_compressed_model(comp: str):
    from models.autoencoder import FingerprintAutoencoder
    if comp not in _compressed_models:
        input_dim = 2048
        latent = int(input_dim * (int(comp) / 100))
        model = FingerprintAutoencoder(input_dim=input_dim, latent_dim=latent)
        ckpt = COMPRESSED_FP_PATHS[comp]
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        _compressed_models[comp] = model.to(DEVICE)
    return _compressed_models[comp]


# 4. Feature Extraction
def extract_features(smiles_list: List[str], feature_type: str, batch_size: int = 256) -> np.ndarray:
    """
    Returns features on CPU as numpy array.
    Models/tensors are moved to DEVICE only during computation.
    """
    if feature_type == "molformer":
        molformer, tok = get_molformer()  # keep your existing loader
        outputs = []
        n = len(smiles_list)
        if n == 0:
            return np.zeros((0, molformer.config.hidden_size))  # safe empty return

        for start in range(0, n, batch_size):
            batch = smiles_list[start : start + batch_size]

            # Tokenize with padding/truncation -> produce torch tensors
            inputs = tok(batch, padding=True, truncation=True, return_tensors="pt")
            # Move only the input tensors to DEVICE (avoid moving python dict keys etc.)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                out = molformer(**inputs).pooler_output  # (batch_size, hidden_dim)

            # detach, move to CPU and convert to numpy
            outputs.append(out.detach().cpu().numpy())

            # cleanup to free GPU memory
            del out
            del inputs
            torch.cuda.empty_cache()

        # stack all batches into one array
        features = np.vstack(outputs) if len(outputs) > 0 else np.zeros((0, molformer.config.hidden_size))
        return features
    if feature_type == "fingerprint":
        return smis2torch_fp(smiles_list).numpy()

    if feature_type == "stereo_fingerprint":
        return stereo_fingerprint(smiles_list)

    if feature_type == "mapc":
        return mapc(smiles_list)

    if feature_type.startswith("compressed_fp"):
        comp = feature_type.split("_")[-1]
        model = get_compressed_model(comp)
        fps = smis2torch_fp(smiles_list).to(DEVICE)
        with torch.no_grad():
            z = model.encoder(fps)
        return z.cpu().numpy()

    if feature_type.startswith("molformer+"):
        base, extra = feature_type.split("+")
        f1 = extract_features(smiles_list, base)
        f2 = extract_features(smiles_list, extra)
        return np.concatenate([f1, f2], axis=1)

    raise ValueError(f"Unknown feature type: {feature_type}")


# 5. Prediction Heads
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int]):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden + [1]
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_model(head: str, input_dim: int):
    if head == "mlp_1": return MLP(input_dim, [128])
    if head == "mlp_2": return MLP(input_dim, [256, 128])
    if head == "mlp_3": return MLP(input_dim, [512, 256, 128])
    if head == "random_forest": return RandomForestRegressor(n_estimators=100, random_state=SEED)
    raise ValueError(head)


def train_torch_model(model, X_train, y_train, global_step=0, metric_key: Optional[str] = None, log_wandb: bool = False):
    """
    Train a PyTorch MLP model on the given features and labels.
    Returns the trained model in eval mode.
    """
    model = model.to(DEVICE).train()
    ds = TensorDataset(
        torch.from_numpy(X_train).float().to(DEVICE),
        torch.from_numpy(y_train).float().unsqueeze(1).to(DEVICE)
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    scheduler = MultiStepLR(opt, milestones=[25], gamma=0.1)  # 1e-3 → 1e-4 at epoch 25

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            preds = model(xb)
            loss = crit(preds, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)

        # log to a single chart, multiple series = fold_0, fold_1, ...
        if log_wandb and metric_key is not None:
            wandb.log({metric_key: avg_loss}, step=global_step)
        global_step += 1
        scheduler.step()

    return model, global_step


# 6. Experiment Runner
def run_experiments(df: pd.DataFrame, precomputed_folds: List[dict], smile_col: str, label_col: str) -> List[Dict]:
    wandb.init(
        project="Inflamnet Ablation",
        name="Quantile Transformation",
        config={
            "seed": SEED,
            "n_splits": N_SPLITS,
            "feature_extractors": FEATURE_EXTRACTORS,
            "prediction_heads": PREDICTION_HEADS
        }
    )

    results = []
    total = len(FEATURE_EXTRACTORS) * len(PREDICTION_HEADS)
    outer = tqdm(total=total, desc="Combos")
    global_step = 0
    for feat in FEATURE_EXTRACTORS:
        smiles = df[smile_col].tolist()
        X = extract_features(smiles, feat)
        y = df[label_col].values  # original labels, 1D

        for head in PREDICTION_HEADS:
            fold_metrics = []
            for fold_idx, fold in enumerate(precomputed_folds):
                tr = fold["train"]
                te = fold["test"]

                X_tr, y_tr = X[tr], y[tr]
                X_te, y_te = X[te], y[te]
                model = get_model(head, input_dim=X.shape[1])

                # ------- fit QuantileTransformer on training labels only -------
                n_samples_tr = max(2, len(y_tr))
                n_quantiles = min(1000, n_samples_tr)
                qt = QuantileTransformer(output_distribution="normal",
                                        n_quantiles=n_quantiles,
                                        random_state=SEED)
                # reshape to 2D for sklearn (n_samples, n_features=1)
                y_tr_2d = np.asarray(y_tr).reshape(-1, 1)
                qt.fit(y_tr_2d)
                y_tr_trans = qt.transform(y_tr_2d).reshape(-1)  # Transform back to 1D

                # transform test labels to transformed space for transformed-metric calc
                y_te_2d = np.asarray(y_te).reshape(-1, 1)
                y_te_trans = qt.transform(y_te_2d).reshape(-1)

                # ------- train model on transformed labels -------
                if head.startswith("mlp"):
                    # PyTorch training: pass transformed labels
                    metric_key = f"train_loss/{feat}/{head}/fold_{fold_idx}"
                    # assume train_torch_model accepts numpy y (1D) or will convert internally
                    model, global_step = train_torch_model(model, X_tr, y_tr_trans, global_step, metric_key, log_wandb=True)
                    model.eval()
                    with torch.no_grad():
                        preds_trans = model(torch.from_numpy(X_te).float().to(DEVICE))
                        preds_trans = preds_trans.cpu().numpy().squeeze()
                else:
                    # scikit-learn style model: fit on transformed labels (ravel)
                    model.fit(X_tr, y_tr_trans.ravel())
                    preds_trans = model.predict(X_te)

                # ------- metrics in transformed space -------
                transformed_mse = mean_squared_error(y_te_trans, preds_trans)
                transformed_rmse = float(np.sqrt(transformed_mse))

                # ------- inverse-transform predictions to original scale -------
                # ensure 2D for inverse_transform
                preds_trans_2d = np.asarray(preds_trans).reshape(-1, 1)
                preds_orig_2d = qt.inverse_transform(preds_trans_2d)
                preds_orig = preds_orig_2d.reshape(-1)

                # compute true-space metrics (original units)
                true_mse = mean_squared_error(y_te, preds_orig)
                true_rmse = float(np.sqrt(true_mse))
                true_r2 = float(r2_score(y_te, preds_orig))
                true_mae = float(mean_absolute_error(y_te, preds_orig))

                fold_metrics.append({
                    "transformed_rmse": transformed_rmse,
                    "true_rmse": true_rmse,
                    "true_r2": true_r2,
                    "true_mae": true_mae
                })

                # free GPU memory for MLPs
                if head.startswith("mlp"):
                    del model
                    torch.cuda.empty_cache()

            # aggregate across folds
            t_rmses = [m["transformed_rmse"] for m in fold_metrics]
            true_rmses = [m["true_rmse"] for m in fold_metrics]
            r2s = [m["true_r2"] for m in fold_metrics]
            maes = [m["true_mae"] for m in fold_metrics]

            entry = {
                "feature": feat, "head": head,
                "transformed_rmse_mean": np.mean(t_rmses), "transformed_rmse_std": np.std(t_rmses),
                "true_rmse_mean": np.mean(true_rmses), "true_rmse_std": np.std(true_rmses),
                "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
                "mae_mean": np.mean(maes), "mae_std": np.std(maes),
            }
            results.append(entry)

            # log to WandB (both transformed and true-space metrics)
            wandb.log({
                f"{feat}/{head}/transformed_rmse": entry["transformed_rmse_mean"],
                f"{feat}/{head}/true_rmse": entry["true_rmse_mean"],
                f"{feat}/{head}/r2": entry["r2_mean"],
                f"{feat}/{head}/mae": entry["mae_mean"],
            })

            print(f"{feat} + {head} → "
                  f"Transformed RMSE {entry['transformed_rmse_mean']:.3f}±{entry['transformed_rmse_std']:.3f}, "
                  f"True RMSE {entry['true_rmse_mean']:.3f}±{entry['true_rmse_std']:.3f}, "
                  f"R² {entry['r2_mean']:.3f}±{entry['r2_std']:.3f}, "
                  f"MAE {entry['mae_mean']:.3f}±{entry['mae_std']:.3f}")
            outer.update(1)

    outer.close()
    wandb.finish()
    return results


# 7. Entry Point
def main():
    parser = argparse.ArgumentParser(description="Run experiments with precomputed splits.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store output results.")
    parser.add_argument("--precomputed_splits_file", type=str, required=True,
                        help="Path to the precomputed splits JSON file.")
    parser.add_argument("--smiles_column_name", type=str, required=True, help="Name of the SMILES column in the CSV.")
    parser.add_argument("--label_column_name", type=str, required=True, help="Name of the label column in the CSV.")

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_file)

    # Load precomputed splits
    with open(args.precomputed_splits_file, "r") as f:
        precomputed_folds = json.load(f)

    # Run experiments
    res = run_experiments(df, precomputed_folds, args.smiles_column_name, args.label_column_name)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "results_inflamnet_quantile_transformation.json")
    with open(output_path, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
