import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from experiment import extract_features, get_model, train_torch_model  # reuse from experiment.py

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Optional: use RDKit to canonicalize SMILES for better matching if available
try:
    from rdkit import Chem

    def canonical_smiles(smi: str):
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                return None
            return Chem.MolToSmiles(m, isomericSmiles=True)
        except Exception:
            return None

    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

    def canonical_smiles(smi: str):
        # fallback: return the original string (matching will be exact string match)
        return smi


def lookup_true_labels(test_smiles, labeled_df, smiles_col, label_col):
    """
    For every SMILES in test_smiles, try to find a matching entry in labeled_df.
    Matching is done using canonicalized SMILES (if RDKit available) then exact string match.
    If multiple matches are found, picks the first non-null label.
    Returns a list of labels where missing entries are represented by np.nan.
    """
    # Build lookup dict from canonicalized smiles -> label (first non-null)
    lookup = {}
    # iterate through labeled_df rows
    for _, row in labeled_df[[smiles_col, label_col]].iterrows():
        s = row[smiles_col]
        lab = row[label_col]
        if pd.isna(s):
            continue
        cs = canonical_smiles(s) if RDKit_AVAILABLE else s
        if cs is None:
            cs = s  # fallback
        if cs not in lookup:
            # store first non-null label if available
            lookup[cs] = lab if not pd.isna(lab) else np.nan
        else:
            # if existing is NaN and this one has value, update
            if pd.isna(lookup[cs]) and not pd.isna(lab):
                lookup[cs] = lab

    results = []
    for s in test_smiles:
        if pd.isna(s):
            results.append(np.nan)
            continue
        cs = canonical_smiles(s) if RDKit_AVAILABLE else s
        if cs is None:
            cs = s
        val = lookup.get(cs, np.nan)
        results.append(val)
    return results


def main():
    parser = argparse.ArgumentParser(description="Train on labeled dataset and predict on separate test SMILES list.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the labeled dataset (e.g., full lipo.csv) containing labels.")
    parser.add_argument("--input_smiles_column", type=str, required=True,
                        help="SMILES column name in the labeled dataset.")
    parser.add_argument("--label_column_name", type=str, required=True,
                        help="Label column name (lipophilicity) in the labeled dataset.")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to the test dataset (may not contain labels).")
    parser.add_argument("--test_smiles_column_name", type=str, required=True,
                        help="SMILES column name in the test dataset.")
    parser.add_argument("--model1_feat", type=str, required=True)
    parser.add_argument("--model1_head", type=str, required=True)
    parser.add_argument("--model2_feat", type=str, required=True)
    parser.add_argument("--model2_head", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save output CSV with predictions.")
    args = parser.parse_args()

    # Load labeled data (full dataset with labels)
    df_full = pd.read_csv(args.input_file)
    if args.input_smiles_column not in df_full.columns:
        raise ValueError(f"SMILES column '{args.input_smiles_column}' not found in {args.input_file}")
    if args.label_column_name not in df_full.columns:
        raise ValueError(f"Label column '{args.label_column_name}' not found in {args.input_file}")

    # Load test dataset early so we can exclude overlapping SMILES from training
    df_test = pd.read_csv(args.test_file)
    if args.test_smiles_column_name not in df_test.columns:
        raise ValueError(f"Test SMILES column '{args.test_smiles_column_name}' not found in {args.test_file}")
    smiles_test = df_test[args.test_smiles_column_name].dropna().astype(str).tolist()

    # Build set of test canonical SMILES (if canonical_smiles exists it will be used)
    def _canon(s):
        try:
            cs = canonical_smiles(s)  # uses existing function if present
        except Exception:
            cs = None
        return cs if cs is not None else s

    test_set = set(_canon(s) for s in smiles_test)

    # Create boolean mask indicating which full-df rows appear in the test set (via canonical SMILES)
    full_smiles_series = df_full[args.input_smiles_column].fillna("").astype(str)
    in_test_mask = full_smiles_series.apply(lambda s: _canon(s) in test_set)

    # Train = all rows in df_full NOT present in test file
    train_df = df_full.loc[~in_test_mask].copy()
    excluded = int(in_test_mask.sum())
    print(f"[INFO] Excluding {excluded} rows from training because they appear in the test file; using {len(train_df)} training rows.")

    # Drop any training rows with missing labels
    before = len(train_df)
    train_df = train_df.dropna(subset=[args.label_column_name])
    if len(train_df) != before:
        print(f"[WARN] Dropped {before - len(train_df)} training rows with missing labels.")

    smiles_train = train_df[args.input_smiles_column].astype(str).tolist()
    y_train = train_df[args.label_column_name].values

    # Extract features for training
    X1_train = extract_features(smiles_train, args.model1_feat)
    X2_train = extract_features(smiles_train, args.model2_feat)

    # Train model 1
    model1 = get_model(args.model1_head, X1_train.shape[1])
    if args.model1_head.startswith("mlp"):
        model1, _ = train_torch_model(model1, X1_train, y_train)
    else:
        model1.fit(X1_train, y_train)

    # Train model 2
    model2 = get_model(args.model2_head, X2_train.shape[1])
    if args.model2_head.startswith("mlp"):
        model2, _ = train_torch_model(model2, X2_train, y_train)
    else:
        model2.fit(X2_train, y_train)

    # Extract features for test smiles and predict
    X1_test = extract_features(smiles_test, args.model1_feat)
    X2_test = extract_features(smiles_test, args.model2_feat)

    if args.model1_head.startswith("mlp"):
        model1.eval()
        with torch.no_grad():
            preds1 = model1(torch.from_numpy(X1_test).float().to(DEVICE)).detach().cpu().numpy().squeeze()
    else:
        preds1 = model1.predict(X1_test)

    if args.model2_head.startswith("mlp"):
        model2.eval()
        with torch.no_grad():
            preds2 = model2(torch.from_numpy(X2_test).float().to(DEVICE)).detach().cpu().numpy().squeeze()
    else:
        preds2 = model2.predict(X2_test)

    # Lookup true labels from full labeled dataset for each test SMILES
    true_labels = lookup_true_labels(smiles_test, df_full, args.input_smiles_column, args.label_column_name)

    # Prepare output DataFrame
    col_pred_1 = f"{args.model1_feat}+{args.model1_head}"
    col_pred_2 = f"{args.model2_feat}+{args.model2_head}"
    out_df = pd.DataFrame({
        args.test_smiles_column_name: smiles_test,
        "true_label": true_labels,
        col_pred_1: preds1,
        col_pred_2: preds2
    })

    # Ensure output directory exists and save
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"[INFO] CSV saved to {args.output_csv}")


if __name__ == "__main__":
    main()
