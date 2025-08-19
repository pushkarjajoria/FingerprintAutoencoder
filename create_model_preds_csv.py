import argparse
import json

import numpy as np
import pandas as pd
import torch
from experiment import extract_features, get_model, train_torch_model  # reuse from experiment.py

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--precomputed_splits_file", type=str, required=True)
    parser.add_argument("--fold_idx", type=int, default=0, help="Which fold to use (0-based index)")
    parser.add_argument("--smiles_column_name", type=str, required=True)
    parser.add_argument("--label_column_name", type=str, required=True)
    parser.add_argument("--model1_feat", type=str, required=True)
    parser.add_argument("--model1_head", type=str, required=True)
    parser.add_argument("--model2_feat", type=str, required=True)
    parser.add_argument("--model2_head", type=str, required=True)
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_file)
    smiles = df[args.smiles_column_name].tolist()
    y_true_all = df[args.label_column_name].values

    # Load precomputed folds
    with open(args.precomputed_splits_file, "r") as f:
        precomputed_folds = json.load(f)

    if args.fold_idx >= len(precomputed_folds):
        raise ValueError(f"Fold index {args.fold_idx} out of range (max fold index: {len(precomputed_folds)-1})")

    # Get train/test indices
    train_idx = np.array(precomputed_folds[args.fold_idx]["train"])
    test_idx = np.array(precomputed_folds[args.fold_idx]["test"])

    # Train/Test split
    smiles_train, smiles_test = [smiles[i] for i in train_idx], [smiles[i] for i in test_idx]
    y_train, y_test = y_true_all[train_idx], y_true_all[test_idx]

    # Extract features for both models
    X1_train = extract_features(smiles_train, args.model1_feat)
    X1_test = extract_features(smiles_test, args.model1_feat)
    X2_train = extract_features(smiles_train, args.model2_feat)
    X2_test = extract_features(smiles_test, args.model2_feat)

    # Train model 1
    model1 = get_model(args.model1_head, X1_train.shape[1])
    if args.model1_head.startswith("mlp"):
        model1, _ = train_torch_model(model1, X1_train, y_train)
        preds1 = model1(torch.from_numpy(X1_test).float().to(DEVICE)).detach().cpu().numpy().squeeze()
    else:
        model1.fit(X1_train, y_train)
        preds1 = model1.predict(X1_test)

    # Train model 2
    model2 = get_model(args.model2_head, X2_train.shape[1])
    if args.model2_head.startswith("mlp"):
        model2, _ = train_torch_model(model2, X2_train, y_train)
        preds2 = model2(torch.from_numpy(X2_test).float().to(DEVICE)).detach().cpu().numpy().squeeze()
    else:
        model2.fit(X2_train, y_train)
        preds2 = model2.predict(X2_test)

    # Save results CSV
    out_df = pd.DataFrame({
        args.smiles_column_name: smiles_test,
        "true_labels": y_test,
        "model1_name": args.model1_head,
        "model1_preds": preds1,
        "model2_name": args.model2_head,
        "model2_preds": preds2
    })
    out_df.to_csv(args.output_csv, index=False)
    print(f"[INFO] CSV saved to {args.output_csv}")


if __name__ == "__main__":
    main()
