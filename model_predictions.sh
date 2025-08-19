#!/bin/bash

# Source the GPU renaming script
source /nethome/pjajoria/condor_setup.sh

$PYTHON_BIN/python /nethome/pjajoria/Github/FingerprintAutoencoder/create_model_prediction.py \
    --input_file /nethome/pjajoria/Github/thesis/data/processed/lipo.csv \
    --output_csv /nethome/pjajoria/Github/FingerprintAutoencoder/results/lipo_model_preds.csv \
    --precomputed_splits_file /nethome/pjajoria/Github/thesis/data/processed/lipo-splits.json \
    --fold_idx 0 \
    --smiles_column_name smiles \
    --label_column_name label \
    --model1_feat molformer+fingerprint \
    --model1_head mlp_3 \
    --model2_feat molformer+compressed_fp_5 \
    --model2_head mlp_3
