#!/bin/bash

# Source the GPU renaming script
source /nethome/pjajoria/condor_setup.sh

$PYTHON_BIN/python /nethome/pjajoria/Github/FingerprintAutoencoder/create_model_preds_on_german_drugs.py \
    --input_file /nethome/pjajoria/Github/thesis/data/processed/lipo.csv \
    --input_smiles_column smiles \
    --label_column_name label \
    --test_file /nethome/pjajoria/Downloads/Germany_drugs_wt_labels.csv \
    --test_smiles_column_name SMILES \
    --model1_feat molformer+fingerprint \
    --model1_head mlp_3 \
    --model2_feat molformer+compressed_fp_5 \
    --model2_head mlp_3 \
    --output_csv /nethome/pjajoria/Github/FingerprintAutoencoder/results/lipo_model_preds_germany_drugs.csv
