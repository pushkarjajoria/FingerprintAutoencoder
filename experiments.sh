#!/bin/bash

# Source the GPU renaming script
source /nethome/pjajoria/condor_setup.sh

# Call training script
$PYTHON_BIN/python /nethome/pjajoria/Github/FingerprintAutoencoder/experiment.py \
    --input_file /nethome/pjajoria/Github/thesis/data/processed/inflamnat-dedup.csv \
    --output_dir /nethome/pjajoria/Github/FingerprintAutoencoder/results \
    --precomputed_splits_file /nethome/pjajoria/Github/thesis/data/processed/inflamnat-dedup.splits.json \
    --smiles_column_name SMILES \
    --label_column_name pNO

# For Lipo
#DATA_PATH = "/nethome/pjajoria/Github/thesis/data/processed/lipo.csv"
#SPLITS_PATH = "/nethome/pjajoria/Github/thesis/data/processed/lipo-splits.json"
#OUTPUT_DIR = "/nethome/pjajoria/Github/FingerprintAutoencoder"
