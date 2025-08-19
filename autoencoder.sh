#!/bin/bash

# Source the GPU renaming script
source /nethome/pjajoria/condor_setup.sh

# Read compression argument
COMPRESSION="$1"

# Input CSV path
CSV="/data/users/pjajoria/molecule_screening/autoencoder/merged.csv"

# Root output directory
BASE_OUTPUT_DIR="/data/users/pjajoria/model_checkpoints/autoencoder"

# Call training script
$PYTHON_BIN/python /nethome/pjajoria/Github/FingerprintAutoencoder/train.py \
    --compression "$COMPRESSION" \
    --csv "$CSV" \
    --output_dir "$BASE_OUTPUT_DIR"
