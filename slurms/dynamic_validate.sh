#!/bin/bash
#SBATCH --account=cis250169-ai
#SBATCH --job-name=BRIDGE_dynamic_validate
#SBATCH --output=./slurms/logs/dynamic_validate/%x_%j.out
#SBATCH --error=./slurms/logs/dynamic_validate/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=500G
#SBATCH --time=10:00:00
#SBATCH -p ai
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=mgurnani@ucdavis.edu

# Dataset stem: loader expects <DATA_FILE>_pos.fa and <DATA_FILE>_neg.fa under --data_path
DATA_FILE=${1:-AUH_HepG2}

# Load in conda
module --force purge
module load intel-mkl
module load conda

# Activate your virtual environment
conda activate BRIDGE

# Dynamic prediction (GPU) — cross cell-line: resolves the alternate model
# (e.g. K562<->HepG2 swap) via resolve_dynamic_model_name and evaluates on this dataset
python main.py \
    --dynamic_predict \
    --data_path ./dataset \
    --data_file "${DATA_FILE}" \
    --device_num 0 \
    --seed 42 \
    --Transformer_path ./RBPformer \
    --model_save_path ./results/model
