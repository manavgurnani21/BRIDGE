#!/bin/bash
#SBATCH --account=cis250169-ai
#SBATCH --job-name=BRIDGE_train
#SBATCH --output=./slurms/logs/train/%x_%j.out
#SBATCH --error=./slurms/logs/train/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --time=2:00:00
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

# Train (GPU) — mirrors README "1) Train" command
python main.py \
    --train \
    --data_path ./dataset \
    --data_file "${DATA_FILE}" \
    --device_num 0 \
    --seed 42 \
    --early_stopping 20 \
    --Transformer_path ./RBPformer \
    --model_save_path ./results/model \
    --lr 0.001
