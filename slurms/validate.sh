#!/bin/bash
#SBATCH --account=cis250169-gpu
#SBATCH --job-name=BRIDGE_validate
#SBATCH --output=./slurms/logs/validate/%x_%j.out
#SBATCH --error=./slurms/logs/validate/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --time=10:00:00
#SBATCH -p gpu
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=mgurnani@ucdavis.edu
#
# Runs on -p gpu (A100, sm_80), NOT -p ai (H100, sm_90): this repo's pinned torch==2.0.1+cu11
# build only ships kernels up to sm_86, so it hard-fails on H100 with "no kernel image is
# available for execution on the device". See slurms/validate_pretrained.sh / slurms/ablation.sh
# for the same fix applied for the same reason.

# Dataset stem: loader expects <DATA_FILE>_pos.fa and <DATA_FILE>_neg.fa under --data_path
DATA_FILE=${1:-AUH_HepG2}

# Load in conda
module --force purge
module load intel-mkl
module load conda

# Activate your virtual environment
conda activate BRIDGE

# Validate (GPU) — loads saved checkpoint and evaluates on the test split
python main.py \
    --validate \
    --data_path ./dataset \
    --data_file "${DATA_FILE}" \
    --device_num 0 \
    --seed 42 \
    --Transformer_path ./RBPformer \
    --model_save_path ./results/model
