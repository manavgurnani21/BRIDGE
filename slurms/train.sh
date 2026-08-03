#!/bin/bash
# BRIDGE_JOB_CLASS: gpu
#SBATCH --job-name=BRIDGE_train
#SBATCH --output=./slurms/logs/train/%x_%j.out
#SBATCH --error=./slurms/logs/train/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --time=2:00:00
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=mgurnani@ucdavis.edu
#
# --account/--partition are supplied at submit time by slurms/submit.sh (see
# slurms/cluster/*.sh) -- they differ per cluster and can't be shell-expanded
# in #SBATCH lines. Submit this with `slurms/submit.sh train.sh ...`, not a
# bare `sbatch train.sh ...`.

# Dataset stem: loader expects <DATA_FILE>_pos.fa and <DATA_FILE>_neg.fa under --data_path
DATA_FILE=${1:-AUH_HepG2}

# shellcheck source=/dev/null
source "${BRIDGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurms/cluster/common.sh"
bridge_assert_gpu_partition
bridge_load_base_modules
bridge_activate_env

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
