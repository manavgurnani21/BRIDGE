#!/bin/bash
# BRIDGE_JOB_CLASS: gpu
#SBATCH --job-name=BRIDGE_validate
#SBATCH --output=./slurms/logs/validate/%x_%j.out
#SBATCH --error=./slurms/logs/validate/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --time=10:00:00
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=mgurnani@ucdavis.edu
#
# --account/--partition come from slurms/submit.sh at submit time (see
# slurms/cluster/*.sh) -- submit with `slurms/submit.sh validate.sh ...`, not
# a bare `sbatch validate.sh ...`. This must land on a GPU generation this
# repo's pinned torch==2.0.1+cu117 build supports (sm_80/86 only, no sm_90+);
# bridge_assert_gpu_partition below enforces that per-cluster.

# Dataset stem: loader expects <DATA_FILE>_pos.fa and <DATA_FILE>_neg.fa under --data_path
DATA_FILE=${1:-AUH_HepG2}

# shellcheck source=/dev/null
source "${BRIDGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurms/cluster/common.sh"
bridge_assert_gpu_partition
bridge_load_base_modules
bridge_activate_env

# Validate (GPU) — loads saved checkpoint and evaluates on the test split
python main.py \
    --validate \
    --data_path ./dataset \
    --data_file "${DATA_FILE}" \
    --device_num 0 \
    --seed 42 \
    --Transformer_path ./RBPformer \
    --model_save_path ./results/model
