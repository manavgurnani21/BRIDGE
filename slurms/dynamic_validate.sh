#!/bin/bash
# BRIDGE_JOB_CLASS: gpu
#SBATCH --job-name=BRIDGE_dynamic_validate
#SBATCH --output=./slurms/logs/dynamic_validate/%x_%j.out
#SBATCH --error=./slurms/logs/dynamic_validate/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=500G
#SBATCH --time=10:00:00
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=mgurnani@ucdavis.edu
#
# --account/--partition come from slurms/submit.sh at submit time (see
# slurms/cluster/*.sh) -- submit with `slurms/submit.sh dynamic_validate.sh ...`,
# not a bare `sbatch dynamic_validate.sh ...`. This previously ran on `-p ai`
# (Anvil's H100 partition), which is actually INCOMPATIBLE with this repo's
# pinned torch==2.0.1+cu117 build (no kernels past sm_86); bridge_assert_gpu_partition
# below now enforces landing on a supported partition instead.

# Dataset stem: loader expects <DATA_FILE>_pos.fa and <DATA_FILE>_neg.fa under --data_path
DATA_FILE=${1:-AUH_HepG2}

# shellcheck source=/dev/null
source "${BRIDGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurms/cluster/common.sh"
bridge_assert_gpu_partition
bridge_load_base_modules
bridge_activate_env

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
