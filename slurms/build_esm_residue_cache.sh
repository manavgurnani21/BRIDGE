#!/bin/bash
# BRIDGE_JOB_CLASS: gpu
#SBATCH --job-name=BRIDGE_esm_residue_cache
#SBATCH --output=./slurms/logs/esm_residue_cache/%x_%j.out
#SBATCH --error=./slurms/logs/esm_residue_cache/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=mgurnani@ucdavis.edu
#
# --account/--partition come from slurms/submit.sh at submit time -- submit with
# `slurms/submit.sh build_esm_residue_cache.sh`, not a bare `sbatch slurms/build_esm_residue_cache.sh`.
#
# One-off precompute of the per-residue ESM-2 cache consumed by the "attn_protein" ablation
# config (see utils/protein_features.py, ablation/build_esm_residue_cache.py). Idempotent:
# ablation/build_esm_residue_cache.py skips any {RBP}.npy that already exists, so re-running
# this job (e.g. after adding new FASTAs) only computes what's missing.
#
# fair-esm is not part of the tracked BRIDGE conda env (reqs.txt/BRIDGE.yml) since nothing
# else in the pipeline needs it. Rather than build a whole second env, this job just pip
# installs it into the existing BRIDGE env at runtime (it only needs torch + einops, both
# already present) -- skipped if already importable, so this is a no-op on repeat runs.
#
# Must land on a GPU generation this repo's pinned torch==2.0.1+cu117 build supports
# (sm_80/86 only) -- see slurms/ablation.sh for the fuller explanation of this constraint.

set -euo pipefail

FASTA_DIR=${FASTA_DIR:-/quobyte/savirangrp/manav/dataset/protein}

# shellcheck source=/dev/null
source "${BRIDGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurms/cluster/common.sh"
bridge_assert_gpu_partition
bridge_load_base_modules
bridge_activate_env

CACHE_DIR=${CACHE_DIR:-${BRIDGE_ESM_RESIDUE_CACHE_DIR:?set BRIDGE_ESM_RESIDUE_CACHE_DIR in slurms/cluster/${BRIDGE_CLUSTER}.sh, or pass CACHE_DIR=... explicitly}}

echo "[$(date)] host=$(hostname) SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

python -c "import esm" 2>/dev/null || {
    echo "[$(date)] fair-esm not importable in ${BRIDGE_CONDA_ENV}; installing"
    pip install --quiet fair-esm
}

echo "[$(date)] building per-residue ESM-2 cache: fasta_dir=${FASTA_DIR} cache_dir=${CACHE_DIR}"
python -m ablation.build_esm_residue_cache \
    --fasta_dir "${FASTA_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --device cuda
echo "[$(date)] done"
