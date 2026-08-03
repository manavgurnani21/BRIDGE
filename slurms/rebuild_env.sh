#!/bin/bash
# BRIDGE_JOB_CLASS: cpu
#SBATCH --job-name=BRIDGE_env_rebuild
#SBATCH --output=./slurms/logs/BRIDGE_env_rebuild_%j.out
#SBATCH --error=./slurms/logs/BRIDGE_env_rebuild_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# --account/--partition come from slurms/submit.sh at submit time (see
# slurms/cluster/*.sh) -- submit with `slurms/submit.sh rebuild_env.sh`, not a
# bare `sbatch rebuild_env.sh`.
#
# Create-or-rebuild the BRIDGE conda env on a compute node (not the login
# node, which kills long/heavy foreground conda operations). Originally
# written as a one-off after the env's package cache got corrupted on Anvil
# (widespread missing .py source files under matplotlib, statsmodels, and
# stdlib packages like encodings/asyncio/email, .pyc-only left behind) --
# `conda clean --all` was run beforehand there. Idempotent by default: if the
# env already exists (e.g. a build already in progress or completed), this
# skips `conda env create` unless FORCE=1 is passed.

set -euo pipefail

# shellcheck source=/dev/null
source "${BRIDGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurms/cluster/common.sh"
cd "${BRIDGE_ROOT}"

module --force purge
module load "${BRIDGE_MODULE_CONDA}"

if conda env list | awk '{print $1}' | grep -qx "${BRIDGE_CONDA_ENV}"; then
    if [[ "${FORCE:-0}" == "1" ]]; then
        echo "[$(date)] FORCE=1: removing existing '${BRIDGE_CONDA_ENV}' env and recreating"
        conda env remove -n "${BRIDGE_CONDA_ENV}" -y
        conda env create -f BRIDGE.yml
    else
        echo "[$(date)] conda env '${BRIDGE_CONDA_ENV}' already exists -- skipping create (pass FORCE=1 to recreate)"
    fi
else
    echo "[$(date)] starting conda env create -f BRIDGE.yml"
    conda env create -f BRIDGE.yml
    echo "[$(date)] conda env create finished"
fi

conda activate "${BRIDGE_CONDA_ENV}"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'cuda_available:', torch.cuda.is_available())"
python -c "import torch_geometric; print('torch-geometric:', torch_geometric.__version__)"
python -c "import matplotlib; from matplotlib.cbook import ls_mapper; print('matplotlib OK:', matplotlib.__version__)"
echo "[$(date)] sanity checks passed"
