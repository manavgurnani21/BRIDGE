#!/bin/bash
#SBATCH --account=cis250169
#SBATCH --job-name=BRIDGE_env_rebuild
#SBATCH --output=./slurms/logs/BRIDGE_env_rebuild_%j.out
#SBATCH --error=./slurms/logs/BRIDGE_env_rebuild_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -p shared
#
# One-off rebuild of the BRIDGE conda env after its package cache/env got
# corrupted (widespread missing .py source files under matplotlib, statsmodels,
# and stdlib packages like encodings/asyncio/email, .pyc-only left behind).
# ~/.conda/pkgs was already cleared with `conda clean --all` beforehand.
# Runs on a compute node (not the login node) because the login node kills
# long/heavy foreground conda operations.

set -euo pipefail
cd /anvil/scratch/x-mgurnani/BRIDGE

module --force purge
module load conda

echo "[$(date)] starting conda env create -f BRIDGE.yml"
conda env create -f BRIDGE.yml
echo "[$(date)] conda env create finished"

conda activate BRIDGE
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'cuda_available:', torch.cuda.is_available())"
python -c "import torch_geometric; print('torch-geometric:', torch_geometric.__version__)"
python -c "import matplotlib; from matplotlib.cbook import ls_mapper; print('matplotlib OK:', matplotlib.__version__)"
echo "[$(date)] sanity checks passed"
