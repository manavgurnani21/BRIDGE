#!/bin/bash
# BRIDGE_JOB_CLASS: gpu
#SBATCH --job-name=BRIDGE_ablation
#SBATCH --output=./slurms/logs/ablation/%x_%A_%a.out
#SBATCH --error=./slurms/logs/ablation/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=mgurnani@ucdavis.edu
#
# --account/--partition come from slurms/submit.sh at submit time (see
# slurms/cluster/*.sh) -- submit with `slurms/submit.sh ablation.sh ...`, not a
# bare `sbatch slurms/ablation.sh ...`.
#
# Must land on a GPU generation this repo's pinned torch==2.0.1+cu117 build
# supports (sm_80/86 only): it hard-fails on sm_90+ with "no kernel image is
# available for execution on the device" (confirmed: jobs 19448485/19448645
# died this way on every array task, right at RBPformer/BERT embedding build,
# before any ablation config ran, when misrouted to Anvil's H100 partition).
# bridge_assert_gpu_partition below enforces the safe partition list per
# cluster (slurms/cluster/*.sh: BRIDGE_GPU_SAFE_PARTITIONS) instead of relying
# on a hardcoded partition name here. See slurms/validate_pretrained.sh for the
# same fix applied earlier.
#
# Feature + module ablation for BRIDGE. Each array task trains ALL configs
# (baseline + 5 feature drops + kan_to_mlp + adpnet_to_gap) for a SHARD of
# SHARD_SIZE datasets (one Python process loops the shard sequentially, reusing
# the features it builds per dataset). Runs are idempotent: a requeued/preempted
# task skips configs whose row-file already exists, regardless of shard membership.
#
# Batching multiple datasets per task, plus the stagger + retry below, exist to
# avoid a shared-filesystem race: when many array tasks call `conda activate` and
# launch `python` in the same instant, Python's interpreter bootstrap can die with
# `Fatal Python error: init_fs_encoding` under filesystem metadata-server
# contention. Batching cuts the number of concurrent launches; stagger spreads
# them out; retry recovers any race that still slips through.
#
# ---- Usage -------------------------------------------------------------------
# 1) Single dataset (or explicit comma-separated list):
#      slurms/submit.sh ablation.sh AUH_HepG2
#      slurms/submit.sh ablation.sh AUH_HepG2,AARS_K562
#
# 2) Array over a manifest (SHARD_SIZE datasets per task; task index -> line range):
#      # pilot (3 datasets):
#      MANIFEST=ablation/datasets_pilot.txt slurms/submit.sh --array=0-2%3 --time=02:00:00 ablation.sh
#      # full sweep (261 datasets, SHARD_SIZE=7 -> 38 shards, <=20 concurrent):
#      SHARD_SIZE=7 slurms/submit.sh --array=0-37%20 --time=14:00:00 ablation.sh
#
# 3) After the array finishes, collate to master CSV/long/parquet + deltas:
#      python -m ablation.collate_results --manifest ablation/datasets.txt
#
# Resubmitting the SAME array command is always safe: idempotency is keyed per
# dataset+config row-file on disk, not by task ID, so already-completed datasets
# (from a prior run, even under a different shard mapping) are skipped instantly.
#
# Tune --mem / --time from a pilot run: feature-building holds the RBPformer
# embeddings in memory, and each task trains up to 8 models sequentially per
# dataset in its shard.
# ------------------------------------------------------------------------------

set -euo pipefail

# Overridable knobs (env vars).
MANIFEST=${MANIFEST:-ablation/datasets.txt}
MODE=${MODE:-all}
SEED=${SEED:-42}
MAX_EPOCHS=${MAX_EPOCHS:-200}
EARLY_STOPPING=${EARLY_STOPPING:-10}
DATA_PATH=${DATA_PATH:-./dataset}
TRANSFORMER_PATH=${TRANSFORMER_PATH:-./RBPformer}
OUT_DIR=${OUT_DIR:-./results/ablation}
SHARD_SIZE=${SHARD_SIZE:-7}
STAGGER_WINDOW=${STAGGER_WINDOW:-60}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
RETRY_BACKOFF=${RETRY_BACKOFF:-15}

# Pick the dataset(s): explicit $1 wins (comma-separated allowed); otherwise map
# array index -> a SHARD_SIZE-line range of the manifest.
if [[ $# -ge 1 && -n "${1:-}" ]]; then
    DATASETS="$1"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    START=$(( SLURM_ARRAY_TASK_ID * SHARD_SIZE + 1 ))
    END=$(( START + SHARD_SIZE - 1 ))
    # sed range naturally clamps past EOF, so the last (ragged) shard just gets fewer lines.
    DATASETS=$(sed -n "${START},${END}p" "${MANIFEST}" | tr '\n' ',' | sed 's/,$//')
else
    echo "ERROR: pass dataset stem(s) as \$1, or submit with --array to read ${MANIFEST}" >&2
    exit 1
fi

if [[ -z "${DATASETS}" ]]; then
    echo "ERROR: empty shard (array index ${SLURM_ARRAY_TASK_ID:-?} of ${MANIFEST}, shard_size=${SHARD_SIZE})" >&2
    exit 1
fi
echo "[$(date)] ablation task ${SLURM_ARRAY_TASK_ID:-single} -> datasets [${DATASETS}] (mode=${MODE})"

# shellcheck source=/dev/null
source "${BRIDGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurms/cluster/common.sh"
bridge_assert_gpu_partition
bridge_load_base_modules

# Cached whole-protein ESM-2 embeddings for the "protein" config (mode=feature/all only;
# see utils/protein_features.py). Cluster-specific -- comes from slurms/cluster/<cluster>.sh,
# same pattern as BRIDGE_PRETRAINED_MODEL_DIR in slurms/validate_pretrained.sh.
ESM_CACHE_DIR=${ESM_CACHE_DIR:-${BRIDGE_ESM_CACHE_DIR:?set BRIDGE_ESM_CACHE_DIR in slurms/cluster/${BRIDGE_CLUSTER}.sh, or pass ESM_CACHE_DIR=... explicitly}}

# Stagger conda-activate/python-launch across a window so array tasks scheduled
# together don't all hit the shared filesystem's metadata server in the same
# instant (the root cause of the init_fs_encoding race described above).
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    JITTER_DS=$(( RANDOM % (STAGGER_WINDOW * 10) ))
    SLEEP_S=$(awk -v j="${JITTER_DS}" 'BEGIN{printf "%.1f", j/10.0}')
    echo "[$(date)] staggering: sleeping ${SLEEP_S}s before conda activate"
    sleep "${SLEEP_S}"
fi

# Retry conda-activate + the python launch on the specific transient
# filesystem-contention signature (fast failure + init_fs_encoding in stderr).
# Any other failure (real bugs, data errors, later-stage crashes) is not
# retried and exits immediately with its real code.
ATTEMPT=1
STDERR_TMP="${SLURM_TMPDIR:-/tmp}/ablation_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}_attempt.err"

while true; do
    echo "[$(date)] attempt ${ATTEMPT}/${MAX_ATTEMPTS}: bridge_activate_env (module load ${BRIDGE_MODULE_CONDA}; conda activate ${BRIDGE_CONDA_ENV})"
    START_TS=$(date +%s)

    set +e
    (
        bridge_activate_env
        echo "[$(date)] host=$(hostname) SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
        nvidia-smi -L
        python -c "import torch; print(f'[gpu_diag] device_count={torch.cuda.device_count()} current_device={torch.cuda.current_device()} name={torch.cuda.get_device_name(0)}')"
        python -m ablation.run_ablation \
            --data_files "${DATASETS}" \
            --data_path "${DATA_PATH}" \
            --Transformer_path "${TRANSFORMER_PATH}" \
            --esm_cache_dir "${ESM_CACHE_DIR}" \
            --mode "${MODE}" \
            --seed "${SEED}" \
            --max_epochs "${MAX_EPOCHS}" \
            --early_stopping "${EARLY_STOPPING}" \
            --out_dir "${OUT_DIR}" \
            --device_num 0
    ) 2> >(tee "${STDERR_TMP}" >&2)
    RC=$?
    set -e
    END_TS=$(date +%s)
    ELAPSED=$(( END_TS - START_TS ))

    if [[ ${RC} -eq 0 ]]; then
        echo "[$(date)] attempt ${ATTEMPT} succeeded (${ELAPSED}s)"
        break
    fi

    IS_FS_RACE=0
    if [[ ${ELAPSED} -le 30 ]] && grep -qE 'init_fs_encoding|Fatal Python error|no codec search functions registered' "${STDERR_TMP}" 2>/dev/null; then
        IS_FS_RACE=1
    fi

    if [[ ${IS_FS_RACE} -eq 1 && ${ATTEMPT} -lt ${MAX_ATTEMPTS} ]]; then
        BACKOFF=$(( RETRY_BACKOFF * ATTEMPT + (RANDOM % 10) ))
        echo "[$(date)] attempt ${ATTEMPT} hit the fs-encoding race (rc=${RC}, ${ELAPSED}s) — retrying in ${BACKOFF}s"
        sleep "${BACKOFF}"
        ATTEMPT=$(( ATTEMPT + 1 ))
        continue
    fi

    echo "[$(date)] attempt ${ATTEMPT} failed with rc=${RC} (${ELAPSED}s), not a retryable fs-race or attempts exhausted" >&2
    rm -f "${STDERR_TMP}"
    exit "${RC}"
done
rm -f "${STDERR_TMP}"
