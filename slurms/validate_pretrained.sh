#!/bin/bash
# BRIDGE_JOB_CLASS: gpu
#SBATCH --job-name=BRIDGE_validate_pretrained
#SBATCH --output=./slurms/logs/validate_pretrained/%x_%A_%a.out
#SBATCH --error=./slurms/logs/validate_pretrained/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --time=1:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=mgurnani@ucdavis.edu
#
# --account/--partition come from slurms/submit.sh at submit time (see
# slurms/cluster/*.sh) -- submit with `slurms/submit.sh validate_pretrained.sh ...`,
# not a bare `sbatch slurms/validate_pretrained.sh ...`.
#
# Wrapper around slurms/validate.sh's underlying `main.py --validate` command: batches it
# over many datasets in one job (array-over-manifest, SHARD_SIZE per task) so we can validate
# the paper's released checkpoints across all RBP-cell-line datasets without one SLURM job per
# dataset. slurms/validate.sh itself is left untouched — this script exists purely to point
# --model_save_path at an arbitrary checkpoint directory (e.g. the paper's release) and loop.
#
# Must land on a GPU generation this repo's pinned torch==2.0.1+cu117 build supports
# (sm_80/86 only): it hard-fails on sm_90+ with "no kernel image is available for execution
# on the device" (confirmed directly: job 19427724 on an h-node hit this, when misrouted to
# Anvil's H100 partition). bridge_assert_gpu_partition below enforces the safe partition list
# per cluster (slurms/cluster/*.sh: BRIDGE_GPU_SAFE_PARTITIONS) instead of a hardcoded name here.
#
# ---- Usage ----------------------------------------------------------------------
# Validate all 261 datasets against the paper's released checkpoints (MODEL_SAVE_PATH
# defaults to BRIDGE_PRETRAINED_MODEL_DIR from slurms/cluster/<cluster>.sh; override
# per-invocation if needed):
#   slurms/submit.sh --array=0-17%10 validate_pretrained.sh
#
# (261 datasets / SHARD_SIZE=15 -> 18 shards, 0-17; %10 caps concurrency.)
#
# Single dataset / explicit list (still routes through the same batching logic):
#   slurms/submit.sh validate_pretrained.sh AUH_HepG2
#   slurms/submit.sh validate_pretrained.sh AUH_HepG2,AARS_K562
#
# Each dataset's result line ("<DATA_FILE> auc: ... acc: ... auprc: ... mcc: ...", printed by
# main.py) lands in this task's .out log; grep across slurms/logs/validate_pretrained/*.out
# to collect them afterward. A dataset whose checkpoint is missing under MODEL_SAVE_PATH is
# logged and skipped (not a job failure), so the rest of the shard still runs.
# -----------------------------------------------------------------------------------

set -uo pipefail  # (not -e: one dataset's failure must not abort the rest of the shard)

# shellcheck source=/dev/null
source "${BRIDGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurms/cluster/common.sh"
bridge_assert_gpu_partition

MANIFEST=${MANIFEST:-ablation/datasets.txt}
DATA_PATH=${DATA_PATH:-./dataset}
TRANSFORMER_PATH=${TRANSFORMER_PATH:-./RBPformer}
# No file-level default: an empty BRIDGE_PRETRAINED_MODEL_DIR (e.g. not yet staged on this
# cluster, see slurms/cluster/hive.sh) fails loudly here instead of silently falling through
# to ./results/model.
MODEL_SAVE_PATH=${MODEL_SAVE_PATH:-${BRIDGE_PRETRAINED_MODEL_DIR:?set BRIDGE_PRETRAINED_MODEL_DIR in slurms/cluster/${BRIDGE_CLUSTER}.sh, or pass MODEL_SAVE_PATH=... explicitly}}
SEED=${SEED:-42}
SHARD_SIZE=${SHARD_SIZE:-15}
STAGGER_WINDOW=${STAGGER_WINDOW:-60}

# Pick dataset(s): explicit $1 wins (comma-separated allowed); otherwise map array index -> a
# SHARD_SIZE-line range of the manifest (same scheme as slurms/ablation.sh).
if [[ $# -ge 1 && -n "${1:-}" ]]; then
    DATASETS="$1"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    START=$(( SLURM_ARRAY_TASK_ID * SHARD_SIZE + 1 ))
    END=$(( START + SHARD_SIZE - 1 ))
    DATASETS=$(sed -n "${START},${END}p" "${MANIFEST}" | tr '\n' ',' | sed 's/,$//')
else
    echo "ERROR: pass dataset stem(s) as \$1, or submit with --array to read ${MANIFEST}" >&2
    exit 1
fi

if [[ -z "${DATASETS}" ]]; then
    echo "ERROR: empty shard (array index ${SLURM_ARRAY_TASK_ID:-?} of ${MANIFEST}, shard_size=${SHARD_SIZE})" >&2
    exit 1
fi
echo "[$(date)] validate_pretrained task ${SLURM_ARRAY_TASK_ID:-single} -> datasets [${DATASETS}] model_save_path=${MODEL_SAVE_PATH}"

bridge_load_base_modules

# Stagger conda-activate/python-launch across array tasks scheduled together, to avoid the
# shared-filesystem metadata-server race documented in slurms/ablation.sh
# (Fatal Python error: init_fs_encoding under concurrent launches).
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    JITTER_DS=$(( RANDOM % (STAGGER_WINDOW * 10) ))
    SLEEP_S=$(awk -v j="${JITTER_DS}" 'BEGIN{printf "%.1f", j/10.0}')
    echo "[$(date)] staggering: sleeping ${SLEEP_S}s before conda activate"
    sleep "${SLEEP_S}"
fi

# Retry conda-activate itself against the same fs-encoding race ablation.sh guards against.
# Stagger alone isn't sufficient — it just spreads out *when* the race can be hit, it doesn't
# recover from it. Verify with a cheap `python -c "import torch"` probe before committing to
# the (possibly 15-dataset) loop below, so a broken activation is caught and retried in
# seconds rather than burning the whole shard on every dataset failing identically (as
# happened in job 19427550, tasks 0 and 1: all 15 datasets in each shard failed instantly with
# the same init_fs_encoding signature, because conda activate itself never worked that run).
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
RETRY_BACKOFF=${RETRY_BACKOFF:-15}
STDERR_TMP="${SLURM_TMPDIR:-/tmp}/validate_pretrained_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}_attempt.err"

ATTEMPT=1
while true; do
    echo "[$(date)] attempt ${ATTEMPT}/${MAX_ATTEMPTS}: bridge_activate_env (module load ${BRIDGE_MODULE_CONDA}; conda activate ${BRIDGE_CONDA_ENV})"
    bridge_activate_env
    python -c "import torch; print(f'[env_check] ok torch={torch.__version__}')" 2> "${STDERR_TMP}"
    RC=$?
    if [[ ${RC} -eq 0 ]]; then
        echo "[$(date)] attempt ${ATTEMPT}: environment OK"
        break
    fi

    IS_FS_RACE=0
    if grep -qE 'init_fs_encoding|Fatal Python error|no codec search functions registered' "${STDERR_TMP}" 2>/dev/null; then
        IS_FS_RACE=1
    fi

    if [[ ${IS_FS_RACE} -eq 1 && ${ATTEMPT} -lt ${MAX_ATTEMPTS} ]]; then
        BACKOFF=$(( RETRY_BACKOFF * ATTEMPT + (RANDOM % 10) ))
        echo "[$(date)] attempt ${ATTEMPT} hit the fs-encoding race (rc=${RC}) — retrying in ${BACKOFF}s"
        conda deactivate 2>/dev/null || true
        sleep "${BACKOFF}"
        ATTEMPT=$(( ATTEMPT + 1 ))
        continue
    fi

    echo "[$(date)] attempt ${ATTEMPT} failed with rc=${RC}, not a retryable fs-race or attempts exhausted" >&2
    cat "${STDERR_TMP}" >&2
    rm -f "${STDERR_TMP}"
    exit "${RC}"
done
rm -f "${STDERR_TMP}"

FAILED=0
IFS=',' read -ra DATASET_ARR <<< "${DATASETS}"
for DATA_FILE in "${DATASET_ARR[@]}"; do
    CKPT="${MODEL_SAVE_PATH}/${DATA_FILE}.pth"
    if [[ ! -f "${CKPT}" ]]; then
        echo "[$(date)] SKIP ${DATA_FILE}: no checkpoint at ${CKPT}"
        continue
    fi

    echo "[$(date)] validating ${DATA_FILE}"
    # Same command as slurms/validate.sh, just looped and pointed at MODEL_SAVE_PATH.
    python main.py \
        --validate \
        --data_path "${DATA_PATH}" \
        --data_file "${DATA_FILE}" \
        --device_num 0 \
        --seed "${SEED}" \
        --Transformer_path "${TRANSFORMER_PATH}" \
        --model_save_path "${MODEL_SAVE_PATH}"
    RC=$?
    if [[ ${RC} -ne 0 ]]; then
        echo "[$(date)] FAILED ${DATA_FILE} (rc=${RC})"
        FAILED=$(( FAILED + 1 ))
    fi
done

if [[ ${FAILED} -gt 0 ]]; then
    echo "[$(date)] shard finished with ${FAILED} failure(s)"
    exit 1
fi
echo "[$(date)] shard finished, all datasets OK"
