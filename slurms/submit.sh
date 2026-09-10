#!/bin/bash
# Cluster-portable drop-in replacement for `sbatch`.
#
# Detects which cluster we're on (slurms/cluster/common.sh), loads that
# cluster's account/partition (slurms/cluster/<cluster>.sh), and injects them
# as `sbatch` CLI overrides -- this is required rather than optional, because
# `#SBATCH --account=...`/`-p ...` lines are parsed by sbatch as literal text
# before bash ever runs, so shell variables can't appear inside them.
#
# Usage:
#   slurms/submit.sh [sbatch opts] <script.sh> [script args]
#
# Examples:
#   slurms/submit.sh train.sh AUH_HepG2
#   SHARD_SIZE=7 slurms/submit.sh --array=0-37%20 --time=14:00:00 ablation.sh
#
# Escape hatches (no file edits needed):
#   BRIDGE_ACCOUNT=publicgrp BRIDGE_PARTITION=low slurms/submit.sh train.sh AUH_HepG2
#   BRIDGE_CLUSTER=hive slurms/submit.sh train.sh AUH_HepG2   # force detection
#
# A job script opts into the CPU account/partition (instead of the GPU one)
# with a marker comment anywhere in the file: `# BRIDGE_JOB_CLASS: cpu`.

set -euo pipefail

BRIDGE_ROOT=${BRIDGE_ROOT:-$PWD}
# shellcheck source=/dev/null
source "${BRIDGE_ROOT}/slurms/cluster/common.sh"

SBATCH_OPTS=()
SCRIPT=""
SCRIPT_ARGS=()
for a in "$@"; do
    if [[ -z "$SCRIPT" ]]; then
        if [[ -f "$a" ]]; then
            SCRIPT="$a"
        elif [[ -f "${BRIDGE_ROOT}/slurms/$a" ]]; then
            SCRIPT="${BRIDGE_ROOT}/slurms/$a"
        else
            SBATCH_OPTS+=("$a")
        fi
    else
        SCRIPT_ARGS+=("$a")
    fi
done

if [[ -z "$SCRIPT" ]]; then
    echo "usage: slurms/submit.sh [sbatch opts] <script.sh> [script args]" >&2
    exit 2
fi

CLASS=$(awk -F': *' '/^# *BRIDGE_JOB_CLASS:/{print $2; exit}' "$SCRIPT")
CLASS=${CLASS:-gpu}

if [[ "$CLASS" == cpu ]]; then
    ACCT=$BRIDGE_CPU_ACCOUNT
    PART=$BRIDGE_CPU_PARTITION
else
    ACCT=$BRIDGE_GPU_ACCOUNT
    PART=$BRIDGE_GPU_PARTITION
fi
# Per-submission overrides.
ACCT=${BRIDGE_ACCOUNT:-$ACCT}
PART=${BRIDGE_PARTITION:-$PART}

# Make sure the job's #SBATCH --output=/--error= directories exist (they're
# gitignored, so a fresh clone/clean checkout won't have them yet).
LOG_PATH=$(grep -m1 -E '^#SBATCH +--output=' "$SCRIPT" | sed -E 's/^#SBATCH +--output=//')
if [[ -n "${LOG_PATH:-}" ]]; then
    mkdir -p "${BRIDGE_ROOT}/$(dirname "${LOG_PATH}")"
fi

export BRIDGE_CLUSTER BRIDGE_ROOT

echo "[submit] cluster=${BRIDGE_CLUSTER} class=${CLASS} account=${ACCT} partition=${PART} script=${SCRIPT}"
exec sbatch --account="${ACCT}" --partition="${PART}" "${SBATCH_OPTS[@]}" "$SCRIPT" "${SCRIPT_ARGS[@]}"
