# Sourced (not executed) by slurms/submit.sh and every slurms/*.sh job script.
#
# Resolves which cluster we're on and loads its config from
# slurms/cluster/<cluster>.sh, then exposes helpers so job scripts don't
# hardcode account/partition/module names that differ between clusters.
#
# Detection precedence: explicit override > SLURM's own idea (works inside a
# running job) > `scontrol show config` (works on a login node) > $CLUSTER_NAME.
bridge_detect_cluster() {
    if [[ -n "${BRIDGE_CLUSTER:-}" ]]; then echo "${BRIDGE_CLUSTER}"; return; fi
    if [[ -n "${SLURM_CLUSTER_NAME:-}" ]]; then echo "${SLURM_CLUSTER_NAME}"; return; fi
    local c
    c=$(scontrol show config 2>/dev/null | awk -F'=' '/^ClusterName/{gsub(/[[:space:]]/,"",$2); print $2}')
    if [[ -n "$c" ]]; then echo "$c"; return; fi
    echo "${CLUSTER_NAME:-unknown}"
}

# SLURM copies the batch script to a spool path on the compute node, so
# $0/$BASH_SOURCE can't be used to locate the repo. SLURM_SUBMIT_DIR is the
# directory `sbatch` was invoked from; fall back to $PWD for an interactive
# (srun) run from the repo root.
BRIDGE_ROOT=${BRIDGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
BRIDGE_CLUSTER=$(bridge_detect_cluster)
BRIDGE_CLUSTER_CONF="${BRIDGE_ROOT}/slurms/cluster/${BRIDGE_CLUSTER}.sh"

if [[ ! -f "${BRIDGE_CLUSTER_CONF}" ]]; then
    echo "ERROR: no cluster config for '${BRIDGE_CLUSTER}' (looked for ${BRIDGE_CLUSTER_CONF})." >&2
    echo "       Set BRIDGE_CLUSTER=<name> or add slurms/cluster/<name>.sh" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${BRIDGE_CLUSTER_CONF}"
export BRIDGE_ROOT BRIDGE_CLUSTER

# MKL is loaded once up front; conda load + activate is what the fs-race retry
# loops in ablation.sh/validate_pretrained.sh re-attempt, so it's a separate
# function.
bridge_load_base_modules() {
    module --force purge
    module load "${BRIDGE_MODULE_MKL}"
}

bridge_activate_env() {
    module load "${BRIDGE_MODULE_CONDA}"
    conda activate "${BRIDGE_CONDA_ENV}"
}

# Fail immediately instead of dying partway through a run on an incompatible
# GPU arch (this repo's pinned torch==2.0.1+cu117 build has no kernels past
# sm_86, so an sm_90/sm_120 node fails with "no kernel image is available").
bridge_assert_gpu_partition() {
    local p="${SLURM_JOB_PARTITION:-}"
    [[ -z "$p" ]] && return 0
    case " ${BRIDGE_GPU_SAFE_PARTITIONS} " in
        *" $p "*) return 0 ;;
        *)
            echo "ERROR: partition '$p' is not a known sm_80/86-safe partition on ${BRIDGE_CLUSTER}." >&2
            echo "       Safe partitions: ${BRIDGE_GPU_SAFE_PARTITIONS}" >&2
            exit 1
            ;;
    esac
}
