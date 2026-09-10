# Cluster config: Anvil (Purdue ACCESS allocation cis250169).
BRIDGE_GPU_ACCOUNT=cis250169-gpu
BRIDGE_GPU_PARTITION=gpu           # A100, sm_80. NOT 'ai' (H100, sm_90 -- incompatible,
                                   # see bridge_assert_gpu_partition in common.sh).
BRIDGE_CPU_ACCOUNT=cis250169
BRIDGE_CPU_PARTITION=shared

BRIDGE_GPU_SAFE_PARTITIONS="gpu"

BRIDGE_MODULE_MKL=intel-mkl
BRIDGE_MODULE_CONDA=conda
BRIDGE_CONDA_ENV=BRIDGE

# Released checkpoints, used by slurms/validate_pretrained.sh.
BRIDGE_PRETRAINED_MODEL_DIR=/anvil/scratch/x-mgurnani/BRIDGE_Source_Files/model/model

# Cached whole-protein ESM-2 embeddings, used by ablation/run_ablation.py's "protein" config
# (see utils/protein_features.py). PaRPI_BIP's original cache location.
BRIDGE_ESM_CACHE_DIR=/anvil/projects/x-cis250169/PreprocessedPaRPIData/esm

# Cached per-residue ESM-2 embeddings, used by ablation/run_ablation.py's "attn_protein" config
# (see utils/protein_features.py, ablation/build_esm_residue_cache.py). Built by this repo, not
# inherited from PaRPI_BIP.
BRIDGE_ESM_RESIDUE_CACHE_DIR=/anvil/projects/x-cis250169/BRIDGE_esm_residue_cache
