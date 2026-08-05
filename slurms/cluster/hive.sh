# Cluster config: Hive (UC Davis HPC).
BRIDGE_GPU_ACCOUNT=genome-center-grp
BRIDGE_GPU_PARTITION=gpu-a100      # A100, sm_80. The only Hive partition guaranteed
                                   # sm_80 -- the default 'high' partition also has
                                   # sm_120 (Blackwell) and other GPU generations.
BRIDGE_CPU_ACCOUNT=genome-center-grp
BRIDGE_CPU_PARTITION=high          # no 'shared' partition on Hive; 'low' is preemptible.

BRIDGE_GPU_SAFE_PARTITIONS="gpu-a100 gpu-a100-40gb gpu-a6000"

BRIDGE_MODULE_MKL=intel-oneapi-mkl
BRIDGE_MODULE_CONDA=conda
BRIDGE_CONDA_ENV=BRIDGE

# Released checkpoints, used by slurms/validate_pretrained.sh.
# TODO: set once BRIDGE_Source_Files is staged on Hive.
BRIDGE_PRETRAINED_MODEL_DIR=

# Cached whole-protein ESM-2 embeddings, used by ablation/run_ablation.py's "protein" config
# (see utils/protein_features.py). Copied over from Anvil's PreprocessedPaRPIData/esm cache.
BRIDGE_ESM_CACHE_DIR=/quobyte/savirangrp/manav/esm

# Cached per-residue ESM-2 embeddings, used by ablation/run_ablation.py's "attn_protein" config
# (see utils/protein_features.py, ablation/build_esm_residue_cache.py). Built by this repo, not
# inherited from PaRPI_BIP.
BRIDGE_ESM_RESIDUE_CACHE_DIR=/quobyte/savirangrp/manav/esm_residue
