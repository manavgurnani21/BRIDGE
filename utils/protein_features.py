"""
Loader for cached whole-protein ESM-2 embeddings, used by the "protein" ablation control.

The embeddings themselves are not part of this repo: they live in
``/anvil/projects/x-cis250169/PreprocessedPaRPIData/esm/``, precomputed by the PaRPI_BIP
project (ESM-2 ``esm2_t33_650M_UR50D``, mean-pooled over residues -> one (1280,) float32
vector per RBP). File names match BRIDGE's own ``{RBP}_{CellLine}`` dataset stems, so a
cached embedding is looked up directly by ``data_file``.

Coverage is 260/262 BRIDGE dataset stems by exact name; the two exceptions are aliased below.
"""

import os

import numpy as np
import torch

ESM_CACHE_DIR = "/anvil/projects/x-cis250169/PreprocessedPaRPIData/esm"

# BRIDGE dataset stems with no exact-name match in the cache:
#   - AUH_HepG2_small is a dev-size subset of AUH_HepG2 (same protein, same embedding).
#   - PTBP1PTBP2_Hela's embedding was cached under the alias PTBP2_Hela.
_NAME_ALIASES = {
    "AUH_HepG2_small": "AUH_HepG2",
    "PTBP1PTBP2_Hela": "PTBP2_Hela",
}


def load_protein_embedding(data_file, cache_dir=ESM_CACHE_DIR):
    """Return the cached (1280,) whole-protein ESM-2 embedding for a BRIDGE dataset stem.

    Args:
        data_file: BRIDGE dataset stem, e.g. ``"AUH_HepG2"``.
        cache_dir: directory holding ``{stem}.npy`` files.

    Returns:
        torch.FloatTensor of shape (1280,).
    """
    stem = _NAME_ALIASES.get(data_file, data_file)
    path = os.path.join(cache_dir, f"{stem}.npy")
    vec = np.load(path)
    return torch.from_numpy(vec).float().view(-1)
