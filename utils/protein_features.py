"""
Loaders for cached ESM-2 protein embeddings, used by the "protein" and "attn_protein" ablation
controls.

Two caches exist:

- Whole-protein (mean-pooled): ``/anvil/projects/x-cis250169/PreprocessedPaRPIData/esm/``,
  precomputed by the PaRPI_BIP project (ESM-2 ``esm2_t33_650M_UR50D``, mean-pooled over
  residues -> one (1280,) float32 vector per RBP). File names match BRIDGE's own
  ``{RBP}_{CellLine}`` dataset stems, so a cached embedding is looked up directly by
  ``data_file``. Coverage is 260/262 BRIDGE dataset stems by exact name; the two exceptions
  are aliased below.
- Per-residue: built by ``ablation/build_esm_residue_cache.py`` from FASTA sequences (not
  mean-pooled -> one (P, 1280) float32 matrix per RBP, P = protein length). Keyed by RBP name
  rather than dataset stem (161 of BRIDGE's 261 dataset stems share an RBP across cell lines),
  so lookup here goes dataset stem -> RBP name via the same ``_NAME_ALIASES`` plus a
  cell-line-suffix strip.
"""

import os
import re

import numpy as np
import torch

ESM_CACHE_DIR = "/anvil/projects/x-cis250169/PreprocessedPaRPIData/esm"
ESM_RESIDUE_CACHE_DIR = "/anvil/projects/x-cis250169/BRIDGE_esm_residue_cache"

# BRIDGE dataset stems with no exact-name match in the cache:
#   - AUH_HepG2_small is a dev-size subset of AUH_HepG2 (same protein, same embedding).
#   - PTBP1PTBP2_Hela's embedding was cached under the alias PTBP2_Hela.
_NAME_ALIASES = {
    "AUH_HepG2_small": "AUH_HepG2",
    "PTBP1PTBP2_Hela": "PTBP2_Hela",
}

# Strips a trailing "_{CellLine}" segment off a canonicalized dataset stem to get the RBP name
# the per-residue cache and FASTA directory are keyed by, e.g. "AUH_HepG2" -> "AUH".
_RBP_SUFFIX_RE = re.compile(r"_[A-Za-z0-9]+$")


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


def load_protein_residue_embedding(data_file, cache_dir=ESM_RESIDUE_CACHE_DIR):
    """Return the cached (P, 1280) per-residue ESM-2 embedding for a BRIDGE dataset stem.

    Args:
        data_file: BRIDGE dataset stem, e.g. ``"AUH_HepG2"``.
        cache_dir: directory holding ``{RBP}.npy`` files (see
            ``ablation/build_esm_residue_cache.py``).

    Returns:
        torch.FloatTensor of shape (P, 1280), P = protein length.
    """
    stem = _NAME_ALIASES.get(data_file, data_file)
    rbp = _RBP_SUFFIX_RE.sub("", stem)
    path = os.path.join(cache_dir, f"{rbp}.npy")
    mat = np.load(path)
    return torch.from_numpy(mat).float()
