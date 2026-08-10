"""
Loaders for cached ESM-2 protein embeddings, used by the "protein", "attn_protein", and
"attn_protein_perm" ablation controls.

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


# Seed for the RBP derangement used by the ``attn_protein_perm`` control. Fixed so the
# mapping is identical across array tasks, shards, and reruns -- every process must derive
# the same permutation independently, since shards never see the full dataset list.
PROTEIN_PERMUTATION_SEED = 20260809


def build_rbp_derangement(rbp_names, seed=PROTEIN_PERMUTATION_SEED):
    """Map each RBP name to a *different* RBP's name, deterministically.

    Uses Sattolo's algorithm, which produces a uniformly random single cyclic permutation --
    guaranteeing zero fixed points (no RBP is ever mapped to itself). A plain shuffle would
    leave ~1/e of entries as fixed points on average, which would silently leak real
    protein-RNA correspondence into the control and weaken the experiment.

    Args:
        rbp_names: iterable of RBP names. Sorted internally, so callers may pass any order
            (e.g. ``os.listdir`` output) and still get the same mapping.
        seed: RNG seed; defaults to ``PROTEIN_PERMUTATION_SEED``.

    Returns:
        dict mapping each RBP name to a different RBP name.
    """
    names = sorted(rbp_names)
    if len(names) < 2:
        raise ValueError(f"need >=2 RBPs to build a derangement, got {len(names)}")
    shuffled = list(names)
    rng = np.random.default_rng(seed)
    # Sattolo: for i from n-1 down to 1, swap with a strictly-lower index.
    for i in range(len(shuffled) - 1, 0, -1):
        j = int(rng.integers(0, i))  # 0 <= j < i, strictly less than i
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    mapping = dict(zip(names, shuffled))
    fixed = [k for k, v in mapping.items() if k == v]
    if fixed:  # unreachable via Sattolo; guards against a future algorithm swap
        raise AssertionError(f"derangement has fixed points: {fixed}")
    return mapping


def load_permuted_protein_residue_embedding(
    data_file, cache_dir=ESM_RESIDUE_CACHE_DIR, seed=PROTEIN_PERMUTATION_SEED
):
    """Return a *different* RBP's (P, 1280) per-residue ESM-2 embedding -- the negative
    control for the ``attn_protein`` family.

    The permutation universe is every ``{RBP}.npy`` in ``cache_dir`` (not just the datasets in
    the current shard), so the mapping is stable no matter how the sweep is sharded.

    Why permute rather than randomize: substituting Gaussian noise would move the keys/values
    off the ESM-2 embedding manifold, so any AUC change would confound "protein identity
    mattered" with "the inputs became out-of-distribution". Swapping in another real protein's
    embedding preserves the marginal distribution exactly and destroys only the RNA<->protein
    correspondence, which is the single variable under test.

    Args:
        data_file: BRIDGE dataset stem, e.g. ``"AUH_HepG2"``.
        cache_dir: directory holding ``{RBP}.npy`` files.
        seed: RNG seed for the derangement.

    Returns:
        torch.FloatTensor of shape (P', 1280), P' = the *substituted* protein's length
        (generally != this dataset's own protein length; the model handles variable P).
    """
    stem = _NAME_ALIASES.get(data_file, data_file)
    rbp = _RBP_SUFFIX_RE.sub("", stem)
    available = sorted(
        os.path.splitext(f)[0] for f in os.listdir(cache_dir) if f.endswith(".npy")
    )
    if rbp not in available:
        raise FileNotFoundError(
            f"RBP {rbp!r} (from dataset {data_file!r}) has no {rbp}.npy in {cache_dir}"
        )
    substitute = build_rbp_derangement(available, seed=seed)[rbp]
    mat = np.load(os.path.join(cache_dir, f"{substitute}.npy"))
    return torch.from_numpy(mat).float()
