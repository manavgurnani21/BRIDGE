"""
Ablation configuration registry.

Each entry describes one model variant to train and evaluate. A config maps to
``BRIDGE(**kwargs)`` (see :mod:`utils.BRIDGE`), so adding a new ablation is a one-line
change here with no engine modification.

Config fields:
    name             : short unique id, used in filenames (e.g. ``sequence``, ``kan_to_mlp``).
    ablation_type    : ``baseline`` | ``feature`` | ``module`` (grouping column in the CSV).
    component_removed: human-readable component tag (feature key, ``kan``, ``adpnet``, ``none``).
    kwargs           : keyword args passed to ``BRIDGE(...)``.
"""

from utils.BRIDGE import FEATURE_CHANNELS, PROTEIN_ATTN_CHANNELS, PROTEIN_CHANNELS

# Full-model baseline (shared by every mode; deltas are computed against it).
BASELINE = {
    "name": "none",
    "ablation_type": "baseline",
    "component_removed": "none",
    "kwargs": {},
}

# Feature ablations: drop one input-feature branch (auto-shrinks the head input).
FEATURE_CONFIGS = [
    {
        "name": feat,
        "ablation_type": "feature",
        "component_removed": feat,
        "kwargs": {"drop_feature": feat},
    }
    for feat in ("gcn", "sequence", "structure", "motif", "biochem")
]

# Protein-awareness control (not a real ablation candidate): adds a whole-protein ESM-2
# embedding as an extra, additive input branch (512 -> 512 + PROTEIN_CHANNELS). Because BRIDGE
# trains one model per single RBP, this vector is identical across every sample in a run, so
# it can only be absorbed as a learned bias -- this config exists to test that prediction
# empirically (expected: ~no change vs baseline), not because it's expected to help. See
# ``utils.BRIDGE.BRIDGE``'s ``add_protein`` docstring for the full reasoning.
PROTEIN_CONFIG = {
    "name": "protein",
    "ablation_type": "feature",
    "component_removed": "protein",
    "kwargs": {"add_protein": True},
}

# Real cross-attention alternative to PROTEIN_CONFIG: RNA per-position fused features attend
# over per-residue protein ESM-2 embeddings (512 -> 512 + PROTEIN_ATTN_CHANNELS). Unlike
# add_protein, attention weights vary by RNA position and by protein content -- not absorbable
# as a learned bias. PROTEIN_ATTN_CHANNELS == PROTEIN_CHANNELS by design, so any AUC delta
# against PROTEIN_CONFIG isolates "real attention" as the only varying factor. See
# ``utils.BRIDGE.BRIDGE``'s ``attn_protein`` docstring for the full reasoning.
ATTN_PROTEIN_CONFIG = {
    "name": "attn_protein",
    "ablation_type": "feature",
    "component_removed": "protein_attn",
    "kwargs": {"attn_protein": True},
}

# Scoped variant of ATTN_PROTEIN_CONFIG: the cross-attention query comes from the RNA-BERT
# branch alone (x0, pre-fusion with structure/motif/biochem), instead of all RNA branches --
# i.e. one pretrained sequence-LM embedding (RNA-BERT) attending directly over another
# (ESM-2), with no engineered-feature context mixed in. Same PROTEIN_ATTN_CHANNELS budget as
# ATTN_PROTEIN_CONFIG, so a three-way AUC comparison (add_protein / attn_protein /
# attn_protein_seq) isolates: additive bias vs. real attention vs. real attention restricted
# to sequence-level complementarity. See ``utils.BRIDGE.BRIDGE``'s ``attn_protein_scope``
# docstring for the full reasoning.
ATTN_PROTEIN_SEQ_CONFIG = {
    "name": "attn_protein_seq",
    "ablation_type": "feature",
    "component_removed": "protein_attn_seq",
    "kwargs": {"attn_protein": True, "attn_protein_scope": "sequence"},
}
# Smoke test passed (job 20032145: guards + non-degeneracy + a few real training epochs, same
# bar ATTN_PROTEIN_CONFIG was held to) -- wired into get_configs() below.

# Negative control for the whole protein-fusion family. Architecturally IDENTICAL to
# ATTN_PROTEIN_CONFIG (same attn_protein=True kwargs -> same modules, same parameter count,
# same PROTEIN_ATTN_CHANNELS fusion width); the only difference is that ``run_ablation`` loads
# a *different* RBP's per-residue embedding as the attention keys/values, via a fixed
# derangement (see utils.protein_features.load_permuted_protein_residue_embedding).
#
# What it tests: the three protein-fusion arms (protein / attn_protein / attn_protein_seq) all
# landed inside noise (mean dAUC -0.0007 / +0.0005 / +0.0008 over 258 datasets). Two
# explanations are consistent with that: (a) the protein content genuinely carries no usable
# signal here -- expected, since BRIDGE trains one model per RBP, so the protein input is
# constant within a run and absorbable as a learned bias; or (b) it does carry signal but the
# effect is small. This config separates them. Because the protein is swapped for an unrelated
# one, any real RNA<->protein correspondence is destroyed while capacity and input statistics
# are held fixed.
#
# RESULT (full sweep, job 20270053, 258/258 datasets, collated 2026-08-10): destroying the
# correspondence costs a paired mean of only 0.0009 AUC (dAUC(attn_protein) - dAUC(perm), 95%
# CI [-0.0004, +0.0023], t p=0.17, Wilcoxon p=0.067); real protein beats the shuffled-protein
# control on just 55.4% of datasets, not distinguishable from chance. Direction is consistent
# with a small real effect -- so (a) is NOT confirmed as a bare "protein contributes exactly
# nothing" -- but the experiment is underpowered to resolve an effect this small (MDE @80% power
# at n=258 is 0.0019 AUC; would need ~1087 datasets to resolve 0.0009). What IS established: a
# TOST equivalence bound of +-0.005 AUC at p<0.00001. Conclusion to cite: protein identity
# contributes <0.005 AUC in this per-RBP training setup -- a bounded null, not a bare one.
#
# Deliberately NOT mutually exclusive with the other protein configs in the same way they are
# with each other -- it is a variant of attn_protein, not a fourth fusion mechanism.
ATTN_PROTEIN_PERM_CONFIG = {
    "name": "attn_protein_perm",
    "ablation_type": "feature",
    "component_removed": "protein_attn_perm",
    "kwargs": {"attn_protein": True},
    # Read by ablation/run_ablation.py, NOT passed to BRIDGE (it is not a model kwarg).
    "permute_protein": True,
}
# Smoke test passed (job 20269988: derangement soundness, identical trainable param count vs
# ATTN_PROTEIN_CONFIG at 21,778,145, non-degenerate attention over the substituted residues,
# and a few real training epochs) -- wired into get_configs() below. Full sweep also completed
# (job 20270053, 258/258 datasets) -- see the RESULT note above ATTN_PROTEIN_PERM_CONFIG.

# Lean BRIDGE: combines the four feature drops whose individual ablations landed at or above
# baseline (gcn +0.0001, motif +0.0007, biochem +0.0009, sequence +0.0031 mean dAUC over the
# 258-dataset sweep -- see docs/ablation_pipeline.md), keeping only the one branch that showed
# a large, near-universal cost when dropped (structure, -0.0442, worse on 97% of datasets),
# plus kan_to_mlp (also +0.0031 alone, i.e. free). ADPNet is left untouched, since both of its
# replacements (adpnet_to_gap/adpnet_to_attnpool) were the *other* large, consistent cost
# (~-0.026 each) -- the one part of the architecture this config deliberately does not touch.
#
# This is NOT validated as a joint effect -- each of the four drops was only ever measured
# against the full 5-branch baseline, one at a time. Dropping all four simultaneously is a
# different regime (interaction effects are possible even though each drop looks like noise in
# isolation), so this config exists to test the combination directly rather than assume the
# deltas sum. Fusion width: 512 - (32 + 256 + 64 + 32) = 128 (structure only). Deliberately
# excludes every protein kwarg (add_protein/attn_protein) -- this is a baseline-vs-lean
# comparison, not a protein-awareness experiment.
LEAN_CONFIG = {
    "name": "lean",
    "ablation_type": "feature",
    "component_removed": "gcn+sequence+motif+biochem",
    "kwargs": {
        # A list, not a set: run_ablation.py's best.json writer does a plain json.dump of
        # config["kwargs"], and a set isn't JSON-serializable. BRIDGE.__init__ normalizes any
        # iterable (list/set/tuple) to a frozenset internally, so this is behaviorally
        # identical -- just JSON-safe.
        "drop_feature": ["gcn", "sequence", "motif", "biochem"],
        "kan_to_mlp": True,
    },
}

# Module ablations: swap an internal mechanism, keeping inputs + 512 fusion fixed.
MODULE_CONFIGS = [
    {
        "name": "kan_to_mlp",
        "ablation_type": "module",
        "component_removed": "kan",
        "kwargs": {"kan_to_mlp": True},
    },
    {
        "name": "adpnet_to_gap",
        "ablation_type": "module",
        "component_removed": "adpnet",
        "kwargs": {"adpnet_to_gap": True},
    },
    {
        "name": "adpnet_to_attnpool",
        "ablation_type": "module",
        "component_removed": "adpnet",
        "kwargs": {"adpnet_to_attnpool": True},
    },
]


def get_configs(mode="all"):
    """Return the ordered config list for a mode. ``none`` is always first so the baseline
    is trained before the ablations (useful for early delta sanity checks)."""
    protein_configs = [
        PROTEIN_CONFIG,
        ATTN_PROTEIN_CONFIG,
        ATTN_PROTEIN_SEQ_CONFIG,
        ATTN_PROTEIN_PERM_CONFIG,
    ]
    if mode == "feature":
        return [BASELINE] + FEATURE_CONFIGS + protein_configs
    if mode == "module":
        return [BASELINE] + MODULE_CONFIGS
    if mode == "lean":
        # Deliberately just baseline + lean -- not folded into "feature"/"all", which also
        # pull in the protein configs; this mode exists so a baseline-vs-lean sweep never
        # trains protein branches it doesn't need.
        return [BASELINE, LEAN_CONFIG]
    if mode == "all":
        return [BASELINE] + FEATURE_CONFIGS + protein_configs + MODULE_CONFIGS
    raise ValueError(
        f"Unknown mode {mode!r}; expected one of 'feature', 'module', 'lean', 'all'"
    )


def channels_removed(config):
    """Net channels removed from the 512-wide fusion (negative = net channels added).

    0 for module ablations and baseline; sum of ``FEATURE_CHANNELS[f]`` over the dropped
    branch(es) for a feature-drop config (``drop_feature`` may be a single feature name or an
    iterable of several, e.g. ``LEAN_CONFIG``); ``-PROTEIN_CHANNELS``/``-PROTEIN_ATTN_CHANNELS``
    for the additive ``protein``/``attn_protein`` configs.
    """
    drop = config["kwargs"].get("drop_feature")
    if drop is None:
        dropped = []
    elif isinstance(drop, str):
        dropped = [drop]
    else:
        dropped = list(drop)
    added = 0
    if config["kwargs"].get("add_protein"):
        added += PROTEIN_CHANNELS
    if config["kwargs"].get("attn_protein"):
        added += PROTEIN_ATTN_CHANNELS
    return sum(FEATURE_CHANNELS[f] for f in dropped) - added


def fusion_channels(config):
    """ADPNet/GAP/attention-pool input width for a config."""
    return 512 - channels_removed(config)
