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

from utils.BRIDGE import FEATURE_CHANNELS, PROTEIN_CHANNELS

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
    if mode == "feature":
        return [BASELINE] + FEATURE_CONFIGS + [PROTEIN_CONFIG]
    if mode == "module":
        return [BASELINE] + MODULE_CONFIGS
    if mode == "all":
        return [BASELINE] + FEATURE_CONFIGS + [PROTEIN_CONFIG] + MODULE_CONFIGS
    raise ValueError(f"Unknown mode {mode!r}; expected one of 'feature', 'module', 'all'")


def channels_removed(config):
    """Net channels removed from the 512-wide fusion (negative = net channels added).

    0 for module ablations and baseline; ``FEATURE_CHANNELS[drop_feature]`` for a dropped
    branch; ``-PROTEIN_CHANNELS`` for the additive ``protein`` config.
    """
    drop = config["kwargs"].get("drop_feature")
    added = PROTEIN_CHANNELS if config["kwargs"].get("add_protein") else 0
    return FEATURE_CHANNELS.get(drop, 0) - added


def fusion_channels(config):
    """ADPNet/GAP/attention-pool input width for a config."""
    return 512 - channels_removed(config)
