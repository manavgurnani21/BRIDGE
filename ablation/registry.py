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

from utils.BRIDGE import FEATURE_CHANNELS

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
        return [BASELINE] + FEATURE_CONFIGS
    if mode == "module":
        return [BASELINE] + MODULE_CONFIGS
    if mode == "all":
        return [BASELINE] + FEATURE_CONFIGS + MODULE_CONFIGS
    raise ValueError(f"Unknown mode {mode!r}; expected one of 'feature', 'module', 'all'")


def fusion_channels(config):
    """ADPNet/GAP input width for a config (512 minus any dropped feature's channels)."""
    drop = config["kwargs"].get("drop_feature")
    return 512 - FEATURE_CHANNELS.get(drop, 0)


def channels_removed(config):
    """Channels removed from the 512-wide fusion (0 for module ablations and baseline)."""
    drop = config["kwargs"].get("drop_feature")
    return FEATURE_CHANNELS.get(drop, 0)
