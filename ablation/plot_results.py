"""
Graph the BRIDGE ablation results from the collated master table.

Reads ``results/ablation/ablation_results.csv`` (produced by ``collate_results.py``) and
renders three publication-style figure families, one per metric (AUC / ACC / AUPRC / MCC):

  1. delta-by-component   -- distribution of Δmetric per ablated component across datasets
                             (box + per-dataset points), ordered by mean effect. The headline
                             "which component matters most" view.
  2. heatmap              -- datasets x components, colored by Δmetric on a diverging scale
                             centered at 0 (mirrors the paper's heatmap_*.xlsx).
  3. feature-vs-module    -- the two ablation flavors (5 feature drops vs 2 module swaps) in
                             separate panels on a shared Δ axis, so they are not conflated.

Δmetric = ablated_metric - baseline_metric (the dataset's ``none`` row). NEGATIVE means the
model got WORSE when the component was removed -> that component contributes. Baseline rows
carry Δ=0 and are excluded from the plots.

Colors are colorblind-safe (dataviz skill reference palette): a blue<->red diverging scale
with a neutral gray midpoint for signed deltas, and a blue/orange categorical split for the
feature/module distinction.

Usage:
    python -m ablation.plot_results
    python -m ablation.plot_results --results results/ablation/ablation_results.csv \
        --out_dir figures/ablation --metrics auc mcc --fmt pdf
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless / SLURM-safe
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

# --- palette (dataviz reference instance) -------------------------------------------------
INK = "#0b0b0b"        # primary text
MUTED = "#898781"      # axis / labels
GRID = "#e1e0d9"       # hairline gridline
BASELINE = "#c3c2b7"   # zero / axis line
FEATURE_C = "#2a78d6"  # categorical slot 1 (blue)  -> feature ablations
MODULE_C = "#eb6834"   # categorical slot 8 (orange) -> module ablations
# Diverging scale for signed deltas: red (hurt / negative) <- gray -> blue (helped / positive).
DIVERGING = LinearSegmentedColormap.from_list(
    "delta_div", ["#c0392b", "#e34948", "#f0efec", "#2a78d6", "#184f95"]
)

# metric key -> (column suffix, display label). "prc" is AUPRC in the paper's naming.
METRIC_LABELS = {"auc": "AUC", "acc": "ACC", "prc": "AUPRC", "mcc": "MCC"}

FLAVOR_COLOR = {"feature": FEATURE_C, "module": MODULE_C}


def _style_axes(ax):
    """Recessive chrome: drop top/right spines, mute the rest, hairline y-grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def load_results(path):
    """Load the collated table; return only the ablated (non-baseline) rows for plotting."""
    df = pd.read_csv(path)
    ablated = df[df["ablation_type"] != "baseline"].copy()
    if ablated.empty:
        raise SystemExit(f"No non-baseline rows in {path} -- nothing to plot yet.")
    return df, ablated


def component_order(ablated, metric):
    """Fixed flavor grouping (feature before module); within a flavor, most-harmful first.

    'Most harmful' = most negative mean Δ, so the strongest contributors sort to the top.
    Returns a list of (name, flavor) pairs.
    """
    delta = f"delta_{metric}"
    means = ablated.groupby(["name", "ablation_type"])[delta].mean().reset_index()
    order = []
    for flavor in ("feature", "module"):
        sub = means[means["ablation_type"] == flavor].sort_values(delta)
        order.extend((r["name"], flavor) for _, r in sub.iterrows())
    return order


def fig_delta_by_component(ablated, metric, order, out_path, dpi):
    """Box + per-dataset points of Δmetric for each component, ordered by mean effect."""
    delta = f"delta_{metric}"
    names = [n for n, _ in order]
    flavors = [f for _, f in order]

    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(names)), 5))
    _style_axes(ax)
    ax.axhline(0, color=BASELINE, linewidth=1.4, zorder=1)  # baseline reference

    rng = np.random.default_rng(0)
    for i, (name, flavor) in enumerate(order):
        vals = ablated.loc[ablated["name"] == name, delta].dropna().values
        color = FLAVOR_COLOR[flavor]
        bp = ax.boxplot(
            vals, positions=[i], widths=0.55, patch_artist=True,
            showfliers=False, zorder=2,
            medianprops=dict(color=INK, linewidth=1.4),
            whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
            boxprops=dict(facecolor="none", edgecolor=color, linewidth=1.6),
        )
        for b in bp["boxes"]:
            b.set_alpha(0.9)
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, color=color,
                   alpha=0.55, edgecolor="none", zorder=3)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", color=INK)
    ax.set_ylabel(f"Δ {METRIC_LABELS[metric]}  (ablated − baseline)", color=INK)
    n_ds = ablated["dataset"].nunique()
    ax.set_title(f"Effect of removing each component on {METRIC_LABELS[metric]}"
                 f"   (n = {n_ds} dataset{'s' if n_ds != 1 else ''})",
                 color=INK, fontsize=12, loc="left")
    ax.legend(handles=[Patch(facecolor="none", edgecolor=FEATURE_C, label="feature"),
                       Patch(facecolor="none", edgecolor=MODULE_C, label="module")],
              frameon=False, loc="lower right", labelcolor=INK)
    fig.text(0.01, 0.01, "below 0 = component contributes (removing it hurt)",
             color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(ablated, metric, order, out_path, dpi):
    """datasets x components heatmap of Δmetric, diverging scale centered at 0."""
    delta = f"delta_{metric}"
    names = [n for n, _ in order]
    mat = ablated.pivot_table(index="dataset", columns="name", values=delta)
    mat = mat.reindex(columns=names)
    # order datasets by overall impact (mean Δ across components), worst at top
    mat = mat.loc[mat.mean(axis=1).sort_values().index]

    vmax = float(np.nanmax(np.abs(mat.values))) or 1e-6
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    # Distinguish "not yet run" (NaN) from a genuine near-zero delta: mask NaNs and
    # paint them a clearly-grey tone rather than the near-white diverging midpoint.
    cmap = DIVERGING.copy()
    cmap.set_bad("#cfcdc6")
    masked = np.ma.masked_invalid(mat.values)

    n_ds = len(mat)
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(names)), max(4, 0.22 * n_ds + 1.5)))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", color=INK)
    # Only show every dataset label when there are few; otherwise thin them out.
    if n_ds <= 40:
        ax.set_yticks(range(n_ds))
        ax.set_yticklabels(mat.index, color=INK, fontsize=8)
    else:
        step = int(np.ceil(n_ds / 40))
        ax.set_yticks(range(0, n_ds, step))
        ax.set_yticklabels(mat.index[::step], color=INK, fontsize=7)
    ax.tick_params(colors=MUTED)
    ax.set_title(f"Δ {METRIC_LABELS[metric]} by dataset and component",
                 color=INK, fontsize=12, loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Δ {METRIC_LABELS[metric]}  (red = hurt, blue = helped)", color=INK)
    cbar.ax.tick_params(colors=MUTED)
    if np.ma.is_masked(masked) and masked.mask.any():
        fig.text(0.01, 0.01, "grey = not yet run", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fig_feature_vs_module(ablated, metric, order, out_path, dpi):
    """Two panels (feature | module) of Δmetric distributions on a shared axis."""
    delta = f"delta_{metric}"
    feat = [n for n, f in order if f == "feature"]
    mod = [n for n, f in order if f == "module"]

    lo = float(ablated[delta].min())
    hi = float(ablated[delta].max())
    pad = 0.05 * (hi - lo or 1.0)
    ylim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 5), sharey=True,
        gridspec_kw={"width_ratios": [max(1, len(feat)), max(1, len(mod))]},
    )
    for ax, group, color, title in (
        (axes[0], feat, FEATURE_C, "Feature ablations"),
        (axes[1], mod, MODULE_C, "Module ablations"),
    ):
        _style_axes(ax)
        ax.axhline(0, color=BASELINE, linewidth=1.4, zorder=1)
        rng = np.random.default_rng(0)
        for i, name in enumerate(group):
            vals = ablated.loc[ablated["name"] == name, delta].dropna().values
            ax.boxplot(vals, positions=[i], widths=0.55, patch_artist=True,
                       showfliers=False, zorder=2,
                       medianprops=dict(color=INK, linewidth=1.4),
                       whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                       boxprops=dict(facecolor="none", edgecolor=color, linewidth=1.6))
            jitter = rng.uniform(-0.14, 0.14, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, color=color,
                       alpha=0.55, edgecolor="none", zorder=3)
        ax.set_xticks(range(len(group)))
        ax.set_xticklabels(group, rotation=30, ha="right", color=INK)
        ax.set_title(title, color=INK, fontsize=12, loc="left")
        ax.set_ylim(*ylim)
    axes[0].set_ylabel(f"Δ {METRIC_LABELS[metric]}  (ablated − baseline)", color=INK)
    n_ds = ablated["dataset"].nunique()
    fig.suptitle(f"Feature vs module ablation — {METRIC_LABELS[metric]}"
                 f"   (n = {n_ds} dataset{'s' if n_ds != 1 else ''})",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Graph BRIDGE ablation results")
    p.add_argument("--results", default="results/ablation/ablation_results.csv",
                   help="collated master CSV from collate_results.py")
    p.add_argument("--out_dir", default="figures/ablation", help="directory for figures")
    p.add_argument("--metrics", nargs="+", default=list(METRIC_LABELS),
                   choices=list(METRIC_LABELS), help="metrics to plot")
    p.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--order_by", default="auc", choices=list(METRIC_LABELS),
                   help="metric used to fix the component order shared across all figures")
    args = p.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(f"{results_path} not found -- run `python -m ablation.collate_results` first.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, ablated = load_results(results_path)
    n_ds = ablated["dataset"].nunique()
    print(f"[plot] {len(ablated)} ablated rows across {n_ds} datasets -> {out_dir}")

    # Fixed component order shared across every metric/figure, so panels are comparable
    # side by side instead of each metric re-sorting components by its own mean Δ.
    order = component_order(ablated, args.order_by)
    print(f"[order] fixed by delta_{args.order_by}: {[n for n, _ in order]}")

    for metric in args.metrics:
        if f"delta_{metric}" not in ablated.columns:
            print(f"[skip] no delta_{metric} column")
            continue
        for stem, fn in (
            ("delta_by_component", fig_delta_by_component),
            ("heatmap", fig_heatmap),
            ("feature_vs_module", fig_feature_vs_module),
        ):
            out_path = out_dir / f"{stem}_{metric}.{args.fmt}"
            fn(ablated, metric, order, out_path, args.dpi)
            print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
