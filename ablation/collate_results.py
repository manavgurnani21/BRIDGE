"""
Collate per-run ablation row-files into master tables + compute deltas + coverage report.

Run this after the SLURM array finishes. It:
  - concatenates every ``results/ablation/rows/*.csv`` (one row per dataset x config),
  - fills ``baseline_test_auc`` and ``delta_*`` for each row against its dataset's ``none``
    (baseline) row  (delta = ablated_metric - baseline_metric; negative = performance dip),
  - writes ``ablation_results.csv`` (wide), ``ablation_results_long.csv`` (tidy, for plotting),
    and ``ablation_results.parquet`` (fast filtering),
  - prints a coverage report of missing / failed (dataset, config) cells.

Example:
    python -m ablation.collate_results --out_dir ./results/ablation
    python -m ablation.collate_results --out_dir ./results/ablation --manifest ablation/datasets.txt
"""

import argparse
import glob
import os
from pathlib import Path

import pandas as pd

from ablation.registry import get_configs

METRICS = ["auc", "acc", "prc", "mcc"]  # metrics that get a delta column

# Fixture datasets (e.g. AUH_HepG2_small) exist only to smoke-test the pipeline and must
# never appear in collated results. Mirrors how ablation/datasets.txt is built (grep -v '_small$').
FIXTURE_SUFFIXES = ("_small",)


def is_fixture(dataset):
    return str(dataset).endswith(FIXTURE_SUFFIXES)


def load_rows(rows_dir):
    files = sorted(glob.glob(str(rows_dir / "*.csv")))
    if not files:
        raise SystemExit(f"No row-files found in {rows_dir}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    fixtures = df["dataset"].map(is_fixture)
    if fixtures.any():
        dropped = sorted(df.loc[fixtures, "dataset"].unique())
        print(f"[skip fixtures] excluding {int(fixtures.sum())} rows from {dropped}")
        df = df[~fixtures].reset_index(drop=True)
    return df


def fill_deltas(df):
    """Fill baseline_test_auc + delta_* per (dataset, seed) against the baseline row."""
    df = df.copy()
    for (dataset, seed), grp in df.groupby(["dataset", "seed"]):
        base = grp[grp["ablation_type"] == "baseline"]
        if base.empty:
            # No baseline yet for this dataset -> leave deltas blank.
            continue
        b = base.iloc[0]
        idx = grp.index
        df.loc[idx, "baseline_test_auc"] = b["test_auc"]
        for m in METRICS:
            df.loc[idx, f"delta_{m}"] = df.loc[idx, f"test_{m}"] - b[f"test_{m}"]
    return df


def to_long(df):
    """Tidy/long form: one row per (dataset, seed, config, metric)."""
    records = []
    for _, r in df.iterrows():
        for m in METRICS:
            records.append({
                "dataset": r["dataset"],
                "seed": r["seed"],
                "ablation_type": r["ablation_type"],
                "component_removed": r["component_removed"],
                "name": r["name"],
                "metric": m,
                "value": r[f"test_{m}"],
                "delta": r.get(f"delta_{m}", ""),
            })
    return pd.DataFrame.from_records(records)


def coverage_report(df, manifest):
    expected_names = [c["name"] for c in get_configs("all")]
    if manifest and os.path.exists(manifest):
        with open(manifest) as f:
            datasets = [ln.strip() for ln in f if ln.strip()]
    else:
        datasets = sorted(df["dataset"].unique())
    datasets = [d for d in datasets if not is_fixture(d)]  # fixtures excluded from results

    print("\n=== Coverage report ===")
    print(f"datasets: {len(datasets)}  configs/dataset: {len(expected_names)}  "
          f"expected runs: {len(datasets) * len(expected_names)}")
    print(f"rows found: {len(df)}")
    missing_total = 0
    for ds in datasets:
        have = set(df[df["dataset"] == ds]["name"])
        missing = [n for n in expected_names if n not in have]
        if missing:
            missing_total += len(missing)
            print(f"  [{ds}] missing {len(missing)}: {missing}")
    if missing_total == 0:
        print("  all expected cells present.")
    else:
        print(f"  TOTAL missing cells: {missing_total}")


def main():
    parser = argparse.ArgumentParser(description="Collate BRIDGE ablation results")
    parser.add_argument("--out_dir", default="./results/ablation", type=str)
    parser.add_argument("--manifest", default="", type=str,
                        help="Optional datasets.txt to report coverage against the full set")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows_dir = out_dir / "rows"

    df = load_rows(rows_dir)
    df = fill_deltas(df)
    df = df.sort_values(["dataset", "ablation_type", "name"]).reset_index(drop=True)

    wide_csv = out_dir / "ablation_results.csv"
    long_csv = out_dir / "ablation_results_long.csv"
    df.to_csv(wide_csv, index=False)
    to_long(df).to_csv(long_csv, index=False)
    print(f"[write] {wide_csv}  ({len(df)} rows)")
    print(f"[write] {long_csv}")

    try:
        parquet = out_dir / "ablation_results.parquet"
        df.to_parquet(parquet, index=False)
        print(f"[write] {parquet}")
    except Exception as e:  # pyarrow/fastparquet may be absent — CSV is the source of truth
        print(f"[skip parquet] {e}")

    coverage_report(df, args.manifest)


if __name__ == "__main__":
    main()
