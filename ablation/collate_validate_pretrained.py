"""Collate slurms/validate_pretrained.sh output logs into a single results CSV.

Each dataset's result line ("<DATA_FILE> auc: ... acc: ... auprc: ... mcc: ...") is printed
by main.py --validate into its task's .out log (see slurms/validate_pretrained.sh). This
script greps all logs under a directory for that line format and writes one row per dataset
to a CSV, keeping the newest result if a dataset appears more than once (e.g. a requeued task
re-validating the same shard).

Usage:
    python -m ablation.collate_validate_pretrained \
        --log_dir slurms/logs/validate_pretrained \
        --out results/validate_pretrained/results.csv
"""

import argparse
import csv
import re
from pathlib import Path

RESULT_RE = re.compile(
    r"^(?P<dataset>\S+) auc: (?P<auc>[\d.]+) acc: (?P<acc>[\d.]+) "
    r"auprc: (?P<auprc>[\d.]+) mcc: (?P<mcc>[\d.]+)"
)


def parse_logs(log_dir: Path) -> dict:
    rows = {}
    for out_file in sorted(log_dir.glob("*.out"), key=lambda p: p.stat().st_mtime):
        for line in out_file.read_text(errors="replace").splitlines():
            m = RESULT_RE.match(line.strip())
            if not m:
                continue
            d = m.groupdict()
            rows[d["dataset"]] = {
                "dataset": d["dataset"],
                "auc": float(d["auc"]),
                "acc": float(d["acc"]),
                "auprc": float(d["auprc"]),
                "mcc": float(d["mcc"]),
                "source_log": out_file.name,
            }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="slurms/logs/validate_pretrained")
    ap.add_argument("--out", default="results/validate_pretrained/results.csv")
    ap.add_argument("--manifest", default="ablation/datasets.txt",
                     help="Full dataset list, used only to print a coverage report")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    rows = parse_logs(log_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "auc", "acc", "auprc", "mcc", "source_log"])
        writer.writeheader()
        for dataset in sorted(rows):
            writer.writerow(rows[dataset])

    print(f"Wrote {len(rows)} dataset result(s) to {out_path}")

    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        all_datasets = [l.strip() for l in manifest_path.read_text().splitlines() if l.strip()]
        missing = [d for d in all_datasets if d not in rows]
        print(f"Coverage: {len(rows)}/{len(all_datasets)} datasets in manifest have a result.")
        if missing:
            print(f"Missing ({len(missing)}): {', '.join(missing[:20])}"
                  + (" ..." if len(missing) > 20 else ""))


if __name__ == "__main__":
    main()
