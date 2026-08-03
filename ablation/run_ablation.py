"""
Ablation driver — one SLURM array task trains all configs for one or more datasets (a "shard").

For each dataset this:
  1. builds the train/val/test loaders once (expensive feature step done a single time),
  2. iterates the ablation configs from :mod:`ablation.registry`,
  3. skips any config whose per-run row-file already exists (idempotent / requeue-safe),
  4. for each remaining config: retrains a fresh ``BRIDGE(**kwargs)`` with ``fit_bridge``,
     evaluates the best checkpoint on the sealed test split, and writes a one-row CSV plus
     ``best.json`` and ``model.pth``.

Each run writes its *own* row-file, so concurrent array tasks never touch the same file
(no shared-CSV races). Deltas-vs-baseline are intentionally NOT computed here — they need the
whole dataset's baseline row and are filled by ``ablation/collate_results.py``.

Passing multiple datasets (``--data_files``) runs them sequentially in one process, so a shard
of datasets shares a single conda-activate / interpreter-startup event instead of one per
dataset — this is what lets ``slurms/ablation.sh`` batch several datasets per array task to cut
down on concurrent process launches. A failure on one dataset in the shard is logged and
skipped so the rest of the shard still completes.

Examples:
    python -m ablation.run_ablation --data_file AUH_HepG2 --mode all --seed 42
    python -m ablation.run_ablation --data_files AUH_HepG2,AARS_K562 --mode all --seed 42
"""

import argparse
import csv
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from utils.BRIDGE import BRIDGE
from utils.data_pipeline import build_split_loaders, fix_seed
from utils.protein_features import ESM_CACHE_DIR, load_protein_embedding
from utils.train_loop import fit_bridge, validate
from ablation.registry import get_configs, fusion_channels, channels_removed

# Column order for per-run row-files and the collated master CSV.
ROW_HEADER = [
    "dataset", "seed", "ablation_type", "component_removed", "name",
    "channels_removed", "adpnet_input_channels", "best_epoch",
    "test_auc", "test_acc", "test_prc", "test_mcc", "test_f1", "test_loss",
    "tp", "tn", "fp", "fn",
    "baseline_test_auc", "delta_auc", "delta_acc", "delta_prc", "delta_mcc",
    "checkpoint_path", "timestamp",
]


def row_file(rows_dir, dataset, config):
    return rows_dir / f"{dataset}__{config['ablation_type']}__{config['name']}.csv"


def write_row(path, row):
    """Write a single-row CSV (with header) atomically via a temp file + rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_HEADER)
        w.writeheader()
        w.writerow(row)
    os.replace(tmp, path)


def run_config(config, loaders, device, args, dataset, rows_dir, runs_dir):
    """Train + evaluate one ablation config; write its row-file, best.json, model.pth."""
    train_loader, val_loader, test_loader = loaders
    name = config["name"]

    run_dir = runs_dir / dataset / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "model.pth"
    log_path = run_dir / "run.log"
    log_fp = open(log_path, "a", encoding="utf-8")

    def log_fn(msg):
        log_fp.write(msg + "\n")
        log_fp.flush()

    # Fresh, identically-seeded init + training for every config (fair comparison).
    fix_seed(args.seed)
    model_kwargs = dict(config["kwargs"])
    if model_kwargs.get("add_protein"):
        # Looked up per-dataset at run time (not stored in the static registry kwargs) since
        # each dataset has its own cached whole-protein embedding.
        model_kwargs["protein_vector"] = load_protein_embedding(dataset, cache_dir=args.esm_cache_dir)
    model = BRIDGE(**model_kwargs).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-6
    )

    tag = f"[{dataset}/{name}]"
    log_fn(f"{tag} start: kwargs={config['kwargs']}")
    best = fit_bridge(
        model, device, train_loader, val_loader, criterion, optimizer,
        max_epochs=args.max_epochs,
        early_stopping=args.early_stopping,
        ckpt_path=ckpt_path,
        log_fn=log_fn,
        tag=tag,
    )

    # Guard: if val AUC never improved above 0, no checkpoint was saved — persist current model.
    if not ckpt_path.exists():
        torch.save(model.state_dict(), ckpt_path)

    # Evaluate the best checkpoint on the sealed test split.
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    met, _, _ = validate(model, device, test_loader, criterion)

    row = {
        "dataset": dataset,
        "seed": args.seed,
        "ablation_type": config["ablation_type"],
        "component_removed": config["component_removed"],
        "name": name,
        "channels_removed": channels_removed(config),
        "adpnet_input_channels": fusion_channels(config),
        "best_epoch": best["best_epoch"],
        "test_auc": met.auc,
        "test_acc": met.acc,
        "test_prc": met.prc,
        "test_mcc": met.mcc,
        "test_f1": met.f1,
        "test_loss": float(met.other[0]) if len(met.avg) > 9 else "",
        "tp": met.tp, "tn": met.tn, "fp": met.fp, "fn": met.fn,
        # baseline_test_auc + delta_* are filled at collate time.
        "baseline_test_auc": "", "delta_auc": "", "delta_acc": "",
        "delta_prc": "", "delta_mcc": "",
        "checkpoint_path": str(ckpt_path.resolve()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    best_summary = {
        "dataset": dataset, "name": name,
        "ablation_type": config["ablation_type"],
        "component_removed": config["component_removed"],
        "kwargs": config["kwargs"],
        "seed": args.seed,
        **best,
        "test_auc": met.auc, "test_acc": met.acc,
        "test_prc": met.prc, "test_mcc": met.mcc,
        "checkpoint": str(ckpt_path.resolve()),
    }
    with open(run_dir / "best.json", "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    write_row(row_file(rows_dir, dataset, config), row)
    log_fn(f"{tag} done: test_auc={met.auc:.4f} acc={met.acc:.4f} prc={met.prc:.4f} mcc={met.mcc:.4f}")
    log_fp.close()
    return met


def main():
    parser = argparse.ArgumentParser(description="BRIDGE ablation driver (one or more datasets)")
    parser.add_argument("--data_file", type=str, default=None,
                         help="Single dataset stem (e.g. AUH_HepG2). Back-compat single-dataset mode.")
    parser.add_argument("--data_files", type=str, default=None,
                         help="Comma-separated dataset stems for batched (shard) mode, "
                              "e.g. AARS_K562,AATF_HepG2,ABCF1_K562")
    parser.add_argument("--data_path", default="./dataset", type=str)
    parser.add_argument("--Transformer_path", default="./RBPformer", type=str)
    parser.add_argument("--esm_cache_dir", default=ESM_CACHE_DIR, type=str,
                         help="Dir of cached {dataset}.npy whole-protein ESM-2 embeddings "
                              "for the 'protein' config. Defaults to the Anvil path; pass "
                              "slurms/cluster/<cluster>.sh's BRIDGE_ESM_CACHE_DIR elsewhere.")
    parser.add_argument("--mode", default="all", choices=["feature", "module", "all"])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--early_stopping", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch_size", default=2048, type=int, help="RBPformer embedding batch size")
    parser.add_argument("--out_dir", default="./results/ablation", type=str)
    parser.add_argument("--device_num", default=0, type=int)
    parser.add_argument("--use_cpu", action="store_true")
    args = parser.parse_args()

    if args.data_files:
        dataset_list = [d.strip() for d in args.data_files.split(",") if d.strip()]
    elif args.data_file:
        dataset_list = [args.data_file]
    else:
        parser.error("one of --data_file or --data_files is required")

    if args.use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(args.device_num)

    out_dir = Path(args.out_dir)
    rows_dir = out_dir / "rows"
    runs_dir = out_dir / "runs"
    rows_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    configs = get_configs(args.mode)

    failed_datasets = []
    for dataset in dataset_list:
        try:
            # Idempotent resume: skip configs already recorded; skip feature-building if all done.
            pending = [c for c in configs if not row_file(rows_dir, dataset, c).exists()]
            if not pending:
                print(f"[{dataset}] all {len(configs)} configs already complete; nothing to do.")
                continue

            print(f"[{dataset}] {len(pending)}/{len(configs)} configs pending: "
                  f"{[c['name'] for c in pending]}")

            t0 = time.time()
            loaders = build_split_loaders(
                data_file=dataset,
                data_path=args.data_path,
                transformer_path=args.Transformer_path,
                device=device,
                seed=args.seed,
                transformer_batch_size=args.batch_size,
            )
            print(f"[{dataset}] features built in {(time.time()-t0)/60:.1f} min")

            for config in configs:
                rp = row_file(rows_dir, dataset, config)
                if rp.exists():
                    print(f"[{dataset}] skip {config['name']} (row exists)")
                    continue
                tc = time.time()
                met = run_config(config, loaders, device, args, dataset, rows_dir, runs_dir)
                print(f"[{dataset}] {config['name']}: test_auc={met.auc:.4f} "
                      f"({(time.time()-tc)/60:.1f} min)")

            print(f"[{dataset}] all done in {(time.time()-t0)/60:.1f} min")
        except Exception:
            print(f"[{dataset}] ERROR — skipping rest of this dataset, continuing shard:")
            traceback.print_exc()
            failed_datasets.append(dataset)
            continue

    if failed_datasets:
        print(f"[shard] finished {len(dataset_list)} dataset(s), "
              f"{len(failed_datasets)} FAILED: {failed_datasets}")
        raise SystemExit(1)

    print(f"[shard] finished {len(dataset_list)} dataset(s)")


if __name__ == "__main__":
    main()
