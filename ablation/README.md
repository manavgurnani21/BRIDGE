# BRIDGE Ablation Pipeline

Feature-wise and module-wise ablation of BRIDGE, retrained from scratch per config and
evaluated on the sealed 15% test split (see `../docs/three_way_split_changes.md`).

## What gets ablated

**Feature ablations** (drop an input branch, auto-shrink the head input from 512):

| config | drops | head input |
|--------|-------|-----------|
| `none` | nothing (baseline) | 512 |
| `gcn` | RBPformer-attention GCN branch | 480 |
| `sequence` | RBPformer-embedding branch | 256 |
| `structure` | icSHAPE branch | 384 |
| `motif` | STREME motif-prior branch | 448 |
| `biochem` | biochemical k-mer branch | 480 |

**Module ablations** (swap an internal mechanism, keep inputs + 512 fusion):

| config | swap |
|--------|------|
| `kan_to_mlp` | all four `multiscaleKAN` blocks → `multiscaleMLP` (Conv1d, same topology) |
| `adpnet_to_gap` | `ADPNet` head → global-average-pool + `Linear(512→1)` |
| `adpnet_to_attnpool` | `ADPNet` head → learned attention pool (`Linear(512→1)` scores, softmax over length) + `Linear(512→1)` |

All variants are just `BRIDGE(**kwargs)` (see `registry.py`); `BRIDGE()` defaults reproduce
the exact baseline. Add a new ablation by adding one entry to `registry.py`.

## Run it

Single dataset (all 8 configs), e.g. on a GPU node:

```bash
python -m ablation.run_ablation --data_file AUH_HepG2 --mode all --seed 42
```

- `--mode {feature,module,all}` selects the config set.
- Each config writes its own row-file `results/ablation/rows/{dataset}__{type}__{name}.csv`
  plus `results/ablation/runs/{dataset}/{name}/{best.json,model.pth,run.log}`.
- **Idempotent:** a config whose row-file already exists is skipped, so a requeued/preempted
  job resumes cleanly. Delete a row-file to force that config to re-run.

## Sweep all 261 datasets (SLURM)

Job script: `slurms/ablation.sh` (same conventions as `slurms/train.sh`). Works as a
single-dataset job or as an array over `datasets.txt`.

```bash
mkdir -p slurms/logs/ablation
# single dataset:
sbatch slurms/ablation.sh AUH_HepG2
# pilot first (validate walltime/mem + collate):
MANIFEST=ablation/datasets_pilot.txt sbatch --array=0-2%3 --time=02:00:00 slurms/ablation.sh
# then the full sweep (<=40 concurrent):
sbatch --array=0-260%40 slurms/ablation.sh
```

Override knobs via env vars: `MODE` (feature|module|all), `SEED`, `MAX_EPOCHS`,
`EARLY_STOPPING`, `OUT_DIR`, `MANIFEST`.

`datasets.txt` (261 stems) and `datasets_pilot.txt` are regenerated with:

```bash
ls dataset/*_pos.fa | sed 's#.*/##; s/_pos.fa//' | grep -v '_small$' | sort > ablation/datasets.txt
```

## Collate results

```bash
python -m ablation.collate_results --manifest ablation/datasets.txt
```

Produces, under `results/ablation/`:
- `ablation_results.csv` — wide table, one row per (dataset, config), with `delta_*`
  (metric − baseline) filled per dataset. Negative delta = performance dip.
- `ablation_results_long.csv` — tidy form (`…, metric, value, delta`) for plotting.
- `ablation_results.parquet` — fast filtering (if pyarrow/fastparquet available).
- a coverage report of any missing (dataset, config) cells.
