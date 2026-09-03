# BRIDGE Ablation Pipeline (feature + module)

## Context

We quantify how much each part of BRIDGE contributes to performance by removing **one thing
at a time**, retraining from scratch, and measuring the **performance dip from the full-model
baseline** on the sealed 15% test split (see [`three_way_split_changes.md`](three_way_split_changes.md)).

Two ablation flavors, one shared engine:

- **Feature ablation** — remove an *input feature* and its whole branch, then auto-shrink the
  classifier head input from 512 by the dropped branch's width.
- **Module ablation** — swap an *internal mechanism* while keeping inputs and the 512-channel
  fusion fixed.

Because the two flavors share ~80% of their machinery (build features once → loop configs →
retrain → evaluate on the sealed test split → append to a shared CSV), the pipeline is built
by making the **existing** model/training code plug-and-play (a backward-compatible `BRIDGE`
config + two small refactors) plus one thin driver — not a parallel package.

## What gets ablated

### Feature ablations — the five branch outputs fused into ADPNet
(highlighted in [`feature_ablation_image.png`](feature_ablation_image.png); fusion at
`utils/BRIDGE.py`, `torch.cat([x, x0, x1, x2, x3], dim=1)`)

| config (`name`) | drops branch | channels | head input |
|-----------------|--------------|----------|-----------|
| `none` | — (baseline) | — | 512 |
| `gcn` | RBPformer-attention `GCNConv(512→32)` | 32 | 480 |
| `sequence` | RBPformer-embedding (`conv_bert`+`multiscale_bert`) | 256 | 256 |
| `structure` | icSHAPE (`conv_str`+`multiscale_str`) | 128 | 384 |
| `motif` | STREME prior (`conv_motif`+`multiscale_motif`) | 64 | 448 |
| `biochem` | biochemical k-mer (`conv_biochem`+`multiscale_biochem`) | 32 | 480 |

`bert_embedding` feeds both `gcn` (GCN node features) and `sequence` (`conv_bert`); `attn`
feeds only `gcn`. They are independent branches, so dropping one does not affect the other.

### Lean BRIDGE — combined drop, `mode=lean`

`drop_feature` now accepts either a single feature name or an iterable of several (normalized
internally to a `frozenset`), so a config can drop more than one branch at once. `LEAN_CONFIG`
(`ablation/registry.py`) combines the four feature drops that individually landed at or above
baseline in the full sweep (`gcn` +0.0001, `motif` +0.0007, `biochem` +0.0009, `sequence`
+0.0031 mean ΔAUC) with `kan_to_mlp` (also +0.0031 alone), keeping only `structure` — the one
branch with a large, consistent cost when dropped (−0.0442, worse on 97% of datasets) — and
leaving `ADPNet` untouched (both of its replacements cost ~−0.026, the other large, consistent
effect in the sweep). Fusion width: 512 − (32+256+64+32) = 128.

This is **not validated as a joint effect** — each drop was only ever measured one at a time
against the full 5-branch baseline; dropping all four together is a different regime, so
`LEAN_CONFIG` exists to test the combination directly. `run_ablation.py --mode lean` trains
only `[none, lean]` (not folded into `feature`/`all`, which also pull in the protein configs —
this is a baseline-vs-lean comparison, deliberately excluding every protein kwarg).

### Module ablations — swap a mechanism, keep inputs + 512 fusion
(highlighted in [`module_ablation_image.png`](module_ablation_image.png): green = the four
`multiscaleKAN` blocks; cyan = `ADPNet`)

| config (`name`) | swap |
|-----------------|------|
| `kan_to_mlp` | all four `multiscaleKAN` blocks → `multiscaleMLP`: identical 2-path residual topology but the project's `Conv1d` (Conv+BN+ReLU) replaces the KAN operator at the same kernel sizes (k=1, k=3). `2*out==in` invariant preserved, so channels are unchanged — isolates KAN-vs-conv. |
| `adpnet_to_gap` | entire `ADPNet` head → global-average-pool over the length axis (`512×101 → 512` vector) then `Linear(512→1)`. |
| `adpnet_to_attnpool` | entire `ADPNet` head → learned attention pool: per-position score (`Linear(512→1)`), softmax over the length axis, weighted sum (`512×101 → 512`), then the same `Linear(512→1)` classifier. Parameter count is close to `adpnet_to_gap` (one extra scoring `Linear`), so a delta vs. GAP isolates content-based weighting from added capacity — the first of three planned ADPNet-replacement experiments (GAP already covers naive uniform pooling as the other end of that spectrum). |

## Design

All variants are just `BRIDGE(**kwargs)`; `BRIDGE()` at defaults reproduces the exact baseline.

### A. Parametrized model — `utils/BRIDGE.py`
```python
BRIDGE(k=3, drop_feature=None, kan_to_mlp=False, adpnet_to_gap=False, adpnet_to_attnpool=False)
```
- `FEATURE_CHANNELS = {"gcn":32,"sequence":256,"structure":128,"motif":64,"biochem":32}`.
- `__init__` builds only the enabled branches; `fusion_ch = 512 - FEATURE_CHANNELS.get(drop_feature, 0)`;
  uses `multiscaleMLP` when `kan_to_mlp`; head is `GAPHead(fusion_ch)` when `adpnet_to_gap`,
  `AttnPoolHead(fusion_ch)` when `adpnet_to_attnpool`, else `ADPNet(fusion_ch, ...)`
  (`adpnet_to_gap` and `adpnet_to_attnpool` are mutually exclusive — one head swap at a time).
  **Module construction order is preserved** so baseline fixed-seed init is byte-identical to
  the pre-change model.
- `forward` builds the fusion list in the original order, omitting the dropped branch.
- New classes: `multiscaleMLP`, `GAPHead`, `AttnPoolHead`.

### B. Reusable training/data — light refactor
- `utils/data_pipeline.py` — `build_split_loaders(...)` (feature build → 70/15/15 split →
  loaders) and shared `fix_seed`. Used by both `main.py` and the driver.
- `utils/train_loop.py` — `fit_bridge(...)` encapsulates the epoch loop (warm-up + step-decay
  LR, val-AUC checkpoint selection, early stopping), reusing `train`/`validate`.
- `main.py` — all three modes now call `build_split_loaders` + `fit_bridge` (behavior
  preserved; DRYs up the previously-triplicated feature building).

### C. Thin ablation layer — `ablation/`
- `registry.py` — configs as `{name, ablation_type, component_removed, kwargs}` +
  `get_configs(mode)`, `fusion_channels`, `channels_removed`.
- `run_ablation.py` — per-dataset driver: build loaders once, loop configs, **skip any config
  whose row-file already exists** (idempotent/requeue-safe), retrain, evaluate on the sealed
  test split, write a one-row CSV + `best.json` + `model.pth`. Each run writes its own
  row-file, so concurrent tasks never race on a shared file.
- `collate_results.py` — concatenate all row-files → `ablation_results.csv` (wide),
  `_long.csv` (tidy), `.parquet`; fill `delta_*` (metric − baseline) per dataset; print a
  coverage report of missing cells.

### D. Scale-out — 261 datasets × 9 configs = 2,349 runs
- **Sharded SLURM array** (~38 tasks, `SHARD_SIZE=7` datasets/shard): each array task loops
  its shard of datasets in one Python process, building features once per dataset and
  training all 9 configs. Batching datasets per task (instead of one task per dataset) exists
  to fix an observed failure mode: when many array tasks call `conda activate` + launch
  `python` in the same instant, Python's interpreter bootstrap can crash with `Fatal Python
  error: init_fs_encoding` under shared-filesystem metadata-server contention (137/261 tasks
  failed this way in the original per-dataset array). Batching cuts the number of concurrent
  launches ~7x; `slurms/ablation.sh` additionally staggers activation with a jittered sleep
  and retries specifically on that failure signature (fast exit + `init_fs_encoding` in
  stderr) before giving up.
- Job script `slurms/ablation.sh` (single dataset, comma-separated list, or sharded array),
  manifests `ablation/datasets.txt` (261) and `ablation/datasets_pilot.txt` (currently empty —
  pre-existing gap; use a slice of `datasets.txt` for pilot runs instead).
- **Idempotent per dataset+config row-file, not per task/shard** — a requeued/preempted task,
  or a full array resubmission under a different shard mapping, always skips whatever's
  already on disk. Resubmitting the same `slurms/submit.sh --array=...` command is therefore
  always safe and is the standard recovery path after any partial failure — no need to track or
  filter down to just the failed datasets.

### Shared CSV schema
```
dataset, seed, ablation_type, component_removed, name, channels_removed,
adpnet_input_channels, best_epoch,
test_auc, test_acc, test_prc, test_mcc, test_f1, test_loss, tp, tn, fp, fn,
baseline_test_auc, delta_auc, delta_acc, delta_prc, delta_mcc,
checkpoint_path, timestamp
```
`ablation_type ∈ {baseline, feature, module}`. `delta_*` negative = performance dip
(filled at collate against the `none` row). Feature + module rows share one table for joint
plotting.

## How to run

```bash
# one dataset, all 9 configs
python -m ablation.run_ablation --data_file AUH_HepG2 --mode all --seed 42

# multiple datasets in one process (what a sharded array task does internally)
python -m ablation.run_ablation --data_files AUH_HepG2,AARS_K562 --mode all --seed 42

# SLURM: pilot, then full sweep, then collate
# (slurms/submit.sh auto-detects the cluster -- Anvil or Hive -- and supplies the right
# --account/--partition; see slurms/cluster/*.sh. Don't call `sbatch slurms/ablation.sh`
# directly, since #SBATCH can't carry cluster-specific values.)
MAX_EPOCHS=2 EARLY_STOPPING=2 SHARD_SIZE=7 slurms/submit.sh --array=0-5%6 --time=00:30:00 ablation.sh
SHARD_SIZE=7 slurms/submit.sh --array=0-37%20 --time=14:00:00 ablation.sh
python -m ablation.collate_results --manifest ablation/datasets.txt
```

`--mode {feature,module,all}`; env overrides for the sbatch: `MODE, SEED, MAX_EPOCHS,
EARLY_STOPPING, OUT_DIR, MANIFEST, SHARD_SIZE, STAGGER_WINDOW, MAX_ATTEMPTS, RETRY_BACKOFF`.
Tune `--mem`/`--time` from the pilot — `--time` should scale with `SHARD_SIZE` since each
task now trains all configs for every dataset in its shard sequentially. Resubmitting the
same `--array=...` command after a partial failure is always safe (see Scale-out above).

## Files

- **Edited:** `utils/BRIDGE.py`, `utils/train_loop.py`, `main.py`
- **New:** `utils/data_pipeline.py`, `ablation/{registry,run_ablation,collate_results}.py`,
  `ablation/{datasets.txt,datasets_pilot.txt,README.md,__init__.py}`, `slurms/ablation.sh`

## Verification (completed)

| Check | Result |
|-------|--------|
| All 8 configs forward-pass; head widths = `512 − dropped channels`; output `(2,1)` | pass |
| Baseline init reproducible (two seed-42 inits agree) | pass |
| Existing checkpoint loads into refactored `BRIDGE()` (`strict=True`, 0 missing/unexpected) | pass — baseline architecturally identical; `--validate`/`--dynamic_predict` on old checkpoints unaffected |
| Registry / fusion-channel math for all 8 configs | pass |
| End-to-end smoke (module mode, `AUH_HepG2_small`, 2 epochs): features built, 3 configs trained, row-files + `best.json` + checkpoints written (GAP=3MB vs baseline=76MB) | pass |
| Collate → wide/long/parquet + deltas + coverage report | pass |
| Idempotent resume: full re-run skips all; deleting one row re-runs only that config and reproduces the AUC | pass |

## Open note

The diagram labels STREME motif *"only used during training, not inference,"* but the current
code feeds `motif` in all modes. This pipeline treats `motif` as a normal input branch
(consistent with the code). Worth one line of team confirmation; does not change the design.

## Attention-pooling addition (`ablation-attention-pooling` branch)

First of three planned ADPNet-replacement experiments (see `MODULE_CONFIGS` in
`ablation/registry.py`): `adpnet_to_attnpool` swaps the `ADPNet` head for `AttnPoolHead`
(`utils/BRIDGE.py`) — a per-position `Linear(C,1)` score, softmax-normalized over the length
axis, weighted-summed to `(B, C)`, then the same final `Linear(C, 1)` classifier `GAPHead`
uses. Parameter count is close to `GAPHead`'s, so comparing `adpnet_to_attnpool` against
`adpnet_to_gap` isolates whether content-based weighting (vs. uniform averaging) recovers
performance lost by dropping the ADPNet pyramid, independent of added model capacity. No
engine changes were needed — `run_ablation.py`/`collate_results.py`/`plot_results.py` already
iterate whatever `ablation/registry.get_configs()` returns.

- **Edited:** `utils/BRIDGE.py` (new `AttnPoolHead`, new `adpnet_to_attnpool` kwarg, mutually
  exclusive with `adpnet_to_gap`), `ablation/registry.py` (new `MODULE_CONFIGS` entry).
