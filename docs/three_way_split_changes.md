# Three-Way Train/Validation/Test Split

## Motivation

Previously BRIDGE used a **two-way** stratified split (80% train / 20% test) built by
`split_dataset()` from a fixed seed. Because the seeding and preprocessing consume NumPy's
RNG deterministically, the split was identical across separate runs. This created a
methodological flaw:

- During `--train`, the 20% "test" split was evaluated every epoch and drove **both**
  early stopping **and** best-checkpoint selection.
- `--validate` then reloaded that checkpoint and re-scored the **same** 20% split.

So the reported "validation" metrics were computed on data used for model selection — an
optimistically biased estimate, not a true generalization measure.

**Goal:** add a sealed **test** partition that is never observed during training or model
selection, so `--validate` (and `--dynamic_predict`) report genuine held-out performance.

## Split ratio

**70% train / 15% validation / 15% test**, stratified per class (positive/negative).

- Training selects checkpoints and early-stops on the **15% validation** partition.
- `--validate` evaluates on the **15% test** partition only.
- `--dynamic_predict` (cross cell-line) also evaluates on the **15% test** partition.

## Code changes

### `utils/utils.py` — `split_dataset()`

- New signature: `split_dataset(..., valid_frac=0.15, test_frac=0.15)`.
- Returns **three** partitions `(train, valid, test)` instead of two; each is the same
  `[X1, X2, X3, X4, X5, Y]` list of modalities as before.
- Within each class, the permuted indices are sliced into three contiguous chunks:
  `test | valid | train`. A small `_gather` helper concatenates the positive-then-negative
  rows for all six arrays.
- Still consumes exactly **two** `np.random.permutation` calls (one per class), so the
  split remains fully reproducible for a fixed RNG seed.

### `main.py` — `--train` block

- Unpacks three partitions; builds `train_set` (from `train_*`) and `val_set` (from
  `val_*`).
- The per-epoch `validate()` call now uses `val_loader`, so early stopping and
  checkpoint selection watch the **validation** set.
- The `test_*` arrays are left unused (sealed) during training.

### `main.py` — `--validate` block

- Unpacks three partitions; evaluates on the sealed `test_*` split (matches the partition
  held out during `--train`).

### `main.py` — `--dynamic_predict` block

- Unpacks three partitions; evaluates cross cell-line predictions on the sealed `test_*`
  split, consistent with `--validate`.

### Not touched

- `utils/datautils.py` `split_dataset` — a separate legacy 2-arg helper, unused by
  `main.py`.
- `fix_seed`, preprocessing, model, and training-loop internals.

## Verification

- **Compilation:** `py_compile` passes on `main.py` and `utils/utils.py`.
- **Split unit test** (N=5000, standalone): partitions are reproducible across runs,
  mutually disjoint with full coverage, sized exactly **3500 / 750 / 750
  (0.70 / 0.15 / 0.15)**, and the test split is class-balanced (375 pos / 375 neg).
- **Remaining (needs GPU/SLURM):** end-to-end `--train` on `AUH_HepG2_small`, then
  `--validate` to confirm test-set metrics differ from the training-run's best validation
  AUC (confirming the sealed set is genuinely separate), plus a `--dynamic_predict` smoke
  test.
