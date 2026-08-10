# BUG: Protein FASTA contamination in ESM-2 caches

| | |
|---|---|
| **Status** | Open — not yet fixed |
| **Severity** | Low impact on existing conclusions / **blocking** for future protein work |
| **Found** | 2026-08-09 |
| **Affects** | 25 of 261 datasets (9.6%), 18 RBPs |
| **Affects configs** | `protein`, `attn_protein`, `attn_protein_seq` (all three, identically) |
| **Machine-readable list** | [`protein_fasta_defects.csv`](protein_fasta_defects.csv) |

---

## Summary

The protein FASTA set inherited from PaRPI_BIP at
`/quobyte/savirangrp/manav/dataset/protein/{RBP}.fasta` contains, for a subset of RBPs,
sequences that are either **a different protein entirely** or **a substantially incomplete
fragment** of the intended protein.

Those sequences were embedded by ESM-2 and cached, so the model was trained on embeddings of
the wrong biological entity for the affected datasets.

**There is no bug in the embedding code, the caches, the loaders, or the model.** Every
component behaved correctly on the input it was given. ESM-2 produced *correct* embeddings of
*wrong sequences*, filed under the *right* names. This is why the failure is completely silent:
no exception, no shape mismatch, no log warning.

### Explicitly NOT the problem

- ❌ Not a dimension/shape bug. `esm_residue/AGO.npy` is `(707, 1280)`, which is exactly
  correct for a 707-residue input.
- ❌ Not a filename/key lookup mismatch. Every lookup resolves and returns a file.
- ❌ Not unused/dead files. Both caches are actively read by the training pipeline.

---

## Root cause

Both caches derive from the same contaminated FASTA directory:

- `ablation/build_esm_residue_cache.py:73` → `--fasta_dir /quobyte/savirangrp/manav/dataset/protein`
- PaRPI_BIP `main.py:190,427` → `read_protein_fasta("dataset/protein/{}.fasta")`

The defect is in the **data**, one layer upstream of any BRIDGE code.

---

## Evidence

### Tier 1 — wrong protein / organism (4 datasets)

| RBP | Cached content | Verification | Confidence |
|---|---|---|---|
| `AGO` | **FBXW7** (UniProt Q969H0), 707 aa | Byte-exact full-length string match to FBXW7; zero substring overlap with AGO1 (Q9UL18) / AGO2 (Q9UKV8) / AGO3 (Q9H9G7) | **Proven** |
| `PUS1` | ***Candida africana*** PUS1, 448 aa | k-mer (k=10) coverage vs human PUS1 (Q9Y606) = **0.000**; header self-declares `[Candida africana]` | **Proven** |
| `FMR1` | 65 aa fragment | 65 aa vs 632 aa canonical (Q06787); k-mer coverage only **0.143** | Strong |

### Tier 2 — correct protein, incomplete coverage (21 datasets)

k-mer coverage against UniProt canonical is 0.89–0.999, i.e. these *are* the intended protein,
just a short isoform or fragment. Degraded input, not wrong input.

Coverage worst-first: SF3B1 38%, NIPBL 42%, FAM120A 44%, HLTF 45%, SLTM 49%, ILF3 49%,
LARP7 49%, C17ORF85 54%, RPS3 55%, SAFB2 55%, MOV10 56%, AGGF1 58%, NOP56 75%, CPSF3 81%,
NOP58 99%.

### Both caches proven identical

The whole-protein cache (`/quobyte/savirangrp/manav/esm`, used by `add_protein`) and the
per-residue cache (`/quobyte/savirangrp/manav/esm_residue`, used by `attn_protein*`) are two
views of the *same* ESM run over the *same* sequences — verified numerically, not inferred:

```
cos(meanpool(residue_cache), whole_cache):
  AGO/AGO_HEK293    1.000000      PUS1/PUS1_K562    1.000000
  FMR1/FMR1_HEK293  1.000000      FMR1/FMR1_K562    1.000000
  AUH/AUH_HepG2     1.000000      FUS/FUS_K562      1.000000

control (different proteins, confirms the metric discriminates):
  cos(AGO meanpool, AUH_HepG2 whole) = 0.940676
  cos(FUS meanpool, AUH_HepG2 whole) = 0.874035
```

Cosine of exactly 1.000000 to six decimals, against a 0.87–0.94 cross-protein baseline
(ESM-2 mean-pooled embeddings are anisotropic, so 0.94 is the "unrelated" floor, not a match).

---

## Affected datasets

### Tier 1 — wrong protein (4)

```
AGO_HEK293      FMR1_HEK293      FMR1_K562      PUS1_K562
```

### Tier 2 — incomplete coverage (21)

```
AGGF1_HepG2   AGGF1_K562     C17ORF85_HEK293  CPSF3_HEK293   FAM120A_HepG2
FAM120A_K562  HLTF_HepG2     HLTF_K562        ILF3_K562      LARP7_HepG2
LARP7_K562    MOV10_HEK293   NIPBL_K562       NOP56_HEK293   NOP58_HEK293
RPS3_HepG2    RPS3_K562      SAFB2_K562       SF3B1_K562     SLTM_HepG2
SLTM_K562
```

> **`MOV10_HEK293` has no rows** in any sweep — it is one of the 3 datasets in the standing
> `METTL14_Hela` / `METTL3_Hela` / `MOV10_HEK293` gap. So **24 of the 25 actually carry
> results** (4 Tier-1 + 20 Tier-2).

### False positive — NOT affected

**`CPSF4_HEK293`**. Its FASTA header says `partial`, but the sequence is full-length
(269/269 vs UniProt O95639). Flagged on header text in a first pass, then cleared by the
substring test. Header text alone is **not** sufficient evidence — 18 files say "partial" and
only some are genuinely incomplete.

---

## Impact assessment

**On existing conclusions: negligible. Do not re-run the completed sweeps over this.**

1. Only 4/261 datasets (1.5%) are genuinely wrong; 25/261 (9.6%) touched at all. That cannot
   manufacture or mask an effect across 258 datasets.
2. More fundamentally, the protein branch had no discriminative information to contribute
   regardless — BRIDGE trains one model per RBP, so the protein input is constant within a
   dataset and absorbable as a learned bias (see `ablation/registry.py:37-38`). All three
   protein-fusion arms landed inside noise for that structural reason, not this one.
3. The contamination is **common-mode**: identical across all three arms on the same datasets.
   Head-to-head comparisons (`attn_protein` vs `protein`, `attn_protein_seq` vs
   `attn_protein`) remain valid as *relative* results. Only absolute per-dataset numbers on
   the affected 24 are suspect.

**On future work: blocking.** Any protein-structure feature (AlphaFold, ESM contact maps,
DSSP, pLDDT/disorder) predicted from a wrong or 10%-complete sequence is worse than no feature
at all. Fix before proceeding.

**On publication:** worth a caveat/methods note if this work is written up.

---

## Secondary defect (separate, minor)

PaRPI_BIP's `utils/esm.py:18` mean-pools `token_representations[0, 1 : len(pro_seq)-1]`.
With tokens laid out `[BOS, r1..rL, EOS]`, that slice covers residues 1..L-2 — it
**silently drops the last two residues of every protein**, in all 261 whole-protein embeddings.

`ablation/build_esm_residue_cache.py` already fixes this on the per-residue side, so the two
caches disagree at the C-terminus by construction. Consequence: `protein` and the two
`attn_protein` arms were not fed strictly identical protein information — a small asymmetry
in what was designed as a controlled comparison. Does not change any result.

---

## Remediation

1. **Re-fetch the 18 defective FASTAs** from UniProt canonical (reviewed, human, `organism_id:9606`)
   rather than the mixed RefSeq/GenBank records currently in use. Note only 100/172 current
   files carry a RefSeq `NP_`/`XP_` accession; 72 are GenBank (`AAH...`, `EAW...`, `CAG...`)
   which is where the partial records concentrate.
2. **Add a validation step** to `build_esm_residue_cache.py`: assert each sequence matches its
   UniProt canonical length (or log a warning), so this class of defect fails loudly.
3. **Rebuild both caches** after the fix (`slurms/build_esm_residue_cache.sh`; remember Hive
   compute nodes have no outbound internet — pre-warm `~/.cache/torch/hub/checkpoints/` from a
   login node first).
4. **Optionally** fix the C-terminal off-by-one when regenerating the whole-protein cache.

## Reproducing the checks

All verification used only UniProt REST (`rest.uniprot.org`, reachable from Hive login nodes)
plus the existing local caches — no GPU and no ESM re-run required. Methods, in increasing
order of evidential strength:

1. FASTA header text — hypothesis generator only, produces false positives (see CPSF4).
2. Length vs. UniProt canonical — confounded by isoforms.
3. Exact-substring test — distinguishes clean truncation.
4. k-mer (k=10) coverage — distinguishes *right protein, different isoform* (≈1.0) from
   *wrong protein* (≈0.0). **This is the test that matters.**
5. Byte-exact full-sequence comparison against candidates — definitive (used for AGO).
6. Cache-shape fingerprinting + cross-cache cosine — proves propagation into training inputs.
