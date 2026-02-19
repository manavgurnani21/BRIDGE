#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
variant_aware.py

Unified *variant-aware* scoring entry point for BRIDGE.

The script reads sequence windows from a FASTA file, optionally applies an allele
substitution (REF -> ALT) based on metadata encoded in each FASTA header, and
runs BRIDGE checkpoints to produce per-record prediction scores.

Pipelines
---------
The behavior is selected by optional CLI flags:

1) GWAS mode (default):

   - Activated when no other pipeline flag is provided (or when `--gwas` is set).

   - Uses a single BRIDGE checkpoint whose name is derived from the FASTA filename
     stem (i.e., `<model_save_path>/<fasta_stem>.pth`).

   - Output format (one line per record):

     .. code-block:: text

        <header_without_>\tPrediction_score:<float>

2) Ribosnitch mode (`--ribosnitch` / `--ribosnitches`):

   - For each FASTA record, extracts the last two tokens in the header as
     candidate cell lines and scores the sequence against every checkpoint in
     `--model_save_path` whose filename ends with `_<cell_line>.pth`.

   - If `--variation_mode=after` (or `--ribosnitch_after_variation`) is active,
     the ALT allele is substituted before scoring (strand-aware: bases are
     complemented on '-' strand).

   - Output format (one line per (record, checkpoint)):

     .. code-block:: text

        <header_without_>\t<checkpoint_stem>\t<float>

     Results are written under:

     .. code-block:: text

        <ribosnitch_out_dir>/{before_mut,after_mut}/<basename(variant_out_file)>

3) Variant-catalog mode (`--genomic_variants` / `--variant_catalog`):

   - For curated variant collections such as ClinVar / TCGA / 1000 Genomes, where
     FASTA headers include:

       * a region token:  chr:start-end(strand)
       * an SNV token:    POS:REF>ALT
       * model-id fields: typically "... <PROTEIN> in <CELL_LINE>"

   - Provides robust parsing and optional off-by-one handling when locating the
     variant within the window.

   - Output format matches the standalone catalog script:

     .. code-block:: text

        <header_without_>\tmodel_id=<...>\tmode=<before|after>\tPrediction_score:<float>

Common inputs
-------------
- `--fasta_sequence_path` : FASTA of window sequences (wrapped/multi-line FASTA is supported).
- `--variation_mode`      : `before` scores the input sequence as-is; `after` attempts ALT substitution.
- `--Transformer_path`    : transformer used by `build_Transformer_embeddings`.
- `--model_save_path`     : directory containing BRIDGE `.pth` checkpoints.
- `--variant_out_file`    : path to append results.

"""

from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

# ---------------------------------------------------------------------------
# Third-party / project-specific utilities (assumed to exist in PYTHONPATH)
# ---------------------------------------------------------------------------
from utils.BRIDGE import BRIDGE
from utils.gen_transformer_embedding import build_Transformer_embeddings
from utils.train_loop import validate_without_sigmoid
from utils.utils import RBPInferDataset
from utils.FeatureEncoding import dealwithdata2
from utils.variant import read_fasta, open_output, parse_variant_block, apply_complement, substitute_base, ModelHub, parse_variant_block_flexible


# ---------------------------------------------------------------------------
# GWAS / BRIDGE core pipeline
# ---------------------------------------------------------------------------
def process_sequences_gwas(
    names: List[str],
    seqs: List[str],
    args: argparse.Namespace,
    hub: ModelHub,
) -> None:
    """Process each FASTA record and append GWAS/BRIDGE variant-aware predictions.

    For each (header, seq):
      1) Parse (variant_pos, REF, ALT, strand, seq_start) from header.
      2) If strand is '-', complement REF/ALT bases.
      3) Compute 0-based index in the window: idx0 = variant_pos - seq_start.
      4) Validate idx0 bounds and REF base match inside the window.
      5) Choose sequence:
         - before: `modified_seq = seq`
         - after : `modified_seq = seq` with idx0 substituted to ALT
      6) Build multimodal inputs expected by BRIDGE:
         - transformer embedding, attn, struct, motif priors, and biochemical features.
      7) Load checkpoint named by FASTA stem, run `validate_without_sigmoid`, write output.

    Important implied constraints
    -----------------------------
    - The fixed tensors `(1, 1, 101)` imply window length ≈ 101 in typical usage.
    """
    out_fp = open_output(Path(args.variant_out_file))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0, device=hub.device))

    with out_fp.open("a") as fout:
        for header, seq in zip(names, seqs):
            try:
                var_pos, ref, alt, strand, seq_start = parse_variant_block(header)
            except ValueError as err:
                logging.error("%s → %s", header, err)
                continue

            if strand == "-":
                ref, alt = apply_complement(ref), apply_complement(alt)

            idx0 = var_pos - seq_start
            if idx0 < 0 or idx0 >= len(seq):
                logging.error("Variant index out of bounds (%s)", header)
                continue
            if seq[idx0] != ref:
                logging.error("Ref base mismatch (%s) — skip", header)
                continue

            modified_seq = substitute_base(seq, idx0, alt) if args.variation_mode == "after" else seq

            test_emb, _ = build_Transformer_embeddings(
                sequences=[modified_seq],
                transformer_path=str(args.Transformer_path),
                device=hub.device,
                k=1,
                transpose_to_ch_first=True,
            )
            N = int(test_emb.shape[0])
            test_attn = np.zeros((N, 101, 103))
            struct = np.zeros((N, 1, 101))
            motif = np.zeros((N, 1, 101))
            bio_chem = dealwithdata2(modified_seq).transpose([0, 2, 1])

            dataset = RBPInferDataset(
                embedding=test_emb,
                attn=test_attn,
                struct=struct,
                motif=motif,
                biochem=bio_chem,
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            filename_stem = Path(args.fasta_sequence_path).stem
            bridge = hub.load_bridge(Path(args.model_save_path), filename_stem)
            if bridge is None:
                continue

            prob = validate_without_sigmoid(bridge, hub.device, loader, criterion).item()
            fout.write(f"{header.lstrip('>')}\tPrediction_score:{prob:.6f}\n")


# ---------------------------------------------------------------------------

# Backward-compatible alias: keep the old function name if other scripts import it.
process_sequences = process_sequences_gwas


# ---------------------------------------------------------------------------
# ClinVar / TCGA / 1000 Genomes style FASTA batches ("catalog variants")
# ---------------------------------------------------------------------------
# These datasets typically store variant windows directly in FASTA, where the header
# contains (chrom:start-end(strand)) and a SNV token like POS:REF>ALT.
#
# This branch is activated by passing `--catalog_variants` (alias: --genomic_variants).
# The default model naming strategy is `<PROTEIN>_<CELL>` parsed from the header.
# ---------------------------------------------------------------------------

_REGION_RE = re.compile(r"^(chr[^:]+):(\d+)-(\d+)\(([+-])\)")
_VAR_RE = re.compile(r"^(\d+):([ACGTN])>([ACGTN])$")


@dataclass
class ParsedHeader:
    "Parsed representation of a variant-window FASTA header."
    header_raw: str
    chrom: str
    start: int
    end: int
    strand: str
    var_pos: int
    ref: str
    alt: str
    protein: Optional[str]
    cell_line: Optional[str]


def parse_protein_cell_line(fields: List[str]) -> Tuple[Optional[str], Optional[str]]:
    "Heuristically parse protein & cell line from header tokens."
    protein: Optional[str] = None
    cell: Optional[str] = None

    # Prefer "... <PROTEIN> in <CELL>"
    if "in" in fields:
        # use the last "in" to be robust if "in" appears elsewhere
        idx_in = len(fields) - 1 - list(reversed(fields)).index("in")
        if idx_in + 1 < len(fields):
            cell = fields[idx_in + 1]
        if idx_in - 1 >= 0:
            protein = fields[idx_in - 1]

    # Fallbacks (match the old "[-3],[-1]" convention)
    if cell is None and len(fields) >= 1:
        cell = fields[-1]
    if protein is None:
        if len(fields) >= 3 and fields[-2] == "in":
            protein = fields[-3]
        elif len(fields) >= 3:
            protein = fields[-3]
        elif len(fields) >= 2:
            protein = fields[-2]

    return protein, cell


def parse_header_catalog(header_raw: str) -> ParsedHeader:
    "Parse a catalog-variant header with flexible token positions."
    fields = header_raw.split()

    region_tok: Optional[str] = None
    var_tok: Optional[str] = None

    for tok in fields:
        if tok.startswith("chr") and ":" in tok and "(" in tok and ")" in tok:
            region_tok = tok
            break
    if region_tok is None:
        for tok in fields:
            if _REGION_RE.match(tok):
                region_tok = tok
                break
    if region_tok is None:
        raise ValueError(f"Cannot find region token like chr:start-end(strand) in header: {header_raw}")

    for tok in fields:
        if _VAR_RE.match(tok):
            var_tok = tok
            break
    if var_tok is None:
        raise ValueError(f"Cannot find SNV token like POS:REF>ALT in header: {header_raw}")

    m_r = _REGION_RE.match(region_tok)
    if m_r is None:
        raise ValueError(f"Region token doesn't match expected pattern: {region_tok}")
    chrom, start, end, strand = m_r.group(1), int(m_r.group(2)), int(m_r.group(3)), m_r.group(4)

    m_v = _VAR_RE.match(var_tok)
    assert m_v is not None
    var_pos, ref, alt = int(m_v.group(1)), m_v.group(2), m_v.group(3)

    protein, cell = parse_protein_cell_line(fields)

    return ParsedHeader(
        header_raw=header_raw,
        chrom=chrom,
        start=start,
        end=end,
        strand=strand,
        var_pos=var_pos,
        ref=ref,
        alt=alt,
        protein=protein,
        cell_line=cell,
    )


def find_variant_index(
    seq: str,
    seq_start: int,
    var_pos: int,
    ref: str,
    alt: str,
    try_off_by_one: bool = True,
) -> Tuple[Optional[int], str]:
    "Locate the 0-based variant index inside the window sequence."
    candidates = [var_pos - seq_start]
    if try_off_by_one:
        candidates.append(var_pos - seq_start - 1)

    for idx0 in candidates:
        if idx0 < 0 or idx0 >= len(seq):
            continue
        base = seq[idx0]
        if base == ref:
            return idx0, "ref"
        if base == alt:
            return idx0, "alt"

    return None, "none"


def process_sequences_catalog_variants(
    headers: List[str],
    seqs: List[str],
    args: argparse.Namespace,
    hub: ModelHub,
) -> None:
    "Variant-aware scoring for ClinVar/TCGA/1000G-style FASTA batches (SNVs)."
    out_fp = open_output(args.variant_out_file)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(args.pos_weight), device=hub.device))

    with out_fp.open("a") as fout:
        for header_raw, seq in zip(headers, seqs):
            try:
                ph = parse_header_catalog(header_raw)
            except Exception as e:
                logging.error("[catalog_variants] Header parse failed: %s | %s", header_raw, e)
                continue

            ref, alt = ph.ref, ph.alt
            if ph.strand == "-":
                ref, alt = apply_complement(ref), apply_complement(alt)

            idx0, state = find_variant_index(
                seq=seq,
                seq_start=ph.start,
                var_pos=ph.var_pos,
                ref=ref,
                alt=alt,
                try_off_by_one=(not bool(args.disable_off_by_one)),
            )
            if state == "none":
                logging.warning("[catalog_variants] Cannot match REF/ALT at site: %s", ph.header_raw)
                if bool(args.strict_ref_match):
                    continue

            # choose checkpoint name
            if args.model_id_strategy == "from_fasta_stem":
                model_id = Path(args.fasta_sequence_path).stem
            else:
                if not ph.protein or not ph.cell_line:
                    logging.warning("[catalog_variants] Cannot parse protein/cell line for model id: %s", ph.header_raw)
                    continue
                model_id = f"{ph.protein}_{ph.cell_line}"

            model = hub.load_bridge(Path(args.model_save_path), model_id)
            if model is None:
                continue

            # build modified_seq depending on variation_mode and what we see in input
            if args.variation_mode == "before":
                if state == "alt":
                    logging.info("[catalog_variants] Input already ALT at site; scoring as-is in BEFORE: %s", ph.header_raw)
                modified_seq = seq
            else:  # after
                if idx0 is None:
                    modified_seq = seq
                elif state == "ref":
                    modified_seq = substitute_base(seq, idx0, alt)
                else:
                    modified_seq = seq

            emb, _ = build_Transformer_embeddings(
                sequences=[modified_seq],
                transformer_path=str(args.Transformer_path),
                device=hub.device,
                k=int(args.k),
                transpose_to_ch_first=True,
            )
            N = int(emb.shape[0])

            attn = np.zeros((N, 101, 103))
            struct = np.zeros((N, 1, 101))
            motif = np.zeros((N, 1, 101))
            biochem = dealwithdata2(modified_seq).transpose([0, 2, 1])

            dataset = RBPInferDataset(
                embedding=emb,
                attn=attn,
                struct=struct,
                motif=motif,
                biochem=biochem,
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            score = validate_without_sigmoid(model, hub.device, loader, criterion).item()

            # Output keeps genomic_variants.py style (adds model_id/mode)
            fout.write(
                f"{ph.header_raw}\tmodel_id={model_id}"
                f"\tmode={args.variation_mode}\tPrediction_score:{score:.6f}\n"
            )


# Ribosnitches pipeline
# ---------------------------------------------------------------------------
def _lazy_import_symbol(module_name: str, symbol_name: str):
    """Import `symbol_name` from `module_name` dynamically (helper for optional deps)."""
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _extract_cell_lines(header_wo_gt: str) -> Tuple[str, str]:
    """Extract the last two tokens as (cell_line1, cell_line2).

    This matches the user-provided ribosnitches code and assumes the FASTA header
    ends with two cell-line names.
    """
    fields = header_wo_gt.split()
    if len(fields) < 2:
        raise ValueError("Header has fewer than 2 tokens; cannot extract cell lines.")
    return fields[-2], fields[-1]


def _maybe_mutate_sequence_from_header(header: str, seq: str) -> str:
    """Apply ALT substitution based on header tokens (used by ribosnitches-after).

    We use the flexible parser to support headers where the variant token is not
    at a fixed position (e.g. when the header ends with cell-line names).
    """
    var_pos, ref, alt, strand, seq_start = parse_variant_block_flexible(header)

    if strand == "-":
        # The user requested: "only complement, do not reverse" because the variant is centered.
        ref, alt = apply_complement(ref), apply_complement(alt)

    idx0 = var_pos - seq_start
    if idx0 < 0 or idx0 >= len(seq):
        raise ValueError(f"Variant index out of bounds (idx0={idx0}, len={len(seq)})")
    if seq[idx0] != ref:
        raise ValueError(f"Reference base mismatch at idx0={idx0}: seq={seq[idx0]} vs ref={ref}")
    return substitute_base(seq, idx0, alt)


def run_ribosnitches(
    names: List[str],
    seqs: List[str],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Run ribosnitches scoring using the *same* BRIDGE pipeline components as GWAS mode.

    What changes compared to GWAS mode is *only* the checkpoint selection rule:
    we select all `.pth` files under `--model_save_path` that end with either
    `_<cell_line1>.pth` or `_<cell_line2>.pth`, where the two cell lines are taken
    from the last two tokens of each FASTA header.

    Mutation behavior is controlled by:
    - `--variation_mode` (before/after), OR
    - `--ribosnitches_after_variation` (force after)

    Output is appended as TSV:
        <header_without_>  <model_stem>  <score>
    """

    # Decide whether we should substitute ALT in the window
    do_after = bool(args.ribosnitch_after_variation) or (
        bool(args.ribosnitch) and args.variation_mode == "after"
    )

    out_subdir = "after_mut" if do_after else "before_mut"
    out_path = open_output(args.variant_out_file)

    # Loss is only used because validate_without_sigmoid expects it
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0, device=device))

    model_dir = Path(args.model_save_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"--model_save_path does not exist: {model_dir}")

    # Reuse BRIDGE checkpoint caching logic from the GWAS branch
    hub = ModelHub(Path(args.Transformer_path), device)

    logging.info("[ribosnitches] Mode=%s | writing to %s", "after" if do_after else "before", out_path)

    with out_path.open("a") as fout:
        for header, seq in zip(names, seqs):
            header_wo_gt = header.lstrip(">")

            # 1) Determine which checkpoints to apply (by cell line suffix)
            try:
                cell_line1, cell_line2 = _extract_cell_lines(header_wo_gt)
            except ValueError as e:
                logging.warning("[ribosnitches] %s -> %s (skip)", header_wo_gt, e)
                continue

            model_files = [
                fn for fn in os.listdir(model_dir)
                if fn.endswith(f"_{cell_line1}.pth") or fn.endswith(f"_{cell_line2}.pth")
            ]
            if not model_files:
                logging.warning("[ribosnitches] No BRIDGE checkpoints for '%s' (skip)", header_wo_gt)
                continue

            # 2) Possibly mutate the input sequence
            try:
                seq_in = _maybe_mutate_sequence_from_header(header, seq) if do_after else seq
            except Exception as e:
                logging.warning("[ribosnitches] %s -> cannot apply variant: %s (skip)", header_wo_gt, e)
                continue

            # 3) Build BRIDGE inputs ONCE per record (then reuse across all checkpoints)
            test_emb, _ = build_Transformer_embeddings(
                sequences=[seq_in],
                transformer_path=str(args.Transformer_path),
                device=hub.device,
                k=1,
                transpose_to_ch_first=True,
            )

            # Keep placeholder tensors consistent with the existing GWAS workflow
            N = int(test_emb.shape[0])
            test_attn = np.zeros((N, 101, 103))
            struct = np.zeros((N, 1, 101))
            motif = np.zeros((N, 1, 101))
            bio_chem = dealwithdata2(seq_in).transpose([0, 2, 1])

            dataset = RBPInferDataset(
                embedding=test_emb,
                attn=test_attn,
                struct=struct,
                motif=motif,
                biochem=bio_chem,
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            # 4) Score with each matching checkpoint
            for filename in model_files:
                stem = Path(filename).stem
                bridge = hub.load_bridge(model_dir, stem)
                if bridge is None:
                    continue

                score = validate_without_sigmoid(bridge, hub.device, loader, criterion).item()
                fout.write(f"{header_wo_gt}\t{stem}\t{score}\n")

    logging.info("[ribosnitches] Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Variant-aware scoring with BRIDGE. Supports three pipelines:\n"
            "  (1) GWAS windows (legacy variant_aware.py behavior; default)\n"
            "  (2) Ribosnitch scoring (BRIDGE; per-record checkpoint selection)\n"
            "  (3) Catalog variants (ClinVar/TCGA/1000G-style FASTA batches; SNVs)\n"
        )
    )

    # ------------------------------------------------------------------
    # Common arguments (shared across pipelines)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--variation_mode",
        choices=["before", "after"],
        required=True,
        help="Score sequences before variation (reference) or after variation (mutated).",
    )
    parser.add_argument("--fasta_sequence_path", required=True, type=Path)
    parser.add_argument("--variant_out_file", required=True, type=Path)
    parser.add_argument("--Transformer_path", required=True, type=Path)
    parser.add_argument("--model_save_path", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Pipeline selection flags
    #
    # Backward compatibility:
    # - If you pass none of these flags, we default to GWAS mode.
    # - `--ribosnitches` is accepted as an alias for `--ribosnitch`.
    # - `--genomic_variants` is accepted as an alias for `--catalog_variants`.
    # ------------------------------------------------------------------
    pipe = parser.add_mutually_exclusive_group(required=False)
    pipe.add_argument(
        "--gwas",
        action="store_true",
        help="Force GWAS window scoring (default if no pipeline flag is provided).",
    )
    pipe.add_argument(
        "--ribosnitch",
        "--ribosnitches",
        dest="ribosnitch",
        action="store_true",
        help="Run ribosnitch scoring (BRIDGE).",
    )
    pipe.add_argument(
        "--catalog_variants",
        "--genomic_variants",
        dest="catalog_variants",
        action="store_true",
        help="Run ClinVar/TCGA/1000G-style FASTA batch scoring (SNVs).",
    )

    # ------------------------------------------------------------------
    # Ribosnitch-specific options
    # ------------------------------------------------------------------
    parser.add_argument(
        "--ribosnitch_after_variation",
        "--ribosnitches_after_variation",
        dest="ribosnitch_after_variation",
        action="store_true",
        help="Force ALT substitution for ribosnitch scoring, regardless of --variation_mode.",
    )
    parser.add_argument(
        "--ribosnitch_out_dir",
        "--ribosnitches_out_dir",
        dest="ribosnitch_out_dir",
        type=Path,
        default=Path("./results/ribosnitches"),
        help="Root output directory for ribosnitch results.",
    )

    # ------------------------------------------------------------------
    # Catalog-variants options (also used by genomic_variants.py)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--model_id_strategy",
        choices=["from_header", "from_fasta_stem"],
        default="from_header",
        help=(
            "How to choose checkpoint name for catalog variants. "
            "from_header: <PROTEIN>_<CELL> parsed from header; "
            "from_fasta_stem: use FASTA filename stem."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="K-mer / stride parameter forwarded to build_Transformer_embeddings (catalog variants branch).",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=2.0,
        help="Positive class weight for BCEWithLogitsLoss (catalog variants branch).",
    )
    parser.add_argument(
        "--strict_ref_match",
        action="store_true",
        help="If set, skip records when REF/ALT cannot be matched inside the window (catalog variants branch).",
    )
    parser.add_argument(
        "--disable_off_by_one",
        action="store_true",
        help="Disable +/-1 position fallback when locating the SNV inside the window (catalog variants branch).",
    )

    return parser


def main() -> None:
    args = build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(args.device)

    logging.info("Loading FASTA from %s", args.fasta_sequence_path)
    headers, sequences = read_fasta(args.fasta_sequence_path)

    # Pipeline selection validation.
    # - Default: GWAS (for backward compatibility) when no pipeline flag is provided.
    selected = []
    if bool(getattr(args, "gwas", False)):
        selected.append("gwas")
    if bool(getattr(args, "ribosnitch", False)) or bool(getattr(args, "ribosnitch_after_variation", False)):
        selected.append("ribosnitch")
    if bool(getattr(args, "catalog_variants", False)):
        selected.append("catalog_variants")

    if len(selected) > 1:
        raise SystemExit(f"Conflicting pipeline flags: {selected}. Please choose only one.")

    pipeline = selected[0] if selected else "gwas"

    hub = ModelHub(args.Transformer_path, device)

    if pipeline == "catalog_variants":
        logging.info("Running catalog variants pipeline (%s_variation)", args.variation_mode)
        process_sequences_catalog_variants(headers, sequences, args, hub)
    elif pipeline == "ribosnitch":
        logging.info("Running ribosnitch pipeline (%s_variation)", args.variation_mode)
        run_ribosnitches(headers, sequences, args, device)
    else:
        logging.info("Running GWAS pipeline (%s_variation)", args.variation_mode)
        process_sequences_gwas(headers, sequences, args, hub)

    logging.info("Finished. Results appended to %s", args.variant_out_file)


if __name__ == "__main__":
    main()