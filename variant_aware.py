#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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

from __future__ import annotations  # allow forward-referenced/PEP 604-style type hints on older Python

import argparse  # CLI argument parsing
import logging  # structured info/warning/error reporting used throughout this script
import os  # filesystem operations (listing checkpoint directories, etc.)
import re  # regex parsing of FASTA header tokens (region/variant patterns)
from dataclasses import dataclass  # used to define the ParsedHeader record type
from pathlib import Path  # path handling for FASTA/model/output locations
from typing import List, Tuple, Dict, Optional, Callable  # type hints for function signatures

import numpy as np  # placeholder feature arrays (attn/struct/motif zeros)
import torch  # tensors, device handling, loss functions
from torch import nn  # loss module (BCEWithLogitsLoss)
from torch.utils.data import DataLoader  # wraps the single-sequence inference dataset for batched model calls
from transformers import BertTokenizer, BertModel  # imported for type/availability parity with the embedding pipeline; not directly instantiated here

# ---------------------------------------------------------------------------
# Third-party / project-specific utilities (assumed to exist in PYTHONPATH)
# ---------------------------------------------------------------------------
from utils.BRIDGE import BRIDGE  # BRIDGE model class, loaded via checkpoints through ModelHub
from utils.gen_transformer_embedding import build_Transformer_embeddings  # produces per-sequence transformer embeddings used as BRIDGE input
from utils.train_loop import validate_without_sigmoid  # runs a loader through a model and returns raw (pre-sigmoid) prediction scores
from utils.utils import RBPInferDataset  # thin Dataset wrapper bundling embedding + placeholder feature tensors for inference
from utils.FeatureEncoding import dealwithdata2  # computes biochemical one-hot/derived features for a sequence
from utils.variant import read_fasta, open_output, parse_variant_block, apply_complement, substitute_base, ModelHub, parse_variant_block_flexible  # FASTA I/O, variant parsing/substitution helpers, and the checkpoint-caching ModelHub


# ---------------------------------------------------------------------------
# GWAS / BRIDGE core pipeline
# ---------------------------------------------------------------------------
def process_sequences_gwas(
    names: List[str],
    seqs: List[str],
    args: argparse.Namespace,
    hub: ModelHub,
) -> None:
    """
    Process each FASTA record and append GWAS/BRIDGE variant-aware predictions.

    Args:
        names (List[str]): List of FASTA record headers (sequence identifiers).
        seqs (List[str]): List of FASTA sequence strings.
        args (argparse.Namespace): Command line arguments containing configurations like variation mode and output file.
        hub (ModelHub): A ModelHub object to manage model loading and device handling.

    Returns:
        None. Results are written to the file specified in `args.variant_out_file`.

    Exceptions:
        Logs errors when:

            - Parsing the variant information fails.

            - Variant position is out of bounds or mismatches the REF base.

    Example:
        >>> process_sequences_gwas(names, sequences, args, hub)
    """
    out_fp = open_output(Path(args.variant_out_file))  # resolve/prepare the output file path for appending
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0, device=hub.device))  # loss used only so validate_without_sigmoid has a criterion to satisfy its signature

    with out_fp.open("a") as fout:  # open the output file in append mode so repeated runs accumulate results
        for header, seq in zip(names, seqs):  # iterate over each FASTA record (header, sequence) pair
            try:
                var_pos, ref, alt, strand, seq_start = parse_variant_block(header)  # extract variant position/REF/ALT/strand/window-start encoded in the header
            except ValueError as err:
                logging.error("%s → %s", header, err)  # header didn't match the expected variant encoding
                continue  # skip this record and move to the next

            if strand == "-":
                ref, alt = apply_complement(ref), apply_complement(alt)  # on the minus strand, complement REF/ALT to match the sequence's coding orientation

            idx0 = var_pos - seq_start  # convert genomic variant position to a 0-based index within this sequence window
            if idx0 < 0 or idx0 >= len(seq):
                logging.error("Variant index out of bounds (%s)", header)  # the computed index falls outside the window; header/coords are inconsistent
                continue
            if seq[idx0] != ref:
                logging.error("Ref base mismatch (%s) — skip", header)  # sanity check failed: base at idx0 doesn't match the expected REF allele
                continue

            modified_seq = substitute_base(seq, idx0, alt) if args.variation_mode == "after" else seq  # apply the ALT substitution only when scoring the mutated sequence

            test_emb, _ = build_Transformer_embeddings(
                sequences=[modified_seq],  # embed the (possibly mutated) sequence
                transformer_path=str(args.Transformer_path),  # path to the pretrained RNA transformer
                device=hub.device,  # compute embeddings on the same device as the BRIDGE model
                k=1,  # k-mer/stride parameter for tokenization
                transpose_to_ch_first=True,  # reorder embedding dims to channel-first for BRIDGE's conv layers
            )
            N = int(test_emb.shape[0])  # batch size (always 1 here: a single sequence per call)
            test_attn = np.zeros((N, 101, 103))  # placeholder attention-prior tensor (unused signal, zeros) matching BRIDGE's expected input shape
            struct = np.zeros((N, 1, 101))  # placeholder RNA structure feature tensor (zeros; not computed in this pipeline)
            motif = np.zeros((N, 1, 101))  # placeholder motif-prior feature tensor (zeros; not computed in this pipeline)
            bio_chem = dealwithdata2(modified_seq).transpose([0, 2, 1])  # biochemical one-hot features for the sequence, reordered to match BRIDGE's expected axis order

            dataset = RBPInferDataset(
                embedding=test_emb,  # transformer embedding for this sequence
                attn=test_attn,  # placeholder attention feature
                struct=struct,  # placeholder structure feature
                motif=motif,  # placeholder motif feature
                biochem=bio_chem,  # computed biochemical feature
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False)  # wrap the single-example dataset in a DataLoader for the model's forward pass

            filename_stem = Path(args.fasta_sequence_path).stem  # derive the checkpoint name from the input FASTA's filename (GWAS mode convention)
            bridge = hub.load_bridge(Path(args.model_save_path), filename_stem)  # load (or fetch cached) BRIDGE checkpoint matching this dataset
            if bridge is None:
                continue  # no matching checkpoint found; skip scoring this record

            prob = validate_without_sigmoid(bridge, hub.device, loader, criterion).item()  # run the model and extract the scalar raw prediction score
            fout.write(f"{header.lstrip('>')}\tPrediction_score:{prob:.6f}\n")  # write header (without leading '>') and score as a TSV line


# ---------------------------------------------------------------------------

# Backward-compatible alias: keep the old function name if other scripts import it.
process_sequences = process_sequences_gwas  # alias so older callers referencing `process_sequences` keep working


# ---------------------------------------------------------------------------
# ClinVar / TCGA / 1000 Genomes style FASTA batches ("catalog variants")
# ---------------------------------------------------------------------------
# These datasets typically store variant windows directly in FASTA, where the header
# contains (chrom:start-end(strand)) and a SNV token like POS:REF>ALT.
#
# This branch is activated by passing `--catalog_variants` (alias: --genomic_variants).
# The default model naming strategy is `<PROTEIN>_<CELL>` parsed from the header.
# ---------------------------------------------------------------------------

_REGION_RE = re.compile(r"^(chr[^:]+):(\d+)-(\d+)\(([+-])\)")  # matches a "chrN:start-end(strand)" region token
_VAR_RE = re.compile(r"^(\d+):([ACGTN])>([ACGTN])$")  # matches a "POS:REF>ALT" SNV token


@dataclass
class ParsedHeader:
    "Parsed representation of a variant-window FASTA header."
    header_raw: str  # the original, unparsed header string
    chrom: str  # chromosome name extracted from the region token
    start: int  # window start coordinate
    end: int  # window end coordinate
    strand: str  # '+' or '-' strand of the window
    var_pos: int  # genomic position of the variant (from the SNV token)
    ref: str  # reference allele
    alt: str  # alternate allele
    protein: Optional[str]  # parsed RBP/protein name, if present in the header
    cell_line: Optional[str]  # parsed cell-line name, if present in the header


def parse_protein_cell_line(fields: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Heuristically parses protein and cell line from header tokens in a FASTA header.

    This function identifies the protein and cell line from a list of header tokens.
    It prefers the format "... <PROTEIN> in <CELL>" but falls back on older conventions
    where the protein and cell line are assumed to be at fixed positions in the header.

    Args:
        fields (List[str]): A list of strings representing the tokens parsed from a FASTA header.

    Returns:
        Tuple[Optional[str], Optional[str]]:

            - `protein` (Optional[str]): The parsed protein name, or None if not found.

            - `cell` (Optional[str]): The parsed cell line name, or None if not found.

    Exceptions:
        None. If the header format does not match expectations, the function will attempt to
        fall back on different header conventions.

    Example:
        >>> fields = ["GeneA", "in", "CellLineA"]
        >>> protein, cell = parse_protein_cell_line(fields)
        >>> print(protein, cell)
        "GeneA", "CellLineA"
    """
    protein: Optional[str] = None  # default: protein not yet found
    cell: Optional[str] = None  # default: cell line not yet found

    # Prefer "... <PROTEIN> in <CELL>"
    if "in" in fields:
        # use the last "in" to be robust if "in" appears elsewhere
        idx_in = len(fields) - 1 - list(reversed(fields)).index("in")  # index of the last occurrence of the literal token "in"
        if idx_in + 1 < len(fields):
            cell = fields[idx_in + 1]  # token right after "in" is the cell line
        if idx_in - 1 >= 0:
            protein = fields[idx_in - 1]  # token right before "in" is the protein name

    # Fallbacks (match the old "[-3],[-1]" convention)
    if cell is None and len(fields) >= 1:
        cell = fields[-1]  # no "in" token found: assume the last token is the cell line
    if protein is None:
        if len(fields) >= 3 and fields[-2] == "in":
            protein = fields[-3]  # "<PROTEIN> in <CELL>" pattern found near the end
        elif len(fields) >= 3:
            protein = fields[-3]  # fallback: assume protein is 3rd-from-last token
        elif len(fields) >= 2:
            protein = fields[-2]  # shorter header: assume protein is 2nd-from-last token

    return protein, cell  # possibly None if the header didn't contain enough tokens


def parse_header_catalog(header_raw: str) -> ParsedHeader:
    """
    Parse a catalog-variant header (ClinVar, TCGA, 1000G style) from a FASTA header.

    Args:
        header_raw (str): The raw header string to parse.

    Returns:
        ParsedHeader: A dataclass containing parsed region, variant, and model information.

    Exceptions:
        Raises ValueError if the header does not contain expected region or variant information.

    Example:
        >>> header = ">chr1:100-200(+) 123:A>T ProteinA in CellLineA"
        >>> parsed_header = parse_header_catalog(header)
    """
    fields = header_raw.split()  # tokenize the header on whitespace

    region_tok: Optional[str] = None  # will hold the "chr:start-end(strand)" token once found
    var_tok: Optional[str] = None  # will hold the "POS:REF>ALT" token once found

    for tok in fields:
        if tok.startswith("chr") and ":" in tok and "(" in tok and ")" in tok:
            region_tok = tok  # quick heuristic match on a region-shaped token
            break
    if region_tok is None:
        for tok in fields:
            if _REGION_RE.match(tok):
                region_tok = tok  # fall back to strict regex match if the heuristic above found nothing
                break
    if region_tok is None:
        raise ValueError(f"Cannot find region token like chr:start-end(strand) in header: {header_raw}")  # header is missing required region info

    for tok in fields:
        if _VAR_RE.match(tok):
            var_tok = tok  # first token matching "POS:REF>ALT" is treated as the SNV token
            break
    if var_tok is None:
        raise ValueError(f"Cannot find SNV token like POS:REF>ALT in header: {header_raw}")  # header is missing the variant token

    m_r = _REGION_RE.match(region_tok)  # re-run the regex to capture groups from the region token
    if m_r is None:
        raise ValueError(f"Region token doesn't match expected pattern: {region_tok}")  # defensive check; should not happen given prior matching
    chrom, start, end, strand = m_r.group(1), int(m_r.group(2)), int(m_r.group(3)), m_r.group(4)  # unpack chromosome, window start/end, and strand

    m_v = _VAR_RE.match(var_tok)  # re-run the regex to capture groups from the SNV token
    assert m_v is not None  # guaranteed non-None since var_tok was found via the same regex
    var_pos, ref, alt = int(m_v.group(1)), m_v.group(2), m_v.group(3)  # unpack variant position, reference and alternate alleles

    protein, cell = parse_protein_cell_line(fields)  # heuristically extract protein/cell-line names from the remaining tokens

    return ParsedHeader(
        header_raw=header_raw,  # keep the original header for output/logging
        chrom=chrom,  # parsed chromosome
        start=start,  # parsed window start
        end=end,  # parsed window end
        strand=strand,  # parsed strand
        var_pos=var_pos,  # parsed variant genomic position
        ref=ref,  # parsed reference allele
        alt=alt,  # parsed alternate allele
        protein=protein,  # parsed protein name (may be None)
        cell_line=cell,  # parsed cell-line name (may be None)
    )


def find_variant_index(
    seq: str,
    seq_start: int,
    var_pos: int,
    ref: str,
    alt: str,
    try_off_by_one: bool = True,
) -> Tuple[Optional[int], str]:
    """
    Locate the 0-based variant index inside the window sequence.

    This function searches for the variant position within the given sequence. It returns
    the index of the matching base (REF or ALT) in the sequence. Optionally, it can attempt
    to handle the case where the variant position is off by one (e.g., due to off-by-one errors).

    Args:
        seq (str): The sequence string in which to find the variant position.
        seq_start (int): The starting position of the sequence window.
        var_pos (int): The position of the variant (1-based index).
        ref (str): The reference base at the variant position.
        alt (str): The alternative base at the variant position.
        try_off_by_one (bool, optional): Whether to try the adjacent position (var_pos - 1) if the exact variant position is not found. Defaults to True.

    Returns:
        Tuple[Optional[int], str]:
            - The index (0-based) of the base in the sequence that matches either REF or ALT.
            - A string indicating whether the base at the index is "ref", "alt", or "none" if neither match.

    Exceptions:
        None. If no match is found, the function returns `None, "none"` without raising any errors.

    Example:
        >>> find_variant_index("AGCTG", 0, 3, "T", "G")
        (2, "ref")
    """
    candidates = [var_pos - seq_start]  # primary candidate: straightforward genomic-to-window index conversion
    if try_off_by_one:
        candidates.append(var_pos - seq_start - 1)  # secondary candidate: guards against a common off-by-one in coordinate conventions

    for idx0 in candidates:
        if idx0 < 0 or idx0 >= len(seq):
            continue  # candidate index falls outside the sequence; try the next one
        base = seq[idx0]  # base actually present in the window at this candidate index
        if base == ref:
            return idx0, "ref"  # window currently holds the reference allele at this index
        if base == alt:
            return idx0, "alt"  # window already holds the alternate allele at this index

    return None, "none"  # neither candidate index matched REF or ALT


def process_sequences_catalog_variants(
    headers: List[str],
    seqs: List[str],
    args: argparse.Namespace,
    hub: ModelHub,
) -> None:
    """
    Variant-aware scoring for ClinVar/TCGA/1000G-style FASTA batches (SNVs).

    Args:
        headers (List[str]): List of FASTA record headers (with variant information).
        seqs (List[str]): List of FASTA sequences.
        args (argparse.Namespace): Command-line arguments specifying paths and scoring options.
        hub (ModelHub): ModelHub object for model loading.

    Returns:
        None. Results are written to the file specified in `args.variant_out_file`.

    Exceptions:
        Logs warnings and skips records when:
        - Variant position is out of bounds.
        - REF/ALT mismatch occurs.

    Example:
        >>> process_sequences_catalog_variants(headers, sequences, args, hub)
    """
    out_fp = open_output(args.variant_out_file)  # resolve/prepare the output file path for appending
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(args.pos_weight), device=hub.device))  # loss with a user-configurable positive-class weight (only used to satisfy validate_without_sigmoid's signature)

    with out_fp.open("a") as fout:  # open the output file in append mode
        for header_raw, seq in zip(headers, seqs):  # iterate over each FASTA record
            try:
                ph = parse_header_catalog(header_raw)  # parse region/variant/protein/cell-line info from the header
            except Exception as e:
                logging.error("[catalog_variants] Header parse failed: %s | %s", header_raw, e)  # header didn't match the expected catalog format
                continue

            ref, alt = ph.ref, ph.alt  # local copies so strand complementing below doesn't mutate the dataclass
            if ph.strand == "-":
                ref, alt = apply_complement(ref), apply_complement(alt)  # complement alleles to match the minus-strand sequence orientation

            idx0, state = find_variant_index(
                seq=seq,  # the window sequence to search within
                seq_start=ph.start,  # genomic coordinate of the window's first base
                var_pos=ph.var_pos,  # genomic coordinate of the variant
                ref=ref,  # (possibly complemented) reference allele
                alt=alt,  # (possibly complemented) alternate allele
                try_off_by_one=(not bool(args.disable_off_by_one)),  # allow the +/-1 fallback unless the user disabled it
            )
            if state == "none":
                logging.warning("[catalog_variants] Cannot match REF/ALT at site: %s", ph.header_raw)  # neither REF nor ALT found at the expected/nearby index
                if bool(args.strict_ref_match):
                    continue  # in strict mode, skip records we can't confidently locate

            # choose checkpoint name
            if args.model_id_strategy == "from_fasta_stem":
                model_id = Path(args.fasta_sequence_path).stem  # use the input FASTA filename as the checkpoint identifier
            else:
                if not ph.protein or not ph.cell_line:
                    logging.warning("[catalog_variants] Cannot parse protein/cell line for model id: %s", ph.header_raw)  # header didn't yield enough info to name a checkpoint
                    continue
                model_id = f"{ph.protein}_{ph.cell_line}"  # default naming convention: "<PROTEIN>_<CELL_LINE>"

            model = hub.load_bridge(Path(args.model_save_path), model_id)  # load (or fetch cached) BRIDGE checkpoint for this protein/cell-line
            if model is None:
                continue  # no matching checkpoint on disk; skip this record

            # build modified_seq depending on variation_mode and what we see in input
            if args.variation_mode == "before":
                if state == "alt":
                    logging.info("[catalog_variants] Input already ALT at site; scoring as-is in BEFORE: %s", ph.header_raw)  # note that the "reference" window already carries ALT
                modified_seq = seq  # BEFORE mode never mutates the sequence
            else:  # after
                if idx0 is None:
                    modified_seq = seq  # couldn't locate the variant site; fall back to scoring the unmodified sequence
                elif state == "ref":
                    modified_seq = substitute_base(seq, idx0, alt)  # window currently has REF: substitute in ALT to produce the mutated sequence
                else:
                    modified_seq = seq  # window already has ALT (or unmatched); nothing further to substitute

            emb, _ = build_Transformer_embeddings(
                sequences=[modified_seq],  # embed the (possibly mutated) sequence
                transformer_path=str(args.Transformer_path),  # path to the pretrained RNA transformer
                device=hub.device,  # compute on the same device as the BRIDGE model
                k=int(args.k),  # k-mer/stride parameter, configurable via --k for this pipeline
                transpose_to_ch_first=True,  # reorder embedding dims to channel-first for BRIDGE's conv layers
            )
            N = int(emb.shape[0])  # batch size (always 1: single sequence per call)

            attn = np.zeros((N, 101, 103))  # placeholder attention-prior tensor (zeros; unused signal)
            struct = np.zeros((N, 1, 101))  # placeholder RNA structure feature tensor (zeros)
            motif = np.zeros((N, 1, 101))  # placeholder motif-prior feature tensor (zeros)
            biochem = dealwithdata2(modified_seq).transpose([0, 2, 1])  # biochemical one-hot features, reordered to match BRIDGE's expected axis order

            dataset = RBPInferDataset(
                embedding=emb,  # transformer embedding for this sequence
                attn=attn,  # placeholder attention feature
                struct=struct,  # placeholder structure feature
                motif=motif,  # placeholder motif feature
                biochem=biochem,  # computed biochemical feature
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False)  # wrap the single example for the model's forward pass

            score = validate_without_sigmoid(model, hub.device, loader, criterion).item()  # run the model and extract the scalar raw prediction score

            # Output keeps genomic_variants.py style (adds model_id/mode)
            fout.write(
                f"{ph.header_raw}\tmodel_id={model_id}"  # original header plus which checkpoint scored it
                f"\tmode={args.variation_mode}\tPrediction_score:{score:.6f}\n"  # scoring mode (before/after) and the resulting score
            )


# Ribosnitches pipeline
# ---------------------------------------------------------------------------
def _lazy_import_symbol(module_name: str, symbol_name: str):
    """Import `symbol_name` from `module_name` dynamically (helper for optional deps)."""
    module = importlib.import_module(module_name)  # dynamically import the requested module by name (relies on `importlib` being available in scope)
    return getattr(module, symbol_name)  # fetch the requested attribute/symbol from that module


def _extract_cell_lines(header_wo_gt: str) -> Tuple[str, str]:
    """Extract the last two tokens as (cell_line1, cell_line2).

    This matches the user-provided ribosnitches code and assumes the FASTA header
    ends with two cell-line names.
    """
    fields = header_wo_gt.split()  # tokenize the header (with leading '>' already stripped) on whitespace
    if len(fields) < 2:
        raise ValueError("Header has fewer than 2 tokens; cannot extract cell lines.")  # not enough tokens to contain two cell-line names
    return fields[-2], fields[-1]  # by convention, the last two tokens are the candidate cell lines


def _maybe_mutate_sequence_from_header(header: str, seq: str) -> str:
    """Apply ALT substitution based on header tokens (used by ribosnitches-after).

    We use the flexible parser to support headers where the variant token is not
    at a fixed position (e.g. when the header ends with cell-line names).
    """
    var_pos, ref, alt, strand, seq_start = parse_variant_block_flexible(header)  # parse variant info from a header whose trailing tokens are cell-line names rather than fixed fields

    if strand == "-":
        # The user requested: "only complement, do not reverse" because the variant is centered.
        ref, alt = apply_complement(ref), apply_complement(alt)  # complement (but do not reverse) alleles for minus-strand windows

    idx0 = var_pos - seq_start  # convert genomic variant position to a 0-based index within the window
    if idx0 < 0 or idx0 >= len(seq):
        raise ValueError(f"Variant index out of bounds (idx0={idx0}, len={len(seq)})")  # header/coords inconsistent with the sequence length
    if seq[idx0] != ref:
        raise ValueError(f"Reference base mismatch at idx0={idx0}: seq={seq[idx0]} vs ref={ref}")  # sanity check that the window actually carries the expected REF base
    return substitute_base(seq, idx0, alt)  # produce the mutated sequence with ALT substituted at idx0


def run_ribosnitches(
    names: List[str],
    seqs: List[str],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """
    Run ribosnitch scoring using the BRIDGE pipeline components.

    This function scores a sequence using multiple BRIDGE models based on cell-line-specific checkpoints.
    It applies mutation behavior and computes prediction scores for each (record, checkpoint) pair.

    Args:
        names (List[str]): List of FASTA record headers (sequence identifiers).
        seqs (List[str]): List of FASTA sequences to be scored.
        args (argparse.Namespace): Command-line arguments that define scoring options and model paths.
        device (torch.device): The device (CPU/GPU) to perform computation on.

    Returns:
        None. The results are written to the output file specified by `args.variant_out_file`.

    Raises:
        FileNotFoundError: If the provided model path does not exist.
        ValueError: If the variant substitution cannot be applied due to base mismatches.

    Example:
        >>> run_ribosnitches(names, sequences, args, device)
    """

    # Decide whether we should substitute ALT in the window
    do_after = bool(args.ribosnitch_after_variation) or (
        bool(args.ribosnitch) and args.variation_mode == "after"
    )  # mutate the sequence if explicitly forced, or if ribosnitch mode is combined with --variation_mode=after

    out_subdir = "after_mut" if do_after else "before_mut"  # subdirectory name reflecting whether ALT was substituted (informational; used by callers organizing output)
    out_path = open_output(args.variant_out_file)  # resolve/prepare the output file path for appending

    # Loss is only used because validate_without_sigmoid expects it
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0, device=device))  # dummy loss purely to satisfy validate_without_sigmoid's signature

    model_dir = Path(args.model_save_path)  # directory expected to contain per-cell-line BRIDGE checkpoints
    if not model_dir.exists():
        raise FileNotFoundError(f"--model_save_path does not exist: {model_dir}")  # fail fast if the checkpoint directory is missing

    # Reuse BRIDGE checkpoint caching logic from the GWAS branch
    hub = ModelHub(Path(args.Transformer_path), device)  # caches loaded BRIDGE checkpoints so repeated cell lines aren't reloaded from disk

    logging.info("[ribosnitches] Mode=%s | writing to %s", "after" if do_after else "before", out_path)  # log the resolved mode and destination file

    with out_path.open("a") as fout:  # open the output file in append mode
        for header, seq in zip(names, seqs):  # iterate over each FASTA record
            header_wo_gt = header.lstrip(">")  # strip the leading FASTA '>' marker for cleaner output/parsing

            # 1) Determine which checkpoints to apply (by cell line suffix)
            try:
                cell_line1, cell_line2 = _extract_cell_lines(header_wo_gt)  # candidate cell-line names taken from the header's trailing tokens
            except ValueError as e:
                logging.warning("[ribosnitches] %s -> %s (skip)", header_wo_gt, e)  # header didn't have enough tokens to extract cell lines
                continue

            model_files = [
                fn for fn in os.listdir(model_dir)  # scan the checkpoint directory
                if fn.endswith(f"_{cell_line1}.pth") or fn.endswith(f"_{cell_line2}.pth")  # keep only checkpoints matching either candidate cell line
            ]
            if not model_files:
                logging.warning("[ribosnitches] No BRIDGE checkpoints for '%s' (skip)", header_wo_gt)  # neither cell line has a matching checkpoint on disk
                continue

            # 2) Possibly mutate the input sequence
            try:
                seq_in = _maybe_mutate_sequence_from_header(header, seq) if do_after else seq  # apply ALT substitution only when scoring the "after" condition
            except Exception as e:
                logging.warning("[ribosnitches] %s -> cannot apply variant: %s (skip)", header_wo_gt, e)  # mutation failed (bad coords/REF mismatch); skip this record
                continue

            # 3) Build BRIDGE inputs ONCE per record (then reuse across all checkpoints)
            test_emb, _ = build_Transformer_embeddings(
                sequences=[seq_in],  # embed the (possibly mutated) sequence once
                transformer_path=str(args.Transformer_path),  # path to the pretrained RNA transformer
                device=hub.device,  # compute on the same device as the cached BRIDGE checkpoints
                k=1,  # k-mer/stride parameter for tokenization
                transpose_to_ch_first=True,  # reorder embedding dims to channel-first for BRIDGE's conv layers
            )

            # Keep placeholder tensors consistent with the existing GWAS workflow
            N = int(test_emb.shape[0])  # batch size (always 1: single sequence per call)
            test_attn = np.zeros((N, 101, 103))  # placeholder attention-prior tensor (zeros)
            struct = np.zeros((N, 1, 101))  # placeholder RNA structure feature tensor (zeros)
            motif = np.zeros((N, 1, 101))  # placeholder motif-prior feature tensor (zeros)
            bio_chem = dealwithdata2(seq_in).transpose([0, 2, 1])  # biochemical one-hot features for the (possibly mutated) sequence

            dataset = RBPInferDataset(
                embedding=test_emb,  # transformer embedding shared across all matching checkpoints
                attn=test_attn,  # placeholder attention feature
                struct=struct,  # placeholder structure feature
                motif=motif,  # placeholder motif feature
                biochem=bio_chem,  # computed biochemical feature
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False)  # wrap the single example; reused across every matching checkpoint below

            # 4) Score with each matching checkpoint
            for filename in model_files:  # score this same input against every checkpoint whose cell line matched
                stem = Path(filename).stem  # checkpoint name without the .pth extension, used as an identifier in the output
                bridge = hub.load_bridge(model_dir, stem)  # load (or fetch cached) this specific BRIDGE checkpoint
                if bridge is None:
                    continue  # checkpoint failed to load; skip scoring with it

                score = validate_without_sigmoid(bridge, hub.device, loader, criterion).item()  # run the model and extract the scalar raw prediction score
                fout.write(f"{header_wo_gt}\t{stem}\t{score}\n")  # write header, checkpoint name, and score as a TSV line

    logging.info("[ribosnitches] Done.")  # signal completion of the ribosnitch pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    """
    Build and return the argument parser for the script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Variant-aware scoring with BRIDGE. Supports three pipelines:\n"
            "  (1) GWAS windows (legacy variant_aware.py behavior; default)\n"
            "  (2) Ribosnitch scoring (BRIDGE; per-record checkpoint selection)\n"
            "  (3) Catalog variants (ClinVar/TCGA/1000G-style FASTA batches; SNVs)\n"
        )
    )  # top-level parser with a description summarizing the three supported pipelines

    # ------------------------------------------------------------------
    # Common arguments (shared across pipelines)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--variation_mode",
        choices=["before", "after"],  # only these two scoring conditions are supported
        required=True,  # caller must explicitly choose before/after scoring
        help="Score sequences before variation (reference) or after variation (mutated).",
    )
    parser.add_argument("--fasta_sequence_path", required=True, type=Path)  # input FASTA of sequence windows to score
    parser.add_argument("--variant_out_file", required=True, type=Path)  # output file that results are appended to
    parser.add_argument("--Transformer_path", required=True, type=Path)  # pretrained RNA transformer used to compute embeddings
    parser.add_argument("--model_save_path", required=True, type=Path)  # directory containing BRIDGE .pth checkpoints
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")  # compute device; defaults to GPU if one is available

    # ------------------------------------------------------------------
    # Pipeline selection flags
    #
    # Backward compatibility:
    # - If you pass none of these flags, we default to GWAS mode.
    # - `--ribosnitches` is accepted as an alias for `--ribosnitch`.
    # - `--genomic_variants` is accepted as an alias for `--catalog_variants`.
    # ------------------------------------------------------------------
    pipe = parser.add_mutually_exclusive_group(required=False)  # exactly zero or one pipeline flag may be given
    pipe.add_argument(
        "--gwas",
        action="store_true",
        help="Force GWAS window scoring (default if no pipeline flag is provided).",
    )
    pipe.add_argument(
        "--ribosnitch",
        "--ribosnitches",  # alias
        dest="ribosnitch",  # both flags set the same attribute
        action="store_true",
        help="Run ribosnitch scoring (BRIDGE).",
    )
    pipe.add_argument(
        "--catalog_variants",
        "--genomic_variants",  # alias
        dest="catalog_variants",  # both flags set the same attribute
        action="store_true",
        help="Run ClinVar/TCGA/1000G-style FASTA batch scoring (SNVs).",
    )

    # ------------------------------------------------------------------
    # Ribosnitch-specific options
    # ------------------------------------------------------------------
    parser.add_argument(
        "--ribosnitch_after_variation",
        "--ribosnitches_after_variation",  # alias
        dest="ribosnitch_after_variation",
        action="store_true",
        help="Force ALT substitution for ribosnitch scoring, regardless of --variation_mode.",
    )
    parser.add_argument(
        "--ribosnitch_out_dir",
        "--ribosnitches_out_dir",  # alias
        dest="ribosnitch_out_dir",
        type=Path,
        default=Path("./results/ribosnitches"),  # default root directory for ribosnitch outputs
        help="Root output directory for ribosnitch results.",
    )

    # ------------------------------------------------------------------
    # Catalog-variants options (also used by genomic_variants.py)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--model_id_strategy",
        choices=["from_header", "from_fasta_stem"],  # two supported ways to name the checkpoint to load
        default="from_header",  # default: parse "<PROTEIN>_<CELL>" from the FASTA header
        help=(
            "How to choose checkpoint name for catalog variants. "
            "from_header: <PROTEIN>_<CELL> parsed from header; "
            "from_fasta_stem: use FASTA filename stem."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,  # default k-mer/stride of 1 (single-base resolution)
        help="K-mer / stride parameter forwarded to build_Transformer_embeddings (catalog variants branch).",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=2.0,  # default positive-class weight for the (unused-for-scoring) loss
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

    return parser  # fully configured parser ready for parse_args()


def main() -> None:
    """
    Main entry point for variant-aware scoring using BRIDGE. Processes FASTA sequences
    through one of three available pipelines: GWAS, Ribosnitch, or Catalog Variants.
    """
    args = build_argparser().parse_args()  # parse CLI arguments using the parser defined above
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")  # configure console logging format/verbosity for the whole script
    device = torch.device(args.device)  # resolve the requested compute device (cuda or cpu)

    logging.info("Loading FASTA from %s", args.fasta_sequence_path)  # announce which input file is being read
    headers, sequences = read_fasta(args.fasta_sequence_path)  # parse the input FASTA into parallel header/sequence lists

    # Pipeline selection validation.
    # - Default: GWAS (for backward compatibility) when no pipeline flag is provided.
    selected = []  # collects which pipeline flag(s) the user actually passed
    if bool(getattr(args, "gwas", False)):
        selected.append("gwas")  # explicit --gwas flag requested
    if bool(getattr(args, "ribosnitch", False)) or bool(getattr(args, "ribosnitch_after_variation", False)):
        selected.append("ribosnitch")  # ribosnitch mode requested either directly or implied by --ribosnitch_after_variation
    if bool(getattr(args, "catalog_variants", False)):
        selected.append("catalog_variants")  # catalog-variants mode requested

    if len(selected) > 1:
        raise SystemExit(f"Conflicting pipeline flags: {selected}. Please choose only one.")  # more than one pipeline requested; ambiguous, so abort

    pipeline = selected[0] if selected else "gwas"  # default to GWAS mode when no pipeline flag was given

    hub = ModelHub(args.Transformer_path, device)  # shared checkpoint-caching hub used by whichever pipeline runs

    if pipeline == "catalog_variants":
        logging.info("Running catalog variants pipeline (%s_variation)", args.variation_mode)  # announce the selected pipeline and mode
        process_sequences_catalog_variants(headers, sequences, args, hub)  # run ClinVar/TCGA/1000G-style scoring
    elif pipeline == "ribosnitch":
        logging.info("Running ribosnitch pipeline (%s_variation)", args.variation_mode)  # announce the selected pipeline and mode
        run_ribosnitches(headers, sequences, args, device)  # run per-cell-line ribosnitch scoring
    else:
        logging.info("Running GWAS pipeline (%s_variation)", args.variation_mode)  # announce the selected pipeline and mode
        process_sequences_gwas(headers, sequences, args, hub)  # run the default single-checkpoint GWAS scoring

    logging.info("Finished. Results appended to %s", args.variant_out_file)  # final confirmation of where results were written


if __name__ == "__main__":
    main()  # run the CLI entry point when executed as a script

