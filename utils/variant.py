"""
Variant-aware inference utilities for BRIDGE (GWAS / ribosnitches-style workflows).

This module implements I/O and parsing helpers used by BRIDGE variant scoring pipelines.
It focuses on FASTA inputs whose headers encode variant coordinates and alleles, and
provides utilities to (1) reconstruct the alternate-allele sequence window, and (2)
cache heavy Transformer/BRIDGE models for high-throughput scoring.

Key ideas
---------
1) FASTA parsing with wrapped sequences
   :func:`read_fasta` supports standard multi-line/wrapped FASTA sequences. Each record
   begins with a header line starting with '>' and is followed by one or more sequence
   lines. Sequence lines are concatenated and returned in upper-case.
   
2) Variant metadata encoded in headers

   Two parsers are provided:

   - :func:`parse_variant_block` (legacy / fixed token positions)
     Assumes the region token is at ``fields[1]`` and the variant token is at
     ``fields[-2]``. This matches the original GWAS implementation and is kept
     for backward compatibility.

   - :func:`parse_variant_block_flexible` (robust / token search)
     Searches the header tokens for:

     * a region token that contains ``:``, ``-``, ``(``, and ``)``
     * a variant token matching
       ``^\\d+:[ACGT]>[ACGT]$`` (case-insensitive)

     This is intended for headers where trailing tokens vary (for example,
     extra annotations or cell-line suffixes), and is often used by
     ribosnitches-style inputs.

   Both parsers return the same 5-tuple::

       (variant_pos, ref_base, alt_base, strand, seq_start)

   where:

   - ``variant_pos``: genomic coordinate of the SNV
   - ``ref_base``: reference allele base (A/C/G/T)
   - ``alt_base``: alternate allele base (A/C/G/T)
   - ``strand``: ``'+'`` or ``'-'`` parsed from the region token
   - ``seq_start``: genomic start of the provided sequence window

3) Coordinate conversion: genomic -> window index

   Given::

       idx0 = variant_pos - seq_start

   the index is 0-based into the sequence window returned by
   :func:`read_fasta`.

   Most pipelines then validate that::

       seq[idx0] == ref_base

   (or its complement for the ``'-'`` strand) before substituting the
   alternate base.

4) Strand handling and complements

   This module provides:

   - :data:`COMPLEMENT` mapping for DNA letters ``{A, T, C, G, N}``
   - :func:`apply_complement` to map one base to its Watson-Crick complement
   - :func:`substitute_base` to write an alternate allele at a 0-based window index

   Important:

   - If your sequence window is given on the ``'-'`` strand, typical pipelines
     usually do one of the following:

     * store the window already reverse-complemented
       (then ``ref_base`` / ``alt_base`` can be used directly), or
     * store the window in genomic ``'+'`` orientation
       (then allele complementation may be required)

   - This module only provides the complement primitive. The exact policy should
     be enforced by the caller (that is, the caller decides whether to
     complement ``ref_base`` / ``alt_base`` when ``strand == '-'``).

   - The complement mapping uses ``'T'`` (DNA). If your windows are RNA
     (``'U'``), consider extending ``COMPLEMENT`` with ``{"U": "A"}`` and
     adjusting parsing/validation accordingly.

5) High-throughput model reuse via :class:`ModelHub`

   BRIDGE variant scoring typically requires:

   - a tokenizer + Transformer encoder (BERT-like) for k-mer embeddings
   - a BRIDGE checkpoint per experiment/model name

   :class:`ModelHub` caches these heavy components:

   - loads tokenizer and Transformer once from ``transformer_path``
   - caches BRIDGE checkpoints by ``filename_stem`` to avoid repeated disk I/O
     when a FASTA file contains many records spanning multiple models

Constants
---------
COMPLEMENT
    DNA Watson–Crick complement mapping used by :func:`apply_complement`.

RIBOSNITCHES_MAX_LEN
    Default fixed window length (101) used by ribosnitches-derived pipelines.
    Many downstream feature builders assume length 101; if you deviate, ensure you
    also update shape-dependent modules.

I/O helpers
-----------
read_fasta(fasta_path)
    Read headers and sequences from a FASTA file. Supports wrapped sequences.

open_output(out_path)
    Create parent directories and return a `Path` suitable for writing/appending.

Variant utilities
-----------------
parse_variant_block(fasta_header)
    Fixed-position parser (legacy GWAS rule).

parse_variant_block_flexible(fasta_header)
    Search-based parser (robust to header token drift).

apply_complement(base)
    Complement a single base (A/T/C/G), returning unchanged for unknown letters.

substitute_base(seq, pos0, alt)
    Replace the base at 0-based index pos0 with alt and return the new string.

ModelHub
--------
ModelHub(transformer_path, device)
    Loads tokenizer/Transformer once, and caches BRIDGE checkpoints.

ModelHub.load_bridge(model_dir, filename_stem)
    Load `<model_dir>/<filename_stem>.pth` into a BRIDGE model on the hub device.
    Returns None if the checkpoint is missing.

Common failure modes and recommendations
---------------------------------------
- Header format drift:
    If parse_variant_block() raises ValueError or yields wrong tokens, switch to
    parse_variant_block_flexible() or call it as a fallback.

- Variant token alphabet:
    The default regex accepts only A/C/G/T. If your headers can contain 'U'
    (e.g., A>U), extend `_VARIANT_TOKEN_RE`.

- Bounds checking:
    Always check `0 <= (variant_pos - seq_start) < len(window_seq)` before indexing.

- Ref allele validation:
    Before writing alt allele, validate that the observed base matches the expected
    ref allele (possibly after complementing for '-' strand depending on your policy).

Logging
-------
This module uses the standard `logging` module. The caller should configure logging
handlers/levels (e.g., via `logging.basicConfig`) if runtime diagnostics are desired.

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
from utils.BRIDGE import BRIDGE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMPLEMENT: Dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
RIBOSNITCHES_MAX_LEN: int = 101  # matches the shapes in the provided ribosnitches code


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def read_fasta(fasta_path: Path) -> Tuple[List[str], List[str]]:
    """Read a FASTA file (supports wrapped / multi-line sequences).

    This reader supports multi-line (wrapped) FASTA sequences. Each record begins with a
    header line starting with '>' and is followed by one or more sequence lines. Sequence
    lines are concatenated and returned in upper-case.

    Args:
        fasta_path (Path):
            Path to a FASTA file on disk.

    Returns:
        Tuple[List[str], List[str]]:
            A tuple ``(headers, seqs)`` where:

            - **headers**: List of header lines (including the leading '>'), one per record.
            - **seqs**: List of concatenated, upper-cased sequences, one per record.

    Raises:
        FileNotFoundError:
            If ``fasta_path`` does not exist.
        OSError:
            If the file cannot be opened/read.

    Notes:
        - Empty/blank lines are ignored.
        - This function does not validate alphabet (A/C/G/T/U/N). If you need strict
          validation, do it downstream.
    """
    headers: List[str] = []
    seqs: List[str] = []

    cur_header: Optional[str] = None
    cur_seq_parts: List[str] = []

    with open(fasta_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_header is not None:
                    headers.append(cur_header)
                    seqs.append("".join(cur_seq_parts).upper())
                cur_header = line
                cur_seq_parts = []
            else:
                cur_seq_parts.append(line)

    if cur_header is not None:
        headers.append(cur_header)
        seqs.append("".join(cur_seq_parts).upper())

    return headers, seqs


def open_output(out_path: os.PathLike | str) -> Path:
    """Create parent directories and return a `Path` for appending outputs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


# ---------------------------------------------------------------------------
# Variant utilities
# ---------------------------------------------------------------------------
def parse_variant_block(fasta_header: str) -> Tuple[int, str, str, str, int]:
    """Parse a FASTA header and extract variant coordinates.

    This is the original parsing rule used by the GWAS branch, kept intact
    for backward compatibility.

    Expected header (example):
        .. code-block:: text

            >variant_1 chr1:27891903-27892003(-)[...]{NA} 27891953:T>A ...

    Token usage in the original implementation:
        .. code-block:: text

            fields = fasta_header.lstrip('>').split()

        - ``fields[1]`` is the region token like: ``chr1:27891903-27892003(-)[...]``
          We parse:
            * **strand**: text between '(' and ')', e.g. '+' or '-'
            * **seq_start**: window start coordinate, the first number after ':'

        - ``fields[-2]`` is the variant token like: ``27891953:T>A``
          We parse:
            * **variant_pos**: genomic position (int)
            * **ref_base**: reference base (str)
            * **alt_base**: alternate base (str)

    Args:
        fasta_header (str):
            FASTA header line including the leading '>'.

    Returns:
        Tuple[int, str, str, str, int]:
            ``(variant_pos, ref_base, alt_base, strand, seq_start)`` where:

            - **variant_pos** (int): Genomic coordinate of the variant.
            - **ref_base** (str): Reference allele base (A/C/G/T).
            - **alt_base** (str): Alternate allele base (A/C/G/T).
            - **strand** (str): '+' or '-' parsed from the region token.
            - **seq_start** (int): Genomic coordinate of the window start (used to compute
              0-based index into the sequence).

    Raises:
        ValueError:
            If the header does not contain enough tokens to parse with this rule
            (e.g., fewer than 3 whitespace-separated fields).

    Notes:
        - This parser assumes **fixed token positions**. If your headers contain extra
          trailing tokens (e.g., cell line names), consider using
          ``parse_variant_block_flexible`` as a fallback.
    """
    fields = fasta_header.lstrip(">").split()
    if len(fields) < 3:
        raise ValueError("Unexpected FASTA header format")

    region = fields[1]                             # chr1:27891903-27892003(-)[...]
    strand = region.split("(")[1].split(")")[0]    # + / -
    seq_start = int(region.split(":")[1].split("-")[0])

    var_info = fields[-2]                          # 27891953:T>A
    variant_pos = int(var_info.split(":")[0])
    ref_base, alt_base = var_info.split(":")[1].split(">")

    return variant_pos, ref_base, alt_base, strand, seq_start


_VARIANT_TOKEN_RE = re.compile(r"^\d+:[ACGT]>[ACGT]$", re.IGNORECASE)


def _find_variant_token(fields: List[str]) -> Optional[str]:
    """
    Find a variant token like '11120205:T>C' in a split FASTA header.

    Args:
        fields (List[str]):
            Tokens from `fasta_header.lstrip('>').split()`.

    Returns:
        Optional[str]:
            The first token matching the variant pattern (case-insensitive),
            or None if not found.

    Notes:
        - Regex is ``^\\d+:[ACGT]>[ACGT]$`` (case-insensitive).
        - 'U' is not accepted by this regex. If you expect RNA tokens like ``A>U``,
          extend the pattern accordingly.
    """
    for tok in fields:
        if _VARIANT_TOKEN_RE.match(tok):
            return tok
    return None


def _find_region_token(fields: List[str]) -> Optional[str]:
    """Find a region token like ``chr_num:start-end(strand)[...]`` in a split header.

    Args:
        fields (List[str]):
            Tokens from a FASTA header split by whitespace.

    Returns:
        Optional[str]:
            The first token that contains ':', '-', '(' and ')' (heuristic match),
            or ``None`` if not found.
    """
    for tok in fields:
        if ":" in tok and "-" in tok and "(" in tok and ")" in tok:
            # This is intentionally permissive; the exact bracket payload can vary.
            return tok
    return None


def parse_variant_block_flexible(fasta_header: str) -> Tuple[int, str, str, str, int]:
    """Parse a FASTA header and extract variant coordinates (flexible token search rule).

    This parser is designed for headers where the variant token is not necessarily at a fixed index (e.g. when the last two tokens are cell-line names). It is used by the ribosnitches-after branch, but can also serve as a fallback when `parse_variant_block()` fails.

    Parsing strategy:
        1) Split header into tokens:
           ``fields = fasta_header.lstrip('>').split()``
        2) Locate:
           - region token: a token containing ':', '-', '(' and ')'
           - variant token: matches ``^\\d+:[ACGT]>[ACGT]$`` (case-insensitive)
        3) Extract:
           - strand and seq_start from region token
           - variant_pos/ref/alt from variant token

    Args:
        fasta_header (str):
            FASTA header line including the leading '>'.

    Returns:
        Tuple[int, str, str, str, int]:
            ``(variant_pos, ref_base, alt_base, strand, seq_start)`` where each field has the same meaning as in ``parse_variant_block``.
    """
    fields = fasta_header.lstrip(">").split()
    if len(fields) < 3:
        raise ValueError("Unexpected FASTA header format")

    region = _find_region_token(fields)
    var_info = _find_variant_token(fields)

    if region is None or var_info is None:
        raise ValueError("Cannot locate region token and/or variant token in FASTA header")

    strand = region.split("(")[1].split(")")[0]
    seq_start = int(region.split(":")[1].split("-")[0])

    variant_pos = int(var_info.split(":")[0])
    ref_base, alt_base = var_info.split(":")[1].split(">")

    return variant_pos, ref_base, alt_base, strand, seq_start


def apply_complement(base: str) -> str:
    """Return Watson-Crick complement for A/T/C/G; otherwise return `base` unchanged."""
    return COMPLEMENT.get(base, base)


def substitute_base(seq: str, pos0: int, alt: str) -> str:
    """Return a new sequence where `seq[pos0]` is replaced by `alt`.

    Parameters
    ----------
    seq : str
        Input sequence (window).
    pos0 : int
        0-based index *into the window*.
    alt : str
        Alternate allele to write at `pos0`.

    Notes
    -----
    - If `seq[pos0]` already equals `alt`, we return the original string.
    """
    if seq[pos0] == alt:
        return seq
    seq_list = list(seq)
    seq_list[pos0] = alt
    return "".join(seq_list)


# ---------------------------------------------------------------------------
# Model loaders (with caching)
# ---------------------------------------------------------------------------
class ModelHub:
    """Caches heavy models & tokenizers for the GWAS workflow.

    The tokenizer/transformer are loaded once and held for reuse. BRIDGE checkpoints
    are cached by filename stem to avoid repeated disk loads in long FASTA batches.
    
    Attributes:
        device (torch.device):
            Inference device (CPU/CUDA).
        tokenizer (BertTokenizer):
            Tokenizer loaded from ``transformer_path``.
        transformer (BertModel):
            Transformer encoder loaded from ``transformer_path`` and set to ``eval()``.
        bridge_cache (Dict[str, BRIDGE]):
            Cache mapping ``filename_stem`` to loaded BRIDGE models.
    """

    def __init__(self, transformer_path: Path, device: torch.device) -> None:
        """Initialize the hub.

        Parameters
        ----------
        transformer_path : Path
            Path to a directory compatible with `BertTokenizer.from_pretrained`
            and `BertModel.from_pretrained`.
        device : torch.device
            CPU or CUDA device for inference.
        """
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(transformer_path, do_lower_case=False)
        self.transformer = BertModel.from_pretrained(transformer_path).to(device).eval()
        self.bridge_cache: Dict[str, BRIDGE] = {}

    def load_bridge(self, model_dir: Path, filename_stem: str) -> Optional[BRIDGE]:
        """Load (or reuse cached) BRIDGE checkpoint: `<model_dir>/<filename_stem>.pth`.

        Parameters
        ----------
        model_dir : Path
            Directory containing `.pth` checkpoints.
        filename_stem : str
            Stem used to construct checkpoint name.

        Returns
        -------
        Optional[BRIDGE]
            Loaded `BRIDGE` model in `.eval()` mode, or None if the file does not exist.
        """
        if filename_stem in self.bridge_cache:
            return self.bridge_cache[filename_stem]

        model_file = model_dir / f"{filename_stem}.pth"
        if not model_file.exists():
            logging.warning("Model not found for %s → skip", filename_stem)
            return None

        model = BRIDGE().to(self.device)
        model.load_state_dict(torch.load(model_file, map_location=self.device))
        model.eval()
        self.bridge_cache[filename_stem] = model
        return model
