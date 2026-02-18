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
    """
    Read a FASTA file (supports wrapped / multi-line sequences).

    A record starts with a header line beginning with ``>`` followed by one or more
    sequence lines. Wrapped sequences are concatenated and returned upper-cased.

    :param fasta_path:
        Path to a FASTA file on disk.
    :type fasta_path: pathlib.Path

    :returns:
        A tuple ``(headers, seqs)``.
        - ``headers``: header lines (including the leading ``>``), one per record.
        - ``seqs``: concatenated, upper-cased sequences, one per record.
    :rtype: tuple[list[str], list[str]]

    :raises FileNotFoundError:
        If ``fasta_path`` does not exist.
    :raises OSError:
        If the file cannot be opened/read.

    .. note::
        - Empty/blank lines are ignored.
        - No alphabet validation is performed here; downstream code may validate A/C/G/T/U/N.
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
    """Parse a FASTA header and extract variant coordinates (legacy rule).

    This is the original parsing rule used by the GWAS/BRIDGE branch, kept intact
    for backward compatibility.

    Expected header (example)
    -------------------------
    >variant_1 chr1:27891903-27892003(-)[...]{NA} 27891953:T>A ...

    Token usage in the original implementation
    ------------------------------------------
    fields = fasta_header.lstrip('>').split()

    - fields[1] is the region token like: chr1:27891903-27892003(-)[...]
      We parse:
        * strand    : text between '(' and ')', e.g. '+' or '-'
        * seq_start : the window start coordinate, the first number after ':'

    - fields[-2] is the variant token like: 27891953:T>A
      We parse:
        * variant_pos : the genomic position (integer)
        * ref_base    : REF base (string)
        * alt_base    : ALT base (string)

    Returns
    -------
    variant_pos : int
        Genomic coordinate of the variant.
    ref_base : str
        Reference allele base (A/C/G/T).
    alt_base : str
        Alternate allele base (A/C/G/T).
    strand : str
        '+' or '-' parsed from the region token.
    seq_start : int
        Genomic coordinate of the window start (used to compute 0-based index into the sequence).

    Raises
    ------
    ValueError
        If the header does not contain enough tokens to parse with the above rules.
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
    """
    for tok in fields:
        if _VARIANT_TOKEN_RE.match(tok):
            return tok
    return None


def _find_region_token(fields: List[str]) -> Optional[str]:
    """Find a region token like 'chr19:11120155-11120255(+)[...]' anywhere in a split header."""
    for tok in fields:
        if ":" in tok and "-" in tok and "(" in tok and ")" in tok:
            # This is intentionally permissive; the exact bracket payload can vary.
            return tok
    return None


def parse_variant_block_flexible(fasta_header: str) -> Tuple[int, str, str, str, int]:
    """Parse a FASTA header and extract variant coordinates (robust rule).

    This parser is designed for headers where the variant token is not necessarily
    at a fixed index (e.g. when the last two tokens are cell-line names).

    It is used by the ribosnitches-after branch, but can also serve as a fallback
    when `parse_variant_block()` fails.

    Returns (same fields as `parse_variant_block`).
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
    """Caches heavy models & tokenizers for the GWAS/BRIDGE workflow.

    The tokenizer/transformer are loaded once and held for reuse. BRIDGE checkpoints
    are cached by filename stem to avoid repeated disk loads in long FASTA batches.
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
