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

from __future__ import annotations  # allows forward-referenced/PEP 604 type hints (e.g. `os.PathLike | str`) on older Python

import argparse  # imported for CLI-style extension; unused by this module's active code
import logging  # used to warn when a requested BRIDGE checkpoint is missing
import os  # path-like type hints and filesystem path handling
import re  # regex matching for variant/region tokens in FASTA headers
from dataclasses import dataclass  # imported for potential struct-like extension; unused directly here
from pathlib import Path  # path manipulation for FASTA/checkpoint file locations
from typing import List, Tuple, Dict, Optional, Callable  # type hints for the functions and class below

import numpy as np  # imported for array ops used by callers of this module; not directly used here
import torch  # model device placement, checkpoint loading, and eval-mode inference
from torch import nn  # imported for type/extension purposes; not directly used here
from torch.utils.data import DataLoader  # imported for callers batching variant records; not directly used here
from transformers import BertTokenizer, BertModel  # loads the pretrained tokenizer + Transformer encoder used for embeddings
from utils.BRIDGE import BRIDGE  # the BRIDGE model class, instantiated and checkpoint-loaded by ModelHub

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMPLEMENT: Dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}  # DNA Watson-Crick complement lookup, used when a '-' strand window needs allele complementing
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
    headers: List[str] = []  # one header string per record, collected in file order
    seqs: List[str] = []  # one concatenated, upper-cased sequence per record, aligned with `headers`

    cur_header: Optional[str] = None  # header of the record currently being accumulated (None until the first '>' line)
    cur_seq_parts: List[str] = []  # wrapped sequence lines for the record currently being accumulated

    with open(fasta_path, "r", encoding="utf-8") as f:  # open the FASTA file as UTF-8 text
        for raw in f:  # iterate the file line by line
            line = raw.strip()  # drop the trailing newline (and any surrounding whitespace)
            if not line:  # skip blank lines
                continue
            if line.startswith(">"):  # a new record's header line
                if cur_header is not None:  # flush the previous record (if any) before starting a new one
                    headers.append(cur_header)  # store the just-finished record's header
                    seqs.append("".join(cur_seq_parts).upper())  # store the just-finished record's concatenated, upper-cased sequence
                cur_header = line  # start tracking the new record's header
                cur_seq_parts = []  # reset the sequence-line accumulator for the new record
            else:
                cur_seq_parts.append(line)  # accumulate this wrapped sequence line for the current record

    if cur_header is not None:  # flush the final record after the loop ends (no trailing '>' to trigger it)
        headers.append(cur_header)  # store the last record's header
        seqs.append("".join(cur_seq_parts).upper())  # store the last record's concatenated, upper-cased sequence

    return headers, seqs  # parallel lists: headers[i] corresponds to seqs[i]


def open_output(out_path: os.PathLike | str) -> Path:
    """Create parent directories and return a `Path` for appending outputs."""
    out_path = Path(out_path)  # normalize the input to a pathlib.Path
    out_path.parent.mkdir(parents=True, exist_ok=True)  # ensure the containing directory tree exists
    return out_path  # path ready to be opened for writing/appending


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
    fields = fasta_header.lstrip(">").split()  # drop the leading '>' and split the header into whitespace-separated tokens
    if len(fields) < 3:  # this fixed-position parser needs at least a region token and a variant token
        raise ValueError("Unexpected FASTA header format")  # header is too short to match the expected layout

    region = fields[1]                             # chr1:27891903-27892003(-)[...]
    strand = region.split("(")[1].split(")")[0]    # + / -
    seq_start = int(region.split(":")[1].split("-")[0])  # genomic start coordinate of the sequence window (text after ':' and before '-')

    var_info = fields[-2]                          # 27891953:T>A
    variant_pos = int(var_info.split(":")[0])  # genomic coordinate of the variant (text before ':')
    ref_base, alt_base = var_info.split(":")[1].split(">")  # reference and alternate alleles, split on '>'

    return variant_pos, ref_base, alt_base, strand, seq_start  # (position, ref, alt, strand, window start)


_VARIANT_TOKEN_RE = re.compile(r"^\d+:[ACGT]>[ACGT]$", re.IGNORECASE)  # matches a "<pos>:<ref>><alt>" variant token, e.g. "27891953:T>A"


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
    for tok in fields:  # scan header tokens in order
        if _VARIANT_TOKEN_RE.match(tok):  # check whether this token looks like "<pos>:<ref>><alt>"
            return tok  # first matching token is taken as the variant descriptor
    return None  # no token in this header matched the variant pattern


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
    for tok in fields:  # scan header tokens in order
        if ":" in tok and "-" in tok and "(" in tok and ")" in tok:  # heuristic: a region token carries all four of these characters
            # This is intentionally permissive; the exact bracket payload can vary.
            return tok  # first matching token is taken as the region descriptor
    return None  # no token in this header matched the region heuristic


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
    fields = fasta_header.lstrip(">").split()  # drop the leading '>' and split the header into whitespace-separated tokens
    if len(fields) < 3:  # need at least enough tokens to plausibly contain a region and a variant token
        raise ValueError("Unexpected FASTA header format")  # header is too short to be well-formed

    region = _find_region_token(fields)  # locate the region token by content rather than fixed position
    var_info = _find_variant_token(fields)  # locate the variant token by regex rather than fixed position

    if region is None or var_info is None:  # both tokens are required to compute the return values
        raise ValueError("Cannot locate region token and/or variant token in FASTA header")  # neither fallback heuristic found what it needed

    strand = region.split("(")[1].split(")")[0]  # strand symbol between the parentheses in the region token
    seq_start = int(region.split(":")[1].split("-")[0])  # genomic start coordinate of the sequence window

    variant_pos = int(var_info.split(":")[0])  # genomic coordinate of the variant
    ref_base, alt_base = var_info.split(":")[1].split(">")  # reference and alternate alleles, split on '>'

    return variant_pos, ref_base, alt_base, strand, seq_start  # (position, ref, alt, strand, window start)


def apply_complement(base: str) -> str:
    """Return Watson-Crick complement for A/T/C/G; otherwise return `base` unchanged."""
    return COMPLEMENT.get(base, base)  # look up the complement; pass through unrecognized letters as-is


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
    if seq[pos0] == alt:  # already the alternate allele: nothing to change
        return seq  # return the original string unmodified (avoids an unnecessary copy)
    seq_list = list(seq)  # strings are immutable in Python, so convert to a mutable list of characters
    seq_list[pos0] = alt  # overwrite the base at the target window position with the alternate allele
    return "".join(seq_list)  # reassemble the mutated sequence back into a string


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
        self.device = device  # remember the target device for tensors/models loaded through this hub
        self.tokenizer = BertTokenizer.from_pretrained(transformer_path, do_lower_case=False)  # load the k-mer tokenizer once, case-sensitive (nucleotide tokens are uppercase)
        self.transformer = BertModel.from_pretrained(transformer_path).to(device).eval()  # load the pretrained Transformer encoder, move to device, and fix it in inference mode
        self.bridge_cache: Dict[str, BRIDGE] = {}  # lazily-populated cache of loaded BRIDGE models, keyed by checkpoint filename stem

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
        if filename_stem in self.bridge_cache:  # avoid reloading a checkpoint already read from disk
            return self.bridge_cache[filename_stem]  # return the previously cached model instance

        model_file = model_dir / f"{filename_stem}.pth"  # expected checkpoint path for this experiment/model name
        if not model_file.exists():  # nothing to load for this stem
            logging.warning("Model not found for %s → skip", filename_stem)  # surface the miss so batch scoring can skip it visibly
            return None  # signal to the caller that this model is unavailable

        model = BRIDGE().to(self.device)  # instantiate a fresh BRIDGE model on the target device
        model.load_state_dict(torch.load(model_file, map_location=self.device))  # load the trained weights from the checkpoint file
        model.eval()  # fix the model in inference mode (disables dropout, etc.)
        self.bridge_cache[filename_stem] = model  # cache the loaded model so subsequent calls skip the disk read
        return model  # ready-to-use BRIDGE model for this experiment/model name
