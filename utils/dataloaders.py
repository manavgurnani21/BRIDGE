"""
FASTA + structure dataset readers (3-line records).

This module implements strict parsers for a simple FASTA-like format where each record
contains:

1) a header line starting with ``>``
2) a nucleotide sequence line
3) a structure-score line containing comma-separated numeric tokens (one per base)

It is designed for classification datasets where each record has both sequence and
per-position structure features and where labels are assigned at the file level
(e.g., all sequences in one file are negatives, the other are positives).

Who this is for
---------------
- Users preparing binary classification datasets from paired ``(neg, pos)`` files.
- Pipelines that need both sequence strings and aligned per-base structure scores
  (kept as raw strings for downstream parsing).

This module does not build tensors, tokenize sequences, or pad/truncate lengths; it only
reads and validates file structure.

File format specification
-------------------------
Record layout (strict)
    Each record occupies exactly **3 non-empty lines**:

    1. Header line starting with ``>``
    2. Sequence line (uppercase recommended)
    3. Structure line: comma-separated tokens, one per sequence position

Example
    .. code-block:: text

        >record_001 optional metadata
        ACGTACGT
        0.12,0.03,0.50,0.10,0.22,0.18,0.07,0.09

Parsing rules
    - Blank/empty lines are ignored.
    - The total number of non-empty lines must be a multiple of 3.
    - Header lines must begin with ``>``.
    - Sequence validation uses the regex ``r"[ACGTUN]+"`` (uppercase only):
      accepts DNA (``T``), RNA (``U``), and ``N`` for unknown.
    - Structure lines are *not* converted to floats here. They are returned as raw strings.
      A length check is performed by comparing:
      ``len(sequence)`` vs. ``len(struct_str.split(","))``.

Returned data conventions
-------------------------
The readers return:

- ``sequences``: sequence strings (variable length allowed across records).
- ``structs``: raw structure strings (comma-separated; length matches each sequence).
- ``labels``: float32 array of shape ``(N, 1)`` with file-level constant label values.

Function summary
----------------
``read_fasta_with_struct_single(path, label_val)``
    Reads one file and assigns the same ``label_val`` to all records.

``read_fasta(neg_path, pos_path)``
    Reads a negative file (label 0) and a positive file (label 1), then concatenates:

    - ordering is ``[pos] + [neg]`` by default
    - outputs are ``np.ndarray`` objects:
      ``sequences`` and ``structs`` have dtype ``object``

How to use
----------
Read one file:

.. code-block:: python

    from readers import read_fasta_with_struct_single
    seqs, structs, y = read_fasta_with_struct_single("neg.fa", label_val=0)

Read paired neg/pos files:

.. code-block:: python

    from readers import read_fasta
    sequences, structs, labels = read_fasta("neg.fa", "pos.fa")
    # sequences: (N,), dtype object
    # structs:   (N,), dtype object
    # labels:    (N, 1), float32

Parse structure strings into numeric arrays downstream:

.. code-block:: python

    import numpy as np
    struct_vec = np.array(structs[0].split(","), dtype=np.float32)  # shape (L,)

Notes and caveats
-----------------
- Uppercase-only validation:
  If sequences may contain lowercase letters, either uppercase them before writing the file
  or modify the reader to apply ``seq = seq.upper()`` prior to validation.
- Variable-length sequences:
  This reader allows different record lengths across the file(s) as long as each record’s
  sequence length matches its structure-token count. If your model requires fixed length,
  pad/truncate consistently downstream.

"""

from typing import List, Tuple
import numpy as np
import re

def read_fasta_with_struct_single(
    path: str,
    label_val: int,
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Read a single FASTA-like file containing sequence + per-base structure scores.

    File format (strict):
        Each record occupies exactly 3 non-empty lines:
          1) Header line starting with '>' (FASTA-style).
          2) Sequence line: a string of nucleotide characters.
          3) Structure line: comma-separated numeric tokens, one per nucleotide position.

    Example record:

    .. code-block:: text

        >record_001 optional metadata
        ACGTACGT
        0.12,0.03,0.50,0.10,0.22,0.18,0.07,0.09

    Parsing behavior:
        - Empty/blank lines are ignored.
        - The total number of non-empty lines must be a multiple of 3.
        - Headers are validated to start with '>'.
        - Sequences are validated by regex: r"[ACGTUN]+" (uppercase only).
          This accepts both DNA ('T') and RNA ('U') plus 'N' for unknown.
        - The structure line is kept as a raw string (e.g., "0.1,0.2,...") because some
          downstream code expects to call `.split(',')`. Length validation is performed
          by comparing len(sequence) to len(struct_str.split(',')).

    Args:
        path (str):
            Path to the input FASTA-like file on disk.
        label_val (int):
            Label value assigned to all records in this file (e.g., 0 for negative, 1 for positive).
            The returned label array will be float32.

    Returns:
        Tuple[List[str], List[str], np.ndarray]:
            - sequences: List of sequence strings, length N (number of records). Each string has length L_i.
            - structs: List of raw comma-separated structure score strings, length N. Each has L_i items.
            - labels: Array of shape (N, 1), dtype float32, filled with label_val.

    Raises:
        FileNotFoundError:
            If `path` does not exist or cannot be opened.
        ValueError:
            - If the non-empty line count is not a multiple of 3.
            - If a header line does not start with '>'.
            - If a sequence contains characters outside [ACGTUN] (uppercase).
            - If sequence length does not match the number of comma-separated structure values.

    Notes:
        - This function does not parse structure scores into floats; it only validates length.
          If you need numeric arrays, parse each `struct_str` downstream:
              np.array(struct_str.split(","), dtype=np.float32)
        - If your files contain lowercase letters, you may want to `.upper()` the sequence
          before validation (not done here to keep behavior explicit).
    """
    sequences: List[str] = []
    structs: List[str] = []

    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) % 3 != 0:
        raise ValueError(f"{path}: line count is not a multiple of 3")

    for i in range(0, len(lines), 3):
        hdr, seq, struct_str = lines[i], lines[i+1], lines[i+2]
        if not hdr.startswith(">"):
            raise ValueError(f"{path}: invalid header at line {i}")
        if not re.fullmatch(r"[ACGTUN]+", seq):
            raise ValueError(f"{path}: invalid sequence at block {i}")

        arr_len = len(struct_str.split(","))
        if len(seq) != arr_len:
            raise ValueError(f"{path}: length mismatch (seq={len(seq)}, struct={arr_len})")

        sequences.append(seq)
        structs.append(struct_str)

    labels = np.full((len(sequences), 1), label_val, dtype=np.float32)
    return sequences, structs, labels


def read_fasta(
    neg_path: str, 
    pos_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read paired negative/positive FASTA-like files (sequence + structure) and concatenate them.

    This is a dataset convenience wrapper that:
        1) Reads a negative file assigning label=0.
        2) Reads a positive file assigning label=1.
        3) Concatenates results into a single (sequences, structs, labels) triple.

    Input file format:
        Both `neg_path` and `pos_path` must follow the same strict 3-line-per-record format
        described in `read_fasta_with_struct_single`.

    Output format:
        Matches a common "read_csv()" style used in some pipelines:
          - sequences: np.ndarray of Python strings (dtype=object), shape (N,)
          - structs:   np.ndarray of Python strings (dtype=object), shape (N,)
          - labels:    np.ndarray float32, shape (N, 1)

    Ordering:
        By default, returned arrays are ordered as:
            [all positive records] + [all negative records]
        This is consistent with the code:
            sequences = seq_pos + seq_neg
        If you need a different ordering or shuffling, do it downstream.

    Args:
        neg_path (str):
            Path to the negative-class file.
        pos_path (str):
            Path to the positive-class file.

    Returns:
        sequences (np.ndarray):
            Array of shape (N,), dtype object, containing sequence strings.
        structs (np.ndarray):
            Array of shape (N,), dtype object, containing raw comma-separated structure score strings.
        labels (np.ndarray):
            Array of shape (N, 1), dtype float32, containing labels (1 for pos, 0 for neg).

    Raises:
        FileNotFoundError:
            If either file cannot be opened.
        ValueError:
            Propagated from `read_fasta_with_struct_single` for format/validation errors.

    Notes:
        - If you require randomized mixing of positive/negative samples, apply a permutation
          to all three outputs consistently.
        - If downstream expects equal-length sequences, you must enforce that separately;
          this reader allows variable length across records as long as each record's
          sequence length matches its structure length.
    """
    seq_neg, struct_neg, label_neg = read_fasta_with_struct_single(neg_path, 0)
    seq_pos, struct_pos, label_pos = read_fasta_with_struct_single(pos_path, 1)

    sequences = np.array(seq_pos + seq_neg, dtype=object)
    structs   = np.array(struct_pos + struct_neg, dtype=object)
    labels    = np.vstack([label_pos, label_neg]).astype(np.float32)

    return sequences, structs, labels
