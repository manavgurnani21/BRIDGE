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
        sequences (List[str]):
            List of sequence strings, length N (number of records).
            Each string has length L_i for record i.
        structs (List[str]):
            List of raw comma-separated structure score strings, length N.
            Each entry corresponds to one record and should have exactly L_i comma-separated items.
        labels (np.ndarray):
            Array of shape (N, 1), dtype float32, filled with `label_val`.

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
