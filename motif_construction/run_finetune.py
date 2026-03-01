#!/usr/bin/env python3
# coding: utf-8
"""
Attention visualization + prediction export for HuggingFace sequence classifiers.

This module provides a small, self-contained CLI tool that:
  1) loads a TSV dataset of biological sequences and integer labels,
  2) tokenizes sequences with a HuggingFace tokenizer,
  3) runs a HuggingFace `AutoModelForSequenceClassification` with
     `output_attentions=True`,
  4) converts the last-layer attention into a 1D per-token importance score,
  5) exports attention scores and prediction probabilities to NumPy files.

Typical usage context
---------------------
This script is intended for *post hoc* inspection of attention patterns for a
trained Transformer-based sequence classifier (e.g., DNA/RNA k-mer tokenization
or character-level tokenization). It can be used to:
  - produce per-token "importance-like" vectors derived from [CLS] attention,
  - export model probabilities for downstream ranking/analysis,
  - reproduce legacy pipelines that expect a fixed set of CLI arguments.

Core modules (functions)
------------------------
1) `load_tsv(path, max_len) -> (seqs, labels)`

   Input:
     - `path`: TSV file with lines `<sequence>\\t<label>`
   Output:
     - `seqs`: uppercased sequences, truncated to `max_len` characters
     - `labels`: integer labels aligned with `seqs` (defaults to 0 if missing)
   Notes:
     - blank lines are skipped
     - a header row is skipped if the label field is non-integer

2) `build_dataset(tokenizer, seqs, labels, max_len) -> TensorDataset`

   Input:
     - `tokenizer`: HuggingFace tokenizer
     - `seqs`, `labels`: data returned by `load_tsv`
   Output:
     - `TensorDataset(input_ids, attention_mask, labels)`
       with shapes (N, max_len), (N, max_len), (N,)

3) `attention_scores(attn, kmer) -> Tensor[L]`

   Input:
     - `attn`: attention for a single example from a single layer,
              shaped (heads, L, L)
     - `kmer`: optional smoothing window size (kmer=1 disables smoothing)
   Output:
     - 1D normalized score vector of length L
   Method:
     - head-sum on the CLS row: sum_h attn[h, 0, i]
     - optional k-mer diffusion/smoothing over positions
     - L2 normalization

4) `main()`

   CLI entry point that wires everything together, runs inference in batches,
   and saves outputs.

Main inputs
-----------
Required CLI arguments (actually used by this script):
  --do_visualize
      Must be set; the script asserts visualize-mode execution.
  --tokenizer_name
      HuggingFace tokenizer name or local path.
  --model_name_or_path
      HuggingFace model checkpoint name or local path.
  --visualize_data_dir
      Directory containing `dev.tsv`.
  --max_seq_length
      Maximum length used for both truncation and tokenization.
  --predict_dir
      Output directory for exported arrays and metadata.

Optional (used):
  --per_gpu_pred_batch_size
      DataLoader batch size (default: 8).
  --visualize_models
      Interpreted here as `kmer` smoothing window size (default: 1).

Accepted-but-ignored arguments (kept for interface compatibility):
  --model_type, --task_name, --data_dir, --output_dir, --n_process

Expected input file layout
--------------------------
`<visualize_data_dir>/dev.tsv` with one example per line:
  <sequence>\\t<label>

Example:
  sequence    label
  ACGTACGT    1
  TTGCAA      0

Main outputs
------------
Files written to `<predict_dir>`:
  - `atten.npy`
      NumPy array of shape (N, max_seq_length) containing per-token attention
      scores (L2-normalized).
  - `pred_results.npy`
      NumPy array of shape (N,) containing prediction probabilities:
        * binary: softmax(logits)[:, 1]
        * multi-class: max softmax probability per sample
  - `run_meta.json`
      Small metadata record: { "N": ..., "kmer": ..., "max_len": ... }

Typical command
---------------

.. code-block:: text

    python run_finetune.py \\
    --do_visualize \\
    --tokenizer_name <tokenizer_name_or_path> \\
    --model_name_or_path <model_ckpt_or_path> \\
    --visualize_data_dir <dir_with_dev_tsv> \\
    --max_seq_length 256 \\
    --per_gpu_pred_batch_size 32 \\
    --visualize_models 3 \\
    --predict_dir <output_dir>

Dependencies
------------
- torch
- transformers
- numpy
- tqdm

Reproducibility
---------------
Seeds are fixed in `main()` for torch / random / numpy. Device selection defaults
to CUDA when available.
"""
import argparse, os, json, random, numpy as np, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_tsv(path: str, max_len: int):
    """
    Load sequences and integer labels from a TSV file.

    The TSV is expected to contain one example per line in the form:
        <sequence>\t<label>

    Behavior:
    
        - Skips blank lines.
        
        - Skips a header row if the label field is present but non-integer
        (after stripping a leading '-').

    Sequences are uppercased and truncated to `max_len` characters.

    Args:
        path: Path to the TSV file.
        max_len: Maximum number of sequence characters to keep (hard truncation).

    Returns:
        A tuple (seqs, labels):
        - seqs: List[str], uppercased sequences truncated to `max_len`.
        - labels: List[int], parsed labels. If a line has no label field, the
          label defaults to 0.

    Raises:
        FileNotFoundError: If `path` does not exist.
        UnicodeDecodeError: If the file cannot be decoded with UTF-8.
        IndexError: If a non-empty line has no first column (malformed TSV).
        ValueError: If a label is present but cannot be converted to `int`
          (this will only occur if it passes the header check but is still not
          a valid integer representation).

    Example:
        >>> # dev.tsv content:
        >>> # sequence\tlabel
        >>> # ACGT\t1
        >>> seqs, labels = load_tsv("dev.tsv", max_len=8)
        >>> seqs[0], labels[0]
        ('ACGT', 1)
    """
    seqs, labels = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            seq, *lab = line.rstrip("\n").split("\t")

            # ── handle header or bad label ──
            if lab and not lab[0].lstrip("-").isdigit():
                # treat as header row; just skip it
                continue

            seqs.append(seq.upper()[:max_len])
            labels.append(int(lab[0]) if lab else 0)   # default 0 if no label
    return seqs, labels


def build_dataset(tokenizer, seqs, labels, max_len):
    """
    Tokenize sequences and build a PyTorch TensorDataset.

    This function tokenizes `seqs` using the provided HuggingFace tokenizer and
    returns a `TensorDataset` containing:
        (input_ids, attention_mask, labels)

    Tokenization config:
        - padding="max_length"
        
        - truncation=True
        
        - max_length=max_len
        
        - add_special_tokens=True
        
        - return_tensors="pt"

    Args:
        tokenizer: A HuggingFace tokenizer instance (e.g., AutoTokenizer).
        seqs: List[str] of raw input sequences (already truncated/processed).
        labels: List[int] of labels aligned to `seqs`.
        max_len: Maximum token sequence length used by the tokenizer.

    Returns:
        torch.utils.data.TensorDataset with three tensors:
        - input_ids: LongTensor of shape (N, max_len)
        - attention_mask: LongTensor of shape (N, max_len)
        - labels: LongTensor of shape (N,)

    Raises:
        ValueError: If `len(seqs) != len(labels)`.
        KeyError: If tokenizer output does not contain 'input_ids' or 'attention_mask'.
        TypeError/RuntimeError: If tensors cannot be constructed due to dtype/shape issues.

    Example:
        >>> tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> ds = build_dataset(tok, ["ACGT"], [1], max_len=8)
        >>> len(ds)
        1
    """
    enc = tokenizer(
        seqs,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
        return_tensors="pt",
    )
    lbl = torch.tensor(labels, dtype=torch.long)
    return torch.utils.data.TensorDataset(
        enc["input_ids"], enc["attention_mask"], lbl
    )


def attention_scores(attn, kmer):
    """
    Convert a multi-head attention matrix into a 1D, normalized importance vector.

    The expected attention tensor corresponds to a *single example* from a single
    layer:
        attn shape: (heads, L, L)

    Computation:
    
    1) Head-sum on CLS row:
    
       - Sum attentions across heads: attn.sum(dim=0) -> (L, L)
       
       - Take CLS row (row index 0): row = ...[0] -> (L,)
       
       This yields a per-token score for how much CLS attends to token i.

    2) Optional k-mer smoothing:
    
       - For each window i..i+kmer-1, add window sum to all positions in window,
         then average by counts.

    3) L2 normalization:
    
       - Return scores / (||scores|| + eps)

    Args:
        attn: Attention tensor of shape (heads, L, L) for one sample.
        kmer: Window size for smoothing/diffusion. If kmer==1, no smoothing is
            applied beyond normalization.

    Returns:
        A 1D tensor of shape (L,) containing normalized attention-derived scores.

    Raises:
        ValueError: If `kmer < 1`.
        RuntimeError: If `attn` does not have 3 dimensions or is on an incompatible device.

    Example:
        >>> attn = torch.rand(8, 10, 10)  # 8 heads, length 10
        >>> s = attention_scores(attn, kmer=3)
        >>> s.shape
        torch.Size([10])
    """
    # head-sum on CLS row: Σ_h a[h, 0, i]
    row = attn.sum(dim=0)[0]       # shape (L,)
    # slide a k-mer window & diffuse scores like original implementation
    if kmer == 1:
        return row / (row.norm() + 1e-12)
    tmp = torch.zeros_like(row)
    counts = torch.zeros_like(row)
    for i in range(len(row) - kmer + 1):
        w = row[i : i + kmer]
        tmp[i : i + kmer] += w.sum()
        counts[i : i + kmer] += 1
    scores = tmp / counts.clamp_min(1)
    return scores / (scores.norm() + 1e-12)


def main():
    """
    CLI entry point for attention visualization and prediction export.

    Command-line arguments
    ----------------------
    This script keeps a set of arguments for compatibility. Some are
    accepted but intentionally ignored.

    Used arguments:
        - --do_visualize (required): must be set; otherwise AssertionError.
        - --tokenizer_name: HuggingFace tokenizer name/path.
        - --model_name_or_path: HuggingFace model checkpoint directory/name.
        - --visualize_data_dir: directory containing dev.tsv.
        - --max_seq_length: maximum length for truncation/tokenization.
        - --per_gpu_pred_batch_size: DataLoader batch size.
        - --predict_dir: output directory for .npy and JSON.
        - --visualize_models: interpreted here as `kmer` smoothing window size.

    Ignored arguments (accepted for interface compatibility):
        - --model_type, --task_name, --data_dir, --output_dir, --n_process

    Outputs:
        - atten.npy, pred_results.npy, run_meta.json under `predict_dir`.

    Raises:
        AssertionError: If `--do_visualize` is not provided.
        FileNotFoundError: If `<visualize_data_dir>/dev.tsv` is missing.
        RuntimeError: If CUDA/CPU device placement or model outputs fail.
    """
    parser = argparse.ArgumentParser()
    # keep only the 13 required args ----------------------------------------- #
    parser.add_argument("--model_type")  # ignored
    parser.add_argument("--tokenizer_name", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--task_name")   # ignored
    parser.add_argument("--do_visualize", action="store_true")
    parser.add_argument("--visualize_data_dir", required=True)
    parser.add_argument("--visualize_models", type=int, default=1)
    parser.add_argument("--data_dir")  # ignored
    parser.add_argument("--max_seq_length", type=int, required=True)
    parser.add_argument("--per_gpu_pred_batch_size", type=int, default=8)
    parser.add_argument("--output_dir")  # ignored
    parser.add_argument("--predict_dir", required=True)
    parser.add_argument("--n_process")  # ignored
    args = parser.parse_args()

    assert args.do_visualize, "This script only supports --do_visualize mode."

    # reproducibility -------------------------------------------------------- #
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model + tokenizer ------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, output_attentions=True
    ).to(device).eval()

    # data ------------------------------------------------------------------- #
    tsv_path = os.path.join(args.visualize_data_dir, "dev.tsv")
    seqs, labels = load_tsv(tsv_path, args.max_seq_length)
    dataset = build_dataset(tokenizer, seqs, labels, args.max_seq_length)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.per_gpu_pred_batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # inference -------------------------------------------------------------- #
    all_scores = np.zeros((len(dataset), args.max_seq_length))
    all_probs  = np.zeros(len(dataset))
    softmax = torch.nn.Softmax(dim=1)
    k = args.visualize_models

    with torch.no_grad():
        offset = 0
        for batch in tqdm(loader, desc="Visualising"):
            ids, mask, _ = [t.to(device) for t in batch]
            out = model(input_ids=ids, attention_mask=mask)
            logits, attns = out.logits, out.attentions  # tuple (num_layers, B, heads, L, L)
            # use last layer's attention
            last_attn = attns[-1]                       # shape (B, heads, L, L)

            for b in range(ids.size(0)):
                s = attention_scores(last_attn[b], k)   # tensor (L,)
                all_scores[offset + b] = s.cpu().numpy()

            probs = softmax(logits)[:, 1] if logits.size(-1) == 2 else softmax(logits).max(dim=1).values
            all_probs[offset : offset + ids.size(0)] = probs.cpu().numpy()
            offset += ids.size(0)

    # save ------------------------------------------------------------------- #
    os.makedirs(args.predict_dir, exist_ok=True)
    np.save(os.path.join(args.predict_dir, "atten.npy"), all_scores)
    np.save(os.path.join(args.predict_dir, "pred_results.npy"), all_probs)

    # quick JSON log --------------------------------------------------------- #
    meta = dict(N=len(dataset), kmer=k, max_len=args.max_seq_length)
    with open(os.path.join(args.predict_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved scores → {args.predict_dir}")

if __name__ == "__main__":
    main()
