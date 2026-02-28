"""
Transformer embedding and attention extraction for k-mer tokenized sequences.

This module provides utilities to convert raw nucleotide sequences (RNA/DNA strings)
into whitespace-delimited k-mer "sentences", run a HuggingFace ``BertModel`` to obtain
token-level embeddings, and derive an attention-based token-to-token weight matrix from
the final Transformer layer.

It is primarily used to generate the two BRIDGE inputs:

- ``bert_embedding``: token embeddings, typically shaped ``(B, 512, L)``
- ``attn``: token adjacency/attention weights, typically shaped ``(B, L, L)``

Who this is for
---------------
- Users running BRIDGE training/inference pipelines who need to build Transformer features
  from raw sequences.
- Developers who want to reproduce the exact embedding/attention extraction logic
  (special-token removal, head averaging, etc.).

This module is not a tokenizer trainer and does not build a k-mer vocabulary; it assumes
``transformer_path`` points to a compatible pretrained checkpoint/tokenizer.

Main entry point
----------------
Use ``build_Transformer_embeddings(...)`` to produce embeddings and attention weights:

- loads tokenizer/model from ``transformer_path``
- converts sequences to k-mer token strings via ``seq2kmer``
- runs batched inference via ``rbpformer_encode_batch``
- optionally transposes embeddings to channel-first layout

Input/Output conventions
------------------------
k-mer tokenization
    ``seq2kmer(seq, k)`` converts a sequence into overlapping k-mers (stride 1) separated
    by spaces. If the raw sequence length is ``S``, the token count before special tokens
    is ``S - k + 1``.

Token lengths and array types (important)
    Downstream code often assumes all sequences yield the same token length ``L``.
    If token lengths differ across sequences, the returned NumPy arrays may become
    ``dtype=object`` because ``np.array(list_of_arrays)`` cannot stack ragged arrays.

    If your pipeline requires fixed ``L``, ensure upstream padding/truncation of raw
    sequences so that all inputs have equal length (and use a consistent ``k``).

Embedding shape
    - HuggingFace returns last hidden states as ``(B, L_total, C)``, where ``L_total``
      includes special tokens and padding.
    - This module removes ``[CLS]`` and ``[SEP]`` by slicing ``[1 : seq_len-1]``,
      where ``seq_len`` is computed from ``attention_mask``.

Returned outputs
    ``build_Transformer_embeddings`` returns:

    - ``Transformer_embedding``:
      - if ``transpose_to_ch_first=True``: expected shape ``(N, C, L)``
      - else: expected shape ``(N, L, C)``
    - ``attention_weight``:
      attention-derived matrices aligned to token positions, expected shape ``(N, L, L)``

    Here ``C`` is the Transformer hidden size (often 512 for RBPformer checkpoints).

Attention extraction details
----------------------------
- In ``rbpformer_encode_batch`` the model is called with ``output_attentions=True``.
- The implementation uses the **last layer** attention: ``outputs.attentions[-1]``.
- Attention heads are averaged: ``mean(1)`` resulting in shape ``(B, L_total, L_total)``.
- Special tokens are removed by slicing indices ``[1 : seq_len-1]``.

Performance notes
-----------------
- ``gen_Transformer_embedding`` uses a large DataLoader batch size (2048) for throughput.
  This may exceed GPU memory depending on ``L`` and model size. Reduce batch size if you
  encounter out-of-memory errors.
- Inference runs under ``torch.no_grad()`` and with ``model.eval()``.

How to use
----------
Minimal usage (main entry point):

.. code-block:: python

    import torch
    from transformer_features import build_Transformer_embeddings

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequences = ["ACGU..." , "ACGU..."]  # same length recommended

    embeds, attn = build_Transformer_embeddings(
        sequences=sequences,
        transformer_path="path_or_hf_name",
        device=device,
        k=1,
        transpose_to_ch_first=True,
    )

    # embeds: (N, C, L), attn: (N, L, L) if token lengths are uniform

Notes and caveats
-----------------
- Hidden size assumptions:
  BRIDGE often expects ``C=512``. Ensure the checkpoint at ``transformer_path`` matches
  your model architecture (otherwise channel mismatch errors will occur downstream).
- Tokenizer settings:
  This code uses ``do_lower_case=False``. For nucleotide k-mers this is usually correct.
- Ragged outputs:
  If sequences differ in length, NumPy outputs may be ragged ``dtype=object``. Handle
  padding/truncation before calling this module if you need dense tensors.
- Device placement:
  The model is moved to ``device``; input tensors are also moved accordingly.
"""

from typing import Sequence, Tuple
import numpy as np
import torch
import torch.utils.data
from transformers import BertTokenizer, BertModel


def seq2kmer(seq: str, k: int) -> str:
    """
    Convert a nucleotide sequence into overlapping k-mers separated by spaces.

    This function transforms a raw RNA/DNA string into a whitespace-delimited token string
    so that each k-mer can be treated as a token by a tokenizer.

    Args:
        seq (str):
            Raw nucleotide sequence (e.g., "ACGT..." or "AUGC...").
        k (int):
            k-mer length. Must satisfy 1 <= k <= len(seq).

    Returns:
        str:
            Space-separated k-mers.
            Example: seq="ACGT", k=2 -> "AC CG GT".

    Notes:
        - If the raw sequence length is S, the number of k-mers produced is (S - k + 1).
        - Downstream modules often assume all sequences produce the same token length.
          If lengths vary, later stacking into a numeric NumPy array may produce dtype=object.
    """
    seq_length = len(seq)
    
    # Generate overlapping k-mers with stride 1
    kmer = [seq[x:x + k] for x in range(seq_length - k + 1)]

    # Join k-mers with spaces to match tokenizer input format
    kmers = " ".join(kmer)
    return kmers


def rbpformer_encode_batch(
    dataloader: torch.utils.data.DataLoader,
    model: BertModel,
    tokenizer: BertTokenizer,
    device: torch.device
):
    """
    Run Transformer inference over batches of k-mer token strings.

    This function encodes sequences into token-level embeddings and derives an attention-based
    token-to-token weight matrix from the final Transformer layer.

    Args:
        dataloader (torch.utils.data.DataLoader):
            Yields batches where each element is a whitespace-delimited k-mer string,
            e.g., "AC CG GT ...".
        model (transformers.BertModel):
            HuggingFace BERT model compatible with the tokenizer and k-mer vocabulary.
        tokenizer (transformers.BertTokenizer):
            Tokenizer used to convert k-mer strings into input IDs and masks.
        device (torch.device):
            Device on which the model runs (e.g., torch.device("cuda") or torch.device("cpu")).

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]:
            features:
                List of per-sequence embedding arrays with shape (L_i, C),
                where L_i is the number of valid tokens excluding special tokens,
                and C is the hidden size.
            attn_adj:
                List of per-sequence attention-derived arrays.
                As implemented, each item has shape (L_i, L_i) after removing special tokens.

    Notes:
        - The code uses `output_attentions=True`, takes the last layer attention,
          and averages across attention heads via `.mean(1)`.
        - Special tokens [CLS] and [SEP] are removed by slicing [1 : seq_len-1].
        - `seq_len` is computed from `attention_mask` (number of ones), so padding positions
          are excluded automatically.
    """
    features = []
    seq = []
    attn_adj = []
    
    for sequences in dataloader:
        # sequences: list of space-separated k-mer strings
        seq.append(sequences)
        
        # Tokenize sequences and move tensors to target device
        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        token_type_ids = torch.tensor(ids['token_type_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        
        # Forward pass without gradient tracking
        with torch.no_grad():
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids,
                            output_attentions=True)
            
            # outputs[0]: last hidden states (B, L, C)
            embedding = outputs[0]
            
            # outputs.attentions: tuple of attention matrices from all layers
            attention_w = outputs.attentions
            del outputs
            
        # Move outputs to CPU and convert to NumPy
        embedding = embedding.cpu().numpy()
        
        # Use last layer attention and average over attention heads
        attention_w = attention_w[-1].mean(1)
        attention_w = attention_w.cpu().numpy()
        
        # Remove special tokens ([CLS], [SEP]) and pad positions
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            
            # Token embeddings excluding special tokens
            seq_emd = embedding[seq_num][1:seq_len - 1]
            
            # Corresponding attention submatrix
            seq_attn = attention_w[seq_num][1:seq_len - 1]
            
            features.append(seq_emd)
            attn_adj.append(seq_attn)
            
    return features, attn_adj


def gen_Transformer_embedding(protein, model, tokenizer, device, k, Transformer_batch_size):
    """
    Convenience wrapper: raw sequences -> k-mer strings -> batched Transformer inference.

    Args:
        protein (Sequence[str]):
            Raw nucleotide sequences (strings). Each sequence is stripped and converted to k-mers.
        model (transformers.BertModel):
            Pre-loaded Transformer model (already moved to `device`).
        tokenizer (transformers.BertTokenizer):
            Tokenizer corresponding to the model and k-mer vocabulary.
        device (torch.device):
            Device on which inference runs.
        k (int):
            k-mer length used by `seq2kmer`.
        Transformer_batch_size (int):
            Batch size used for Transformer inference.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            embeds:
                NumPy array built from per-sequence embedding matrices.
                If all sequences yield identical token length L, expected numeric shape is (N, L, C).
                Otherwise, `np.array(list_of_arrays)` may produce dtype=object.
            attns:
                NumPy array built from per-sequence attention matrices.
                If all sequences yield identical token length L, expected numeric shape is (N, L, L).
                Otherwise may become dtype=object.
    """
    sequences1 = protein
    sequences = []
    Transformer_Feature = []
    Attention_adjacent = []
    
    # Convert each raw sequence into space-separated k-mers
    for seq in sequences1:
        seq = seq.strip()
        ss = seq2kmer(seq, k)
        sequences.append(ss)
        
    # Use a large batch size for efficient inference
    dataloader = torch.utils.data.DataLoader(sequences, batch_size=Transformer_batch_size, shuffle=False)
    
    # Run Transformer inference
    Features, Attn_adj = rbpformer_encode_batch(dataloader, model, tokenizer, device)
    
    # Convert lists to NumPy arrays
    for i in Features:
        Feature = np.array(i)
        Transformer_Feature.append(Feature)
        
    for i in Attn_adj:
        attn = np.array(i)
        Attention_adjacent.append(attn)
        
    embeds = np.array(Transformer_Feature)
    attns = np.array(Attention_adjacent)
    
    return embeds, attns


def build_Transformer_embeddings(
    sequences: Sequence[str],
    transformer_path: str,
    device: torch.device,
    k: int = 1,
    transpose_to_ch_first: bool = True,
    Transformer_batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build token-level embeddings and attention weights for input sequences.

    This is the main entry point used by the training/inference pipeline. It:
      1) Loads a tokenizer/model from `transformer_path`,
      2) Converts raw sequences to k-mer token strings,
      3) Runs Transformer inference,
      4) Optionally transposes embeddings to channel-first format.

    Args:
        sequences (Sequence[str]):
            Raw nucleotide sequences.
        transformer_path (str):
            HuggingFace model name or local checkpoint directory.
        device (torch.device):
            Device used for inference (CPU/GPU).
        k (int, optional):
            k-mer size. Default: 1.
            For k=1, token length typically matches the raw sequence length (after special-token removal).
            For k>1, token length is approximately len(seq) - k + 1.
        transpose_to_ch_first (bool, optional):
            If True, transpose embeddings from (N, L, C) to (N, C, L). Default: True.
        Transformer_batch_size (int, optional):
            Batch size used for Transformer inference. Default: 2048.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Transformer_embedding:
                If `transpose_to_ch_first=True`, expected shape (N, C, L).
                Otherwise expected shape (N, L, C).
                If token lengths vary across sequences, may become dtype=object.
            attention_weight:
                Attention matrices aligned to token positions.
                If token lengths are uniform, expected shape (N, L, L).
                If token lengths vary, may become dtype=object.

    Notes:
        - Downstream BRIDGE typically expects:
              bert_embedding: (B, 512, L)
              attn:          (B, L, L)
          Ensure the loaded Transformer hidden size matches the expected C (e.g., 512).
        - This function sets model.eval() and runs under torch.no_grad().
    """
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(transformer_path, do_lower_case=False)
    model = BertModel.from_pretrained(transformer_path).to(device).eval()
    
    # Run embedding extraction without gradient computation
    with torch.no_grad():
        Transformer_embedding, attention_weight = gen_Transformer_embedding(
            list(sequences), model, tokenizer, device, k, Transformer_batch_size
        )

    # Convert to channel-first format if required by downstream modules
    if transpose_to_ch_first:
        Transformer_embedding = Transformer_embedding.transpose([0, 2, 1])

    return Transformer_embedding, attention_weight
