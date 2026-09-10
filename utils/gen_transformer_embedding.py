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

from typing import Sequence, Tuple  # type hints for the public build_Transformer_embeddings signature
import numpy as np  # stacking per-sequence embedding/attention lists into arrays
import torch  # tensors, no_grad, device handling
import torch.utils.data  # DataLoader for batching k-mer strings through the Transformer
from transformers import BertTokenizer, BertModel  # HuggingFace tokenizer/model classes loaded from transformer_path


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
    seq_length = len(seq)  # total sequence length, bounds the sliding window

    # Generate overlapping k-mers with stride 1
    kmer = [seq[x:x + k] for x in range(seq_length - k + 1)]  # slide a length-k window across the sequence with stride 1

    # Join k-mers with spaces to match tokenizer input format
    kmers = " ".join(kmer)  # whitespace-delimited so the tokenizer treats each k-mer as one token
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
    features = []  # accumulates per-sequence embedding arrays across all batches
    seq = []  # accumulates raw batches of k-mer strings (kept for potential debugging/inspection)
    attn_adj = []  # accumulates per-sequence attention-derived arrays across all batches

    for sequences in dataloader:  # iterate over batches of k-mer strings
        # sequences: list of space-separated k-mer strings
        seq.append(sequences)  # keep a record of this batch's raw input strings

        # Tokenize sequences and move tensors to target device
        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True)  # tokenize with [CLS]/[SEP] added, padding to batch max length
        input_ids = torch.tensor(ids['input_ids']).to(device)  # token id tensor, moved to the inference device
        token_type_ids = torch.tensor(ids['token_type_ids']).to(device)  # segment ids (all zero for single-sequence input), moved to device
        attention_mask = torch.tensor(ids['attention_mask']).to(device)  # 1 for real tokens, 0 for padding, moved to device

        # Forward pass without gradient tracking
        with torch.no_grad():  # inference only, no need to build the autograd graph
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_attentions=True)  # request per-layer attention matrices in addition to hidden states

            # outputs[0]: last hidden states (B, L, C)
            embedding = outputs[0]  # final-layer token embeddings for this batch

            # outputs.attentions: tuple of attention matrices from all layers
            attention_w = outputs.attentions  # tuple of (B, heads, L, L) attention tensors, one per Transformer layer
            del outputs  # free the rest of the output object (other layers' hidden states, etc.) promptly

        # Move outputs to CPU and convert to NumPy
        embedding = embedding.cpu().numpy()  # detach from GPU/device, convert to NumPy for downstream storage

        # Use last layer attention and average over attention heads
        attention_w = attention_w[-1].mean(1)  # take only the final Transformer layer, average across attention heads -> (B, L, L)
        attention_w = attention_w.cpu().numpy()  # move to CPU NumPy for downstream storage

        # Remove special tokens ([CLS], [SEP]) and pad positions
        for seq_num in range(len(embedding)):  # process each sequence in this batch individually (lengths may differ)
            seq_len = (attention_mask[seq_num] == 1).sum()  # number of real (non-padding) tokens, including [CLS]/[SEP]

            # Token embeddings excluding special tokens
            seq_emd = embedding[seq_num][1:seq_len - 1]  # drop position 0 ([CLS]) and the last real position ([SEP]); padding beyond seq_len is already excluded by the upper slice bound

            # Corresponding attention submatrix
            seq_attn = attention_w[seq_num][1:seq_len - 1]  # same special-token trim applied to the attention rows (see module docstring: only rows are sliced, not columns)

            features.append(seq_emd)  # store this sequence's trimmed embedding matrix
            attn_adj.append(seq_attn)  # store this sequence's trimmed attention matrix

    return features, attn_adj  # lists of per-sequence arrays (possibly ragged if token lengths differ)


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
    sequences1 = protein  # alias; despite the name, these are the raw nucleotide sequences to embed
    sequences = []  # will hold k-mer-tokenized (space-separated) versions of each sequence
    Transformer_Feature = []  # will hold per-sequence embedding arrays after NumPy conversion
    Attention_adjacent = []  # will hold per-sequence attention arrays after NumPy conversion

    # Convert each raw sequence into space-separated k-mers
    for seq in sequences1:  # tokenize every input sequence
        seq = seq.strip()  # remove leading/trailing whitespace that would corrupt k-mer boundaries
        ss = seq2kmer(seq, k)  # convert to a whitespace-delimited k-mer string
        sequences.append(ss)  # collect the tokenized string

    # Use a large batch size for efficient inference
    dataloader = torch.utils.data.DataLoader(sequences, batch_size=Transformer_batch_size, shuffle=False)  # batch the k-mer strings; no shuffling since order must be preserved for downstream alignment

    # Run Transformer inference
    Features, Attn_adj = rbpformer_encode_batch(dataloader, model, tokenizer, device)  # run the model over every batch, get per-sequence embedding/attention lists

    # Convert lists to NumPy arrays
    for i in Features:  # each i is one sequence's (L_i, C) embedding matrix
        Feature = np.array(i)  # convert to a NumPy array
        Transformer_Feature.append(Feature)  # collect for final stacking

    for i in Attn_adj:  # each i is one sequence's (L_i, L_i) attention matrix
        attn = np.array(i)  # convert to a NumPy array
        Attention_adjacent.append(attn)  # collect for final stacking

    embeds = np.array(Transformer_Feature)  # stack into (N, L, C) if all L_i match, else dtype=object
    attns = np.array(Attention_adjacent)  # stack into (N, L, L) if all L_i match, else dtype=object

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
    tokenizer = BertTokenizer.from_pretrained(transformer_path, do_lower_case=False)  # load the k-mer tokenizer; case preserved since nucleotide k-mers are already uppercase
    model = BertModel.from_pretrained(transformer_path).to(device).eval()  # load pretrained weights, move to device, and set to inference mode

    # Run embedding extraction without gradient computation
    with torch.no_grad():  # no gradients needed anywhere in this feature-extraction path
        Transformer_embedding, attention_weight = gen_Transformer_embedding(
            list(sequences), model, tokenizer, device, k, Transformer_batch_size
        )  # tokenize, batch, and run the model to get raw (N, L, C) embeddings and (N, L, L) attention

    # Convert to channel-first format if required by downstream modules
    if transpose_to_ch_first:  # BRIDGE's conv-based branches expect channels before the sequence axis
        Transformer_embedding = Transformer_embedding.transpose([0, 2, 1])  # (N, L, C) -> (N, C, L)

    return Transformer_embedding, attention_weight
