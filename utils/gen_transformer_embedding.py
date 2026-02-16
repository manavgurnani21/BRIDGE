from typing import Sequence, Tuple
import numpy as np
import torch
import torch.utils.data
from transformers import BertTokenizer, BertModel

def seq2kmer(seq: str, k: int) -> str:
    """
    Convert a nucleotide sequence into overlapping k-mers separated by spaces.

    Tutorial note:
        This function converts a raw RNA/DNA string into a whitespace-delimited token string,
        so that each k-mer is treated as a "token" by the tokenizer.

    Input:
        seq (str): Raw nucleotide sequence, e.g. "ACGT..." or "AUGC...".
        k (int): k-mer length.

    Output:
        str: Space-separated k-mers.
             Example: seq="ACGT", k=2 -> "AC CG GT"

    Token length:
        If the raw sequence length is S, the number of k-mers is (S - k + 1).
        Downstream modules often assume all sequences end up with the same token length L.
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

    What this produces:
        - Token-level embeddings per sequence (after removing special tokens).
        - Attention weights per sequence (returned by the model and post-processed here),
          intended to be usable as an adjacency signal.

    Inputs:
        dataloader:
            Yields batches of sequences, where each element is a whitespace-delimited k-mer string.
            Example element: "AC CG GT ..."

        model/tokenizer:
            Must be compatible with the k-mer vocabulary used to tokenize the sequences.

        device:
            torch.device("cuda") or torch.device("cpu")

    Outputs:
        features: List[np.ndarray]
            Each item is a token embedding matrix of shape (L_i, C)
            where L_i is the number of valid tokens for that sequence (excluding special tokens)
            and C is the model hidden size.

        attn_adj: List[np.ndarray]
            Each item is an attention-derived matrix associated with the sequence tokens.
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


def gen_Transformer_embedding(protein, model, tokenizer, device, k):
    """
    Convenience wrapper:
        raw sequences -> k-mer strings -> batched Transformer inference.

    Inputs:
        - sequences (List[str]): Raw nucleotide sequences.
        - model: Pre-loaded RBPformer model.
        - tokenizer: Corresponding tokenizer.
        - device (torch.device): CUDA / CPU device.
        - k (int, optional): k-mer length. Defaults to 1.

    Outputs:
        embeds, attns:
            np.array(...) built from lists of per-sequence arrays.

    Tutorial caveat:
        If sequences do NOT produce the same token length L (after k-merization/tokenization),
        then np.array(list_of_arrays) may become a dtype=object array instead of a numeric tensor.
        Many downstream operations (including `.transpose([0,2,1])`) assume a numeric 3D array.
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
    dataloader = torch.utils.data.DataLoader(sequences, batch_size=2048, shuffle=False)
    
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build token-level embeddings and attention weights for input sequences.

    Tutorial summary:
        This is the main entry point used by the training/inference pipeline.
        It loads a pretrained tokenizer/model from `transformer_path`, converts raw sequences
        into k-mer token strings, runs Transformer inference, and returns:

            - Transformer_embedding: token-level embeddings
            - attention_weight: attention weights aligned to token positions (verify exact shape)

    Inputs:
        sequences:
            List/tuple of raw sequences (strings).
        transformer_path:
            HuggingFace model name or local checkpoint directory.
        device:
            Where the model runs (CPU/GPU).
        k:
            k-mer size. For k=1, token length typically matches sequence length.
            For k>1, token length changes (~ len(seq) - k + 1).
        transpose_to_ch_first:
            If True, embeddings are transposed to channel-first format (N, C, L).

    Outputs:
        Transformer_embedding:
            Intended shape:
                - if transpose_to_ch_first=True: (N, C, L)
                - else: (N, L, C)
            where C should match downstream expectations (BRIDGE assumes C=512).
            If token lengths vary across sequences, this may become an object array.

        attention_weight:
            Attention weights returned by `gen_Transformer_embedding`.
            Downstream BRIDGE expects attention shaped like (B, L, L) for graph adjacency.

    Downstream usage in BRIDGE:
        Transformer_emb, attention_weight = build_Transformer_embeddings(
            sequences=list(sequences),
            transformer_path=args.Transformer_path,
            device=device,
            k=1,
            transpose_to_ch_first=True
        )
        
        BRIDGE.forward expects:
            bert_embedding: (B, 512, L)
            attn:          (B, L, L)
    """
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(transformer_path, do_lower_case=False)
    model = BertModel.from_pretrained(transformer_path).to(device).eval()
    
    # Run embedding extraction without gradient computation
    with torch.no_grad():
        Transformer_embedding, attention_weight = gen_Transformer_embedding(
            list(sequences), model, tokenizer, device, k
        )

    # Convert to channel-first format if required by downstream modules
    if transpose_to_ch_first:
        Transformer_embedding = Transformer_embedding.transpose([0, 2, 1])

    return Transformer_embedding, attention_weight
