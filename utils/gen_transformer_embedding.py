from typing import Sequence, Tuple
import numpy as np
import torch
import torch.utils.data
from transformers import BertTokenizer, BertModel

def seq2kmer(seq, k):
    """Convert a nucleotide sequence into overlapping k-mers separated by spaces.

    Args:
        seq (str): Raw nucleotide sequence.
        k (int): Length of each k-mer.

    Returns:
        str: String of k-mers separated by a single space.
    """
    seq_length = len(seq)
    
    # Generate overlapping k-mers with stride 1
    kmer = [seq[x:x + k] for x in range(seq_length - k + 1)]

    # Join k-mers with spaces to match tokenizer input format
    kmers = " ".join(kmer)
    return kmers


def rbpformer_encode_batch(dataloader, model, tokenizer, device):
    """Run RBPformer inference on a dataloader of tokenized sequences.

    Args:
        dataloader (DataLoader): Iterable of k-mer-formatted sequences.
        model: Pre-loaded RBPformer model.
        tokenizer: Corresponding tokenizer.
        device (torch.device): CUDA / CPU device for inference.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Lists containing per-sequence
            token embeddings and attentions, respectively.
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
    """Wrapper that k-mer-tokenizes raw sequences and obtains RBPformer outputs.

    Args:
        sequences (List[str]): Raw nucleotide sequences.
        model: Pre-loaded RBPformer model.
        tokenizer: Corresponding tokenizer.
        device (torch.device): CUDA / CPU device.
        k (int, optional): k-mer length. Defaults to 1.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 2048.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of per-sequence embeddings and
            attention matrices.
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
    Build Transformer embeddings and attention weights for input RNA/DNA sequences.

    This function loads a pretrained Transformer tokenizer and model from `transformer_path`,
    runs `gen_Transformer_embedding` to obtain embeddings and attention,
    and (optionally) transposes embeddings to channel-first format.

    Args:
        sequences (Sequence[str]): List/tuple of input sequences (length N).
        transformer_path (str): Path or model name for HuggingFace `from_pretrained`.
        device (torch.device): Target device to place the model on (e.g., torch.device("cuda:0")).
        k (int): k-mer length to be used inside `gen_Transformer_embedding`.
        transpose_to_ch_first (bool): If True, transpose embeddings from (N, L, C) to (N, C, L).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Transformer_embedding: np.ndarray, shape (N, C, L) if `transpose_to_ch_first` else (N, L, C).
            - attention_weight: np.ndarray, attention matrices as returned by `gen_Transformer_embedding`.

    Notes:
        - This function assumes `gen_Transformer_embedding(seqs, model, tokenizer, device, batch_size)`
          returns `(embeddings, attentions)` where `embeddings` has shape (N, L, C).
        - Type hints are for readability and static checking; they do not enforce runtime checks.
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
