import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any, Dict, List, Sequence, Tuple

def seq2kmer(seq, k):
    """
    Convert a nucleotide sequence into overlapping k-mers (stride 1).

    Args:
        seq (str):
            Raw nucleotide sequence (DNA/RNA), e.g., "ACGT..." or "AUGC...".
        k (int):
            k-mer length. Typically 1 <= k <= len(seq).

    Returns:
        list[str]:
            List of k-mer substrings of length `k`.
            Example: seq="ACGT", k=2 -> ["AC", "CG", "GT"].
    """
    seq_length = len(seq)
    sub_seq = 'ATCG'
    import random
    rand1 = random.randint(0, 3)  # [0,3]
    rand2 = random.randint(0, 3)
    # seq = sub_seq[rand1] + seq + sub_seq[rand2]
    kmer = [seq[x:x + k] for x in range(seq_length - k + 1)]
    return kmer


def split_dataset(data1, data2, data3, data_motif, data_plfold, targets, valid_frac=0.2):
    """
    Stratified train/test split for multiple aligned modalities.

    This function splits samples into train and test sets by thresholding targets at 0.5:
        negatives: targets < 0.5
        positives: targets >= 0.5
    and sampling approximately `valid_frac` from each class into the test set.

    Args:
        data1, data2, data3, data_motif, data_plfold (np.ndarray):
            Input arrays for different modalities. Each must share the same first dimension N.
        targets (np.ndarray):
            Target array aligned with the modalities along the first axis. Shape (N,) or (N,1).
        valid_frac (float, optional):
            Fraction of each class assigned to the test split. Default: 0.2.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]:
            (train, test) where each is a list:
                train = [X_train1, X_train2, X_train3, X_train4, X_train5, Y_train]
                test  = [X_test1,  X_test2,  X_test3,  X_test4,  X_test5,  Y_test]

    Notes:
        - Indices are permuted independently within each class using np.random.permutation.
        - The returned order concatenates positives first, then negatives (as implemented).
    """
    ind0 = np.where(targets < 0.5)[0]
    ind1 = np.where(targets >= 0.5)[0]

    n_neg = int(len(ind0) * valid_frac)
    n_pos = int(len(ind1) * valid_frac)

    shuf_neg = np.random.permutation(len(ind0)) 
    shuf_pos = np.random.permutation(len(ind1))

    X_train1 = np.concatenate((data1[ind1[shuf_pos[n_pos:]]], data1[ind0[shuf_neg[n_neg:]]]))
    X_train2 = np.concatenate((data2[ind1[shuf_pos[n_pos:]]], data2[ind0[shuf_neg[n_neg:]]]))
    X_train3 = np.concatenate((data3[ind1[shuf_pos[n_pos:]]], data3[ind0[shuf_neg[n_neg:]]]))
    X_train4 = np.concatenate((data_motif[ind1[shuf_pos[n_pos:]]], data_motif[ind0[shuf_neg[n_neg:]]]))
    X_train5 = np.concatenate((data_plfold[ind1[shuf_pos[n_pos:]]], data_plfold[ind0[shuf_neg[n_neg:]]]))
    Y_train = np.concatenate((targets[ind1[shuf_pos[n_pos:]]], targets[ind0[shuf_neg[n_neg:]]]))
    train = [X_train1, X_train2, X_train3, X_train4, X_train5, Y_train]

    X_test1 = np.concatenate((data1[ind1[shuf_pos[:n_pos]]], data1[ind0[shuf_neg[:n_neg]]]))
    X_test2 = np.concatenate((data2[ind1[shuf_pos[:n_pos]]], data2[ind0[shuf_neg[:n_neg]]]))
    X_test3 = np.concatenate((data3[ind1[shuf_pos[:n_pos]]], data3[ind0[shuf_neg[:n_neg]]]))
    X_test4 = np.concatenate((data_motif[ind1[shuf_pos[:n_pos]]], data_motif[ind0[shuf_neg[:n_neg]]]))
    X_test5 = np.concatenate((data_plfold[ind1[shuf_pos[:n_pos]]], data_plfold[ind0[shuf_neg[:n_neg]]]))
    Y_test = np.concatenate((targets[ind1[shuf_pos[:n_pos]]], targets[ind0[shuf_neg[:n_neg]]]))
    test = [X_test1, X_test2, X_test3, X_test4, X_test5, Y_test]

    return train, test


def param_num(model):
    """
    Print the total/trainable/non-trainable parameter counts of a PyTorch model.
    """
    num_param0 = sum(p.numel() for p in model.parameters())
    num_param1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("---------------------------------")
    print("Total params:", num_param0)
    print("Trainable params:", num_param1)
    print("Non-trainable params:", num_param0 - num_param1)
    print("---------------------------------")


class BaseRBPDataset(Dataset):
    """
    Base class for multimodal RBP datasets.

    Subclasses define `modalities` as an ordered tuple of modality names (singular).
    The constructor expects keyword tensors whose keys match these modality names, and stores
    them as pluralized attributes (e.g., key="embedding" -> self.embeddings).
    
    Attributes:
        modalities (Tuple[str, ...]):
            Names and order of fields returned by __getitem__.
        _length (int):
            Number of samples (N).

    Example:
        >>> class RBPTrainDataset(BaseRBPDataset):
        ...     modalities = ("embedding", "attn", "struct", "motif", "plfold", "label")
        ...
        >>> train_ds = RBPTrainDataset(
        ...     embedding=torch.randn(N, C, L),
        ...     attn=torch.randn(N, L, L),
        ...     struct=torch.randn(N, 1, L),
        ...     motif=torch.randn(N, M),
        ...     plfold=torch.randn(N, P),
        ...     label=torch.randint(0, 2, (N, 1)).float(),
        ... )
        >>> batch = train_ds[0]
        >>> len(batch)
        6

        >>> class RBPInferDataset(BaseRBPDataset):
        ...     modalities = ("embedding", "attn", "struct", "motif", "biochem")
        ...
        >>> infer_ds = RBPInferDataset(
        ...     embedding=torch.randn(N, C, L),
        ...     attn=torch.randn(N, L, L),
        ...     struct=torch.randn(N, 1, L),
        ...     motif=torch.randn(N, M),
        ...     biochem=torch.randn(N, B),
        ... )
        >>> len(infer_ds[0])
        5
    """
    # Ordered list of field names expected in __getitem__ output
    modalities: Tuple[str, ...] = ()

    def __init__(self, **modal_tensors: torch.Tensor) -> None:
        """
        Initialize the dataset with aligned modality tensors.

        Args:
            **modal_tensors (torch.Tensor):
                Keyword tensors whose keys must match `self.modalities`.
                Each tensor must have shape (N, ...), and all tensors must share the same N.
        """
        missing = set(self.modalities) - modal_tensors.keys()
        if missing:
            raise ValueError(f"Missing modalities: {missing}")

        # save tensors as attributes, e.g. self.embeddings, self.attn …
        for k, v in modal_tensors.items():
            setattr(self, f"{k}s", v)   # pluralised as attribute

        self._length = next(iter(modal_tensors.values())).shape[0]

    # --------------------------------------------------------------
    # PyTorch Dataset API
    # --------------------------------------------------------------
    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return tuple(getattr(self, f"{m}s")[idx] for m in self.modalities)


# ------------------------------------------------------------------
# Specific datasets
# ------------------------------------------------------------------
class RBPTrainDataset(BaseRBPDataset):
    """
    Dataset for training/validation that includes labels.
    """
    modalities = ("embedding", "attn", "struct", "motif", "plfold", "label")


class RBPInferDataset(BaseRBPDataset):
    """
    Dataset for inference without labels.
    """
    modalities = ("embedding", "attn", "struct", "motif", "biochem")


class myDataset(Dataset):
    """
    Legacy training dataset wrapper for multiple modalities + label.

    Args:
        bert_embedding (np.ndarray or torch.Tensor):
            Token/channel embeddings per sample.
        attn (np.ndarray or torch.Tensor):
            Attention matrices per sample.
        structure (np.ndarray or torch.Tensor):
            Structure features per sample.
        motif (np.ndarray or torch.Tensor):
            Motif features per sample.
        plfold (np.ndarray or torch.Tensor):
            RNAplfold/biochemical-descriptors per sample.
        label (np.ndarray or torch.Tensor):
            Labels per sample.

    Returns:
        tuple:
            (embedding, attn, struct, motif, plfold, label) for the given index.

    Notes:
        - This class is functionally similar to `RBPTrainDataset` but uses fixed attribute names.
        - Prefer `RBPTrainDataset` for clearer modality control and validation.
    """
    def __init__(self, bert_embedding, attn, structure, motif, plfold, label):
        self.embedding = bert_embedding
        self.attn = attn
        self.structs = structure
        self.motifs = motif
        self.plfolds = plfold
        self.label = label

    def __getitem__(self, index):
        embedding = self.embedding[index]
        attn = self.attn[index]
        struct = self.structs[index]
        motif = self.motifs[index]
        plfold = self.plfolds[index]
        label = self.label[index]

        return embedding, attn, struct, motif, plfold, label

    def __len__(self):
        return len(self.label)


class myDataset2(Dataset):
    """
    Legacy inference dataset wrapper for multiple modalities without labels.

    Args:
        bert_embedding, attn, structure, motif:
            Same semantics as myDataset.
        phys_chem (np.ndarray or torch.Tensor):
            Biochemical features per sample.

    Returns:
        tuple:
            (embedding, attn, struct, motif, phys_chem) for the given index.
    """
    def __init__(self, bert_embedding, attn, structure, motif, phys_chem):
        self.embedding = bert_embedding
        self.attn = attn
        self.structs = structure
        self.motifs = motif
        self.phys_chems = phys_chem

    def __getitem__(self, index):
        embedding = self.embedding[index]
        attn = self.attn[index]
        struct = self.structs[index]
        motif = self.motifs[index]
        phys_chem = self.phys_chems[index]

        return embedding, attn, struct, motif, phys_chem

    def __len__(self):
        return len(self.embedding)

def read_csv(path):
    """
    Read a tab-separated dataset file into sequences, structures, and labels.

    The input file is expected to be TSV with at least 6 columns, where:
        col 0: Type (string; header row uses "Type")
        col 2: Seq (sequence string)
        col 3: Str (structure string)
        col 5: label (0/1 or numeric)

    Args:
        path (str):
            Path to the TSV file.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            sequences:
                Array of sequence strings, shape (N,), dtype object.
            structs:
                Array of structure strings, shape (N,), dtype object.
            targets:
                Float32 label array, shape (N, 1).

    Notes:
        - Rows with df[0] == "Type" are dropped (header-like row).
        - No validation of sequence alphabet or structure length is performed here.
    """
    df = pd.read_csv(path, sep='\t', header=None)
    df = df.loc[df[0] != "Type"]

    Type = 0
    loc = 1
    Seq = 2
    Str = 3
    Score = 4
    label = 5

    rnac_set = df[Type].to_numpy()
    sequences = df[Seq].to_numpy()
    structs = df[Str].to_numpy()
    targets = df[label].to_numpy().astype(np.float32).reshape(-1, 1)
    return sequences, structs, targets


def read_csv_with_name(path):
    """
    Read a tab-separated dataset file and also return record identifiers/names.

    Expected columns (TSV):
        col 1: loc/name/identifier
        col 2: Seq
        col 3: Str
        col 5: label

    Args:
        path (str):
            Path to the TSV file.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            name:
                Array of identifiers (col 1), shape (N,).
            sequences:
                Array of sequence strings, shape (N,).
            structs:
                Array of structure strings, shape (N,).
            targets:
                Float32 label array, shape (N, 1).
    """
    # load sequences
    df = pd.read_csv(path, sep='\t', header=None)
    df = df.loc[df[0] != "Type"]

    Type = 0
    loc = 1
    Seq = 2
    Str = 3
    Score = 4
    label = 5

    name = df[loc].to_numpy()
    sequences = df[Seq].to_numpy()
    structs = df[Str].to_numpy()
    targets = df[label].to_numpy().astype(np.float32).reshape(-1, 1)
    return name, sequences, structs, targets


# def read_h5(file_path):
#     f = h5py.File(file_path)
#     embedding = np.array(f['bert_embedding']).astype(np.float32)
#     structure = np.array(f['structure']).astype(np.float32)
#     label = np.array(f['label']).astype(np.int32)
#     f.close()
#     return embedding, structure, label


def convert_one_hot(sequence, max_length=None):
    """
    Convert DNA/RNA sequences to one-hot encoding with optional centered zero-padding.

    Encoding:
        Channel order: A, C, G, U/T
        - 'A' -> channel 0
        - 'C' -> channel 1
        - 'G' -> channel 2
        - 'U' or 'T' -> channel 3

    Args:
        sequence (Sequence[str]):
            List/array of sequence strings. Characters are uppercased internally.
        max_length (int, optional):
            If provided, sequences are padded (centered) to `max_length` with zeros.

    Returns:
        np.ndarray:
            Array of shape (N, 4, L) if max_length is None,
            otherwise (N, 4, max_length). dtype is float64 by default.

    Notes:
        - Non-ACGU/T characters are left as all-zeros at that position.
        - If you need explicit handling of 'N', add it upstream.
    """
    one_hot_seq = []
    for seq in sequence:
        seq = seq.upper()
        seq_length = len(seq)
        one_hot = np.zeros((4,seq_length))
        index = [j for j in range(seq_length) if seq[j] == 'A']
        one_hot[0,index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'C']
        one_hot[1,index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'G']
        one_hot[2,index] = 1
        index = [j for j in range(seq_length) if (seq[j] == 'U') | (seq[j] == 'T')]
        one_hot[3,index] = 1

        # handle boundary conditions with zero-padding
        if max_length:
            offset1 = int((max_length - seq_length)/2)
            offset2 = max_length - seq_length - offset1

            if offset1:
                one_hot = np.hstack([np.zeros((4,offset1)), one_hot])
            if offset2:
                one_hot = np.hstack([one_hot, np.zeros((4,offset2))])

        one_hot_seq.append(one_hot)

    # convert to numpy array
    one_hot_seq = np.array(one_hot_seq)
    return one_hot_seq


def convert_one_hot2(sequence, attention, max_length=None):
    """
    Convert sequences into an attention-weighted one-hot representation.

    Instead of writing 1.0 at nucleotide positions, this function writes `attention[i]`
    into the nucleotide channel at position i.

    Args:
        sequence (Sequence[str]):
            Sequence strings.
        attention (Sequence[float] or np.ndarray):
            Per-position weights. This implementation uses `attention[i]` where i indexes
            the position in the sequence.
        max_length (int, optional):
            If provided, output is padded (centered) to this length with zeros.

    Returns:
        np.ndarray:
            Array of shape (N, 4, L) or (N, 4, max_length) depending on padding.

    Notes:
        - This function assumes `attention` is compatible with each sequence length.
          If sequences have different lengths, a single shared attention vector may not work.
    """
    one_hot_seq = []
    for seq in sequence:
        seq = seq.upper()
        seq_length = len(seq)
        one_hot = np.zeros((4,seq_length))
        index = [j for j in range(seq_length) if seq[j] == 'A']
        for i in index:
            one_hot[0,i] = attention[i]
        index = [j for j in range(seq_length) if seq[j] == 'C']
        for i in index:
            one_hot[1,i] = attention[i]
        index = [j for j in range(seq_length) if seq[j] == 'G']
        for i in index:
            one_hot[2,i] = attention[i]
        index = [j for j in range(seq_length) if (seq[j] == 'U') | (seq[j] == 'T')]
        for i in index:
            one_hot[3,i] = attention[i]

        # handle boundary conditions with zero-padding
        if max_length:
            offset1 = int((max_length - seq_length)/2)
            offset2 = max_length - seq_length - offset1

            if offset1:
                one_hot = np.hstack([np.zeros((4,offset1)), one_hot])
            if offset2:
                one_hot = np.hstack([one_hot, np.zeros((4,offset2))])

        one_hot_seq.append(one_hot)

    # convert to numpy array
    one_hot_seq = np.array(one_hot_seq)

    return one_hot_seq


class GradualWarmupScheduler(_LRScheduler):
    """
    Gradually warm up (increase) the learning rate, then delegate to another scheduler.

    During warmup (epochs 1..total_epoch):
        lr = base_lr * ( (multiplier - 1) * epoch/total_epoch + 1 )

    After warmup:
        - If `after_scheduler` is provided, it is used for subsequent scheduling.
        - If `after_scheduler` is ReduceLROnPlateau, use `step(metrics=...)`.

    Args:
        optimizer (torch.optim.Optimizer):
            Wrapped optimizer.
        multiplier (float):
            Target learning rate multiplier. Final warmup LR is base_lr * multiplier.
            Must be > 1.0.
        total_epoch (int):
            Number of warmup epochs. Target LR is reached at `total_epoch`.
        after_scheduler (Optional[_LRScheduler or ReduceLROnPlateau]):
            Scheduler used after warmup.

    Raises:
        ValueError:
            If multiplier <= 1.0.
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        """
        Initialize the warmup scheduler.

        Args:
            optimizer (torch.optim.Optimizer):
                Wrapped optimizer whose learning rate will be scheduled.
            multiplier (float):
                Target LR multiplier relative to the optimizer's base learning rates.
                Must be > 1.0.
            total_epoch (int):
                Number of warmup epochs.
            after_scheduler (Optional[_LRScheduler or ReduceLROnPlateau]):
                Scheduler to use after warmup is finished.

        Raises:
            ValueError:
                If `multiplier <= 1.0`.
        """
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        """
        Compute the learning rate(s) for the current epoch.

        Returns:
            List[float]:
                A list of learning rates, one per parameter group in the wrapped optimizer.
        """
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]


    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        """
        Step function for the special case where `after_scheduler` is ReduceLROnPlateau.

        Args:
            metrics (float):
                Monitored metric value (e.g., validation loss) used by ReduceLROnPlateau.
            epoch (int, optional):
                Epoch index. If None, uses `self.last_epoch + 1`.

        Returns:
            None.

        Notes:
            - ReduceLROnPlateau is typically called at the end of an epoch, whereas most schedulers
              are called at the beginning. This method mirrors common warmup implementations:
                - During warmup, it manually sets optimizer.param_groups[*]['lr'].
                - After warmup, it calls `after_scheduler.step(metrics, epoch - total_epoch)`.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)


    def step(self, epoch=None, metrics=None):
        """
        Advance the scheduler by one step.

        Args:
            epoch (int, optional):
                Epoch index to step to. If None, scheduler increments internally.
            metrics (float, optional):
                Metric value required when `after_scheduler` is ReduceLROnPlateau.
                Ignored for most other schedulers.

        Returns:
            None.

        Notes:
            - If `after_scheduler` is not ReduceLROnPlateau:
                - During warmup, this delegates to `_LRScheduler.step`.
                - After warmup, this delegates to `after_scheduler.step`, shifting epochs by `total_epoch`.
            - If `after_scheduler` is ReduceLROnPlateau:
                - This calls `step_ReduceLROnPlateau(metrics, epoch)`.
        """
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


# Determine alternate cell-line model for dynamic prediction
def resolve_dynamic_model_name(name: str) -> str:
    """
    Resolve an alternate cell-line model name for dynamic prediction.

    Heuristic behavior:
        - If `name` ends with one of: {"K562","HEK293","HEK293T","Hela","H9"},
          replace that suffix with "HepG2".
        - If `name` ends with "HepG2", replace it with "K562".

    Args:
        name (str):
            Model or experiment identifier string, typically with a cell-line suffix.

    Returns:
        str:
            The modified name if a known suffix is found.

    Notes:
        - If `name` does not match any recognized suffix, the current implementation
          returns None implicitly (because there is no final `return name`).
          If you want identity behavior, add `return name` at the end.
    """
    for src in ["K562", "HEK293", "HEK293T", "Hela", "H9"]:
        if name.endswith(src):
            return name.replace(src, "HepG2")
    if name.endswith("HepG2"):
        return name.replace("HepG2", "K562")
