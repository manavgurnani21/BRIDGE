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
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

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
    num_param0 = sum(p.numel() for p in model.parameters())
    num_param1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("---------------------------------")
    print("Total params:", num_param0)
    print("Trainable params:", num_param1)
    print("Non-trainable params:", num_param0 - num_param1)
    print("---------------------------------")


class BaseRBPDataset(Dataset):
    """
    Base class for RBP-related multimodal datasets.

    Each sample is returned as a tuple whose exact structure is defined
    by the subclass through `modalities`.
    """

    # Ordered list of field names expected in __getitem__ output
    modalities: Tuple[str, ...] = ()

    def __init__(self, **modal_tensors: torch.Tensor) -> None:
        """
        Parameters
        ----------
        **modal_tensors
            Keyword arguments whose keys must match `self.modalities`
            and whose values are (N, …) tensors with the same first-dim length.
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
    """Used for training / validation with labels."""
    modalities = ("embedding", "attn", "struct", "motif", "plfold", "label")


class RBPInferDataset(BaseRBPDataset):
    """Used for inference: no labels, motif & structure + biochemical features."""
    modalities = ("embedding", "attn", "struct", "motif", "biochem")



class myDataset(Dataset):
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


def read_h5(file_path):
    f = h5py.File(file_path)
    embedding = np.array(f['bert_embedding']).astype(np.float32)
    structure = np.array(f['structure']).astype(np.float32)
    label = np.array(f['label']).astype(np.int32)
    f.close()
    return embedding, structure, label

def convert_one_hot(sequence, max_length=None):
    """convert DNA/RNA sequences to a one-hot representation"""
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
    """convert DNA/RNA sequences to a one-hot representation"""
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
    """Gradually warm-up(increasing) learning rate in optimizer.
    
    Parameters:
    -----
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
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
    for src in ["K562", "HEK293", "HEK293T", "Hela", "H9"]:
        if name.endswith(src):
            return name.replace(src, "HepG2")
    if name.endswith("HepG2"):
        return name.replace("HepG2", "K562")
    