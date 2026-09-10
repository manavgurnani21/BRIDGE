"""
Shared data-loading pipeline for BRIDGE.

This module factors the feature-building + three-way-split + DataLoader construction that
was previously inlined (and duplicated) across the ``--train`` / ``--validate`` /
``--dynamic_predict`` blocks of ``main.py``. Both ``main.py`` and the ablation driver
(``ablation/run_ablation.py``) call :func:`build_split_loaders` so there is a single source
of truth for how a dataset becomes train/val/test loaders.

The heavy, config-independent work (RBPformer embeddings + attention, structure tensor,
biochemical encoding, motif prior) is done exactly once per dataset here; the ablation sweep
then trains all model variants on the resulting loaders.
"""

import os  # for path joining and setting the PYTHONHASHSEED env var
import random  # Python's stdlib RNG, seeded for reproducibility

import numpy as np  # NumPy RNG seeding
import torch  # PyTorch RNG seeding and DataLoader tensor backend
from torch.utils.data import DataLoader  # batches myDataset objects for training/eval

from utils.gen_transformer_embedding import build_Transformer_embeddings  # produces RBPformer sequence embeddings + attention weights
from utils.motif_prior.motif_prior import get_motif_prior_matrix  # loads/builds the motif-prior feature matrix for a dataset
from utils.structureFeatures import build_structure_tensor  # converts RNA secondary-structure strings into a numeric tensor
from utils.FeatureEncoding import dealwithdata  # builds the one-hot/biochemical encoding of raw sequences
from utils.dataloaders import read_fasta  # reads the pos/neg FASTA pair into sequences, structures, and labels
from utils.utils import myDataset, split_dataset  # myDataset wraps tensors as a torch Dataset; split_dataset does the stratified 70/15/15 split


def fix_seed(seed):
    """Seed all relevant RNGs (mirrors ``main.py.fix_seed`` for shared use)."""
    if seed is None:
        seed = random.randint(1, 10000)  # pick a random seed if the caller didn't pin one, so a seed is always defined below
    torch.set_num_threads(1)  # single-threaded CPU ops for deterministic results across runs
    random.seed(seed)  # seed Python's random module (used by e.g. shuffling)
    os.environ['PYTHONHASHSEED'] = str(seed)  # fix hash randomization so dict/set iteration order is reproducible
    np.random.seed(seed)  # seed NumPy's global RNG (used by data splitting/augmentation)
    torch.manual_seed(seed)  # seed PyTorch's CPU RNG (model init, dropout, etc.)
    torch.cuda.manual_seed(seed)  # seed PyTorch's RNG for the current GPU
    torch.cuda.manual_seed_all(seed)  # seed PyTorch's RNG for all GPUs (multi-GPU determinism)


def build_split_loaders(
    data_file,
    data_path,
    transformer_path,
    device,
    seed,
    max_length=101,
    transformer_batch_size=2048,
    train_batch_size=32,
    eval_batch_size=32 * 8,
):
    """
    Build train / validation / test DataLoaders for one dataset.

    Reproduces the feature pipeline of ``main.py`` (read FASTA -> RBPformer embeddings +
    attention -> structure tensor -> biochemical features -> motif prior -> stratified
    70/15/15 split -> ``myDataset`` -> ``DataLoader``). ``fix_seed(seed)`` is called up front
    so the split is identical to a ``main.py`` run with the same seed, and so all ablation
    configs in a sweep share one sealed test partition.

    Args:
        data_file: dataset stem; loader reads ``{data_path}/{data_file}_{pos,neg}.fa``.
        data_path: directory containing the FASTA files.
        transformer_path: path to the pretrained RBPformer.
        device: torch device for embedding generation.
        seed: RNG seed controlling the split (and preprocessing RNG).
        max_length: fixed sequence length (default 101).
        transformer_batch_size: batch size for embedding generation.
        train_batch_size / eval_batch_size: DataLoader batch sizes.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    fix_seed(seed)  # pin all RNGs before any random split/shuffle happens, so results are reproducible for this seed

    neg_path = os.path.join(data_path, data_file + '_neg.fa')  # path to the negative-class FASTA file for this dataset
    pos_path = os.path.join(data_path, data_file + '_pos.fa')  # path to the positive-class FASTA file for this dataset

    sequences, structs, label = read_fasta(neg_path, pos_path)  # parse both FASTA files into raw sequences, structure annotations, and binary labels

    Transformer_emb, attention_weight = build_Transformer_embeddings(
        sequences=list(sequences),  # raw RNA sequences to embed
        transformer_path=transformer_path,  # path to the pretrained RBPformer checkpoint used for embedding
        device=device,  # CPU/GPU device to run the transformer forward pass on
        k=1,  # k-mer tokenization size passed through to the embedding builder
        transpose_to_ch_first=True,  # reorder embedding dims to (channel, length) for downstream conv layers
        Transformer_batch_size=transformer_batch_size,  # batch size used while running sequences through the transformer
    )

    structure = build_structure_tensor(structs, max_length)  # convert structure strings to a fixed-length numeric tensor
    biochem = dealwithdata(data_file).transpose([0, 2, 1])  # build the biochemical/one-hot encoding, then swap to (channel, length) axis order
    motif = get_motif_prior_matrix(data_file)  # load the precomputed motif-prior matrix for this dataset

    [train_emb, train_attn, train_struc, train_motif, train_biochem, train_label], \
    [val_emb, val_attn, val_struc, val_motif, val_biochem, val_label], \
    [test_emb, test_attn, test_struc, test_motif, test_biochem, test_label] = split_dataset(
        Transformer_emb,  # RBPformer sequence embeddings to split
        attention_weight,  # corresponding per-sequence attention weights to split
        structure,  # structure tensor to split
        motif,  # motif-prior matrix to split
        biochem,  # biochemical encoding to split
        label,  # binary labels to split (drives the stratification)
    )  # stratified 70/15/15 split applied identically across all feature tensors and labels

    train_set = myDataset(train_emb, train_attn, train_struc, train_motif, train_biochem, train_label)  # wrap the training split's tensors as a torch Dataset
    val_set = myDataset(val_emb, val_attn, val_struc, val_motif, val_biochem, val_label)  # wrap the validation split's tensors as a torch Dataset
    test_set = myDataset(test_emb, test_attn, test_struc, test_motif, test_biochem, test_label)  # wrap the held-out test split's tensors as a torch Dataset

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)  # training loader: shuffled each epoch
    val_loader = DataLoader(val_set, batch_size=eval_batch_size, shuffle=False)  # validation loader: fixed order, larger eval batch size
    test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False)  # test loader: fixed order, larger eval batch size

    return train_loader, val_loader, test_loader  # hand back the three DataLoaders for training/validation/testing
