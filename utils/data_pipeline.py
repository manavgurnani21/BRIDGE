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

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.gen_transformer_embedding import build_Transformer_embeddings
from utils.motif_prior.motif_prior import get_motif_prior_matrix
from utils.structureFeatures import build_structure_tensor
from utils.FeatureEncoding import dealwithdata
from utils.dataloaders import read_fasta
from utils.utils import myDataset, split_dataset


def fix_seed(seed):
    """Seed all relevant RNGs (mirrors ``main.py.fix_seed`` for shared use)."""
    if seed is None:
        seed = random.randint(1, 10000)
    torch.set_num_threads(1)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    fix_seed(seed)

    neg_path = os.path.join(data_path, data_file + '_neg.fa')
    pos_path = os.path.join(data_path, data_file + '_pos.fa')

    sequences, structs, label = read_fasta(neg_path, pos_path)

    Transformer_emb, attention_weight = build_Transformer_embeddings(
        sequences=list(sequences),
        transformer_path=transformer_path,
        device=device,
        k=1,
        transpose_to_ch_first=True,
        Transformer_batch_size=transformer_batch_size,
    )

    structure = build_structure_tensor(structs, max_length)
    biochem = dealwithdata(data_file).transpose([0, 2, 1])
    motif = get_motif_prior_matrix(data_file)

    [train_emb, train_attn, train_struc, train_motif, train_biochem, train_label], \
    [val_emb, val_attn, val_struc, val_motif, val_biochem, val_label], \
    [test_emb, test_attn, test_struc, test_motif, test_biochem, test_label] = split_dataset(
        Transformer_emb,
        attention_weight,
        structure,
        motif,
        biochem,
        label,
    )

    train_set = myDataset(train_emb, train_attn, train_struc, train_motif, train_biochem, train_label)
    val_set = myDataset(val_emb, val_attn, val_struc, val_motif, val_biochem, val_label)
    test_set = myDataset(test_emb, test_attn, test_struc, test_motif, test_biochem, test_label)

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
