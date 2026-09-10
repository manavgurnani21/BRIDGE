"""
BRIDGE model components: multimodal sequenc-structure network + supporting blocks.

This module defines the core neural network building blocks used in the BRIDGE project,
including:

- ``BRIDGE``: a multimodal fusion network that integrates token embeddings, structure,
  motif priors, biochemical features, and a graph branch derived from token-to-token
  adjacency (e.g., attention).
- ``ADPNet`` / ``ADPNetblock``: an Adaptive Pyramidal Network backbone used as the final
  classifier head.
- ``multiscaleKAN``: a multi-path KAN-based feature extractor used per modality.

Who this is for
---------------
This module is intended for users who:

- want to instantiate the BRIDGE model architecture in PyTorch,
- run training/inference with precomputed modalities aligned to the same token axis,
- understand or modify the modality fusion and graph construction logic.

It is not a standalone dataset/preprocessing script; the caller is responsible for
preparing tensors with the correct shapes and alignment.

Key dependencies
----------------
- ``torch`` / ``torch.nn`` / ``torch.nn.functional``
- ``torch_geometric.nn.GCNConv`` (PyG)
- Project-local layers:
  - ``utils.conv_layer.Conv1d`` and ``utils.conv_layer.SimpleConvKAN_1layer``
  - ``utils.resnet`` (imported with ``*`` in this file)

Input/Output conventions
------------------------
BRIDGE expects **channel-first** tensors and a shared token length ``L`` across modalities.

Common symbols
    - ``B``: batch size
    - ``L``: token length per sequence (often fixed to ``101`` in this project)
    - ``M``: motif length (may differ from ``L``)
    - ``S``: number of structure feature channels (here structure is provided as 1 channel)

Main forward signature
    ``BRIDGE.forward(bert_embedding, attn, structure, motif, biochem) -> logits``

Inputs
    ``bert_embedding``
        Token-aligned Transformer embeddings, shape ``(B, 512, L)``.

    ``attn``
        Token-to-token adjacency source, shape ``(B, L, L)``.
        Edges are constructed by taking non-zero entries via ``.nonzero()``.
        This tensor should therefore be adjacency-like (sparse preferred).

    ``structure``
        Token-aligned structure features, shape ``(B, 1, L)``.

    ``motif``
        Motif prior scores, shape ``(B, 1, M)``. This branch is projected and then
        padded inside ``forward`` to match the fixed target length (currently ``101``).

    ``biochem``
        Token-aligned biochemical features, shape ``(B, 99, L)``.

Output
    A single logit per example, shape ``(B, 1)`` (from ``ADPNet`` classifier).

Architecture summary
--------------------
1) Graph branch (PyG GCN)
    - Node features originate from ``bert_embedding`` tokens.
    - Edge list (``edge_index``) is derived from ``attn[i].nonzero()`` per sample.
    - Graph output channels: ``512 -> 32`` then reshaped back to ``(B, 32, L)``.

2) Per-modality convolution + multiscale KAN
    - Embedding: ``512 -> 256`` then ``multiscaleKAN(256 -> 128)``
    - Structure: ``1 -> 128`` then ``multiscaleKAN(128 -> 64)``
    - Motif: ``1 -> 64`` then ``multiscaleKAN(64 -> 32)`` then pad to length ``101``
    - Biochem: ``99 -> 32`` then ``multiscaleKAN(32 -> 16)``

3) Fusion + ADPNet
    All branches are concatenated along channels and fed into ``ADPNet``:

    - GCN branch: 32 channels
    - Embedding branch: (depends on ``multiscaleKAN`` concat rule; see note below)
    - Structure branch: ...
    - Motif branch: ...
    - Biochem branch: ...

    The concatenated tensor is expected to match the ``ADPNet(filter_num=512, ...)``
    input channel count. If you change any branch widths, you must keep this consistent.

.. important::
   **Channel arithmetic and ``multiscaleKAN`` behavior**

   ``multiscaleKAN`` returns ``torch.cat([x0, x1], dim=1) + x``. This implies:

   - The concatenated tensor ``cat([x0, x1])`` must have the same channel dimension as ``x``
     for the residual addition to be valid.
   - If you modify ``in_channel`` or ``out_channel``, ensure shapes remain compatible.

   If you encounter shape errors, verify the output channel sizes of
   ``SimpleConvKAN_1layer`` and the intended residual path.

   **Graph construction and batching**

   This implementation builds edges by concatenating per-sample ``edge_index`` tensors:

   - ``edge_index`` is created from each ``attn[i].nonzero()`` and concatenated along ``dim=1``.
   - Node features are flattened to ``(B*L, 512)``.

   For correct batching in PyG, edges from sample ``i`` must reference nodes in the range
   ``[i*L, (i+1)*L)``. If your ``edge_index`` is not offset per sample, edges from different
   samples may incorrectly connect across the batch.

   If you see unexpected behavior, consider using ``torch_geometric.data.Batch`` utilities
   or offset indices by ``i * L`` when concatenating.

How to use
----------
Instantiate and run a forward pass (example shapes only):

.. code-block:: python

    import torch
    from my_module import BRIDGE

    B, L, M = 2, 101, 81
    model = BRIDGE()

    bert_embedding = torch.randn(B, 512, L)
    attn = (torch.rand(B, L, L) > 0.95).to(torch.int)   # sparse-ish adjacency
    structure = torch.randn(B, 1, L)
    motif = torch.randn(B, 1, M)
    biochem = torch.randn(B, 99, L)

    logits = model(bert_embedding, attn, structure, motif, biochem)  # (B, 1)

Notes and caveats
-----------------
- Input alignment:
  All token-aligned modalities must share the same ``L``. Any truncation/padding should be
  handled consistently in preprocessing.

"""

from utils.resnet import *  # brings in ResNet building blocks used elsewhere in this file/project
from math import log  # used to compute how many pyramid layers ADPNet needs to shrink L to <=2
import torch  # core tensor library
import torch.nn as nn  # neural network layer/module base classes
import torch.nn.functional as F  # functional ops (e.g. padding) used in forward passes
from utils.conv_layer import Conv1d, SimpleConvKAN_1layer  # project-local conv building blocks (plain conv and KAN-based conv)
from torch_geometric.nn import GCNConv  # graph convolution layer used for the token-adjacency branch
from typing import Iterable, List, Union  # type hints for constructor signatures


# Channel width contributed to the fused ADPNet input by each modality branch.
# Used by feature-ablation: dropping a feature shrinks the fusion (and head) input
# from 512 by the corresponding amount. Sum of all five == 512.
FEATURE_CHANNELS = {
    "gcn": 32,        # graph branch  (GCNConv 512 -> 32)
    "sequence": 256,  # RBPformer embedding branch (conv_bert + multiscale_bert)
    "structure": 128, # icSHAPE branch (conv_str + multiscale_str)
    "motif": 64,      # STREME motif-prior branch (conv_motif + multiscale_motif)
    "biochem": 32,    # biochemical k-mer branch (conv_biochem + multiscale_biochem)
}

# Channel width contributed by the optional "protein" branch (add_protein=True). Additive:
# fusion width becomes 512 + PROTEIN_CHANNELS (minus any dropped feature), not a replacement.
PROTEIN_CHANNELS = 32

# Channel width contributed by the optional protein cross-attention branch (attn_protein=True).
# Set equal to PROTEIN_CHANNELS on purpose, so this config's channel budget matches add_protein's
# and any AUC delta between the two isolates "real cross-attention" vs "additive bias" rather
# than reflecting a difference in fusion width.
PROTEIN_ATTN_CHANNELS = 32
# Internal attention dimensionality and head count for the protein cross-attention branch.
PROTEIN_ATTN_DMODEL = 128
PROTEIN_ATTN_HEADS = 8


class ADPNetblock(nn.Module):
    """
    Adaptive Pyramidal Network (ADPNet) block.

    This block performs:
        1. Constant padding for pooling.
        2. Max pooling (downsampling).
        3. Two convolutional layers with padding.
        4. Residual connection from pooled features to the output.

    Args:
        filter_num (int): Number of input/output channels (kept constant inside the block).
        kernel_size (int): Size of the convolution kernel.
        dilation (int): Dilation factor for the convolution.

    Note:
        - Padding sizes are computed to preserve spatial length after convolution.
        - Max pooling reduces the sequence length before convolution.
    """
    def __init__(self, filter_num: int, kernel_size: int, dilation: int) -> None:
        super(ADPNetblock, self).__init__()  # standard nn.Module init
        self.conv = Conv1d(filter_num, filter_num, kernel_size=kernel_size, stride=1, dilation=dilation, same_padding=False)  # first conv, channels unchanged
        self.conv1 = Conv1d(filter_num, filter_num, kernel_size=kernel_size, stride=1, dilation=dilation, same_padding=False)  # second conv, channels unchanged
        self.max_pooling = nn.MaxPool1d(kernel_size=(3, ), stride=2)  # halves the length (roughly) for downsampling
        self.padding_conv = nn.ConstantPad1d(((kernel_size-1)//2)*dilation, 0)  # zero-pads to keep length constant across the two convs
        self.padding_pool = nn.ConstantPad1d((0, 1), 0)  # pads by 1 on the right so max_pooling has a valid window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ADPNet block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, L_out),
                          where L_out depends on pooling/stride.
        """
        x = self.padding_pool(x)  # (B, C, L) -> (B, C, L+1), so pooling below has an even window
        px = self.max_pooling(x)  # downsample: (B, C, L+1) -> (B, C, L_out); kept for the residual add
        x = self.padding_conv(px)  # pad before first conv to preserve length
        x = self.conv(x)  # (B, C, L_out) -> (B, C, L_out) conv + activation (see Conv1d)
        x = self.padding_conv(x)  # pad again before second conv
        x = self.conv1(x)  # second conv, same shape
        x = x + px  # residual connection: add pre-conv pooled features back
        return x  # (B, C, L_out)


class ADPNet(nn.Module):
    """
    ADPNet for classification.

    Args:
        filter_num (int): Number of convolutional filters (channels).
        number_of_layers (int): Number of pyramid layers.
    """

    def __init__(self, filter_num: int, number_of_layers: int) -> None:
        super(ADPNet, self).__init__()  # standard nn.Module init

        # Predefined kernel size and dilation lists
        self.kernel_size_list: List[int] = [1 + x * 2 for x in range(number_of_layers)]  # computed odd kernel sizes (immediately overridden below)
        self.kernel_size_list = [5, 5, 5, 5, 5, 5]  # hardcoded override: every pyramid layer uses kernel size 5
        self.dilation_list: List[int] = [1, 1, 1, 1, 1, 1]  # no dilation for any pyramid layer

        # Initial convolution layers
        self.conv = Conv1d(
            filter_num, filter_num, self.kernel_size_list[0],
            stride=1, dilation=1, same_padding=False
        )  # first stem conv, channels unchanged, kernel size 5
        self.conv1 = Conv1d(
            filter_num, filter_num, self.kernel_size_list[0],
            stride=1, dilation=1, same_padding=False
        )  # second stem conv, channels unchanged, kernel size 5
        # Max pooling layer for downsampling
        self.pooling = nn.MaxPool1d(kernel_size=(3, ), stride=2)  # unused directly here; ADPNetblock has its own pooling

        # Constant padding layers for convolutions and pooling
        self.padding_conv = nn.ConstantPad1d(((self.kernel_size_list[0] - 1) // 2), 0)  # pad amount to keep length constant for kernel size 5
        self.padding_pool = nn.ConstantPad1d((0, 1), 0)  # pad by 1 for pooling parity (unused directly here)

        # Pyramid blocks
        self.ADPNetblocklist = nn.ModuleList([
            ADPNetblock(
                filter_num,
                kernel_size=self.kernel_size_list[i],
                dilation=self.dilation_list[i]
            )
            for i in range(len(self.kernel_size_list))
        ])  # stack of pooling+conv blocks, each halving the sequence length

        # Final classification layer
        self.classifier = nn.Linear(filter_num, 1)  # maps pooled feature vector to a single logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ADPNet.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L).

        Returns:
            torch.Tensor: Output logits of shape (B, 1).
        """
        # Initial convolution stage
        x = self.padding_conv(x)  # pad to preserve length before first stem conv
        x = self.conv(x)  # first stem conv
        x = self.padding_conv(x)  # pad again before second stem conv
        x = self.conv1(x)  # second stem conv

        # Pyramid convolution blocks with downsampling until length <= 2
        i = 0  # index into ADPNetblocklist
        while x.size()[-1] > 2:  # keep shrinking the sequence length until it's small enough to pool globally
            x = self.ADPNetblocklist[i](x)  # apply next pyramid block (pool + conv + residual)
            i += 1  # advance to the next block for the next iteration

        # Global pooling and classification
        x = x.squeeze(-1).squeeze(-1)  # drop trailing singleton length dims (expects final L to collapse to 1)
        logits = self.classifier(x)  # (B, C) -> (B, 1) final binding-prediction logit
        return logits


class BRIDGE(nn.Module):
    """
    BRIDGE: Multimodal sequence-structure integration network.

    Tutorial overview:
        BRIDGE consumes multiple aligned modalities describing the same sequence tokens.

        Expected common alignment dimension:
            L = number of tokens per sequence (often fixed in this project, e.g., L=101).
            All token-aligned inputs should share the same L:
                - bert_embedding: (B, 512, L)
                - structure:      (B, 1,   L)
                - biochem:        (B, 99,  L)
                - attn:           (B, L,   L)  (graph adjacency source)

        The motif branch may start shorter (M != L) and is padded to L (here: 101) inside forward().

        Alignment requirement:
            All token-aligned inputs must share the same L (typically 101 in this project),
            which is enforced by data preprocessing and padding/truncation outside this module.

    Modalities:
        1) Transformer embedding    s (512 channels)      -> conv_bert -> multiscaleKAN
        2) Structure profiles (1 channel)            -> conv_str  -> multiscaleKAN
        3) Motif scores (1 channel, length M)        -> conv_motif-> multiscaleKAN -> pad to length 101
        4) Biochemical features (99 channels)        -> conv_biochem -> multiscaleKAN
        5) Graph branch from Transformer tokens via GCN:
            - node features come from token embeddings
            - edges are derived from `attn` via `.nonzero()`

    Graph construction note (important for new readers):
        - In this implementation, the graph topology is derived from `attn` (a token-to-token
            connectivity signal passed into `forward`), not from structure/thermodynamic features.
        - Structure and biochemical/thermodynamic features are used as separate token-aligned
            modalities (Conv1d branches) and are fused later via concatenation.

    """
    def __init__(
        self,
        k: int = 3,
        drop_feature: Union[str, Iterable[str], None] = None,
        kan_to_mlp: bool = False,
        adpnet_to_gap: bool = False,
        adpnet_to_attnpool: bool = False,
        add_protein: bool = False,
        protein_vector=None,
        attn_protein: bool = False,
        protein_residue_vector=None,
        attn_protein_scope: str = "full",
    ) -> None:
        """
        Args:
            k: convolution kernel size for the structure/biochem branches (as before).
            drop_feature: if set, removes one or more input-feature branches entirely and
                shrinks the fused ADPNet/GAP/attention-pool input from 512 by the sum of
                ``FEATURE_CHANNELS[f]`` over the dropped branches. Accepts a single feature
                name (``str``), an iterable of feature names (e.g. a ``set``/``list``, for
                dropping several branches at once -- see ``ablation.registry.LEAN_CONFIG``),
                or ``None`` (baseline, nothing dropped). Each name must be one of
                ``{"gcn","sequence","structure","motif","biochem"}``.
            kan_to_mlp: if True, replace every ``multiscaleKAN`` block with the structurally
                identical ``multiscaleMLP`` (Conv1d in place of the KAN operator).
            adpnet_to_gap: if True, replace the ``ADPNet`` head with ``GAPHead`` (global
                average pool over length -> ``Linear``).
            adpnet_to_attnpool: if True, replace the ``ADPNet`` head with ``AttnPoolHead``
                (learned attention pool over length -> ``Linear``). Mutually exclusive with
                ``adpnet_to_gap`` (only one head swap at a time).
            add_protein: if True, adds an extra input branch that projects a whole-protein
                ESM-2 embedding (``protein_vector``) to ``PROTEIN_CHANNELS`` and concatenates
                it into the fusion (additive: 512 -> 512 + PROTEIN_CHANNELS). Since BRIDGE
                trains one model per single RBP, this vector is the SAME for every sample in
                a run -- it can only be absorbed as a learned bias, not a per-sample signal.
                This branch exists to test that prediction empirically, not because it is
                expected to help (see ``ablation/registry.py``'s ``"protein"`` config).
            protein_vector: required when ``add_protein=True``; a ``(1280,)``-shaped
                array/tensor (e.g. from ``utils.protein_features.load_protein_embedding``),
                stored as a non-trainable buffer.
            attn_protein: if True, adds a real multi-head cross-attention branch: RNA
                per-position fused features act as queries, per-residue protein ESM-2
                embeddings (``protein_residue_vector``) act as keys/values. Unlike
                ``add_protein``, attention weights vary by RNA position and by the actual
                content of both sides -- not absorbable as a learned bias. Mutually exclusive
                with ``add_protein`` (see ``ablation/registry.py``'s ``"attn_protein"``
                config).
            protein_residue_vector: required when ``attn_protein=True``; a
                ``(P, 1280)``-shaped array/tensor (P = protein length, e.g. from
                ``utils.protein_features.load_protein_residue_embedding``), stored as a
                non-trainable buffer. Constant across the batch (same RBP every sample) but
                varies across residues, so it can act as a genuine multi-token key/value set.
            attn_protein_scope: which RNA representation feeds the cross-attention query when
                ``attn_protein=True``. ``"full"`` (default): the fused per-position features
                from all (non-dropped) RNA branches -- structure/motif/biochem context can
                shape what attends over the protein. ``"sequence"``: only the sequence/BERT
                branch (``x0``, pre-fusion with the other branches) -- restricts the query to
                the RNA-BERT "language" embedding, so attention is literally one pretrained
                sequence-embedding space (RNA-BERT) attending over another (ESM-2), with no
                engineered-feature context mixed in. Isolates whether an AUC delta comes from
                real sequence-level RNA-protein complementarity vs. extra fusion capacity from
                the other branches. Ignored when ``attn_protein=False``. Requires that
                ``"sequence"`` not be among the dropped features (there is no ``x0`` to attend
                from otherwise).

        With all four (five) at their defaults this reproduces the original baseline model
        exactly (identical module construction order, so fixed-seed initialization is
        unchanged for every config that leaves ``add_protein=False`` and ``attn_protein=False``).
        """
        super().__init__()  # standard nn.Module init
        # Normalize drop_feature to a frozenset of branch names, whether the caller passed
        # a single string, an iterable of strings, or None. A bare string is iterable itself
        # (would silently split into characters), so it must be special-cased.
        if drop_feature is None:
            drop_feature = frozenset()  # nothing dropped: baseline configuration
        elif isinstance(drop_feature, str):
            drop_feature = frozenset({drop_feature})  # wrap single feature name so it isn't iterated char-by-char
        else:
            drop_feature = frozenset(drop_feature)  # normalize any iterable of names to a frozenset
        unknown = drop_feature - FEATURE_CHANNELS.keys()  # any requested names not in the known branch set
        if unknown:
            raise ValueError(
                f"Unknown drop_feature {sorted(unknown)!r}; expected any of "
                f"{list(FEATURE_CHANNELS)}"
            )  # fail fast on typos/unsupported branch names
        if adpnet_to_gap and adpnet_to_attnpool:
            raise ValueError(
                "adpnet_to_gap and adpnet_to_attnpool are mutually exclusive; "
                "set at most one to swap the ADPNet head."
            )  # only one head-swap ablation may be active at a time
        if add_protein and protein_vector is None:
            raise ValueError("add_protein=True requires protein_vector (a (1280,) array/tensor).")  # protein branch needs its input vector
        if attn_protein and protein_residue_vector is None:
            raise ValueError(
                "attn_protein=True requires protein_residue_vector (a (P, 1280) array/tensor)."
            )  # cross-attention branch needs per-residue embeddings
        if add_protein and attn_protein:
            raise ValueError(
                "add_protein and attn_protein are mutually exclusive; pick one protein-fusion "
                "mechanism."
            )  # only one protein-fusion mechanism may be active at a time
        if attn_protein_scope not in ("full", "sequence"):
            raise ValueError(
                f"attn_protein_scope must be 'full' or 'sequence', got {attn_protein_scope!r}."
            )  # guard against invalid scope strings
        if attn_protein and attn_protein_scope == "sequence" and "sequence" in drop_feature:
            raise ValueError(
                "attn_protein_scope='sequence' requires the sequence branch ('sequence' must "
                "not be among the dropped features)."
            )  # sequence-scope attention needs x0, which requires the sequence branch to exist
        self.drop_feature = drop_feature  # store normalized ablation set for use in forward()
        self.kan_to_mlp = kan_to_mlp  # remember whether to use MLP instead of KAN blocks
        self.adpnet_to_gap = adpnet_to_gap  # remember head-swap choice (GAP)
        self.adpnet_to_attnpool = adpnet_to_attnpool  # remember head-swap choice (attention pool)
        self.add_protein = add_protein  # remember whether additive protein branch is active
        self.attn_protein = attn_protein  # remember whether cross-attention protein branch is active
        self.attn_protein_scope = attn_protein_scope  # remember which RNA representation feeds the cross-attention query

        number_of_layers = int(log(101-k+1, 2))  # derive pyramid depth so ADPNet's downsampling reaches length ~1 (unused after kernel_size_list override, kept for constructor argument)
        # Multiscale block class: KAN (baseline) or its MLP (Conv1d) analog.
        ms = multiscaleMLP if kan_to_mlp else multiscaleKAN  # select the per-modality feature extractor class

        # ===== Modality-specific projection layers =====
        # NOTE: construction order below is kept identical to the original model so that,
        # for the baseline config, per-layer RNG consumption (and thus fixed-seed init) is
        # byte-for-byte unchanged. Ablated branches are simply skipped.
        if "sequence" not in drop_feature:
            self.conv_bert = Conv1d(512, 256, kernel_size=(1,), stride=1)  # project Transformer embedding channels 512 -> 256
        if "structure" not in drop_feature:
            self.conv_str = Conv1d(1, 128, kernel_size=(k,), stride=1, same_padding=True)  # project structure channel 1 -> 128, length-preserving
        if "motif" not in drop_feature:
            self.conv_motif = Conv1d(1, 64, kernel_size=(1,), stride=1)  # project motif channel 1 -> 64
        if "biochem" not in drop_feature:
            self.conv_biochem = Conv1d(99, 32, kernel_size=(k,), stride=1, same_padding=True)  # project biochem channels 99 -> 32, length-preserving

        # ===== Multiscale feature extractors (KAN or MLP) =====
        if "sequence" not in drop_feature:
            self.multiscale_bert = ms(256, 128)  # sequence branch: 256 -> (128 + 128) via multiscale residual block
        if "structure" not in drop_feature:
            self.multiscale_str = ms(128, 64)  # structure branch: 128 -> (64 + 64)
        if "motif" not in drop_feature:
            self.multiscale_motif = ms(64, 32)  # motif branch: 64 -> (32 + 32)
        if "biochem" not in drop_feature:
            self.multiscale_biochem = ms(32, 16)  # biochem branch: 32 -> (16 + 16)

        # ===== Protein branch (optional additive control; see add_protein docstring above) =====
        if add_protein:
            protein_vector = torch.as_tensor(protein_vector, dtype=torch.float32).view(-1)  # coerce to a flat float32 tensor of shape (1280,)
            self.register_buffer("protein_vector", protein_vector)  # store as a non-trainable buffer (moves with .to(device), not updated by optimizer)
            self.protein_proj = nn.Sequential(
                nn.Linear(protein_vector.numel(), 64),  # project whole-protein embedding down to 64 dims
                nn.ReLU(),  # nonlinearity
                nn.Linear(64, PROTEIN_CHANNELS),  # project to the channel width added to fusion
            )

        # ===== Protein cross-attention branch (see attn_protein docstring above) =====
        if attn_protein:
            protein_residue_vector = torch.as_tensor(protein_residue_vector, dtype=torch.float32)  # coerce to float32 tensor of shape (P, 1280)
            self.register_buffer("protein_residue_vector", protein_residue_vector)  # store as non-trainable buffer, shared across the batch
            if attn_protein_scope == "sequence":
                q_in_ch = FEATURE_CHANNELS["sequence"]  # query comes from x0 alone: 256 channels
            else:
                q_in_ch = 512 - sum(FEATURE_CHANNELS[f] for f in drop_feature)  # query comes from all fused RNA branches so far (512 minus any dropped)
            self.protein_query_proj = Conv1d(q_in_ch, PROTEIN_ATTN_DMODEL, kernel_size=(1,), stride=1)  # project RNA query features to attention d_model
            self.protein_kv_proj = nn.Linear(protein_residue_vector.shape[-1], PROTEIN_ATTN_DMODEL)  # project per-residue ESM-2 embeddings (1280) to d_model
            self.protein_cross_attn = nn.MultiheadAttention(
                PROTEIN_ATTN_DMODEL, PROTEIN_ATTN_HEADS, dropout=0.1, batch_first=True
            )  # standard multi-head cross-attention: RNA positions attend over protein residues
            self.protein_attn_out_proj = Conv1d(
                PROTEIN_ATTN_DMODEL, PROTEIN_ATTN_CHANNELS, kernel_size=(1,), stride=1
            )  # project attention output back down to the channel width added to fusion

        # ===== Classifier head (ADPNet pyramid, GAP, or attention pool) over the fused features =====
        fusion_ch = (
            512
            - sum(FEATURE_CHANNELS[f] for f in drop_feature)  # subtract channels for any dropped branch
            + (PROTEIN_CHANNELS if add_protein else 0)  # add additive-protein channels if enabled
            + (PROTEIN_ATTN_CHANNELS if attn_protein else 0)  # add cross-attention-protein channels if enabled
        )  # total channel width of the concatenated fused feature tensor fed to the head
        if adpnet_to_gap:
            self.adpnet = GAPHead(fusion_ch)  # ablation: replace pyramid with global-average-pool head
        elif adpnet_to_attnpool:
            self.adpnet = AttnPoolHead(fusion_ch)  # ablation: replace pyramid with learned attention-pool head
        else:
            self.adpnet = ADPNet(fusion_ch, number_of_layers)  # baseline: full ADPNet pyramidal head

        # ===== Graph Convolution =====
        if "gcn" not in drop_feature:
            self.gcn = GCNConv(512, 32)  # single GCN layer projecting token embeddings 512 -> 32 along the attention-derived graph

        # ===== Initialize weights =====
        self._initialize_weights()  # apply custom init scheme to conv/batchnorm/linear submodules

    def _initialize_weights(self) -> None:
        """
        Initialize learnable parameters for key layer types.

        Tutorial notes:
            Weight initialization can significantly affect optimization stability,
            especially in deep networks with ReLU-like nonlinearities.

            This routine applies common, sensible defaults:
            - Convolution layers (Conv1d/Conv2d): Kaiming/He initialization
                suited for ReLU activations (good variance preservation).
            - BatchNorm layers: start as identity transform
                (gamma=1, beta=0), so normalization does not distort features at init.
            - Linear layers: small Gaussian initialization to start near zero.

            Bias terms (when present) are set to zero to avoid introducing
            an initial offset in activations.

        Scope:
            This iterates over `self.modules()`, so it will touch layers inside
            submodules as well (e.g., ADPNet, etc.) *if* they use
            standard PyTorch layer classes (nn.Conv*, nn.BatchNorm*, nn.Linear).

            Layers not explicitly matched here (e.g., GCNConv, custom layers) keep
            their own default initialization unless they internally use nn.Linear
            submodules that appear in `self.modules()`.
        """
        for m in self.modules():  # walk every submodule recursively
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He init tuned for ReLU
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # zero bias
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He init tuned for ReLU
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # zero bias
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # gamma=1 (identity scale at init)
                nn.init.constant_(m.bias, 0)  # beta=0 (no shift at init)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)  # gamma=1 (identity scale at init)
                nn.init.constant_(m.bias, 0)  # beta=0 (no shift at init)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # small Gaussian weights to start near zero output
                nn.init.constant_(m.bias, 0)  # zero bias


    def forward(
        self,
        bert_embedding: torch.Tensor,  # shape: (B, 512, L)
        attn: torch.Tensor,            # shape: (B, L, L), adjacency from attention
        structure: torch.Tensor,       # shape: (B, 1, L)
        motif: torch.Tensor,           # shape: (B, 1, M)
        biochem: torch.Tensor           # shape: (B, 99, L)
    ) -> torch.Tensor:
        """
        Forward pass of BRIDGE.

        Inputs:
            bert_embedding:
                Token-aligned Transformer embeddings.
                Must be channel-first: (B, 512, L).
                This matches build_Transformer_embeddings(..., transpose_to_ch_first=True).

            attn:
                Token-to-token connectivity signal, expected shape (B, L, L).
                Used to build `edge_index` for the GCN branch.
                Expected to be adjacency-like before calling this module, since edges are
                extracted via `.nonzero()`.

            structure:
                Token-aligned structure profile (e.g., pairing probability), (B, 1, L).

            motif:
                Motif prior scores, (B, 1, M). This branch pads motif features to length 101.

            biochem:
                Token-aligned biochemical features, (B, 99, L).

        Output:
            Model prediction produced by ADPNet after multimodal fusion and multiscale KAN feature extractors.
        """
        # Branch outputs are collected in the original fusion order
        # [gcn, sequence, structure, motif, biochem]; an ablated branch is skipped so the
        # concatenated channel count matches the (possibly shrunk) head input width.
        feats = []  # list of (B, C_i, L) tensors to be concatenated along channels before the head

        # ===== Graph branch (GCN over tokens) =====
        if "gcn" not in self.drop_feature:
            node_features = bert_embedding  # (B, 512, L)
            adj = attn                      # (B, L, L)

            # Here, `num_nodes` is taken from adj.shape[1] (i.e., L)
            batch_size, num_nodes, _ = adj.shape  # unpack batch size and per-sample node count (L)

            # Build edge indices per sample by taking all nonzero entries in adjacency.
            edge_index_list = []  # collects each sample's (2, E_i) edge_index tensor
            for i in range(batch_size):
                edge_index = adj[i].nonzero(as_tuple=False).t().contiguous()  # (2, E_i) coordinates of nonzero adjacency entries, transposed to PyG's edge_index layout
                edge_index_list.append(edge_index)  # accumulate this sample's edges
            edge_index = torch.cat(edge_index_list, dim=1)  # concatenate all samples' edges along the edge dimension (see batching caveat in module docstring)

            # Convert node features from (B, 512, L) -> (B*L, 512) to feed into PyG GCNConv
            node_features = node_features.permute(0, 2, 1).contiguous().view(-1, 512)  # flatten batch and token dims into one node axis
            xg = self.gcn(node_features, edge_index)  # (B*L, 512) -> (B*L, 32) graph convolution

            # Convert back to (B, 32, L)
            xg = xg.view(batch_size, num_nodes, -1).permute(0, 2, 1).contiguous()  # unflatten back to per-sample, channel-first layout
            feats.append(xg)  # add graph branch output to the fusion list

        # ===== Embedding branch =====
        if "sequence" not in self.drop_feature:
            x0 = self.conv_bert(bert_embedding)  # (B, 512, L) -> (B, 256, L)
            x0 = self.multiscale_bert(x0)  # multiscale residual extraction, channels widen to 256 (128+128)
            feats.append(x0)  # add sequence branch output to the fusion list

        # ===== Structure branch =====
        if "structure" not in self.drop_feature:
            x1 = self.conv_str(structure)  # (B, 1, L) -> (B, 128, L)
            x1 = self.multiscale_str(x1)  # multiscale residual extraction, channels widen to 128 (64+64)
            feats.append(x1)  # add structure branch output to the fusion list

        # ===== Motif branch =====
        if "motif" not in self.drop_feature:
            x2 = self.conv_motif(motif)  # (B, 1, M) -> (B, 64, M)
            x2 = self.multiscale_motif(x2)  # multiscale residual extraction, channels widen to 64 (32+32), length still M

            # Pad motif features to the fixed target length (101)
            total_padding = 101 - x2.size(2)  # how many positions short of the target length 101
            left_pad = total_padding // 2  # split padding evenly, left half
            right_pad = total_padding - left_pad  # remaining padding goes on the right
            x2 = F.pad(x2, (left_pad, right_pad), "constant", 0)  # zero-pad the length axis so motif aligns with the other branches
            feats.append(x2)  # add motif branch output to the fusion list

        # ===== Biochemical branch =====
        if "biochem" not in self.drop_feature:
            x3 = self.conv_biochem(biochem)  # (B, 99, L) -> (B, 32, L)
            x3 = self.multiscale_biochem(x3)  # multiscale residual extraction, channels widen to 32 (16+16)
            feats.append(x3)  # add biochem branch output to the fusion list

        # ===== Protein cross-attention branch (real, non-degenerate fusion) =====
        if self.attn_protein:
            batch_size = bert_embedding.shape[0]  # needed to broadcast the shared protein embedding across the batch
            # scope="sequence": query is the RNA-BERT branch alone (x0), pre-fusion with the
            # other branches -- two pretrained sequence-LM spaces (RNA-BERT, ESM-2) attending
            # directly on each other. scope="full" (default): query is every RNA branch
            # computed so far. See attn_protein_scope docstring above.
            q_in = x0 if self.attn_protein_scope == "sequence" else torch.cat(feats, dim=1)  # (B, C, L)
            q = self.protein_query_proj(q_in).permute(0, 2, 1)              # (B, L, d_model)
            kv = self.protein_kv_proj(self.protein_residue_vector)          # (P, d_model)
            kv = kv.unsqueeze(0).expand(batch_size, -1, -1).contiguous()    # (B, P, d_model)
            attn_out, attn_weights = self.protein_cross_attn(
                q, kv, kv, need_weights=True, average_attn_weights=True
            )                                                               # attn_out: (B, L, d_model)
            self._last_protein_attn_weights = attn_weights.detach()         # (B, L, P), for diagnostics
            attn_out = self.protein_attn_out_proj(attn_out.permute(0, 2, 1))  # (B, PROTEIN_ATTN_CHANNELS, L)
            feats.append(attn_out)  # add protein cross-attention branch output to the fusion list

        # ===== Protein branch (optional additive control) =====
        if self.add_protein:
            batch_size = bert_embedding.shape[0]  # needed to broadcast the shared protein vector across the batch
            length = bert_embedding.shape[-1]  # needed to broadcast the (per-batch-constant) protein feature across positions
            protein_feat = self.protein_proj(self.protein_vector)          # (PROTEIN_CHANNELS,)
            protein_feat = protein_feat.view(1, -1, 1).expand(batch_size, -1, length)  # broadcast the same vector to every sample and position (learnable bias only)
            feats.append(protein_feat)  # add additive protein branch output to the fusion list

        # ===== Fusion =====
        x = torch.cat(feats, dim=1)  # concatenate all active branches along the channel dimension
        return self.adpnet(x)  # run the fused features through the classifier head to get logits


class multiscaleKAN(nn.Module):
    """
    Multi-scale KAN block with residual connection.

    This module applies:
        - Path 0: A single 1x1 SimpleConvKAN layer.
        - Path 1: A 1x1 SimpleConvKAN followed by a 3x3 SimpleConvKAN.

    The outputs of both paths are concatenated along the channel dimension
    and then added to the original input (residual connection).

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels per path.

    Note:
        Final output channel dimension = in_channel + 2 * out_channel
        only if the residual input's channel size matches the concatenated output's channel size.
    """

    def __init__(self, in_channel: int, out_channel: int) -> None:
        super(multiscaleKAN, self).__init__()  # standard nn.Module init

        self.conv0 = SimpleConvKAN_1layer(in_channel, out_channel, kernel_size=1, same_padding=False, grid_size=2, dropout=0.3)  # path 0: single 1x1 KAN conv
        self.conv1 = nn.Sequential(
            SimpleConvKAN_1layer(in_channel, out_channel, kernel_size=1, same_padding=False, bn=False, grid_size=2, dropout=0.3),  # path 1 stage a: 1x1 KAN conv, no batchnorm
            SimpleConvKAN_1layer(out_channel, out_channel, kernel_size=3, same_padding=True, grid_size=4, dropout=0.3),  # path 1 stage b: 3x3 KAN conv, length-preserving
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L)

        Returns:
            torch.Tensor: Output tensor after multi-scale feature extraction
                          and residual addition. Shape is (B, C_out, L),
                          where C_out = C + 2*out_channel if concatenation
                          changes channels, else same as input.
        """
        x0 = self.conv0(x)  # path 0 output, (B, out_channel, L)
        x1 = self.conv1(x)  # path 1 output, (B, out_channel, L)
        return torch.cat([x0,x1], dim=1) + x  # concat both paths' channels, then residual-add the original input


class multiscaleMLP(nn.Module):
    """
    MLP analog of :class:`multiscaleKAN` for the ``kan_to_mlp`` module ablation.

    Structurally identical to ``multiscaleKAN`` (same 2-path residual, same channel widths),
    but each KAN operator (``SimpleConvKAN_1layer``) is replaced by the project's standard
    ``Conv1d`` block (Conv1d -> BatchNorm -> ReLU -> dropout) at the *same* kernel sizes
    (path0: k=1; path1: k=1 then k=3). Because ``multiscaleKAN`` satisfies ``2*out == in``,
    the concatenated two paths add cleanly to the residual, so the output channel count is
    ``in_channel`` — a drop-in replacement that isolates KAN-vs-conv as the only change.

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Per-path output channels (must satisfy ``2*out_channel == in_channel``).
    """
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super(multiscaleMLP, self).__init__()  # standard nn.Module init
        self.conv0 = Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False)  # path 0: plain 1x1 conv (KAN replacement)
        self.conv1 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False, bn=False),  # path 1 stage a: plain 1x1 conv, no batchnorm
            Conv1d(out_channel, out_channel, kernel_size=(3,), same_padding=True),  # path 1 stage b: plain 3x3 conv, length-preserving
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.conv0(x)  # path 0 output, (B, out_channel, L)
        x1 = self.conv1(x)  # path 1 output, (B, out_channel, L)
        return torch.cat([x0, x1], dim=1) + x  # concat both paths' channels, then residual-add the original input


class GAPHead(nn.Module):
    """
    Global-average-pooling classifier head for the ``adpnet_to_gap`` module ablation.

    Replaces the entire ADPNet pyramidal refinement: averages each channel over the length
    axis (``(B, C, L) -> (B, C)``), then applies a single ``Linear(C, 1)`` to produce the
    binding logit. Mirrors the baseline's final ``Linear`` classifier (which ADPNet also ends
    with) so only the pyramid is ablated, not the classifier itself.

    Args:
        filter_num (int): Number of fused input channels (C).
    """
    def __init__(self, filter_num: int) -> None:
        super(GAPHead, self).__init__()  # standard nn.Module init
        self.classifier = nn.Linear(filter_num, 1)  # maps pooled per-channel means to a single logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=-1)          # (B, C, L) -> (B, C)
        return self.classifier(x)   # (B, 1)


class AttnPoolHead(nn.Module):
    """
    Attention-pooling classifier head for the ``adpnet_to_attnpool`` module ablation.

    Replaces the entire ADPNet pyramidal refinement with a single learned-query attention
    pool: a per-position score (``Linear(C,1)``) is softmax-normalized over the length axis
    to produce content-based weights, which then combine the length axis into one vector
    (``(B, C, L) -> (B, C)``) before the same final ``Linear(C, 1)`` classifier used by
    ``GAPHead``. Parameter count is deliberately close to ``GAPHead`` (one extra
    ``Linear(C, 1)`` scorer) so a delta vs. GAP isolates content-based weighting from the
    pyramid's depth, rather than from added model capacity.

    Args:
        filter_num (int): Number of fused input channels (C).
    """
    def __init__(self, filter_num: int) -> None:
        super(AttnPoolHead, self).__init__()  # standard nn.Module init
        self.score = nn.Linear(filter_num, 1)  # produces one attention logit per position
        self.classifier = nn.Linear(filter_num, 1)  # maps the pooled vector to a single logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)                    # (B, C, L) -> (B, L, C)
        w = torch.softmax(self.score(h), dim=1)  # (B, L, 1) attention weights over L
        pooled = (h * w).sum(dim=1)               # (B, C)
        return self.classifier(pooled)             # (B, 1)
