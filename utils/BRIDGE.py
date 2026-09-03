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

from utils.resnet import *
from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.conv_layer import Conv1d, SimpleConvKAN_1layer
from torch_geometric.nn import GCNConv
from typing import Iterable, List, Union


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
        super(ADPNetblock, self).__init__()
        self.conv = Conv1d(filter_num, filter_num, kernel_size=kernel_size, stride=1, dilation=dilation, same_padding=False)
        self.conv1 = Conv1d(filter_num, filter_num, kernel_size=kernel_size, stride=1, dilation=dilation, same_padding=False)
        self.max_pooling = nn.MaxPool1d(kernel_size=(3, ), stride=2)
        self.padding_conv = nn.ConstantPad1d(((kernel_size-1)//2)*dilation, 0)
        self.padding_pool = nn.ConstantPad1d((0, 1), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ADPNet block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, L_out),
                          where L_out depends on pooling/stride.
        """
        x = self.padding_pool(x)
        px = self.max_pooling(x)
        x = self.padding_conv(px)
        x = self.conv(x)
        x = self.padding_conv(x)
        x = self.conv1(x)
        x = x + px
        return x


class ADPNet(nn.Module):
    """
    ADPNet for classification.

    Args:
        filter_num (int): Number of convolutional filters (channels).
        number_of_layers (int): Number of pyramid layers.
    """
    
    def __init__(self, filter_num: int, number_of_layers: int) -> None:
        super(ADPNet, self).__init__()
        
        # Predefined kernel size and dilation lists
        self.kernel_size_list: List[int] = [1 + x * 2 for x in range(number_of_layers)]
        self.kernel_size_list = [5, 5, 5, 5, 5, 5]
        self.dilation_list: List[int] = [1, 1, 1, 1, 1, 1]
        
        # Initial convolution layers
        self.conv = Conv1d(
            filter_num, filter_num, self.kernel_size_list[0],
            stride=1, dilation=1, same_padding=False
        )
        self.conv1 = Conv1d(
            filter_num, filter_num, self.kernel_size_list[0],
            stride=1, dilation=1, same_padding=False
        )
        # Max pooling layer for downsampling
        self.pooling = nn.MaxPool1d(kernel_size=(3, ), stride=2)
        
        # Constant padding layers for convolutions and pooling
        self.padding_conv = nn.ConstantPad1d(((self.kernel_size_list[0] - 1) // 2), 0)
        self.padding_pool = nn.ConstantPad1d((0, 1), 0)
        
        # Pyramid blocks
        self.ADPNetblocklist = nn.ModuleList([
            ADPNetblock(
                filter_num, 
                kernel_size=self.kernel_size_list[i],
                dilation=self.dilation_list[i]
            )
            for i in range(len(self.kernel_size_list))
        ])
        
        # Final classification layer
        self.classifier = nn.Linear(filter_num, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ADPNet.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L).

        Returns:
            torch.Tensor: Output logits of shape (B, 1).
        """
        # Initial convolution stage
        x = self.padding_conv(x)
        x = self.conv(x)
        x = self.padding_conv(x)
        x = self.conv1(x)
        
        # Pyramid convolution blocks with downsampling until length <= 2
        i = 0
        while x.size()[-1] > 2:
            x = self.ADPNetblocklist[i](x)
            i += 1
            
        # Global pooling and classification
        x = x.squeeze(-1).squeeze(-1)
        logits = self.classifier(x)
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
        super().__init__()
        # Normalize drop_feature to a frozenset of branch names, whether the caller passed
        # a single string, an iterable of strings, or None. A bare string is iterable itself
        # (would silently split into characters), so it must be special-cased.
        if drop_feature is None:
            drop_feature = frozenset()
        elif isinstance(drop_feature, str):
            drop_feature = frozenset({drop_feature})
        else:
            drop_feature = frozenset(drop_feature)
        unknown = drop_feature - FEATURE_CHANNELS.keys()
        if unknown:
            raise ValueError(
                f"Unknown drop_feature {sorted(unknown)!r}; expected any of "
                f"{list(FEATURE_CHANNELS)}"
            )
        if adpnet_to_gap and adpnet_to_attnpool:
            raise ValueError(
                "adpnet_to_gap and adpnet_to_attnpool are mutually exclusive; "
                "set at most one to swap the ADPNet head."
            )
        if add_protein and protein_vector is None:
            raise ValueError("add_protein=True requires protein_vector (a (1280,) array/tensor).")
        if attn_protein and protein_residue_vector is None:
            raise ValueError(
                "attn_protein=True requires protein_residue_vector (a (P, 1280) array/tensor)."
            )
        if add_protein and attn_protein:
            raise ValueError(
                "add_protein and attn_protein are mutually exclusive; pick one protein-fusion "
                "mechanism."
            )
        if attn_protein_scope not in ("full", "sequence"):
            raise ValueError(
                f"attn_protein_scope must be 'full' or 'sequence', got {attn_protein_scope!r}."
            )
        if attn_protein and attn_protein_scope == "sequence" and "sequence" in drop_feature:
            raise ValueError(
                "attn_protein_scope='sequence' requires the sequence branch ('sequence' must "
                "not be among the dropped features)."
            )
        self.drop_feature = drop_feature
        self.kan_to_mlp = kan_to_mlp
        self.adpnet_to_gap = adpnet_to_gap
        self.adpnet_to_attnpool = adpnet_to_attnpool
        self.add_protein = add_protein
        self.attn_protein = attn_protein
        self.attn_protein_scope = attn_protein_scope

        number_of_layers = int(log(101-k+1, 2))
        # Multiscale block class: KAN (baseline) or its MLP (Conv1d) analog.
        ms = multiscaleMLP if kan_to_mlp else multiscaleKAN

        # ===== Modality-specific projection layers =====
        # NOTE: construction order below is kept identical to the original model so that,
        # for the baseline config, per-layer RNG consumption (and thus fixed-seed init) is
        # byte-for-byte unchanged. Ablated branches are simply skipped.
        if "sequence" not in drop_feature:
            self.conv_bert = Conv1d(512, 256, kernel_size=(1,), stride=1)
        if "structure" not in drop_feature:
            self.conv_str = Conv1d(1, 128, kernel_size=(k,), stride=1, same_padding=True)
        if "motif" not in drop_feature:
            self.conv_motif = Conv1d(1, 64, kernel_size=(1,), stride=1)
        if "biochem" not in drop_feature:
            self.conv_biochem = Conv1d(99, 32, kernel_size=(k,), stride=1, same_padding=True)

        # ===== Multiscale feature extractors (KAN or MLP) =====
        if "sequence" not in drop_feature:
            self.multiscale_bert = ms(256, 128)
        if "structure" not in drop_feature:
            self.multiscale_str = ms(128, 64)
        if "motif" not in drop_feature:
            self.multiscale_motif = ms(64, 32)
        if "biochem" not in drop_feature:
            self.multiscale_biochem = ms(32, 16)

        # ===== Protein branch (optional additive control; see add_protein docstring above) =====
        if add_protein:
            protein_vector = torch.as_tensor(protein_vector, dtype=torch.float32).view(-1)
            self.register_buffer("protein_vector", protein_vector)
            self.protein_proj = nn.Sequential(
                nn.Linear(protein_vector.numel(), 64),
                nn.ReLU(),
                nn.Linear(64, PROTEIN_CHANNELS),
            )

        # ===== Protein cross-attention branch (see attn_protein docstring above) =====
        if attn_protein:
            protein_residue_vector = torch.as_tensor(protein_residue_vector, dtype=torch.float32)
            self.register_buffer("protein_residue_vector", protein_residue_vector)
            if attn_protein_scope == "sequence":
                q_in_ch = FEATURE_CHANNELS["sequence"]
            else:
                q_in_ch = 512 - sum(FEATURE_CHANNELS[f] for f in drop_feature)
            self.protein_query_proj = Conv1d(q_in_ch, PROTEIN_ATTN_DMODEL, kernel_size=(1,), stride=1)
            self.protein_kv_proj = nn.Linear(protein_residue_vector.shape[-1], PROTEIN_ATTN_DMODEL)
            self.protein_cross_attn = nn.MultiheadAttention(
                PROTEIN_ATTN_DMODEL, PROTEIN_ATTN_HEADS, dropout=0.1, batch_first=True
            )
            self.protein_attn_out_proj = Conv1d(
                PROTEIN_ATTN_DMODEL, PROTEIN_ATTN_CHANNELS, kernel_size=(1,), stride=1
            )

        # ===== Classifier head (ADPNet pyramid, GAP, or attention pool) over the fused features =====
        fusion_ch = (
            512
            - sum(FEATURE_CHANNELS[f] for f in drop_feature)
            + (PROTEIN_CHANNELS if add_protein else 0)
            + (PROTEIN_ATTN_CHANNELS if attn_protein else 0)
        )
        if adpnet_to_gap:
            self.adpnet = GAPHead(fusion_ch)
        elif adpnet_to_attnpool:
            self.adpnet = AttnPoolHead(fusion_ch)
        else:
            self.adpnet = ADPNet(fusion_ch, number_of_layers)

        # ===== Graph Convolution =====
        if "gcn" not in drop_feature:
            self.gcn = GCNConv(512, 32)

        # ===== Initialize weights =====
        self._initialize_weights()

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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


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
        feats = []

        # ===== Graph branch (GCN over tokens) =====
        if "gcn" not in self.drop_feature:
            node_features = bert_embedding  # (B, 512, L)
            adj = attn                      # (B, L, L)

            # Here, `num_nodes` is taken from adj.shape[1] (i.e., L)
            batch_size, num_nodes, _ = adj.shape

            # Build edge indices per sample by taking all nonzero entries in adjacency.
            edge_index_list = []
            for i in range(batch_size):
                edge_index = adj[i].nonzero(as_tuple=False).t().contiguous()
                edge_index_list.append(edge_index)
            edge_index = torch.cat(edge_index_list, dim=1)

            # Convert node features from (B, 512, L) -> (B*L, 512) to feed into PyG GCNConv
            node_features = node_features.permute(0, 2, 1).contiguous().view(-1, 512)
            xg = self.gcn(node_features, edge_index)

            # Convert back to (B, 32, L)
            xg = xg.view(batch_size, num_nodes, -1).permute(0, 2, 1).contiguous()
            feats.append(xg)

        # ===== Embedding branch =====
        if "sequence" not in self.drop_feature:
            x0 = self.conv_bert(bert_embedding)
            x0 = self.multiscale_bert(x0)
            feats.append(x0)

        # ===== Structure branch =====
        if "structure" not in self.drop_feature:
            x1 = self.conv_str(structure)
            x1 = self.multiscale_str(x1)
            feats.append(x1)

        # ===== Motif branch =====
        if "motif" not in self.drop_feature:
            x2 = self.conv_motif(motif)
            x2 = self.multiscale_motif(x2)

            # Pad motif features to the fixed target length (101)
            total_padding = 101 - x2.size(2)
            left_pad = total_padding // 2
            right_pad = total_padding - left_pad
            x2 = F.pad(x2, (left_pad, right_pad), "constant", 0)
            feats.append(x2)

        # ===== Biochemical branch =====
        if "biochem" not in self.drop_feature:
            x3 = self.conv_biochem(biochem)
            x3 = self.multiscale_biochem(x3)
            feats.append(x3)

        # ===== Protein cross-attention branch (real, non-degenerate fusion) =====
        if self.attn_protein:
            batch_size = bert_embedding.shape[0]
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
            feats.append(attn_out)

        # ===== Protein branch (optional additive control) =====
        if self.add_protein:
            batch_size = bert_embedding.shape[0]
            length = bert_embedding.shape[-1]
            protein_feat = self.protein_proj(self.protein_vector)          # (PROTEIN_CHANNELS,)
            protein_feat = protein_feat.view(1, -1, 1).expand(batch_size, -1, length)
            feats.append(protein_feat)

        # ===== Fusion =====
        x = torch.cat(feats, dim=1)
        return self.adpnet(x)
    

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
        super(multiscaleKAN, self).__init__()

        self.conv0 = SimpleConvKAN_1layer(in_channel, out_channel, kernel_size=1, same_padding=False, grid_size=2, dropout=0.3)
        self.conv1 = nn.Sequential(
            SimpleConvKAN_1layer(in_channel, out_channel, kernel_size=1, same_padding=False, bn=False, grid_size=2, dropout=0.3),
            SimpleConvKAN_1layer(out_channel, out_channel, kernel_size=3, same_padding=True, grid_size=4, dropout=0.3),
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
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        return torch.cat([x0,x1], dim=1) + x


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
        super(multiscaleMLP, self).__init__()
        self.conv0 = Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False)
        self.conv1 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False, bn=False),
            Conv1d(out_channel, out_channel, kernel_size=(3,), same_padding=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        return torch.cat([x0, x1], dim=1) + x


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
        super(GAPHead, self).__init__()
        self.classifier = nn.Linear(filter_num, 1)

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
        super(AttnPoolHead, self).__init__()
        self.score = nn.Linear(filter_num, 1)
        self.classifier = nn.Linear(filter_num, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)                    # (B, C, L) -> (B, L, C)
        w = torch.softmax(self.score(h), dim=1)  # (B, L, 1) attention weights over L
        pooled = (h * w).sum(dim=1)               # (B, C)
        return self.classifier(pooled)             # (B, 1)
