# PaRPI_BIP's RNA–Protein Interaction Module

Notes from reading `../PaRPI_BIP` (sibling repo, one level above `BRIDGE`) to evaluate
whether its RNA–protein fusion mechanism is worth adopting here, as an alternative to
BRIDGE's current additive whole-protein bias (`ablation.registry.PROTEIN_CONFIG`,
`add_protein=True` in `utils.BRIDGE.BRIDGE`).

## Where it lives

- `CrossAttention` class: `PaRPI_BIP/utils/cross_attention.py:7-50`.
- Instantiated in `PaRPI_BIP/utils/PaRPI.py:184`:
  `CrossAttention(source_dim=256, target_dim=256, hidden_dim=256)` (8 heads, no CLI/config
  knobs — everything hardcoded at the call site).
- Invoked in `PaRPI.forward`, `PaRPI_BIP/utils/PaRPI.py:236-244`, as the very last step
  before the 2-layer `MLPReadout` (`256 -> 128 -> 64 -> 1`) classification head.

## What it actually computes

On paper it's full multi-head cross-attention: separate `W_Q`/`W_K`/`W_V` projections, a
per-head learned bilinear key-transform `W_att` (8 x 256 x 256), softmax, output projection,
residual + LayerNorm.

In practice, both inputs are reduced to a **single vector each** before fusion:

- RNA side: RNA-BERT (768-d/position) + a structure scalar are concatenated per-node, run
  through 4 GraphSAGE layers over a base-pair-probability graph, then a
  transformer+CBAM block, then `DPRBP` (`PaRPI_BIP/utils/PaRPI.py:120-172`), which
  repeatedly `MaxPool1d(kernel_size=3, stride=2)`s the 99-position sequence down to length 1
  (99 -> 49 -> 24 -> 12 -> 6 -> 3 -> 1).
- Protein side: ESM-2 (`esm2_t33_650M_UR50D`) layer-33 per-residue reps, mean-pooled over the
  whole sequence to one 1280-d vector (`PaRPI_BIP/utils/esm.py:5-23`), projected 1280 -> 256
  via a 1x1 conv. Same vector is tiled across every RNA sample for that RBP/dataset — protein
  is a static, per-RBP conditioning signal, same as in BRIDGE.

With one query token (N=1) and one key token (P=1), `softmax(attn, dim=-1)` is always exactly
1 regardless of the Q·K score — the attention weighting is mathematically inert. The module
degenerates, in the forward pass, to:

```
x_2 = LayerNorm(rna) + W_O(ELU(W_V(protein_proj)))
```

i.e. a residual/additive injection of a linearly-then-nonlinearly transformed protein vector,
dressed in unused attention machinery (`W_Q`, `W_K`, `W_att` still receive gradient but don't
gate the value added, since there's nothing else to attend over).

## README rationale

None. `README.md` / `BERT_Model/README.md` only cover provenance, licensing, and CLI usage —
no discussion of why cross-attention was chosen over concatenation/gating, no ablation.

## Takeaway vs. BRIDGE's additive bias

Despite the name, this is functionally much closer to BRIDGE's existing additive
whole-protein-bias config than "cross-attention" suggests. The differences that matter:

1. PaRPI_BIP passes the protein vector through extra learned projections (`W_V`, `W_O`,
   ELU) before adding it, rather than adding it near-directly.
2. Fusion happens at the very end of the network (right before the MLP head), after all
   RNA-side processing, rather than as an early additive bias into a 512-wide fusion layer.
3. The RNA branch is pooled to a single vector *before* fusion — same constraint that makes
   attention vacuous here would apply to BRIDGE too if BRIDGE pooled the same way first.

## Implication for adding real cross-attention to BRIDGE

For cross-attention to do something a plain additive bias can't, it needs more than one key
and more than one query — i.e., avoid PaRPI_BIP's mistake of pooling both sides to length-1
before fusing:

- Protein side: use **per-residue** ESM-2 embeddings (not mean-pooled) as the attention
  source, so there's more than one key/value per RBP.
- RNA side: fuse at BRIDGE's **per-position** RNA representation (before BRIDGE's own final
  pooling/ADPNet head), so the query varies by position.

That lets different RNA positions attend differently over the RBP's own residues — a real
signal a uniform additive bias can't express — even though, per-dataset, the protein input
is still constant across samples (BRIDGE trains one model per single RBP; see
`ablation/registry.py`'s `PROTEIN_CONFIG` docstring). It's a bigger change than the current
additive-bias config: per-residue ESM-2 caching (not just one 1280-d vector per protein) and
new attention parameters per model.
