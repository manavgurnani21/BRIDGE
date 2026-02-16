# Documentation
<!-- <h1 align="center">
BRIDGE: Bridging Sequence–Structure Motifs and Genetic Variants for Genome-wide Dynamic RNA–Protein Interaction Profiling
</h1> -->

<p align="center">

<!-- GitHub Actions CI -->

<a href="https://github.com/wangyb97/BRIDGE/actions/workflows/ci.yml">
    <img src="https://github.com/wangyb97/BRIDGE/actions/workflows/ci.yml/badge.svg">
  </a>

<!-- Language -->

<a href="https://github.com/wangyb97/BRIDGE">
    <img src="https://img.shields.io/badge/BRIDGE-Python-blue.svg">
  </a>

<!-- Stars -->

<a href="https://github.com/wangyb97/BRIDGE/stargazers">
    <img src="https://img.shields.io/github/stars/wangyb97/BRIDGE?style=flat&color=yellow">
  </a>

<!-- Forks -->

<a href="https://github.com/wangyb97/BRIDGE/network/members">
    <img src="https://img.shields.io/github/forks/wangyb97/BRIDGE?style=flat&color=orange">
  </a>

<!-- Issues -->

<a href="https://github.com/wangyb97/BRIDGE/issues">
    <img src="https://img.shields.io/github/issues/wangyb97/BRIDGE?style=flat&color=informational">
  </a>

<!-- License -->

<a href="https://github.com/wangyb97/BRIDGE/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/wangyb97/BRIDGE?color=green">
  </a>

<!-- Contributors -->

<a href="https://github.com/wangyb97/BRIDGE#contributors-">
    <img src="https://img.shields.io/badge/All_Contributors-1-purple.svg">
  </a>

</p>

<!-- ## 🔬Overview

<p align="center">
  <img src="_static/framework.png" alt="BRIDGE framework" width="80%">
</p> -->

BRIDGE is an advanced multimodal deep learning framework for predicting dynamic RNA–protein binding landscapes and assessing the functional impact of genetic variants across multiple human cell types. It leverages a unified architecture that integrates:

- **Sequence embeddings** from pretrained Transformer models to capture rich contextual nucleotide representations.
- **RNA secondary structure features** to model the spatial and thermodynamic constraints on RBP binding.
- **Motif priors** derived from *de novo* motif discovery (STREME) to incorporate known binding patterns.
- **Biochemical profiles** capturing experimental signals such as reactivity, accessibility, and conservation.
- **Graph-based attention modeling** to represent long-range dependencies between nucleotides via token-wise relational graphs.

By fusing these complementary modalities, BRIDGE can accurately characterize both conserved and dynamic binding preferences, enabling:

- **End-to-end model training and evaluation** on large-scale eCLIP/HITS-CLIP datasets.
- **Dynamic cross-cell-type transfer prediction**, where the model generalizes to unseen cellular contexts without fine-tuning.
- **Variant-aware inference**, assessing the functional impact of genetic variants (e.g., SNVs) on RBP binding to facilitate disease and trait association studies.
- **Explicit motif extraction** highlighting dynamic sequence–structure patterns learned from the fused modalities.

This multimodal and interpretable design positions BRIDGE as a powerful tool for dissecting post-transcriptional regulation, guiding functional genomics studies, and prioritizing disease-associated variants with potential regulatory impact.

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Installation {octicon}`plug;1em;`
:link: installation
:link-type: doc

New to _BRIDGE_? Check out the installation guide.
:::

:::{grid-item-card} API reference {octicon}`book;1em;`
:link: api/index
:link-type: doc

The API reference contains a detailed description of
the BRIDGE API.
:::

:::{grid-item-card} Tutorials {octicon}`play;1em;`
:link: tutorials/index
:link-type: doc

The tutorials walk you through real-world applications of BRIDGE models.
:::

::::

```{toctree}
:hidden: true
:maxdepth: 2
:titlesonly: true

installation
user_guide/index
tutorials/index
api/index
developer/index
release_notes
changelog.md
GitHub <https://github.com/wangyb97/BRIDGE>
```
