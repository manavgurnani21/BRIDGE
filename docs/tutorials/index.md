# Tutorials

BRIDGE provides notebook-based, task-oriented workflows that map directly to the main use cases in the repository. Each tutorial is designed to be runnable end-to-end, with clearly specified inputs, key parameters, and expected outputs.

```{customlist}
```

## Contents

::::{tab-set}

:::{tab-item} Benchmarking
Benchmarking refers to training and evaluating BRIDGE models under standardized settings (e.g., across multiple RBPs and cell lines), and optionally testing transfer/generalization via dynamic cross-cell-type prediction.
:::

:::{tab-item} Motif discovery
Motif discovery refers to extracting interpretable binding patterns from BRIDGE outputs and comparing/dissecting these patterns with downstream motif tools and visualizations.
:::

:::{tab-item} Variant-aware prediction
Variant-aware prediction refers to scoring genetic variants by comparing model predictions between reference and alternative alleles (or perturbed sequences), producing variant impact scores that can be aggregated for clinical datasets or genome-wide scans.
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

index_benchmarking
index_motif
index_variant_aware
```