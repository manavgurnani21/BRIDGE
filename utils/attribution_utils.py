"""
Utilities for prediction, attribution, and visualization of nucleotide models.

This module provides small, practical helpers for working with PyTorch models that
consume one-hot encoded nucleotide sequences and (optionally) per-position structure
features. It covers three workflows:

- Prediction helpers: forward pass + optional logits→probabilities conversion.
- Attribution helpers: gradient-based explanations via ``igrads`` (IG / Grad×Input).
- Visualization helpers: sequence-logo style plots via ``logomaker``.

Who this is for
---------------
Researchers/engineers who already have a trained PyTorch model and want a lightweight
utility layer for (a) inference, (b) attribution, and (c) quick visualization.

Dependencies
------------
- ``torch``, ``torch.nn.functional``
- ``igrads`` (required for attribution)
- ``matplotlib`` (required for plotting)
- ``pandas``, ``logomaker`` (required for logo plots)

Data conventions
----------------
Nucleotide channel order
    All logo/attribution plotting assumes channel order:
    ``['A', 'C', 'G', 'U']`` corresponding to columns 0..3.

Sequence encoding
    The included encoder uses::

        base2int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    This means the 4th channel is produced from ``'T'``. If your sequences are RNA
    and contain ``'U'``, you should either pre-convert ``U→T`` before encoding, or
    extend the mapping (recommended)::

        base2int = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}

Tensor shapes
    - One-hot sequence:
      - unbatched: ``(L, 4)`` (float32)
      - batched:   ``(B, L, 4)`` (float32)
    - Structure features (optional, attribution workflow):
      - unbatched: ``(L, S)`` (float32)
      - batched:   ``(B, L, S)`` (float32)

Model interface expectations
----------------------------
This module supports two common model styles:

A) Sequence-only prediction models (used by ``predict`` / ``predict_from_sequence``)
    Expected forward signature::

        model(one_hot) -> dict[str, torch.Tensor]

    Output key naming convention (used by ``_to_probs``):
    - keys containing ``'_profile'``            → ``softmax(dim=1)``
    - keys containing ``'_mixing_coefficient'`` → ``sigmoid``

    If ``to_probs=True`` and a key does not match either rule, a ``ValueError`` is raised.

B) Sequence + structure attribution models (used by ``attribution``)
    Expected forward signature::

        model((one_hot, structure)) -> torch.Tensor

    ``attribution`` will add a batch dimension and call::

        (inputs.unsqueeze(0), structure.unsqueeze(0))

.. important::
   Target selection for attributions

   ``igrads.integrated_gradients(...)`` / ``igrads.grad_x_input(...)`` require a clear
   scalar objective or a target specification that matches your output shape.

   The current implementation passes ``target_mask=pred`` (the raw model output). This is
   only appropriate when your model output is already scalar-per-example *or* when your
   ``igrads`` version interprets ``target_mask`` in a compatible way.

   If your model returns multi-dimensional outputs (profiles, multi-task heads, etc.),
   you will usually need to adapt the objective, e.g.:

   - pick an index (task/class) and reduce to a scalar
   - sum/mean over positions for profile outputs

   Example pattern (conceptual)::

        pred = model(inputs)              # shape: (B, ...)
        score = pred[..., idx].sum()      # scalar objective
        # then run IG/Grad×Input against 'score' (depending on igrads API)

Notes and caveats
-----------------
- Device placement:
  Inputs must be on the same device as the model (CPU/GPU).
- Checkpoint loading:
  ``torch.load`` uses pickle; only load checkpoints from trusted sources.

Usage examples
--------------
1) Prediction from raw sequence (sequence-only models)

.. code-block:: python

    model = load_model(MyModel(), "checkpoint.pt")
    pred = model.predict_from_sequence("ACGTACGT...", to_probs=True)

2) Attribution for sequence + structure models

.. code-block:: python

    seq = sequence2onehot("ACGT...").to(device)          # (L, 4)
    struct = torch.zeros(seq.shape[0], S).to(device)     # (L, S)
    attrs_seq, attrs_struct = attribution(seq, struct, model, atype="IG", steps=50)

3) Visualization

.. code-block:: python

    fig1 = visualize_attribution_only(attrs_seq)
    fig2 = visualize_track_attribution(track, attrs_seq, sequence="ACGT...", title="Example")
"""

import igrads  # third-party integrated-gradients / grad-x-input attribution library
import matplotlib.pyplot as plt  # figure/axis creation for attribution and track plots
import logomaker  # renders one-hot/attribution matrices as sequence logos
import pandas as pd  # DataFrame wrapper required by logomaker's plotting API
import torch.nn.functional as F  # softmax/sigmoid/one_hot ops
import torch  # tensors, device/dtype handling, checkpoint loading


def attribution(inputs, structure, model, atype='IG', steps=50):
    """Compute sequence/structure attributions for a PyTorch model.

    Parameters
    ----------
    inputs : torch.Tensor
        One-hot encoded sequence of shape (L, 4), dtype float.
    structure : torch.Tensor
        Structure features of shape (L, S), dtype float.
    model : torch.nn.Module
        A model whose forward accepts (inputs, structure) as a tuple and returns a prediction tensor.
    atype : {"IG", "grad_x_input"}, default="IG"
        Attribution method: Integrated Gradients ("IG") or Grad×Input ("grad_x_input").
    steps : int, default=50
        Number of IG interpolation steps. Only used when atype="IG".

    Returns
    -------
    Any
        Attributions returned by `igrads.*` for (inputs, structure). Typically a tuple of tensors
        with shapes matching the inputs.

    Raises
    ------
    ValueError
        If `atype` is not supported.
    """
    # Combine sequence and structure inputs (assuming structure is one-hot encoded or of suitable shape)
    combined_inputs = (inputs, structure)  # This could be adjusted based on model's input format  # pair the two unbatched tensors (overwritten below, so this line's result is unused)

    # Add batch dimension
    combined_inputs = (inputs.unsqueeze(0), structure.unsqueeze(0))  # add a leading batch axis of size 1 to both tensors, since model expects batched input

    # Make predictions
    pred = model(combined_inputs)  # forward pass; used below as the attribution target

    if atype == 'IG':  # Integrated Gradients requested
        # Compute integrated gradients for both sequence and structure inputs
        return igrads.integrated_gradients(combined_inputs, model, target_mask=pred, steps=steps)  # accumulate gradients along a path from baseline to input, using `pred` as the target
    elif atype == 'grad_x_input':  # Grad×Input requested
        # Compute grad_x_input for both sequence and structure inputs
        return igrads.grad_x_input(combined_inputs, model, target_mask=pred)  # single-step gradient-times-input attribution, using `pred` as the target
    else:  # unsupported atype string
        raise ValueError(f'Unrecognized attribution type {atype}.')  # fail loudly rather than silently skipping attribution


custom_color_scheme = {  # per-nucleotide colors passed to logomaker.Logo for consistent, RNA-specific coloring
    'A': '#268a34',  # Green
    'C': '#2c51aa',  # Blue
    'G': '#f8981c',  # Orange
    'U': '#ea2529'   # Red
}


def make_attribution_figure(a, ax):
    """
    Plot an attribution matrix (L x 4) as a sequence logo on a given Matplotlib axis.

    This function converts an attribution matrix into a pandas DataFrame with
    columns corresponding to nucleotide channels and then uses logomaker to render
    a logo. It also draws a horizontal baseline at y=0 and removes the bottom spine.

    Args:
        a (array-like):
            Attribution matrix with shape (L, 4), where L is sequence length.
            Channel order is assumed to match ['A', 'C', 'G', 'U'].
            Values can be signed (e.g., importance scores) or non-negative (e.g., probabilities).
        ax (matplotlib.axes.Axes):
            Axis on which the logo will be drawn.

    Returns:
        None. The plot is drawn in-place on `ax`.

    Requires:
        - logomaker installed and importable.

    Visual conventions:
        - shade_below/fade_below highlight negative contributions by default.
        - a y=0 baseline is drawn in red.
    """
    df = pd.DataFrame(a, columns=['A', 'C', 'G', 'U'])  # wrap the (L,4) attribution matrix with per-base column labels for logomaker
    # logomaker.Logo(df, shade_below=.5, fade_below=.5, font_name='Arial Rounded MT Bold', ax=ax)
    logomaker.Logo(df, shade_below=.5, fade_below=.5, ax=ax, color_scheme=custom_color_scheme)  # draw the logo, shading/fading negative-valued letters
    ax.spines['bottom'].set_visible(False)  # hide the default bottom axis line (replaced by the y=0 baseline below)
    ax.axhline(0, color='#ea2529', linewidth=1.5)  # draw an explicit red baseline at y=0 to separate positive/negative contributions
    
    
def visualize_track_attribution(track, attribution, sequence=None, title=None):
    """
    Visualize a predicted 1D track together with an attribution logo (and optionally the input sequence logo).

    The output figure stacks panels vertically:
      1) track panel (line plot)
      2) attribution panel (logo)
      3) optional sequence panel (logo of one-hot encoding)

    Args:
        track (array-like):
            1D signal of length L (e.g., predicted binding signal along the sequence).
            Must be plottable by Matplotlib (list, np.ndarray, torch.Tensor, etc.).
        attribution (array-like or torch.Tensor):
            Attribution matrix with shape (L, 4). If a torch.Tensor, it will be detached to CPU numpy.
            Channel order should match ['A','C','G','U'] (see make_attribution_figure).
        sequence (str, optional):
            Optional RNA sequence of length L. If provided, it will be converted to one-hot and shown
            as a logo in the third panel.
            IMPORTANT: `sequence2onehot` (as written below) supports bases in `base2int`.
            Unknown bases will lead to failure in one-hot (see notes in module docstring).
        title (str, optional):
            Title for the top track panel.

    Returns:
        matplotlib.figure.Figure:
            The created figure instance.
    """
    nplots = 3 if sequence is not None else 2  # add a third panel only when a sequence logo is requested
    hratio = [5, 2, 0.3] if sequence is not None else [5, 2]  # track panel gets most height, attribution less, sequence logo a thin strip

    fig, axs = plt.subplots(nplots, 1, figsize=(22, 5), gridspec_kw={'height_ratios': hratio})  # stacked subplots sharing one wide figure
    axs[0].set_title(title)  # title goes on the top (track) panel
    axs[0].plot(track, color='red', label='Pred. Signal', linewidth=2)  # plot the 1D predicted signal along the sequence

    if isinstance(attribution, torch.Tensor):  # attribution may be passed as a tensor still on GPU/autograd graph
        attribution = attribution.detach().cpu().numpy()  # detach from autograd and move to CPU numpy for plotting

    make_attribution_figure(attribution, axs[1])  # render the attribution logo in the middle panel

    if sequence is not None:  # optional third panel showing the raw input sequence as a logo
        make_attribution_figure(sequence2onehot(sequence).numpy(), axs[2])  # one-hot encode the sequence and reuse the same logo renderer

    for ax in axs:  # apply to every panel
        # remove x-axis margins
        ax.margins(x=0.005)  # tighten horizontal padding so the sequence axis lines up tightly across panels


    # remove plot boarder (except for x-axis)
    axs[0].spines['top'].set_visible(False)  # hide top border of the track panel
    axs[0].spines['right'].set_visible(False)  # hide right border of the track panel
    axs[0].spines['bottom'].set_visible(False)  # hide bottom border of the track panel
    axs[0].get_xaxis().set_visible(False)  # hide x-axis ticks/labels on the track panel (shared with panels below)

    axs[1].spines['top'].set_visible(False)  # hide top border of the attribution panel
    axs[1].spines['right'].set_visible(False)  # hide right border of the attribution panel
    axs[1].spines['bottom'].set_visible(False)  # hide bottom border of the attribution panel
    axs[1].get_xaxis().set_visible(False)  # hide x-axis ticks/labels on the attribution panel

    if sequence is not None:  # only the (optional) sequence panel needs its own spine cleanup
        axs[2].spines['top'].set_visible(False)  # hide top border of the sequence panel
        axs[2].spines['left'].set_visible(False)  # hide left border of the sequence panel
        axs[2].spines['right'].set_visible(False)  # hide right border of the sequence panel
        axs[2].spines['bottom'].set_visible(False)  # hide bottom border of the sequence panel
        #axs[2].get_xaxis().set_visible(False)
        axs[2].get_yaxis().set_visible(False)  # hide y-axis ticks/labels (the sequence logo has no meaningful y-scale)

    return fig  # caller can further customize or save the composed figure


def visualize_attribution_only(attribution):
    """
    Visualize only an attribution matrix (L x 4) as a logo.

    Args:
        attribution (array-like or torch.Tensor):
            Attribution matrix with shape (L, 4).

    Returns:
        matplotlib.figure.Figure:
            The created figure instance.
    """
    fig, ax = plt.subplots(1, 1, figsize=(22, 1.5))  # single wide, short panel for a standalone attribution logo

    if isinstance(attribution, torch.Tensor):  # attribution may be passed as a tensor still on GPU/autograd graph
        attribution = attribution.detach().cpu().numpy()  # detach from autograd and move to CPU numpy for plotting

    make_attribution_figure(attribution, ax)  # render the attribution logo on the single axis

    ax.spines['top'].set_visible(False)  # hide top border
    ax.spines['left'].set_visible(False)  # hide left border
    ax.spines['right'].set_visible(False)  # hide right border
    ax.spines['bottom'].set_visible(False)  # hide bottom border
    ax.get_xaxis().set_visible(False)  # hide x-axis ticks/labels (position index isn't shown)
    ax.get_yaxis().set_visible(False)  # hide y-axis ticks/labels (magnitude scale isn't shown)

    return fig  # caller can further customize or save the figure


def _to_probs(value, key):
    """
    Convert raw model outputs (logits) into probabilities based on the output key name.

    This helper applies a key-dependent transformation:
    - If key contains '_profile': apply softmax over dim=1 (commonly used for per-position categorical profiles).
    - If key contains '_mixing_coefficient': apply sigmoid (commonly used for scalar/bounded coefficients).

    Args:
        value (torch.Tensor):
            Raw tensor output from the model (typically logits).
        key (str):
            Output dictionary key that indicates how to interpret `value`.

    Returns:
        torch.Tensor:
            Transformed tensor in probability space.

    Raises:
        ValueError:
            If `key` does not match any recognized pattern.

    Example:
        pred = model(inputs)  # {'something_profile': logits, 'something_mixing_coefficient': logits2}
        pred_probs = {k: _to_probs(v, k) for k, v in pred.items()}
    """
    if '_profile' in key:  # per-position categorical output (e.g. a base-pair profile)
        value = F.softmax(value, dim=1)  # normalize into a probability distribution over dim=1
    elif '_mixing_coefficient' in key:  # a bounded scalar/mixing-weight output
        value = torch.sigmoid(value)  # squash into (0, 1)
    else:  # key doesn't match any known naming convention
        raise ValueError(f'Unknown key: {key}')  # fail rather than silently passing through raw logits
    return value  # transformed (probability-space) tensor


def predict(inputs, model, to_probs=True):
    """
    Run a model forward pass on already-prepared inputs and optionally convert logits to probabilities.

    Assumes `model(inputs)` returns a dictionary mapping output-name -> tensor.

    Args:
        inputs (torch.Tensor):
            Model inputs, typically one-hot encoded sequences of shape:
              - (B, L, 4) for batch size B, length L, 4 channels.
            The exact expected shape depends on your model implementation.
        model (torch.nn.Module):
            PyTorch model that returns a dict of outputs.
        to_probs (bool):
            If True, convert each output tensor from logits to probabilities using `_to_probs`
            based on the output key naming convention.

    Returns:
        dict[str, torch.Tensor]:
            Model outputs; either raw logits (to_probs=False) or probabilities (to_probs=True).

    Raises:
        ValueError:
            If to_probs=True and an output key is not recognized by `_to_probs`.
    """
    pred = model(inputs)  # forward pass; expected to return a dict of named output tensors
    if to_probs:  # caller wants probabilities rather than raw logits
        pred = {key: _to_probs(value, key) for key, value in pred.items()}  # convert every output tensor per its key naming convention
    return pred  # dict of (possibly probability-transformed) model outputs

base2int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}  # nucleotide-to-channel-index mapping used by sequence2int (see module docstring caveat about 'U')


def sequence2int(sequence):
    """
    Convert a nucleotide sequence string into integer indices using `base2int`.

    Args:
        sequence (str):
            RNA sequence. Expected characters are keys of `base2int` (default: A/C/G/T).

    Returns:
        list[int]:
            Integer-encoded sequence of length L.

    Warning:
        Unknown bases are mapped to 999 by default, which will later break one-hot encoding
        (torch.nn.functional.one_hot requires values < num_classes). Ensure sequences contain
        only valid bases before calling downstream helpers.
    """
    return [base2int.get(base, 999) for base in sequence]  # map each character to its channel index; unknown bases become the sentinel 999


def sequences2inputs(sequences):
    """
    Convert one or more sequences into a batch of one-hot encoded tensors.

    Args:
        sequences (str or list[str]):
            If str, treated as a single sequence (length L).
            If list[str], treated as a batch of sequences (all should have equal length L for stacking).

    Returns:
        torch.Tensor:
            One-hot tensor of shape (B, L, 4), dtype float32,
            where B is batch size (1 if input is a single string).

    Raises:
        RuntimeError / ValueError:
            If sequences contain invalid bases (mapped to 999), one_hot will fail.
            If sequences have inconsistent lengths, tensor construction may fail.
    """
    if isinstance(sequences, str):  # normalize a single sequence into a length-1 list
        sequences = [sequences]  # so the stacking logic below is uniform for str/list inputs
    return F.one_hot(torch.tensor([sequence2int(s) for s in sequences]), num_classes=4).float()  # int-encode each sequence, stack into (B, L), one-hot to (B, L, 4), cast to float


def sequence2onehot(sequence):
    """
    Convert a single sequence string into a one-hot encoded tensor.

    Args:
        sequence (str):
            Single RNA sequence of length L.

    Returns:
        torch.Tensor:
            One-hot tensor of shape (L, 4), dtype float32.

    Raises:
        RuntimeError / ValueError:
            If sequence contains invalid bases (mapped to 999), one_hot will fail.
    """
    return F.one_hot(torch.tensor(sequence2int(sequence)), num_classes=4).float()  # int-encode the sequence to (L,), one-hot to (L, 4), cast to float


def predict_from_sequence(sequences, model, **kwargs):
    """
    Convenience wrapper: encode sequence(s) to one-hot, then call `predict`.

    Args:
        sequences (str or list[str]):
            Sequence(s) to predict on.
            - If str: returns outputs with batch dimension squeezed.
            - If list[str]: returns batched outputs.
        model (torch.nn.Module):
            PyTorch model that accepts one-hot inputs and returns a dict of outputs.
        **kwargs:
            Passed through to `predict`, e.g. to_probs=False.

    Returns:
        dict[str, torch.Tensor]:
            Prediction dictionary.
            - If input is a single sequence string, each tensor will be squeezed from (1, ...) to (...).
            - If input is a list, tensors remain batched.
    """
    one_hot = sequences2inputs(sequences)  # encode raw sequence(s) into a batched one-hot tensor (B, L, 4)
    pred = predict(one_hot, model, **kwargs)  # run the standard predict() path (forward + optional logits->probs)

    if isinstance(sequences, str):  # caller passed a single sequence, not a batch
        pred = {key: value.squeeze(0) for key, value in pred.items()}  # drop the batch dim of size 1 from every output tensor

    return pred  # prediction dict, batched or squeezed depending on input type


def __predict(self, inputs, **kwargs):
    """Returns model predictions on inputs with logits to probs."""

    return predict(inputs, model=self, **kwargs)  # `self` is bound as `model` when attached via __get__ below

def __predict_from_sequence(self, sequences, **kwargs):
    """Predicts on RNA/DNA sequences.

    Args:
        sequences (str, list): RNA/DNA sequence(s). If a string, it is assumed to be a single sequence.

    Returns:
        dict: Dictionary of predictions.
    """
    # Assume a preprocessing function is required for sequence input
    return predict_from_sequence(sequences, model=self, **kwargs)  # `self` is bound as `model` when attached via __get__ below

def __explain(self, inputs, **kwargs):
    """Generate attributions or explanations for model predictions."""
    return attribution(inputs, self, **kwargs)  # NOTE: positionally `self` lands in attribution()'s `structure` slot, not `model` — likely a latent bug if used as-is

def __add_attributes_and_bound_methods(model):
    """
    Monkey-patch a PyTorch model instance with convenience methods.

    After calling this, the model instance will have:
        - model.predict(inputs, **kwargs)
        - model.predict_from_sequence(sequences, **kwargs)
        - model.explain(inputs, **kwargs)

    Args:
        model (torch.nn.Module):
            Model instance to be extended in-place.

    Returns:
        None.
    """
    model.predict = __predict.__get__(model)  # bind __predict as a method, so `self`==model inside it
    model.predict_from_sequence = __predict_from_sequence.__get__(model)  # bind __predict_from_sequence as a method
    model.explain = __explain.__get__(model)  # bind __explain as a method
    

def load_model(model, filepath, **kwargs):
    """
    Load a PyTorch model state dict from disk and attach convenience methods.

    Args:
        model (torch.nn.Module):
            Instantiated model object with the same architecture as the saved checkpoint.
        filepath (str):
            Path to a file produced by torch.save(model.state_dict(), filepath).
        **kwargs:
            Reserved for future extensions (currently unused). You may use this to pass
            torch.load kwargs in your own fork (e.g., map_location), but as written
            it is not forwarded.

    Returns:
        torch.nn.Module:
            The same `model` instance with loaded parameters, patched methods, and set to eval() mode.

    Important:
        - This uses `torch.load(filepath)` directly. If you need CPU/GPU mapping, you may want
          to modify to `torch.load(filepath, map_location=...)`.
    """
    model.load_state_dict(torch.load(filepath))  # deserialize the checkpoint (pickle-based) and load weights into `model` in-place
    __add_attributes_and_bound_methods(model)  # attach .predict / .predict_from_sequence / .explain convenience methods
    model.eval()  # switch to inference mode (disables dropout, freezes batchnorm stats)
    return model  # same instance, now loaded, patched, and in eval mode
