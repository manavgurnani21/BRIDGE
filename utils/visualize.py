"""
Saliency visualization utilities for BRIDGE models.

This module provides:
  1) `inference(...)`: batched model inference that returns sigmoid probabilities.
  
  2) PWM/logo rendering utilities (`normalize_pwm`, `get_nt_height`, `seq_logo`):
  
     - Convert a 4xL PWM (A/C/G/U) into an RGB logo image by stacking pre-rendered
       nucleotide glyphs with per-position heights.
       
  3) `plot_saliency(...)`: end-to-end saliency plotting that combines:
  
       - saliency logo (normalized saliency PWM -> logo)
       - saliency heatmap (raw weights resized to a fixed canvas)
       - raw sequence logo
       - optional structure saliency track + structure trace overlay

Expected inputs
---------------
X : np.ndarray
    Input feature matrix with shape:
      - sequence-only mode: (4, L)
      - sequence+structure mode: (>=5, L) where X[4, :] is per-position structure score
        (typically in [0, 1], with higher = more "pairedness"/signal depending on upstream).
    Padding positions are represented by all zeros across X[:4, pos]. The plotting code
    automatically removes such padded columns.

W : np.ndarray
    Saliency/importance weights aligned to X. Must have at least the first 4 rows
    corresponding to nucleotide channels (A/C/G/U or A/C/G/T treated as U).
    Shape should match X for the plotted channels, typically (4, L) or (>=5, L).

str_null : np.ndarray, optional
    Mask marking "null" structure positions (e.g., regions without valid structure scores).
    Required when X has a structure row (X.shape[0] > 4). The plotting code uses
    `str_null.T == 1` to select null positions, so the input should be shaped/broadcastable
    accordingly.

External assets & dependencies
------------------------------
- `./acgu.npz` is required by `seq_logo(...)`.
  It must contain an array under key `'data'` with 4 RGB glyph images (A,C,G,U) that can be
  indexed by nucleotide channel index. Glyphs are resized per position with skimage.

- `skimage.transform.resize` is used for resizing glyphs and heatmaps.

Output
------
`plot_saliency(...)` saves a single PNG image to `outdir`, but note:
  - Despite the name, `outdir` is treated as a *file path* in the current implementation
    (e.g., "results/foo.png"), not a directory.

Backend note
------------
The script sets `mpl.use("pdf")` but saves figures as PNG via `fig.savefig(..., format="png")`.
For headless servers, a more typical choice is `mpl.use("Agg")`. The current behavior is
preserved for compatibility.

Common pitfalls
---------------
- If `acgu.npz` is missing or malformed, `seq_logo` will fail at load time.
- If X/W include negative values, information-content scaling in `get_nt_height` is not
  strictly meaningful unless `norm == 1` (fixed-height mode).
"""

import os, sys  # unused beyond import; kept for parity with original script
import numpy as np  # array ops for PWMs, logo canvases, and heatmap resizing
import matplotlib as mpl  # backend selection and colormap access
mpl.use("pdf")  # set a non-interactive backend (see module docstring: figures are still saved as PNG)
import matplotlib.pyplot as plt  # figure/axis creation and saving
import matplotlib.gridspec as gridspec  # multi-row layout for the saliency figure
from skimage.transform import resize as imresize  # resize glyphs/heatmaps to target pixel dimensions
import torch  # tensors, no_grad, sigmoid for the inference() helper
import utils.datautils as datautils  # imported but not directly referenced in this module
from PIL import Image  # imported but not directly referenced in this module


def inference(args, model, device, test_loader):
    """Run model inference on a DataLoader and return sigmoid probabilities.

    This function:
      1) switches the model to eval mode,
      2) disables gradient computation,
      3) iterates over `test_loader`,
      4) applies `torch.sigmoid` to model outputs (assumes logits),
      5) concatenates all batches into a single NumPy array on CPU.

    Args:
        args:
            Unused in the current implementation. Kept for API compatibility with
            other training/inference entry points.
        model (torch.nn.Module):
            A PyTorch model that maps inputs `x` to logits of shape (B, ...) compatible
            with `torch.sigmoid`.
        device (torch.device):
            Device where inference runs (e.g., `torch.device("cuda")` or `"cpu"`).
        test_loader (torch.utils.data.DataLoader):
            DataLoader that yields batches of `(x0, y0)`. Labels `y0` are moved to `device`
            but are not used in computing the returned probabilities.

    Returns:
        np.ndarray:
            Concatenated probabilities for all samples, with shape matching the model output
            after sigmoid. For a binary classifier that outputs (B, 1), the return shape is (N, 1).

    Notes:
        - The labels are read and moved to device but are not used; this is typical for
          evaluation pipelines that only need predicted probabilities.
        - If the model output is multi-dimensional (e.g., (B, C)), the returned array will
          preserve that shape.
    """
    model.eval()  # switch to inference mode (disables dropout, freezes batchnorm stats)
    p_all = []  # collects per-batch probability arrays
    with torch.no_grad():  # no gradients needed for inference
        for batch_idx, (x0, y0) in enumerate(test_loader):  # iterate (input, label) batches
            x, y = x0.float().to(device), y0.to(device).float()  # cast to float and move to the inference device
            output = model(x)  # forward pass, assumed to return logits
            prob = torch.sigmoid(output)  # convert logits to probabilities

            p_np = prob.to(device='cpu').numpy()  # move to CPU as a NumPy array
            p_all.append(p_np)  # stash this batch's probabilities

    p_all = np.concatenate(p_all)  # flatten all per-batch arrays into one dataset-level array
    return p_all  # concatenated probabilities across the whole loader


def normalize_pwm(pwm, factor=None, MAX=None):
    """Normalize a position weight matrix (PWM) for visualization.

    The function first scales `pwm` by the maximum absolute value, optionally applies
    an exponential sharpening (`exp(pwm * factor)`), and then normalizes each column
    by the L1 norm (sum of absolute values across nucleotides).

    Args:
        pwm (np.ndarray):
            Numeric array of shape (num_nt, num_positions). Typically `num_nt=4` for A/C/G/U(T).
        factor (float, optional):
            If provided, apply `np.exp(pwm * factor)` after scaling. This is often used to
            increase contrast.
        MAX (float, optional):
            If provided, use this value as the divisor instead of `max(abs(pwm))`. This can be
            used to enforce consistent scaling across multiple PWMs.

    Returns:
        np.ndarray:
            Normalized PWM of the same shape as input.

    Notes:
        - Column-wise normalization uses `sum(abs(pwm[:, i]))`. If a column is all zeros,
          this will divide by zero and produce `inf`/`nan`. Ensure input columns have non-zero mass
          or handle zeros upstream.
        - The normalization uses absolute values, which allows negative entries but normalizes by their magnitude.
    """
    if MAX is None:  # no explicit scale given, derive one from the data
        MAX = np.max(np.abs(pwm))  # largest-magnitude entry across the whole PWM
    pwm = pwm/MAX  # scale all entries into roughly [-1, 1]
    if factor:  # optional contrast-sharpening step
        pwm = np.exp(pwm*factor)  # exponentiate to exaggerate differences between large/small values
    norm = np.outer(np.ones(pwm.shape[0]), np.sum(np.abs(pwm), axis=0))  # broadcast each column's L1 norm across all rows
    return pwm/norm  # column-normalize so each position's channel magnitudes sum to 1


def get_nt_height(pwm, height, norm):
    """Convert PWM columns into integer per-nucleotide heights for logo plotting.

    This computes per-position total height and allocates integer heights to each nucleotide
    proportional to `pwm[:, i]`.

    Args:
        pwm (np.ndarray):
            PWM array of shape (num_nt, num_positions). Typically 4 x L.
            Values are treated as probabilities or non-negative weights when computing entropy.
        height (int | float):
            Base height scaling factor used in the logo renderer.
        norm (int):
            Controls whether to use a fixed total height per position.
            - If `norm == 1`, the total height per position is set to `height`.
            - Otherwise, the total height is scaled by information content:
              `(log2(num_nt) - entropy(pwm[:, i])) * height`.

    Returns:
        np.ndarray:
            Integer heights of shape (num_nt, num_positions), dtype `int`.
            Heights are computed with `np.floor(...)`.

    Notes:
        - Entropy is computed only over entries `> 0`.
        - The final per-position total height is clipped by `min(total_height, height*2)`.
        - If `pwm` contains negative values, the entropy/information-content interpretation
          is not strictly valid; this function assumes non-negative columns for that mode.
    """
    def entropy(p):  # Shannon entropy (base 2) of one PWM column, ignoring non-positive entries
        s = 0  # running entropy accumulator
        for i in range(len(p)):  # sum over nucleotide channels
            if p[i] > 0:  # log2(0) is undefined, so zero/negative weights contribute nothing
                s -= p[i]*np.log2(p[i])  # standard -p*log2(p) entropy term
        return s

    num_nt, num_seq = pwm.shape  # number of channels (rows) and sequence positions (columns)
    heights = np.zeros((num_nt,num_seq))  # output height allocation, same shape as pwm
    for i in range(num_seq):  # compute total height and per-channel allocation independently per position
        if norm == 1:  # fixed-height mode: every position gets the same total height
            total_height = height
        else:  # information-content mode: taller stacks at low-entropy (more informative) positions
            total_height = (np.log2(num_nt) - entropy(pwm[:, i]))*height

        heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height*2))  # distribute total height proportionally to each channel's weight, capped at height*2

    return heights.astype(int)  # integer pixel heights for the logo renderer


def seq_logo(pwm, height=30, nt_width=10, norm=0, alphabet='rna', colormap='standard'):
    """Render a sequence/logo image from a PWM as an RGB NumPy array.

    This is a low-level renderer that stacks resized nucleotide glyph images according to
    per-position heights computed from the PWM.

    Args:
        pwm (np.ndarray):
            PWM array of shape (num_nt, num_positions). Commonly (4, L).
        height (int, optional):
            Base height used by the renderer. The internal canvas height is `height*2`.
        nt_width (int, optional):
            Width in pixels allocated per position.
        norm (int, optional):
            Passed to `get_nt_height`. If 1, uses fixed height per position; otherwise uses
            information-content scaling.
        alphabet (str, optional):
            Currently unused. Present for API compatibility (e.g., "rna" vs "dna").
        colormap (str, optional):
            Currently unused. Present for API compatibility.

    Returns:
        np.ndarray:
            RGB image of shape (height*2, ceil(nt_width * num_positions), 3), dtype uint8.

    Notes:
        - This function expects an `acgu.npz` file at `./acgu.npz` containing nucleotide glyphs
          under the key `'data'`. The glyph array is expected to be indexable by nucleotide index.
    """
    acgu_path = './acgu.npz'  # fixed relative path to the pre-rendered nucleotide glyph archive
    chars = np.load(acgu_path,allow_pickle=True)['data']  # RGB glyph images, one per nucleotide channel
    heights = get_nt_height(pwm, height, norm)  # per-position, per-channel pixel heights
    num_nt, num_seq = pwm.shape  # number of channels and sequence positions
    width = np.ceil(nt_width*num_seq).astype(int)  # total canvas width in pixels

    max_height = height*2  # canvas height matches get_nt_height's height cap
    logo = np.ones((max_height, width, 3)).astype(int)*255  # start with an all-white RGB canvas
    for i in range(num_seq):  # render one column (position) at a time
        nt_height = np.sort(heights[:,i])  # this position's channel heights, ascending
        index = np.argsort(heights[:,i])  # corresponding channel indices, so smallest letters are stacked at top
        remaining_height = np.sum(heights[:,i])  # total stack height still left to place, shrinks as we place letters
        offset = max_height-remaining_height  # top padding so the stack is bottom-aligned within the canvas

        for j in range(num_nt):  # place each channel's glyph, smallest first
            if nt_height[j] <=0 :  # zero-height channels contribute nothing to the stack
                continue
            # resized dimensions of image
            nt_img = imresize(chars[index[j]], output_shape=(nt_height[j], nt_width))*255  # resize this channel's glyph to its allotted height
            # determine location of image
            height_range = range(remaining_height-nt_height[j], remaining_height)  # vertical pixel rows this glyph occupies (before offset)
            width_range = range(i*nt_width, i*nt_width+nt_width)  # horizontal pixel columns for this sequence position
            # 'annoying' way to broadcast resized nucleotide image
            if height_range:  # skip degenerate (empty) ranges
                for k in range(3):  # RGB channels
                    for m in range(len(width_range)):  # copy the glyph pixel-by-pixel into the canvas
                        logo[height_range+offset, width_range[m],k] = nt_img[:,m,k]

            remaining_height -= nt_height[j]  # shrink the remaining stack height by this glyph's height

    return logo.astype(np.uint8)  # final RGB logo image, cast to standard image dtype


def plot_saliency(X, W, nt_width=100, norm_factor=3, str_null=None, outdir="results/"):
    """Plot a saliency visualization combining sequence logo and saliency heatmaps.

    This function creates a multi-row figure that typically includes:
      - saliency logo (logo built from normalized saliency PWM)
      - saliency heatmap (resized raw weights)
      - raw sequence logo
      - (optional) structure saliency heatmap + structure trace (if X includes structure)

    Args:
        X (np.ndarray):
            Input features array. Expected shape depends on mode:
            - Sequence-only mode: shape (4, L) where rows correspond to A/C/G/U(T).
            - Sequence+structure mode: shape (>=5, L) where `X[4, :]` stores per-position structure scores.
            Padding positions are expected to be all zeros across `X[:4, :]`.
        W (np.ndarray):
            Saliency/importance weights aligned to `X`. Expected shape matches `X` (at least first 4 rows).
        nt_width (int, optional):
            Pixel width per nucleotide position in rendered images.
        norm_factor (float, optional):
            Sharpening factor passed to `normalize_pwm(..., factor=norm_factor)` for saliency logo.
        str_null (np.ndarray, optional):
            Mask for null structure positions. Required if `X.shape[0] > 4`.
            Expected to be broadcastable such that `str_null.T == 1` selects null positions.
        outdir (str, optional):
            Output filepath used by `fig.savefig`. Despite the name, this argument is treated as a file path
            in the current implementation.

    Returns:
        None:
            The figure is saved to disk and all matplotlib figures are closed.
    """
    # filter out zero-padding
    plot_index = np.where(np.sum(X[:4,:], axis=0)!=0)[0]  # positions where at least one nucleotide channel is nonzero (i.e. not padding)
    num_nt = len(plot_index)  # number of real (non-padded) positions to render
    trace_width = num_nt*nt_width  # total pixel width for heatmap/line traces
    trace_height = 400  # fixed pixel height for heatmap/line traces

    seq_str_mode = False  # default: sequence-only plotting
    if X.shape[0]>4:  # an extra row beyond the 4 nucleotide channels means structure data is present
        seq_str_mode = True  # enable the structure panel and structure line overlay
        assert str_null is not None, "Null region is not provided."  # structure mode requires a null-region mask

    # sequence logo
    img_seq_raw = seq_logo(X[:4, plot_index], height=nt_width, nt_width=nt_width)  # render the raw (non-saliency) sequence as a logo image

    if seq_str_mode:
        # structure line
        str_raw = X[4, plot_index]  # per-position structure score for the real (non-padded) positions
        if str_null.sum() > 0:  # some positions are marked as having no valid structure score
            str_raw[str_null.T==1] = -0.01  # flag those positions with a sentinel value so they render distinctly (see white-line masking below)

        line_str_raw = np.zeros(trace_width)  # per-pixel-column line trace, one block of `nt_width` pixels per sequence position
        for v in range(str_raw.shape[0]):  # fill in each position's block
            line_str_raw[v*nt_width:(v+1)*nt_width] = (1-str_raw[v])*trace_height  # invert and scale the score to pixel-space y-coordinate
            # i+=1

    # sequence saliency logo
    seq_sal = normalize_pwm(W[:4, plot_index], factor=norm_factor)  # normalize+sharpen the nucleotide saliency weights into a logo-ready PWM
    img_seq_sal_logo = seq_logo(seq_sal, height=nt_width*5, nt_width=nt_width)  # render the saliency-weighted sequence as a (taller) logo
    img_seq_sal = imresize(W[:4, plot_index], output_shape=(trace_height, trace_width))  # resize the raw saliency weights into a heatmap-sized image

    if seq_str_mode:
        # structure saliency logo
        str_sal = W[4, plot_index].reshape(1,-1)  # structure-channel saliency as a single row
        img_str_sal = imresize(str_sal, output_shape=(trace_height, trace_width))  # resize into a heatmap-sized image

    # plot
    fig = plt.figure(figsize=(10.1,2))  # wide, short figure to accommodate the stacked panels
    gs = gridspec.GridSpec(nrows=4, ncols=1, height_ratios=[2.5, 1, 0.5, 1])  # 4 stacked rows: saliency logo, saliency heatmap, raw logo, (optional) structure panel
    cmap_reversed = mpl.cm.get_cmap('jet')  # colormap used for the saliency/structure heatmaps

    ax = fig.add_subplot(gs[0, 0])  # top panel: saliency logo
    ax.axis('off')  # hide axis ticks/labels/border, this is an image panel
    ax.imshow(img_seq_sal_logo)  # display the rendered saliency logo
    plt.text(x=trace_width-400,y=10, s='BRIDGE', fontsize=4)  # watermark/label in the top-right corner

    ax = fig.add_subplot(gs[1, 0])   # second panel: raw saliency heatmap
    ax.axis('off')
    ax.imshow(img_seq_sal, cmap=cmap_reversed)  # display the saliency heatmap with the jet colormap

    ax = fig.add_subplot(gs[2, 0])   # third panel: raw sequence logo (no saliency weighting)
    ax.axis('off')
    ax.imshow(img_seq_raw)

    if seq_str_mode:
        ax = fig.add_subplot(gs[3, 0])   # fourth panel (only if structure data present): structure saliency heatmap + structure trace
        ax.axis('off')
        ax.imshow(img_str_sal, cmap=cmap_reversed)  # display the structure saliency heatmap
        ax.plot(line_str_raw, '-', color='r', linewidth=1, scalex=False, scaley=False)  # overlay the raw structure score as a red line trace

        # plot balck line to hide the -1(NULL structure score)
        x = (np.zeros(trace_width) + (1+0.01))*trace_height  +1.5  # a constant y-position just below the plotted range
        ax.plot(x, '-', color='white', linewidth=1.2, scalex=False, scaley=False)  # draw a white line to visually mask the null-structure sentinel dip

    plt.subplots_adjust(wspace=0, hspace=0)  # remove spacing between the stacked panels so they align as one continuous track

    # save figure
    filepath = outdir  # despite the parameter name, this is treated as a full file path (see module docstring)
    fig.savefig(filepath, format='png', dpi=300, bbox_inches='tight')  # save the composed figure as a PNG
    plt.close('all')  # release the figure/axes to avoid accumulating open figures across repeated calls
