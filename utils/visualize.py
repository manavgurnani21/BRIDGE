import os, sys
import numpy as np
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.transform import resize as imresize
import torch
import utils.datautils as datautils
from PIL import Image


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
    model.eval()
    p_all = []
    with torch.no_grad():
        for batch_idx, (x0, y0) in enumerate(test_loader):
            x, y = x0.float().to(device), y0.to(device).float()
            output = model(x)
            prob = torch.sigmoid(output)

            p_np = prob.to(device='cpu').numpy()
            p_all.append(p_np)

    p_all = np.concatenate(p_all)
    return p_all


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
    if MAX is None:
        MAX = np.max(np.abs(pwm))
    pwm = pwm/MAX
    if factor:
        pwm = np.exp(pwm*factor)
    norm = np.outer(np.ones(pwm.shape[0]), np.sum(np.abs(pwm), axis=0))
    return pwm/norm


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
    def entropy(p):
        s = 0
        for i in range(len(p)):
            if p[i] > 0:
                s -= p[i]*np.log2(p[i])
        return s

    num_nt, num_seq = pwm.shape
    heights = np.zeros((num_nt,num_seq))
    for i in range(num_seq):
        if norm == 1:
            total_height = height
        else:
            total_height = (np.log2(num_nt) - entropy(pwm[:, i]))*height
        
        heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height*2))

    return heights.astype(int)


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
    acgu_path = './acgu.npz'
    chars = np.load(acgu_path,allow_pickle=True)['data']
    heights = get_nt_height(pwm, height, norm)
    num_nt, num_seq = pwm.shape
    width = np.ceil(nt_width*num_seq).astype(int)
    
    max_height = height*2
    logo = np.ones((max_height, width, 3)).astype(int)*255
    for i in range(num_seq):
        nt_height = np.sort(heights[:,i])
        index = np.argsort(heights[:,i])
        remaining_height = np.sum(heights[:,i])
        offset = max_height-remaining_height

        for j in range(num_nt):
            if nt_height[j] <=0 :
                continue
            # resized dimensions of image
            nt_img = imresize(chars[index[j]], output_shape=(nt_height[j], nt_width))*255
            # determine location of image
            height_range = range(remaining_height-nt_height[j], remaining_height)
            width_range = range(i*nt_width, i*nt_width+nt_width)
            # 'annoying' way to broadcast resized nucleotide image
            if height_range:
                for k in range(3):
                    for m in range(len(width_range)):
                        logo[height_range+offset, width_range[m],k] = nt_img[:,m,k]

            remaining_height -= nt_height[j]

    return logo.astype(np.uint8)


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
    plot_index = np.where(np.sum(X[:4,:], axis=0)!=0)[0]
    num_nt = len(plot_index)
    trace_width = num_nt*nt_width
    trace_height = 400
    
    seq_str_mode = False
    if X.shape[0]>4:
        seq_str_mode = True
        assert str_null is not None, "Null region is not provided."

    # sequence logo
    img_seq_raw = seq_logo(X[:4, plot_index], height=nt_width, nt_width=nt_width)

    if seq_str_mode:
        # structure line
        str_raw = X[4, plot_index]
        if str_null.sum() > 0:
            str_raw[str_null.T==1] = -0.01

        line_str_raw = np.zeros(trace_width)
        for v in range(str_raw.shape[0]):
            line_str_raw[v*nt_width:(v+1)*nt_width] = (1-str_raw[v])*trace_height 
            # i+=1
    
    # sequence saliency logo
    seq_sal = normalize_pwm(W[:4, plot_index], factor=norm_factor)
    img_seq_sal_logo = seq_logo(seq_sal, height=nt_width*5, nt_width=nt_width)
    img_seq_sal = imresize(W[:4, plot_index], output_shape=(trace_height, trace_width))

    if seq_str_mode:
        # structure saliency logo
        str_sal = W[4, plot_index].reshape(1,-1)
        img_str_sal = imresize(str_sal, output_shape=(trace_height, trace_width))

    # plot    
    fig = plt.figure(figsize=(10.1,2))
    gs = gridspec.GridSpec(nrows=4, ncols=1, height_ratios=[2.5, 1, 0.5, 1])
    cmap_reversed = mpl.cm.get_cmap('jet')

    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    ax.imshow(img_seq_sal_logo)
    plt.text(x=trace_width-400,y=10, s='BRIDGE', fontsize=4)

    ax = fig.add_subplot(gs[1, 0]) 
    ax.axis('off')
    ax.imshow(img_seq_sal, cmap=cmap_reversed)

    ax = fig.add_subplot(gs[2, 0]) 
    ax.axis('off')
    ax.imshow(img_seq_raw)

    if seq_str_mode:
        ax = fig.add_subplot(gs[3, 0]) 
        ax.axis('off')
        ax.imshow(img_str_sal, cmap=cmap_reversed)
        ax.plot(line_str_raw, '-', color='r', linewidth=1, scalex=False, scaley=False)
        
        # plot balck line to hide the -1(NULL structure score)
        x = (np.zeros(trace_width) + (1+0.01))*trace_height  +1.5
        ax.plot(x, '-', color='white', linewidth=1.2, scalex=False, scaley=False)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # save figure
    filepath = outdir
    fig.savefig(filepath, format='png', dpi=300, bbox_inches='tight')
    plt.close('all')
