from __future__ import print_function
from tqdm import tqdm
import numpy as np
import torch
import utils.metrics as metrics



def train(model, device, train_loader, criterion, optimizer, batch_size):
    """Train one epoch and accumulate binary classification metrics.

    This function runs a standard PyTorch training loop over ``train_loader``:
    forward -> loss -> backward -> gradient clipping -> optimizer step. Metrics
    are tracked via ``utils.metrics.MLMetrics(objective="binary")``.

    Notes
    -----
    **Expected batch format**

    - Each batch from ``train_loader`` must be a 6-tuple::

          (x0, x00, x000, x0000, x00000, y0)

    - Semantics (names used inside this function):

      - ``x0``      -> ``x``     : RBPformer feature tensor
      - ``x00``     -> ``attn``  : attention / adjacency-like tensor
      - ``x000``    -> ``s``     : structural tensor
      - ``x0000``   -> ``motif`` : motif tensor
      - ``x00000``  -> ``plfold``: biochemical tensor
      - ``y0``      -> ``y``     : binary labels (0/1)

    **Tensor conventions**

    - Batch dimension is the first axis for all inputs: ``(B, ...)``.
    - Model returns **logits** of shape ``(B,)`` or ``(B, 1)``.
    - ``criterion(output, y)`` must be valid (e.g., ``BCEWithLogitsLoss`` with matching shapes).
    - Probabilities for metrics are computed as ``torch.sigmoid(output)``.

    **Special handling**

    - Degenerate batches are skipped:

      - all-negative: ``y0.sum() == 0``
      - all-positive: ``y0.sum() == batch_size``

      This avoids metric updates and optimization steps on single-class batches.

    Parameters
    ----------
    model : torch.nn.Module
        Model callable with signature ``model(x, attn, s, motif, plfold)`` returning logits.
    device : torch.device
        Target device used to move tensors and model.
    train_loader : torch.utils.data.DataLoader
        Iterable over training batches, each yielding the 6-tuple described above.
    criterion : callable
        Loss function. Common choice is ``torch.nn.BCEWithLogitsLoss`` when outputs are logits.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    batch_size : int
        Expected batch size used for detecting all-positive/all-negative batches.

    Returns
    -------
    utils.metrics.MLMetrics
        Metric accumulator updated over all non-skipped batches. Contains aggregated
        binary-classification metrics and mean loss (as passed via ``met.update``).
    """
    model.train()
    met = metrics.MLMetrics(objective='binary')
    for batch_idx, (x0, x00, x000, x0000, x00000, y0) in enumerate(train_loader):
        x, attn, s, motif, plfold, y = x0.float().to(device), x00.float().to(device), \
                    x000.float().to(device), x0000.float().to(device), x00000.float().to(device), y0.to(device).float()
        if y0.sum() == 0 or y0.sum() == batch_size:
            continue
        optimizer.zero_grad()  
        output = model(x, attn, s, motif, plfold)
        loss = criterion(output, y)
        prob = torch.sigmoid(output)

        y_np = y.to(device='cpu', dtype=torch.long).detach().numpy()
        p_np = prob.to(device='cpu').detach().numpy()
        met.update(y_np, p_np, [loss.item()])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    return met


def validate(model, device, test_loader, criterion):
    """Evaluate a binary classifier and return metrics, labels, and probabilities.

    Runs the model in evaluation mode over ``test_loader``, collecting:
    - concatenated labels ``y_all``
    - concatenated probabilities ``p_all`` (computed as ``sigmoid(logits)``)
    - mean loss across batches

    Notes
    -----
    **Expected batch format**

    - Each batch from ``test_loader`` must be a 6-tuple::

          (x0, x00, x000, x0000, x00000, y0)

    - Semantics:

      - ``x0``      -> ``x``     : RBPformer feature tensor
      - ``x00``     -> ``attn``  : attention / adjacency-like tensor
      - ``x000``    -> ``s``     : structural tensor
      - ``x0000``   -> ``motif`` : motif tensor
      - ``x00000``  -> ``plfold``: biochemical tensor
      - ``y0``      -> ``y``     : binary labels (0/1)

    **Tensor conventions**

    - Model returns logits; probabilities are computed as ``torch.sigmoid(output)``.
    - Arrays are concatenated along the first axis to produce dataset-level outputs.

    Parameters
    ----------
    model : torch.nn.Module
        Model callable with signature ``model(x, attn, s, motif, plfold)`` returning logits.
    device : torch.device
        Target device.
    test_loader : torch.utils.data.DataLoader
        Iterable over evaluation batches.
    criterion : callable
        Loss function compatible with logits and labels.

    Returns
    -------
    met : utils.metrics.MLMetrics
        Metric accumulator updated once with concatenated arrays and mean loss.
    y_all : np.ndarray
        Concatenated labels for all samples. Shape typically ``(N,)`` or ``(N, 1)``.
    p_all : np.ndarray
        Concatenated probabilities for all samples. Shape matches ``y_all``.

    Notes
    -----
    The returned ``met`` is updated with the mean batch loss (simple average).
    """
    model.eval()
    y_all = []
    p_all = []
    l_all = []
    with torch.no_grad():
        for batch_idx, (x0, x00,x000, x0000, x00000, y0) in enumerate(test_loader):
            x, attn, s, motif, plfold, y = x0.float().to(device), x00.float().to(device), \
                    x000.float().to(device), x0000.float().to(device), x00000.float().to(device), y0.to(device).float()
            
            output = model(x, attn, s, motif, plfold)
            loss = criterion(output, y)
            prob = torch.sigmoid(output)

            y_np = y.to(device='cpu', dtype=torch.long).numpy()
            p_np = prob.to(device='cpu').numpy()
            l_np = loss.item()

            y_all.append(y_np)
            p_all.append(p_np)
            l_all.append(l_np)

    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    l_all = np.array(l_all)

    met = metrics.MLMetrics(objective='binary')
    met.update(y_all, p_all, [l_all.mean()])

    return met, y_all, p_all


def validate2(model, device, test_loader, criterion):
    """Run inference and return predicted probabilities only (no labels).

    This function assumes ``test_loader`` yields inputs only (no ``y0``) and returns
    concatenated probabilities computed as ``torch.sigmoid(logits)``.

    Notes
    -----
    **Expected batch format**

    - Each batch from ``test_loader`` must be a 5-tuple::

          (x0, x00, x000, x0000, x00000)

    - Semantics:

      - ``x0``      -> ``x``     : RBPformer feature tensor
      - ``x00``     -> ``attn``  : attention / adjacency-like tensor
      - ``x000``    -> ``s``     : structural tensor
      - ``x0000``   -> ``motif`` : motif tensor
      - ``x00000``  -> ``plfold``: biochemical tensor

    Parameters
    ----------
    model : torch.nn.Module
        Model callable with signature ``model(x, attn, s, motif, plfold)`` returning logits.
    device : torch.device
        Target device.
    test_loader : torch.utils.data.DataLoader
        Iterable over inference batches (no labels).
    criterion : callable
        Unused. Kept for API compatibility with other validation functions.

    Returns
    -------
    np.ndarray
        Concatenated probabilities for all samples. Shape typically ``(N,)`` or ``(N, 1)``.
    """
    model.eval()
    p_all = []
    with torch.no_grad():
        for batch_idx, (x0, x00,x000, x0000, x00000) in enumerate(test_loader):
            x, attn, s, motif, plfold = x0.float().to(device), x00.float().to(device), \
                    x000.float().to(device), x0000.float().to(device), x00000.float().to(device)

            output = model(x, attn, s, motif, plfold)
            prob = torch.sigmoid(output)
            p_np = prob.to(device='cpu').numpy()
            p_all.append(p_np)

    p_all = np.concatenate(p_all)

    return p_all


def validate_without_sigmoid(model, device, test_loader, criterion):
    """Run inference and return raw model outputs (no sigmoid applied).

    This function is identical to :func:`validate2` except it returns the raw model outputs
    directly (i.e., no ``torch.sigmoid``). This is useful when downstream code wants logits,
    applies custom transformations, or when the model already outputs probabilities.

    Notes
    -----
    **Expected batch format**

    - Each batch from ``test_loader`` must be a 5-tuple::

          (x0, x00, x000, x0000, x00000)

    - Semantics:

      - ``x0``      -> ``x``     : RBPformer feature tensor
      - ``x00``     -> ``attn``  : attention / adjacency-like tensor
      - ``x000``    -> ``s``     : structural tensor
      - ``x0000``   -> ``motif`` : motif tensor
      - ``x00000``  -> ``plfold``: biochemical tensor

    Parameters
    ----------
    model : torch.nn.Module
        Model callable with signature ``model(x, attn, s, motif, plfold)`` returning raw outputs.
    device : torch.device
        Target device.
    test_loader : torch.utils.data.DataLoader
        Iterable over inference batches.
    criterion : callable
        Unused. Kept for API compatibility.

    Returns
    -------
    np.ndarray
        Concatenated raw outputs for all samples. Shape typically ``(N,)`` or ``(N, 1)``.

    Notes
    -----
    Despite the name ``p_all``, the returned array may contain logits, scores, or probabilities
    depending on the model implementation.
    """
    model.eval()
    p_all = []
    with torch.no_grad():
        for batch_idx, (x0, x00,x000, x0000, x00000) in enumerate(test_loader):
            x, attn, s, motif, plfold = x0.float().to(device), x00.float().to(device), \
                    x000.float().to(device), x0000.float().to(device), x00000.float().to(device)

            prob = model(x, attn, s, motif, plfold)
            p_np = prob.to(device='cpu').numpy()
            p_all.append(p_np)

    p_all = np.concatenate(p_all)

    return p_all
