"""
Training and evaluation loops for BRIDGE-style binary classification.

This module contains thin PyTorch training/validation utilities used by the BRIDGE
pipeline. It implements:

- :func:`train`:
  One-epoch training loop with gradient clipping and running metric aggregation.
- :func:`validate`:
  Evaluation loop that returns dataset-level metrics plus concatenated labels/probabilities.
- :func:`validate2`:
  Inference-only loop that returns probabilities (sigmoid applied), **no labels** required.
- :func:`validate_without_sigmoid`:
  Inference-only loop that returns **raw outputs** (no sigmoid), useful for logits or
  already-probabilistic models.

Who this module is for
----------------------
- Users training BRIDGE (or BRIDGE-compatible) binary classifiers.
- Developers who want a simple, reproducible training loop consistent with the paper/repo.

Model I/O contract
------------------
The functions here assume the model signature is::

    logits = model(x, attn, s, motif, plfold)

where each input is batch-first (``B`` is batch size). The model output is assumed to be
a **logit** (or logit-like score) per sample.

- Expected output shape: ``(B,)`` or ``(B, 1)``
- Probabilities are computed as ``torch.sigmoid(logits)`` when metrics are computed.

If your model already outputs probabilities, prefer :func:`validate_without_sigmoid`
(or adjust this module to avoid applying sigmoid twice).

DataLoader batch conventions
----------------------------
Two batch formats are supported depending on the function:

**Training / labeled evaluation** (:func:`train`, :func:`validate`)
    Each batch from the loader must be a 6-tuple::

        (x0, x00, x000, x0000, x00000, y0)

    with the following semantics::

        x0      -> x      : Transformer / RBPformer features
        x00     -> attn   : attention / adjacency-like tensor (for graph branch)
        x000    -> s      : structure tensor
        x0000   -> motif  : motif tensor
        x00000  -> plfold : biochemical features tensor
        y0      -> y      : binary labels (0/1)

**Inference only** (:func:`validate2`, :func:`validate_without_sigmoid`)
    Each batch must be a 5-tuple (no labels)::

        (x0, x00, x000, x0000, x00000)

Tensor dtypes and device placement
----------------------------------
All inputs are converted to ``float`` and moved to ``device``. Labels are moved to
``device`` and cast to float for loss computation. For metrics, labels are converted to
CPU integer arrays and predictions to CPU float arrays.

Metrics
-------
Metrics are computed via :class:`utils.metrics.MLMetrics` with ``objective="binary"``.
Internally, this uses::

    prob = sigmoid(logits)

and computes accuracy / ROC-AUC / PR-AUC / F1 / MCC plus confusion counts.

The training loop calls::

    met.update(y_np, p_np, [loss.item()])

so the mean loss for the epoch is tracked as an extra field appended to the metric vector.

Important behavior and caveats
------------------------------
Skipping degenerate batches (train only)
    :func:`train` **skips** batches where labels are single-class:

    - all-negative: ``y0.sum() == 0``
    - all-positive: ``y0.sum() == batch_size``

    This means:
    - those batches do not contribute to optimization updates,
    - and do not contribute to metric aggregation.

    .. warning::
       This behavior is only correct if your training sampling strategy can produce
       single-class batches and you explicitly want to skip them. If you need every
       sample to contribute to training, remove this condition or ensure balanced batching.

Gradient clipping
    :func:`train` applies ``torch.nn.utils.clip_grad_norm_(model.parameters(), 5)`` each step.
    Adjust the max-norm if you change optimizer/loss scaling.

Shape alignment
    ``criterion(output, y)`` must be valid; in practice, ensure ``y`` is shaped like
    ``output`` (e.g., both ``(B, 1)``). If your model outputs ``(B,)`` but labels are
    ``(B, 1)``, you may want to ``y = y.view_as(output)`` (or squeeze) upstream.

Example
-------
.. code-block:: python

    from torch.nn import BCEWithLogitsLoss
    from torch.optim import Adam

    model = BRIDGE(...).to(device)
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    # one epoch
    met_train = train(model, device, train_loader, criterion, optimizer, batch_size=64)

    # evaluation
    met_val, y_val, p_val = validate(model, device, val_loader, criterion)

    # inference only
    p_test = validate2(model, device, test_loader_no_labels, criterion)

"""

from __future__ import print_function  # ensure Python 2-style print() semantics if this ever runs under py2
import math  # used for the pow()/floor() step-decay LR schedule in fit_bridge
from tqdm import tqdm  # progress-bar helper (imported for optional use in loops)
import numpy as np  # array concatenation/aggregation of predictions and labels
import torch  # core tensor ops, autograd, and nn utilities
import utils.metrics as metrics  # BRIDGE's metric accumulator (MLMetrics)


def train(model, device, train_loader, criterion, optimizer, batch_size):
    """Train one epoch and accumulate binary classification metrics.

    This function runs a standard PyTorch training loop over ``train_loader``:
    forward -> loss -> backward -> gradient clipping -> optimizer step. Metrics
    are tracked via ``utils.metrics.MLMetrics(objective="binary")``.

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
    """
    model.train()  # switch model to training mode (enables dropout/batchnorm updates)
    met = metrics.MLMetrics(objective='binary')  # fresh metric accumulator for this epoch
    for batch_idx, (x0, x00, x000, x0000, x00000, y0) in enumerate(train_loader):  # iterate labeled batches
        x, attn, s, motif, plfold, y = x0.float().to(device), x00.float().to(device), \
                    x000.float().to(device), x0000.float().to(device), x00000.float().to(device), y0.to(device).float()  # cast every tensor to float and move to the training device
        if y0.sum() == 0 or y0.sum() == batch_size:  # detect an all-negative or all-positive (single-class) batch
            continue  # skip degenerate batches: no useful gradient signal, don't update metrics or weights
        optimizer.zero_grad()  # clear gradients accumulated from the previous step
        output = model(x, attn, s, motif, plfold)  # forward pass through BRIDGE, produces logits shape (B,) or (B,1)
        loss = criterion(output, y)  # compute scalar loss (e.g. BCEWithLogitsLoss) between logits and labels
        prob = torch.sigmoid(output)  # convert logits to probabilities for metric computation

        y_np = y.to(device='cpu', dtype=torch.long).detach().numpy()  # move labels to CPU as integer numpy array
        p_np = prob.to(device='cpu').detach().numpy()  # move probabilities to CPU as numpy array
        met.update(y_np, p_np, [loss.item()])  # accumulate this batch's labels/probs/loss into the running metrics
        loss.backward()  # backpropagate to compute gradients w.r.t. model parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # clip gradient norm to 5 to prevent exploding gradients
        optimizer.step()  # apply the optimizer update using the (clipped) gradients

    return met  # return the epoch's aggregated training metrics


def fit_bridge(
    model,
    device,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    max_epochs=200,  # hard cap on number of training epochs
    warmup_epochs=40,  # length (in epochs) of the linear LR warm-up phase
    base_lr=0.001,  # target LR reached at the end of warm-up (scaled by warmup_scale)
    warmup_scale=1.6,  # multiplier applied to base_lr during warm-up ramp
    initial_lrate=0.0016,  # LR used as the base for the post-warmup step-decay schedule
    drop=0.8,  # multiplicative decay factor applied every `epochs_drop` epochs
    epochs_drop=5.0,  # number of epochs between successive LR decay steps
    early_stopping=10,  # stop training if val AUC hasn't improved for this many epochs
    ckpt_path=None,  # optional path to save the best-so-far model state_dict
    train_batch_size=32,  # batch size forwarded to train() for degenerate-batch detection
    log_fn=None,  # optional callable(str) used for per-epoch logging
    tag="",  # short label prefixed to log lines
):
    """Train a BRIDGE model with the standard schedule and return the best-epoch metrics.

    This encapsulates the per-run training loop that was previously inlined in
    ``main.py``'s ``--train`` block: warm-up + step-decay LR, val-AUC-driven checkpoint
    selection, and early stopping. It reuses :func:`train` and :func:`validate`. Behavior is
    identical to the original loop at default arguments.

    Args:
        model, device, train_loader, val_loader, criterion, optimizer: standard training objects.
        max_epochs: hard cap on epochs (original loop ran ``range(1, 201)``).
        warmup_epochs: linear warm-up length; LR = ``base_lr * warmup_scale * epoch/warmup_epochs``.
        initial_lrate, drop, epochs_drop: post-warmup step-decay schedule.
        early_stopping: stop when ``epoch - best_epoch > early_stopping``.
        ckpt_path: if set, best-so-far ``state_dict`` is saved here whenever val AUC improves.
        train_batch_size: passed to :func:`train` for degenerate-batch detection.
        log_fn: optional ``callable(str)`` for per-epoch logging (e.g. write to a logfile).
        tag: short label prefixed to log lines (e.g. dataset/config name).

    Returns:
        dict: ``{best_epoch, best_val_auc, best_val_acc, best_val_prc, best_val_mcc,
        stopped_epoch}``.
    """
    def _log(msg):  # small helper so logging is a no-op when log_fn isn't provided
        if log_fn is not None:  # only log if caller supplied a logging callback
            log_fn(msg)  # forward the message to the caller's logger

    best_auc = 0.0  # best validation AUC seen so far (drives checkpointing/early stopping)
    best_acc = 0.0  # validation accuracy at the best-AUC epoch
    best_mcc = 0.0  # validation MCC at the best-AUC epoch
    best_prc = 0.0  # validation PR-AUC at the best-AUC epoch
    best_epoch = 0  # epoch index at which best_auc was achieved
    stopped_epoch = 0  # last epoch actually run (updated each iteration, returned at the end)

    for epoch in range(1, max_epochs + 1):  # 1-indexed epoch loop up to max_epochs
        stopped_epoch = epoch  # record the current epoch in case training stops here
        t_met = train(model, device, train_loader, criterion, optimizer, batch_size=train_batch_size)  # run one training epoch
        v_met, _, _ = validate(model, device, val_loader, criterion)  # evaluate on the validation set (labels/probs discarded here)

        # Warm-up followed by step-wise exponential learning-rate decay.
        if epoch <= warmup_epochs:  # still within the linear warm-up phase
            lr = base_lr * (warmup_scale * epoch / warmup_epochs)  # ramp LR linearly from ~0 up to base_lr*warmup_scale
        else:  # past warm-up: switch to step-decay schedule
            lr = initial_lrate * math.pow(drop, math.floor((epoch - warmup_epochs) / epochs_drop))  # decay LR by `drop` every `epochs_drop` epochs
        for param_group in optimizer.param_groups:  # apply the computed LR to every optimizer param group
            param_group['lr'] = lr  # overwrite this group's learning rate

        if best_auc < v_met.auc:  # this epoch's validation AUC improved on the best seen so far
            best_auc = v_met.auc  # update best AUC
            best_acc = v_met.acc  # record corresponding accuracy
            best_mcc = v_met.mcc  # record corresponding MCC
            best_prc = v_met.prc  # record corresponding PR-AUC
            best_epoch = epoch  # remember which epoch achieved this
            if ckpt_path is not None:  # only checkpoint if a save path was given
                torch.save(model.state_dict(), ckpt_path)  # persist the best-so-far model weights to disk

        # Early stopping based on validation performance.
        if epoch - best_epoch > early_stopping:  # no improvement for more than `early_stopping` epochs
            _log("{} Early stop at {}".format(tag, epoch))  # log the early-stopping decision
            break  # exit the training loop early

        _log(
            "{} Train Epoch: {}  avg.loss: {:.4f} Acc: {:.2f}%, AUC: {:.4f}, PRC: {:.4f}, "
            "MCC: {:.4f}, lr: {:.6f}".format(
                tag, epoch, t_met.other[0], t_met.acc, t_met.auc, t_met.prc, t_met.mcc, lr)
        )  # log this epoch's training metrics and current LR
        _log(
            "{} Valid Epoch: {}  avg.loss: {:.4f} Acc: {:.2f}%, AUC: {:.4f} ({:.4f}), "
            "PRC: {:.4f}, MCC: {:.4f}, best_epoch: {}".format(
                tag, epoch, v_met.other[0], v_met.acc, v_met.auc, best_auc, v_met.prc,
                v_met.mcc, best_epoch)
        )  # log this epoch's validation metrics alongside the best-so-far AUC/epoch

    return {
        "best_epoch": int(best_epoch),  # epoch number that produced the best validation AUC
        "best_val_auc": float(best_auc),  # best validation ROC-AUC achieved
        "best_val_acc": float(best_acc),  # validation accuracy at that best epoch
        "best_val_prc": float(best_prc),  # validation PR-AUC at that best epoch
        "best_val_mcc": float(best_mcc),  # validation MCC at that best epoch
        "stopped_epoch": int(stopped_epoch),  # final epoch index reached (may be < max_epochs due to early stopping)
    }


def validate(model, device, test_loader, criterion):
    """Evaluate a binary classifier and return metrics, labels, and probabilities.

    Runs the model in evaluation mode over ``test_loader``, collecting:
    - concatenated labels ``y_all``
    - concatenated probabilities ``p_all`` (computed as ``sigmoid(logits)``)
    - mean loss across batches

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
    """
    model.eval()  # switch to eval mode (disables dropout, freezes batchnorm stats)
    y_all = []  # collects per-batch label arrays to concatenate at the end
    p_all = []  # collects per-batch probability arrays to concatenate at the end
    l_all = []  # collects per-batch scalar losses to average at the end
    with torch.no_grad():  # disable autograd tracking since we're not backpropagating
        for batch_idx, (x0, x00,x000, x0000, x00000, y0) in enumerate(test_loader):  # iterate labeled evaluation batches
            x, attn, s, motif, plfold, y = x0.float().to(device), x00.float().to(device), \
                    x000.float().to(device), x0000.float().to(device), x00000.float().to(device), y0.to(device).float()  # cast to float and move all tensors to the eval device

            output = model(x, attn, s, motif, plfold)  # forward pass, returns logits
            loss = criterion(output, y)  # compute this batch's loss for monitoring
            prob = torch.sigmoid(output)  # convert logits to probabilities

            y_np = y.to(device='cpu', dtype=torch.long).numpy()  # move labels to CPU as integer numpy array
            p_np = prob.to(device='cpu').numpy()  # move probabilities to CPU as numpy array
            l_np = loss.item()  # extract the scalar loss value as a Python float

            y_all.append(y_np)  # stash this batch's labels
            p_all.append(p_np)  # stash this batch's probabilities
            l_all.append(l_np)  # stash this batch's loss

    y_all = np.concatenate(y_all)  # flatten all per-batch label arrays into one dataset-level array
    p_all = np.concatenate(p_all)  # flatten all per-batch probability arrays into one dataset-level array
    l_all = np.array(l_all)  # convert the list of per-batch losses into a numpy array

    met = metrics.MLMetrics(objective='binary')  # fresh metric accumulator for the full evaluation set
    met.update(y_all, p_all, [l_all.mean()])  # compute dataset-level metrics using the mean loss across batches

    return met, y_all, p_all  # return metrics plus the raw labels/probabilities for downstream use


def validate2(model, device, test_loader, criterion):
    """Run inference and return predicted probabilities only (no labels).

    This function assumes ``test_loader`` yields inputs only (no ``y0``) and returns
    concatenated probabilities computed as ``torch.sigmoid(logits)``.

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
    """
    model.eval()  # switch to eval mode
    p_all = []  # collects per-batch probability arrays
    with torch.no_grad():  # no gradients needed for pure inference
        for batch_idx, (x0, x00,x000, x0000, x00000) in enumerate(test_loader):  # iterate unlabeled inference batches
            x, attn, s, motif, plfold = x0.float().to(device), x00.float().to(device), \
                    x000.float().to(device), x0000.float().to(device), x00000.float().to(device)  # cast to float and move to device

            output = model(x, attn, s, motif, plfold)  # forward pass, returns logits
            prob = torch.sigmoid(output)  # convert logits to probabilities
            p_np = prob.to(device='cpu').numpy()  # move probabilities to CPU as numpy array
            p_all.append(p_np)  # stash this batch's probabilities

    p_all = np.concatenate(p_all)  # flatten all per-batch probability arrays into one dataset-level array

    return p_all  # return the full set of predicted probabilities


def validate_without_sigmoid(model, device, test_loader, criterion):
    """Run inference and return raw model outputs (no sigmoid applied).

    This function is identical to :func:`validate2` except it returns the raw model outputs
    directly (i.e., no ``torch.sigmoid``). This is useful when downstream code wants logits,
    applies custom transformations, or when the model already outputs probabilities.

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
    **Expected batch format**

    - Each batch from ``test_loader`` must be a 5-tuple::

          (x0, x00, x000, x0000, x00000)

    - Semantics:

      - ``x0``      -> ``x``     : RBPformer feature tensor
      - ``x00``     -> ``attn``  : attention / adjacency-like tensor
      - ``x000``    -> ``s``     : structural tensor
      - ``x0000``   -> ``motif`` : motif tensor
      - ``x00000``  -> ``plfold``: biochemical tensor
    """
    model.eval()  # switch to eval mode
    p_all = []  # collects per-batch raw-output arrays
    with torch.no_grad():  # no gradients needed for pure inference
        for batch_idx, (x0, x00,x000, x0000, x00000) in enumerate(test_loader):  # iterate unlabeled inference batches
            x, attn, s, motif, plfold = x0.float().to(device), x00.float().to(device), \
                    x000.float().to(device), x0000.float().to(device), x00000.float().to(device)  # cast to float and move to device

            prob = model(x, attn, s, motif, plfold)  # forward pass; despite the name, this is the raw (non-sigmoid) output
            p_np = prob.to(device='cpu').numpy()  # move raw outputs to CPU as numpy array
            p_all.append(p_np)  # stash this batch's raw outputs

    p_all = np.concatenate(p_all)  # flatten all per-batch arrays into one dataset-level array

    return p_all  # return the full set of raw model outputs
