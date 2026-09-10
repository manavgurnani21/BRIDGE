"""
Evaluation metrics utilities for BRIDGE experiments.

This module provides a small collection of metric functions and a lightweight
accumulator class for tracking classification/regression-style metrics across
training/evaluation steps. It is primarily used by training loops to compute:

- scalar summary metrics (accuracy, ROC-AUC, PR-AUC, F1, MCC)
- confusion-matrix counts (TP, TN, FP, FN)
- correlation/fit metrics for regression-like objectives (Pearson r, R^2-like score, slope)

Who this is for
---------------
- Users running BRIDGE training / validation scripts who need consistent metric reporting.
- Developers extending objectives or adding new tracked scalars (e.g., loss) via the accumulator.

This module assumes NumPy arrays as inputs and relies on scikit-learn for curve metrics.

Key dependencies
----------------
- ``numpy``
- ``scikit-learn``: ``roc_curve``, ``auc``, ``precision_recall_curve``, ``accuracy_score``,
  ``confusion_matrix``, ``f1_score``, ``matthews_corrcoef``
- ``scipy.stats`` for Pearson correlation

Public API
----------
The module exports (see ``__all__``):

- ``pearsonr(label, prediction)``
- ``rsquare(label, prediction)``
- ``accuracy(label, prediction)``
- ``roc(label, prediction)``
- ``pr(label, prediction)``
- ``calculate_metrics(label, prediction, objective)``

It also defines an accumulator class:

- ``MLMetrics``: stores per-step metric vectors and provides running averages/sums.

Input conventions
-----------------
Labels and predictions
    Most functions accept:

    - binary targets: ``label`` shape ``(N,)`` or ``(N, 1)`` or ``(N, K)``
    - predictions: same shape as labels (probabilities/scores in ``[0, 1]`` for binary)

Multi-label behavior
    For 2D inputs (``(N, K)``), metrics are computed per column and then aggregated with
    ``np.nanmean`` / ``np.nanstd`` where applicable.

Objectives in ``calculate_metrics``
-----------------------------------
``calculate_metrics(label, prediction, objective)`` supports:

- ``"binary"`` and ``"hinge"``

  Treats inputs as binary (or multi-label) classification.

  Returns:
  mean: ``[acc, auc_roc, auc_pr, f1, mcc, tp, tn, fp, fn]``
  std : ``[acc_std, auc_roc_std, auc_pr_std, f1_std, mcc_std]``

- ``"categorical"``

  Treats input as multi-class with one-hot labels and predicted class probabilities.

  Returns:
  
    mean: begins with ``[acc, auc_roc, auc_pr]`` (macro over columns),
            then appends per-class ROC-AUC for each column.
            
    std : begins with ``[acc_std, auc_roc_std, auc_pr_std]``,
            then appends the corresponding per-class ROC-AUC standard deviations.

  Note:
    The current implementation appends per-class ROC-AUC only (not per-class PR-AUC).

- ``"squared_error"``, ``"kl_divergence"``, ``"cdf"``

  Treated as regression-like objectives, but labels are thresholded to binary (0/1) first.

  Returns:
    mean: ``[acc, auc_roc, auc_pr, tp, tn, fp, fn, pearsonr_mean, rsquare_mean, slope_mean]``
    
    std : ``[acc_std, auc_roc_std, auc_pr_std, pearsonr_std, rsquare_std, slope_std]``

  Note:
    ``pearsonr``, ``rsquare``, and ``slope`` are computed after label thresholding.

Return value (important)
------------------------
``calculate_metrics`` returns a two-element list:

- ``[mean, std]``

So typical usage is:

.. code-block:: python

    mean, std = calculate_metrics(y_true, y_pred, objective="binary")

Note that the current ``MLMetrics.update`` implementation does::

    met, _ = calculate_metrics(...)

which will set ``met`` to the *mean list* and ignore std.

Notes on individual helpers
---------------------------
``pearsonr``
    - For 1D input, returns ``[stats.pearsonr(label, prediction)]`` (a tuple inside a list).
    - For 2D input, returns a list of coefficients (floats).
    This asymmetry is preserved for backward compatibility.

``rsquare``
    - Computes an R^2-like score using a no-intercept fit (regression through the origin).
      This differs from sklearn's default R^2 which fits an intercept.

``roc`` / ``pr``
    - Return both metric values and the underlying curve points for plotting.
    - For multi-label inputs, curves are computed independently per column.

``MLMetrics``
-------------
``MLMetrics`` stores metric vectors returned by ``calculate_metrics`` and maintains running
average and sum. For objective ``"binary"`` / ``"hinge"``, it also exposes:

- ``acc, auc, prc, f1, mcc`` from the running average
- ``tp, tn, fp, fn`` from the running sum

You may append additional scalars (e.g., loss) by passing them as ``other_lst`` to ``update``.

How to use
----------
Compute metrics for one evaluation run:

.. code-block:: python

    from metrics import calculate_metrics
    mean, std = calculate_metrics(y_true, y_prob, objective="binary")
    acc, auc_roc, auc_pr, f1, mcc, tp, tn, fp, fn = mean

Accumulate metrics across batches:

.. code-block:: python

    from metrics import MLMetrics
    meter = MLMetrics(objective="binary")

    for y_true, y_prob in dataloader:
        meter.update(y_true, y_prob, other_lst=[loss_value])

    print(meter.acc, meter.auc, meter.prc, meter.f1, meter.mcc)

"""


import os, sys  # unused in this module beyond import; kept for parity with original script
import numpy as np  # array math and nan-aware aggregation (nanmean/nanstd)
from six.moves import cPickle  # py2/py3-compatible pickle alias (unused directly here)
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score, confusion_matrix  # curve/scalar metric primitives
from sklearn.metrics import f1_score, matthews_corrcoef  # additional classification metrics
from scipy import stats  # Pearson correlation (stats.pearsonr)

__all__ = [
    "pearsonr",  # Pearson correlation coefficient(s) between label and prediction
    "rsquare",  # no-intercept R^2-like fit metric and slope
    "accuracy",  # thresholded accuracy score
    "roc",  # ROC-AUC and ROC curve points
    "pr",  # PR-AUC and precision-recall curve points
    "calculate_metrics"  # unified per-objective metric dispatcher used by MLMetrics
]


class MLMetrics(object):
    """
    Accumulator for per-step metrics with running average and sums.

    This class wraps `calculate_metrics(...)` and stores the per-update `mean` list returned
    by that function. It exposes common scalar metrics (acc/auc/prc/f1/mcc) and confusion
    matrix counts (tp/tn/fp/fn) computed from the running sum.

    Args:
        objective (str, optional):
            Objective name passed to `calculate_metrics`. Common values:
            'binary', 'hinge', 'categorical', 'squared_error', 'kl_divergence', 'cdf'.
            Default: 'binary'.

    Attributes (after at least one update):
        metrics (list[list[float]]):
            History of per-update metric vectors (the `mean` list from calculate_metrics).
        avg (np.ndarray or list[float]):
            Running average of the stored metric vectors.
        sum (np.ndarray or list[float]):
            Running sum of the stored metric vectors.
        acc, auc, prc, f1, mcc (float):
            Convenience scalars parsed from `avg` at fixed indices (binary/hinge objectives).
        tp, tn, fp, fn (int):
            Confusion-matrix counts parsed from `sum` at fixed indices.

    Notes:
        - This class assumes the `mean` vector ordering used by `calculate_metrics` for
          objective='binary'/'hinge'. If you use other objectives, the index mapping may differ.
        - `other_lst` passed to update() is appended to the metric vector and stored.
    """
    def __init__(self, objective='binary'):
        self.objective = objective  # which calculate_metrics branch to use on each update
        self.metrics = []  # history of per-update mean-metric vectors

    def update(self, label, pred, other_lst):
        """
        Compute metrics for one batch and update running aggregates.

        Args:
            label (np.ndarray):
                Ground-truth labels. Shape and semantics depend on `self.objective`.
            pred (np.ndarray):
                Model predictions (scores/probabilities). Shape should match `label`.
            other_lst (list[float]):
                Optional extra scalar values to append to the metric vector (e.g., loss).

        Returns:
            None. Updates internal state in-place.
        """
        met, _ = calculate_metrics(label, pred, self.objective)  # compute this batch's mean-metric vector (std discarded)
        if len(other_lst) > 0:  # caller passed extra scalars (e.g. loss) to track alongside metrics
            met.extend(other_lst)  # append them to the end of the metric vector
        self.metrics.append(met)  # record this update in the running history
        self.compute_avg()  # refresh avg/sum and the convenience scalar attributes

    def compute_avg(self):
        """
        Recompute running averages and sums over stored metric vectors.

        Returns:
            None. Populates `avg`, `sum`, and convenience fields such as `acc`, `auc`, etc.
        """
        if len(self.metrics) > 1:  # more than one update recorded so far
            self.avg = np.array(self.metrics).mean(axis=0)  # elementwise mean across all stored metric vectors
            self.sum = np.array(self.metrics).sum(axis=0)  # elementwise sum across all stored metric vectors
        else:  # only one update so far, avoid degenerate mean/sum over a single row
            self.avg = self.metrics[0]  # single metric vector is trivially its own average
            self.sum = self.metrics[0]  # single metric vector is trivially its own sum
        self.acc = self.avg[0]  # running-average accuracy (index 0 of the metric vector)
        self.auc = self.avg[1]  # running-average ROC-AUC (index 1)
        self.prc = self.avg[2]  # running-average PR-AUC (index 2)
        self.f1 = self.avg[3]  # running-average F1 score (index 3)
        self.mcc = self.avg[4]  # running-average Matthews correlation coefficient (index 4)
        self.tp = int(self.sum[5])  # cumulative true positives across all updates
        self.tn = int(self.sum[6])  # cumulative true negatives across all updates
        self.fp = int(self.sum[7])  # cumulative false positives across all updates
        self.fn = int(self.sum[8])  # cumulative false negatives across all updates
        if len(self.avg) > 9:  # extra scalars (e.g. loss) were appended via other_lst
            self.other = self.avg[9:]  # expose the running average of those extra scalars


def pearsonr(label, prediction):
    """
    Compute Pearson correlation(s) between labels and predictions.

    Args:
        label (np.ndarray):
            Ground-truth values. Shape (N,) or (N, K).
        prediction (np.ndarray):
            Predicted values. Same shape as `label`.

    Returns:
        list[float]:
            - If input is 1D: a single-element list containing the Pearson correlation coefficient.
            - If input is 2D: a list of length K containing per-column Pearson correlations.

    Notes:
        - For 1D input, this function currently returns `[stats.pearsonr(...)]` (a tuple),
          while for 2D it returns only the coefficient (float). This is preserved as-is.
          If you want strict consistency, convert the 1D case to `stats.pearsonr(...)[0]`.
    """
    ndim = np.ndim(label)  # determine whether label is 1D (single target) or 2D (multi-label)
    if ndim == 1:  # single-column case
        corr = [stats.pearsonr(label, prediction)]  # wrap the (r, p-value) tuple in a list, per the documented asymmetry
    else:  # multi-label case, one column per target
        num_labels = label.shape[1]  # number of target columns
        corr = []  # accumulates per-column Pearson r
        for i in range(num_labels):  # loop over each label column independently
            # corr.append(np.corrcoef(label[:,i], prediction[:,i]))
            corr.append(stats.pearsonr(label[:, i], prediction[:, i])[0])  # keep only the correlation coefficient, drop the p-value

    return corr  # list of Pearson correlation(s), one per column (or wrapped tuple for 1D)


def rsquare(label, prediction):
    """
    Compute an R^2-like metric and slope for a simple linear fit y ≈ m * x (no intercept).

    For each target dimension, this fits:
        m = (x · y) / (x · x)
    and reports:
        R^2 = 1 - ||y - m x||^2 / ||y - mean(y)||^2

    Args:
        label (np.ndarray):
            Ground-truth values. Shape (N,) or (N, K).
        prediction (np.ndarray):
            Predicted values. Same shape as `label`.

    Returns:
        Tuple[list[float], list[float]]:
            metric:
                List of R^2 values (length 1 for 1D input, else length K).
            slope:
                List of slopes m (same length as metric).

    Notes:
        - This is not the standard sklearn R^2 with intercept; it forces the regression through origin.
    """
    ndim = np.ndim(label)  # 1D (single target) vs 2D (multi-label) input
    if ndim == 1:  # single-column case
        y = label  # ground truth values
        X = prediction  # predicted values used as the regressor
        m = np.dot(X, y) / np.dot(X, X)  # least-squares slope for a no-intercept fit y ≈ m*X
        resid = y - m * X;  # residuals of the fitted line
        ym = y - np.mean(y);  # labels centered around their mean (for the R^2 denominator)
        rsqr2 = 1 - np.dot(resid.T, resid) / np.dot(ym.T, ym);  # 1 - (residual sum of squares / total sum of squares)
        metric = [rsqr2]  # wrap the single R^2 value in a list for a uniform return type
        slope = [m]  # wrap the single slope value in a list for a uniform return type
    else:  # multi-label case, one column per target
        num_labels = label.shape[1]  # number of target columns
        metric = []  # accumulates per-column R^2 values
        slope = []  # accumulates per-column slopes
        for i in range(num_labels):  # fit the no-intercept regression independently per column
            y = label[:, i]  # ground truth for this column
            X = prediction[:, i]  # predictions for this column
            m = np.dot(X, y) / np.dot(X, X)  # least-squares slope for this column's no-intercept fit
            resid = y - m * X;  # residuals for this column
            ym = y - np.mean(y);  # centered labels for this column
            rsqr2 = 1 - np.dot(resid.T, resid) / np.dot(ym.T, ym);  # R^2-like score for this column
            metric.append(rsqr2)  # store this column's R^2
            slope.append(m)  # store this column's slope
    return metric, slope  # (R^2 values, slopes), each a list aligned with label columns


def f1_sc(label, prediction):
    """
    Compute F1 score(s) using a 0.5 threshold via np.round(prediction).

    Args:
        label (np.ndarray):
            Binary labels. Shape (N,) or (N, K).
        prediction (np.ndarray):
            Predicted probabilities/scores. Same shape as label.

    Returns:
        np.ndarray:
            - Scalar array for 1D input.
            - Shape (K,) array for 2D input.

    Notes:
        - Uses `np.round`, i.e., threshold at 0.5 with bankers rounding rules for exact .5 values.
    """
    ndim = np.ndim(label)  # 1D vs 2D input
    if ndim == 1:  # single-column case
        metric = np.array(f1_score(label, np.round(prediction)))  # threshold predictions at 0.5 (via rounding) and score F1
    else:  # multi-label case
        num_labels = label.shape[1]  # number of target columns
        metric = np.zeros((num_labels))  # preallocate per-column F1 scores
        for i in range(num_labels):  # score each column independently
            metric[i] = f1_score(label[:, i], np.round(prediction[:, i]))  # F1 for this column's thresholded predictions
    return metric  # scalar (1D input) or per-column array (2D input) of F1 scores


def mcc_sc(label, prediction):
    """
    Compute Matthews correlation coefficient (MCC) using np.round(prediction) as the classifier.

    Args:
        label (np.ndarray):
            Binary labels. Shape (N,) or (N, K).
        prediction (np.ndarray):
            Predicted probabilities/scores. Same shape as label.

    Returns:
        np.ndarray:
            - Scalar array for 1D input.
            - Shape (K,) array for 2D input.
    """
    ndim = np.ndim(label)  # 1D vs 2D input
    if ndim == 1:  # single-column case
        metric = np.array(matthews_corrcoef(label, np.round(prediction)))  # threshold at 0.5 (via rounding) and compute MCC
    else:  # multi-label case
        num_labels = label.shape[1]  # number of target columns
        metric = np.zeros((num_labels))  # preallocate per-column MCC scores
        for i in range(num_labels):  # score each column independently
            metric[i] = matthews_corrcoef(label[:, i], np.round(prediction[:, i]))  # MCC for this column's thresholded predictions
    return metric  # scalar (1D input) or per-column array (2D input) of MCC scores


def accuracy(label, prediction):
    """
    Compute accuracy using np.round(prediction) as the classifier.

    Args:
        label (np.ndarray):
            Binary labels. Shape (N,) or (N, K).
        prediction (np.ndarray):
            Predicted probabilities/scores. Same shape as label.

    Returns:
        np.ndarray:
            - Scalar array for 1D input.
            - Shape (K,) array for 2D input.
    """
    ndim = np.ndim(label)  # 1D vs 2D input
    if ndim == 1:  # single-column case
        metric = np.array(accuracy_score(label, np.round(prediction)))  # threshold at 0.5 (via rounding) and compute accuracy
    else:  # multi-label case
        num_labels = label.shape[1]  # number of target columns
        metric = np.zeros((num_labels))  # preallocate per-column accuracy scores
        for i in range(num_labels):  # score each column independently
            metric[i] = accuracy_score(label[:, i], np.round(prediction[:, i]))  # accuracy for this column's thresholded predictions
    return metric  # scalar (1D input) or per-column array (2D input) of accuracy scores


def roc(label, prediction):
    """
    Compute ROC-AUC and ROC curves.

    Args:
        label (np.ndarray):
            Binary labels. Shape (N,) or (N, K).
        prediction (np.ndarray):
            Predicted scores/probabilities. Same shape as label.

    Returns:
        Tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
            metric:
                ROC-AUC value(s). Scalar array for 1D, or shape (K,) for 2D.
            curves:
                List of (fpr, tpr) arrays, one per label dimension.

    Notes:
        - Uses sklearn.metrics.roc_curve and auc.
        - For multi-label (2D), ROC is computed independently per label column.
    """
    ndim = np.ndim(label)  # 1D vs 2D input
    if ndim == 1:  # single-column case
        fpr, tpr, thresholds = roc_curve(label, prediction)  # false/true positive rates swept across score thresholds
        score = auc(fpr, tpr)  # area under the ROC curve
        metric = np.array(score)  # wrap as an array for a consistent return type with the 2D branch
        curves = [(fpr, tpr)]  # single-element list of (fpr, tpr) curve points
    else:  # multi-label case
        num_labels = label.shape[1]  # number of target columns
        curves = []  # accumulates per-column (fpr, tpr) curves
        metric = np.zeros((num_labels))  # preallocate per-column ROC-AUC values
        for i in range(num_labels):  # compute ROC independently per column
            fpr, tpr, thresholds = roc_curve(label[:, i], prediction[:, i])  # curve points for this column
            score = auc(fpr, tpr)  # ROC-AUC for this column
            metric[i] = score  # store this column's AUC
            curves.append((fpr, tpr))  # store this column's curve points
    return metric, curves  # (AUC value(s), curve points) for downstream plotting/aggregation


def pr(label, prediction):
    """
    Compute PR-AUC and precision-recall curves.

    Args:
        label (np.ndarray):
            Binary labels. Shape (N,) or (N, K).
        prediction (np.ndarray):
            Predicted scores/probabilities. Same shape as label.

    Returns:
        Tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
            metric:
                PR-AUC value(s), computed as AUC(recall, precision).
            curves:
                List of (precision, recall) arrays, one per label dimension.
    """
    ndim = np.ndim(label)  # 1D vs 2D input
    if ndim == 1:  # single-column case
        precision, recall, thresholds = precision_recall_curve(label, prediction)  # precision/recall swept across score thresholds
        score = auc(recall, precision)  # area under the precision-recall curve (x=recall, y=precision)
        metric = np.array(score)  # wrap as an array for a consistent return type with the 2D branch
        curves = [(precision, recall)]  # single-element list of (precision, recall) curve points
    else:  # multi-label case
        num_labels = label.shape[1]  # number of target columns
        curves = []  # accumulates per-column (precision, recall) curves
        metric = np.zeros((num_labels))  # preallocate per-column PR-AUC values
        for i in range(num_labels):  # compute PR independently per column
            precision, recall, thresholds = precision_recall_curve(label[:, i], prediction[:, i])  # curve points for this column
            score = auc(recall, precision)  # PR-AUC for this column
            metric[i] = score  # store this column's AUC
            curves.append((precision, recall))  # store this column's curve points
    return metric, curves  # (PR-AUC value(s), curve points) for downstream plotting/aggregation


def tfnp(label, prediction):
    """
    Compute confusion-matrix counts (TP, TN, FP, FN) for binary classification.

    Args:
        label (np.ndarray):
            Ground-truth binary labels, shape (N,).
        prediction (np.ndarray or list[bool/int]):
            Predicted binary class labels, shape (N,).

    Returns:
        Tuple[int, int, int, int]:
            (tp, tn, fp, fn). If confusion_matrix fails, returns zeros.

    Notes:
        - This function calls sklearn.metrics.confusion_matrix(label, prediction).ravel().
        - Any exception triggers a fallback (0,0,0,0).
    """
    try:
        tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()  # sklearn's 2x2 confusion matrix, flattened in row-major order
    except Exception:  # e.g. a single class present in this batch, confusion_matrix shape degenerates
        tp, tn, fp, fn = 0, 0, 0, 0  # fall back to all-zero counts rather than raising

    return tp, tn, fp, fn  # confusion-matrix counts for this batch/column


def calculate_metrics(label, prediction, objective):
    """
    Unified metric computation for different learning objectives.

    Depending on objective, this function computes a set of metrics and returns:
        mean: list of aggregated metrics (nanmean over label dimensions where applicable)
        std:  list of metric standard deviations (nanstd over label dimensions)

    Args:
        label (np.ndarray):
            Ground-truth labels/targets.
            - binary/hinge: shape (N,) or (N,1) or (N,K) for multi-label.
            - categorical: shape (N,K), typically one-hot.
            - squared_error/kl_divergence/cdf: numeric targets; internally thresholded at 0.5.
        prediction (np.ndarray):
            Model outputs.
            - binary/hinge: probabilities/scores in [0,1], same shape as label.
            - categorical: probabilities/logits post-processed to probabilities, shape (N,K).
            - squared_error/kl_divergence/cdf: numeric predictions aligned to label.
        objective (str):
            One of:
              - "binary" or "hinge"
              - "categorical"
              - "squared_error", "kl_divergence", or "cdf"
            Other values return (0, 0).

    Returns:
        Tuple[list[float], list[float]]:
            mean, std:
                For objective == "binary" or "hinge":
                    mean = [acc, auc_roc, auc_pr, f1, mcc, tp, tn, fp, fn]
                    std  = [acc_std, auc_roc_std, auc_pr_std, f1_std, mcc_std]

                For objective == "categorical":
                    mean starts as [acc, auc_roc, auc_pr] and then appends per-class ROC-AUC:
                        mean = [acc, auc_roc_macro, auc_pr_macro, auc_roc_class0, ..., auc_roc_class(K-1)]
                    std similarly starts as [acc_std, auc_roc_std, auc_pr_std] and appends per-class std.

                For objective in {"squared_error","kl_divergence","cdf"}:
                    The labels are thresholded into {0,1} before classification metrics.
                    mean = [acc, auc_roc, auc_pr, tp, tn, fp, fn, pearsonr_mean, rsquare_mean, slope_mean]
                    std  = [acc_std, auc_roc_std, auc_pr_std, pearsonr_std, rsquare_std, slope_std]

    Notes:
        - For binary/hinge and regression-like objectives, confusion counts are computed using:
              pred_class = prediction > 0.5
          while accuracy/F1/MCC use np.round(prediction).
        - If label is 2D with shape (N,1), the function flattens to 1D before confusion counts.
        - Multi-label (2D) metrics are computed per column and aggregated with nanmean/nanstd.
    """
    if (objective == "binary") | (objective == 'hinge'):  # binary / multi-label binary classification path
        ndim = np.ndim(label)  # remember whether label is 1D or 2D before it gets reduced below
        correct = accuracy(label, prediction)  # per-column (or scalar) accuracy
        auc_roc, roc_curves = roc(label, prediction)  # per-column (or scalar) ROC-AUC
        auc_pr, pr_curves = pr(label, prediction)  # per-column (or scalar) PR-AUC
        f1 = f1_sc(label, prediction)  # per-column (or scalar) F1 score
        mcc = mcc_sc(label, prediction)  # per-column (or scalar) MCC
        if ndim == 2:  # multi-label input: confusion counts are only computed for the first column
            prediction = prediction[:, 0]  # reduce to the first target column
            label = label[:, 0]  # reduce to the first target column
        pred_class = prediction > 0.5  # hard binary decision at the 0.5 threshold (independent of np.round used elsewhere)
        tp, tn, fp, fn = tfnp(label, pred_class)  # confusion-matrix counts for the (first) column
        mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr), np.nanmean(f1), np.nanmean(mcc), tp, tn, fp, fn]  # aggregate mean metrics across columns, then append raw confusion counts
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr), np.nanstd(f1), np.nanstd(mcc)]  # aggregate std across columns for the scalar metrics only

    elif objective == "categorical":  # multi-class, one-hot labels / class-probability predictions

        correct = np.mean(np.equal(np.argmax(label, axis=1), np.argmax(prediction, axis=1)))  # fraction of samples where the predicted argmax class matches the true class
        auc_roc, roc_curves = roc(label, prediction)  # per-class ROC-AUC (one column per class)
        auc_pr, pr_curves = pr(label, prediction)  # per-class PR-AUC (one column per class)
        mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr)]  # macro-averaged accuracy/ROC-AUC/PR-AUC
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr)]  # corresponding macro std values
        for i in range(label.shape[1]):  # then append each class's own ROC-AUC individually
            label_c, prediction_c = label[:, i], prediction[:, i]  # this class's one-hot column and predicted probability
            auc_roc, roc_curves = roc(label_c, prediction_c)  # recompute ROC restricted to this single class
            mean.append(np.nanmean(auc_roc))  # append per-class AUC to the mean list
            std.append(np.nanstd(auc_roc))  # append per-class AUC std to the std list


    elif (objective == 'squared_error') | (objective == 'kl_divergence') | (objective == 'cdf'):  # regression-like objectives, evaluated via a binarized label
        ndim = np.ndim(label)  # remember whether label is 1D or 2D before it gets reduced below
        label[label < 0.5] = 0  # threshold the (continuous) label into a binary class...
        label[label >= 0.5] = 1  # ...so classification metrics below are well-defined

        correct = accuracy(label, prediction)  # per-column (or scalar) accuracy against the thresholded label
        auc_roc, roc_curves = roc(label, prediction)  # per-column (or scalar) ROC-AUC against the thresholded label
        auc_pr, pr_curves = pr(label, prediction)  # per-column (or scalar) PR-AUC against the thresholded label
        if ndim == 2:  # multi-label input: confusion counts only computed for the first column
            prediction = prediction[:, 0]  # reduce to the first target column
            label = label[:, 0]  # reduce to the first target column
        pred_class = prediction > 0.5  # hard binary decision at the 0.5 threshold
        tp, tn, fp, fn = tfnp(label, pred_class)  # confusion-matrix counts for the (first) column

        # squared_error
        corr = pearsonr(label, prediction)  # Pearson correlation between (binarized) label and raw prediction
        rsqr, slope = rsquare(label, prediction)  # no-intercept R^2 and slope between (binarized) label and raw prediction

        mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr), tp, tn, fp, fn, np.nanmean(corr),
                np.nanmean(rsqr), np.nanmean(slope)]  # aggregate classification metrics, confusion counts, then regression-fit metrics
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr), np.nanstd(corr), np.nanstd(rsqr),
               np.nanstd(slope)]  # corresponding std values (confusion counts have no std since they're scalars)

    else:  # unrecognized objective string
        mean = 0  # sentinel value signaling "no metrics computed"
        std = 0  # sentinel value signaling "no metrics computed"

    return [mean, std]  # [mean metric vector, std metric vector], per the objective-specific layout documented above