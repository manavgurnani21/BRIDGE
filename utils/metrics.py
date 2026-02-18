"""
Metrics utilities for model evaluation.

This module provides:
- Scalar metrics for binary / multi-label / regression-like objectives:
  accuracy, ROC-AUC, PR-AUC, F1, MCC, Pearson correlation, R^2
- Confusion-matrix derived counts (TP/TN/FP/FN)
- `calculate_metrics(...)` as a unified entry point producing:
    mean: list of summary metrics (order depends on objective)
    std:  list of metric standard deviations (order depends on objective)
- `MLMetrics` accumulator that stores per-batch metrics and exposes running averages.

Input conventions:
- For binary tasks:
    label: shape (N,) or (N, 1), values in {0,1}
    prediction: shape (N,) or (N, 1), values in [0,1] (probabilities/scores)
- For multi-label tasks:
    label: shape (N, K)
    prediction: shape (N, K)
- For categorical tasks:
    label/prediction: one-hot or probability matrices with shape (N, K)
- For regression-like objectives ('squared_error', 'kl_divergence', 'cdf'):
    label/prediction are converted into binary labels by thresholding label at 0.5 before classification metrics.

Important:
- Several functions round predictions via `np.round(prediction)` for classification decisions.
  Confusion-matrix counts in `calculate_metrics` use `prediction > 0.5` as the class threshold.
- The `mean` list layout is relied upon by `MLMetrics` (fixed indices for acc/auc/prc/f1/mcc and TP/TN/FP/FN).
  See `calculate_metrics` docstring for the exact ordering per objective.
"""

import os, sys
import numpy as np
from six.moves import cPickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import f1_score, matthews_corrcoef
from scipy import stats

__all__ = [
    "pearsonr",
    "rsquare",
    "accuracy",
    "roc",
    "pr",
    "calculate_metrics"
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
        self.objective = objective
        self.metrics = []

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
        met, _ = calculate_metrics(label, pred, self.objective)
        if len(other_lst) > 0:
            met.extend(other_lst)
        self.metrics.append(met)
        self.compute_avg()

    def compute_avg(self):
        """
        Recompute running averages and sums over stored metric vectors.

        Returns:
            None. Populates `avg`, `sum`, and convenience fields such as `acc`, `auc`, etc.
        """
        if len(self.metrics) > 1:
            self.avg = np.array(self.metrics).mean(axis=0)
            self.sum = np.array(self.metrics).sum(axis=0)
        else:
            self.avg = self.metrics[0]
            self.sum = self.metrics[0]
        self.acc = self.avg[0]
        self.auc = self.avg[1]
        self.prc = self.avg[2]
        self.f1 = self.avg[3]
        self.mcc = self.avg[4]
        self.tp = int(self.sum[5])
        self.tn = int(self.sum[6])
        self.fp = int(self.sum[7])
        self.fn = int(self.sum[8])
        if len(self.avg) > 9:
            self.other = self.avg[9:]


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
    ndim = np.ndim(label)
    if ndim == 1:
        corr = [stats.pearsonr(label, prediction)]
    else:
        num_labels = label.shape[1]
        corr = []
        for i in range(num_labels):
            # corr.append(np.corrcoef(label[:,i], prediction[:,i]))
            corr.append(stats.pearsonr(label[:, i], prediction[:, i])[0])

    return corr


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
    ndim = np.ndim(label)
    if ndim == 1:
        y = label
        X = prediction
        m = np.dot(X, y) / np.dot(X, X)
        resid = y - m * X;
        ym = y - np.mean(y);
        rsqr2 = 1 - np.dot(resid.T, resid) / np.dot(ym.T, ym);
        metric = [rsqr2]
        slope = [m]
    else:
        num_labels = label.shape[1]
        metric = []
        slope = []
        for i in range(num_labels):
            y = label[:, i]
            X = prediction[:, i]
            m = np.dot(X, y) / np.dot(X, X)
            resid = y - m * X;
            ym = y - np.mean(y);
            rsqr2 = 1 - np.dot(resid.T, resid) / np.dot(ym.T, ym);
            metric.append(rsqr2)
            slope.append(m)
    return metric, slope


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
    ndim = np.ndim(label)
    if ndim == 1:
        metric = np.array(f1_score(label, np.round(prediction)))
    else:
        num_labels = label.shape[1]
        metric = np.zeros((num_labels))
        for i in range(num_labels):
            metric[i] = f1_score(label[:, i], np.round(prediction[:, i]))
    return metric


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
    ndim = np.ndim(label)
    if ndim == 1:
        metric = np.array(matthews_corrcoef(label, np.round(prediction)))
    else:
        num_labels = label.shape[1]
        metric = np.zeros((num_labels))
        for i in range(num_labels):
            metric[i] = matthews_corrcoef(label[:, i], np.round(prediction[:, i]))
    return metric


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
    ndim = np.ndim(label)
    if ndim == 1:
        metric = np.array(accuracy_score(label, np.round(prediction)))
    else:
        num_labels = label.shape[1]
        metric = np.zeros((num_labels))
        for i in range(num_labels):
            metric[i] = accuracy_score(label[:, i], np.round(prediction[:, i]))
    return metric


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
    ndim = np.ndim(label)
    if ndim == 1:
        fpr, tpr, thresholds = roc_curve(label, prediction)
        score = auc(fpr, tpr)
        metric = np.array(score)
        curves = [(fpr, tpr)]
    else:
        num_labels = label.shape[1]
        curves = []
        metric = np.zeros((num_labels))
        for i in range(num_labels):
            fpr, tpr, thresholds = roc_curve(label[:, i], prediction[:, i])
            score = auc(fpr, tpr)
            metric[i] = score
            curves.append((fpr, tpr))
    return metric, curves


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
    ndim = np.ndim(label)
    if ndim == 1:
        precision, recall, thresholds = precision_recall_curve(label, prediction)
        score = auc(recall, precision)
        metric = np.array(score)
        curves = [(precision, recall)]
    else:
        num_labels = label.shape[1]
        curves = []
        metric = np.zeros((num_labels))
        for i in range(num_labels):
            precision, recall, thresholds = precision_recall_curve(label[:, i], prediction[:, i])
            score = auc(recall, precision)
            metric[i] = score
            curves.append((precision, recall))
    return metric, curves


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
        tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()
    except Exception:
        tp, tn, fp, fn = 0, 0, 0, 0

    return tp, tn, fp, fn


def calculate_metrics(label, prediction, objective):
    """
    Unified metric computation for different learning objectives.

    Depending on `objective`, this function computes a set of metrics and returns:
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
                    mean = [
                        acc, auc_roc, auc_pr, f1, mcc, tp, tn, fp, fn
                    ]
                    std  = [acc_std, auc_roc_std, auc_pr_std, f1_std, mcc_std]

                For objective == "categorical":
                    mean starts as [acc, auc_roc, auc_pr] and then appends per-class ROC-AUC:
                        mean = [acc, auc_roc_macro, auc_pr_macro, auc_roc_class0, ..., auc_roc_class(K-1)]
                    std similarly starts as [acc_std, auc_roc_std, auc_pr_std] and appends per-class std.

                For objective in {"squared_error","kl_divergence","cdf"}:
                    The labels are thresholded into {0,1} before classification metrics.
                    mean = [
                        acc, auc_roc, auc_pr, tp, tn, fp, fn,
                        pearsonr_mean, rsquare_mean, slope_mean
                    ]
                    std  = [
                        acc_std, auc_roc_std, auc_pr_std,
                        pearsonr_std, rsquare_std, slope_std
                    ]

    Notes:
        - For binary/hinge and regression-like objectives, confusion counts are computed using:
              pred_class = prediction > 0.5
          while accuracy/F1/MCC use np.round(prediction).
        - If label is 2D with shape (N,1), the function flattens to 1D before confusion counts.
        - Multi-label (2D) metrics are computed per column and aggregated with nanmean/nanstd.
    """
    if (objective == "binary") | (objective == 'hinge'):
        ndim = np.ndim(label)
        correct = accuracy(label, prediction)
        auc_roc, roc_curves = roc(label, prediction)
        auc_pr, pr_curves = pr(label, prediction)
        f1 = f1_sc(label, prediction)
        mcc = mcc_sc(label, prediction)
        if ndim == 2:
            prediction = prediction[:, 0]
            label = label[:, 0]
        pred_class = prediction > 0.5
        tp, tn, fp, fn = tfnp(label, pred_class)
        mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr), np.nanmean(f1), np.nanmean(mcc), tp, tn, fp, fn]
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr), np.nanstd(f1), np.nanstd(mcc)]

    elif objective == "categorical":

        correct = np.mean(np.equal(np.argmax(label, axis=1), np.argmax(prediction, axis=1)))
        auc_roc, roc_curves = roc(label, prediction)
        auc_pr, pr_curves = pr(label, prediction)
        mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr)]
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr)]
        for i in range(label.shape[1]):
            label_c, prediction_c = label[:, i], prediction[:, i]
            auc_roc, roc_curves = roc(label_c, prediction_c)
            mean.append(np.nanmean(auc_roc))
            std.append(np.nanstd(auc_roc))


    elif (objective == 'squared_error') | (objective == 'kl_divergence') | (objective == 'cdf'):
        ndim = np.ndim(label)
        label[label < 0.5] = 0
        label[label >= 0.5] = 1

        correct = accuracy(label, prediction)
        auc_roc, roc_curves = roc(label, prediction)
        auc_pr, pr_curves = pr(label, prediction)
        if ndim == 2:
            prediction = prediction[:, 0]
            label = label[:, 0]
        pred_class = prediction > 0.5
        tp, tn, fp, fn = tfnp(label, pred_class)

        # squared_error
        corr = pearsonr(label, prediction)
        rsqr, slope = rsquare(label, prediction)

        mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr), tp, tn, fp, fn, np.nanmean(corr),
                np.nanmean(rsqr), np.nanmean(slope)]
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr), np.nanstd(corr), np.nanstd(rsqr),
               np.nanstd(slope)]

    else:
        mean = 0
        std = 0

    return [mean, std]