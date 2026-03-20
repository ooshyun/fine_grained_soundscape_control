from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def compute_accuracy(
    predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5
) -> float:
    """Multi-label exact-match accuracy.

    Args:
        predictions: Predicted scores, shape ``(N, C)`` or ``(N,)``.
        targets: Binary ground-truth labels, same shape as *predictions*.
        threshold: Decision threshold for binarising scores.

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
    pred_binary = (predictions >= threshold).astype(int)
    target_binary = targets.astype(int)

    if predictions.ndim == 1:
        return float(accuracy_score(target_binary, pred_binary))

    # Multi-label exact match
    exact_match = np.all(pred_binary == target_binary, axis=1)
    return float(np.mean(exact_match))


def compute_precision(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    average: str = "macro",
) -> float:
    """Precision score.

    Args:
        predictions: Predicted scores, shape ``(N, C)`` or ``(N,)``.
        targets: Binary ground-truth labels.
        threshold: Decision threshold.
        average: Averaging strategy (``"macro"``, ``"micro"``, ``"weighted"``).

    Returns:
        Precision as a float in ``[0, 1]``.
    """
    pred_binary = (predictions >= threshold).astype(int)
    return float(
        precision_score(targets, pred_binary, average=average, zero_division=0)
    )


def compute_recall(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    average: str = "macro",
) -> float:
    """Recall score.

    Args:
        predictions: Predicted scores, shape ``(N, C)`` or ``(N,)``.
        targets: Binary ground-truth labels.
        threshold: Decision threshold.
        average: Averaging strategy (``"macro"``, ``"micro"``, ``"weighted"``).

    Returns:
        Recall as a float in ``[0, 1]``.
    """
    pred_binary = (predictions >= threshold).astype(int)
    return float(
        recall_score(targets, pred_binary, average=average, zero_division=0)
    )


def compute_f1(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    average: str = "macro",
) -> float:
    """F1 score.

    Args:
        predictions: Predicted scores, shape ``(N, C)`` or ``(N,)``.
        targets: Binary ground-truth labels.
        threshold: Decision threshold.
        average: Averaging strategy (``"macro"``, ``"micro"``, ``"weighted"``).

    Returns:
        F1 as a float in ``[0, 1]``.
    """
    pred_binary = (predictions >= threshold).astype(int)
    return float(
        f1_score(targets, pred_binary, average=average, zero_division=0)
    )


def compute_map(
    predictions: np.ndarray,
    targets: np.ndarray,
    average: str = "macro",
) -> float:
    """Mean Average Precision (area under the PR curve).

    Args:
        predictions: Predicted scores, shape ``(N, C)``.
        targets: Binary ground-truth labels, shape ``(N, C)``.
        average: Averaging strategy (``"macro"``, ``"micro"``, ``"weighted"``).

    Returns:
        mAP as a float in ``[0, 1]``.
    """
    return float(average_precision_score(targets, predictions, average=average))


def compute_auc_roc(
    predictions: np.ndarray,
    targets: np.ndarray,
    average: str = "macro",
) -> float:
    """Area Under the ROC Curve.

    Args:
        predictions: Predicted scores, shape ``(N, C)`` or ``(N,)``.
        targets: Binary ground-truth labels.
        average: Averaging strategy (``"macro"``, ``"micro"``, ``"weighted"``).

    Returns:
        AUC-ROC as a float in ``[0, 1]``.
    """
    if predictions.ndim == 1:
        return float(roc_auc_score(targets, predictions))
    return float(roc_auc_score(targets, predictions, average=average))


def compute_dprime(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Sensitivity index d-prime.

    ``d' = Z(hit_rate) - Z(false_alarm_rate)`` where *Z* is the inverse
    of the standard normal CDF.  Rates are clipped to ``[0.01, 0.99]`` to
    avoid infinite values.

    For multi-label inputs the metric is computed on the flattened arrays
    (all classes pooled together).

    Args:
        predictions: Predicted scores, shape ``(N, C)`` or ``(N,)``.
        targets: Binary ground-truth labels.
        threshold: Decision threshold.

    Returns:
        d-prime value (higher is better, typically 0--4).
    """
    pred_binary = (predictions >= threshold).astype(int).ravel()
    target_binary = targets.astype(int).ravel()

    tp = int(np.sum((pred_binary == 1) & (target_binary == 1)))
    fn = int(np.sum((pred_binary == 0) & (target_binary == 1)))
    fp = int(np.sum((pred_binary == 1) & (target_binary == 0)))
    tn = int(np.sum((pred_binary == 0) & (target_binary == 0)))

    hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Clip to [0.01, 0.99] to avoid inf from ppf
    hit_rate = float(np.clip(hit_rate, 0.01, 0.99))
    false_alarm_rate = float(np.clip(false_alarm_rate, 0.01, 0.99))

    return float(norm.ppf(hit_rate) - norm.ppf(false_alarm_rate))


# ---------------------------------------------------------------------------
# Aggregate class
# ---------------------------------------------------------------------------

_METRIC_FN = {
    "accuracy": compute_accuracy,
    "precision": compute_precision,
    "recall": compute_recall,
    "f1": compute_f1,
    "map": compute_map,
    "auc_roc": compute_auc_roc,
    "d_prime": compute_dprime,
}

# Metrics that accept an ``average`` kwarg
_AVERAGE_METRICS = {"precision", "recall", "f1", "map", "auc_roc"}

# Metrics that accept a ``threshold`` kwarg
_THRESHOLD_METRICS = {"accuracy", "precision", "recall", "f1", "d_prime"}


class ClassificationMetrics:
    """Unified interface for SED / classification metrics."""

    def compute_all(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.5,
        class_names: Optional[list[str]] = None,
        per_label: bool = False,
    ) -> dict[str, float]:
        """Compute all SED metrics.

        Args:
            predictions: Score array, shape ``(N, C)``.
            targets: Binary label array, shape ``(N, C)``.
            threshold: Decision threshold for binarisation.
            class_names: Optional readable names for each class.
            per_label: If *True*, also return per-class metrics keyed as
                ``"{class_name}_{metric}"``.

        Returns:
            Dictionary with keys ``accuracy``, ``precision``, ``recall``,
            ``f1``, ``map``, ``auc_roc``, ``d_prime`` (and per-class keys
            when *per_label* is *True*).
        """
        results: dict[str, float] = {}

        for name, fn in _METRIC_FN.items():
            kwargs: dict = {}
            if name in _THRESHOLD_METRICS:
                kwargs["threshold"] = threshold
            if name in _AVERAGE_METRICS:
                kwargs["average"] = "macro"
            try:
                results[name] = fn(predictions, targets, **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to compute %s: %s", name, exc)

        if per_label and predictions.ndim == 2:
            num_classes = predictions.shape[1]
            for ci in range(num_classes):
                cname = (
                    class_names[ci]
                    if class_names and ci < len(class_names)
                    else f"label_{ci}"
                )
                p = predictions[:, ci]
                t = targets[:, ci]
                for name, fn in _METRIC_FN.items():
                    kwargs = {}
                    if name in _THRESHOLD_METRICS:
                        kwargs["threshold"] = threshold
                    # Per-class metrics use binary averaging
                    if name in _AVERAGE_METRICS:
                        kwargs["average"] = "binary"
                    try:
                        results[f"{cname}_{name}"] = fn(p, t, **kwargs)
                    except Exception:  # noqa: BLE001
                        pass

        return results
