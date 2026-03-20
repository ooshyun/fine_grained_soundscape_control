from __future__ import annotations

"""Loss functions for multi-label sound event detection."""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MultiLabelBCELoss(nn.Module):
    """Binary cross-entropy loss for multi-label classification.

    Supports per-class ``pos_weight`` to handle class imbalance.
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, reduction=reduction
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss.

        Args:
            logits: Raw model outputs, shape ``(B, C)``.
            targets: Multi-hot binary labels, shape ``(B, C)``.

        Returns:
            Scalar loss.
        """
        return self.loss_fn(logits, targets)


class FocalLoss(nn.Module):
    """Focal loss for addressing extreme class imbalance.

    ``FL = -alpha * (1 - p_t)^gamma * log(p_t)``
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model outputs, shape ``(B, C)``.
            targets: Multi-hot binary labels, shape ``(B, C)``.

        Returns:
            Scalar loss (or per-element if ``reduction='none'``).
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        focal_loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def compute_class_weights(
    dataset: Dataset,
    num_classes: Optional[int] = None,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """Compute positive-class weights for :class:`nn.BCEWithLogitsLoss`.

    ``pos_weight[c] = (neg[c] + smoothing) / (pos[c] + smoothing)``

    Args:
        dataset: Training dataset whose items yield ``(inputs, targets)``
            where ``inputs["labels"]`` is a multi-hot vector.
        num_classes: Number of classes (inferred from first sample if *None*).
        smoothing: Laplace smoothing factor.

    Returns:
        Tensor of shape ``(num_classes,)``.
    """
    logger.info("Computing class weights from dataset (%d samples)...", len(dataset))

    if num_classes is None:
        sample_inputs, _ = dataset[0]
        num_classes = sample_inputs["labels"].shape[0]

    label_counts = np.zeros(num_classes, dtype=np.float64)

    for idx in range(len(dataset)):
        inputs, _ = dataset[idx]
        labels = inputs["labels"]
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        label_counts += labels

    total_samples = len(dataset)
    neg_counts = total_samples - label_counts
    pos_weights = (neg_counts + smoothing) / (label_counts + smoothing)
    pos_weights = torch.FloatTensor(pos_weights)

    logger.info(
        "Class weight stats: min=%.4f, max=%.4f, mean=%.4f, median=%.4f",
        pos_weights.min(),
        pos_weights.max(),
        pos_weights.mean(),
        pos_weights.median(),
    )

    return pos_weights


def get_loss_function(
    config: dict,
    dataset: Optional[Dataset] = None,
    device: str = "cpu",
) -> nn.Module:
    """Create a loss function from *config*.

    Args:
        config: Full training config dict with ``loss.type`` and
            ``loss.pos_weight`` keys.
        dataset: Training dataset (used when ``pos_weight="auto"``).
        device: Device to place loss parameters on.

    Returns:
        A loss module.
    """
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "BCEWithLogitsLoss")

    if loss_type == "BCEWithLogitsLoss":
        pos_weight = None
        if loss_cfg.get("pos_weight") == "auto" and dataset is not None:
            pos_weight = compute_class_weights(
                dataset, num_classes=config.get("model", {}).get("num_classes")
            )
            pos_weight = pos_weight.to(device)

        loss_fn = MultiLabelBCELoss(pos_weight=pos_weight)
        logger.info("Created %s loss (pos_weight=%s)", loss_type,
                     "auto" if pos_weight is not None else "none")

    elif loss_type == "FocalLoss":
        alpha = loss_cfg.get("alpha", 0.25)
        gamma = loss_cfg.get("gamma", 2.0)
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        logger.info("Created FocalLoss (alpha=%.2f, gamma=%.2f)", alpha, gamma)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss_fn
