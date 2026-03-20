from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class TrainerBackend(ABC):
    """Abstract base class for trainer backends.

    All backends expose the same interface so they can be swapped
    transparently.
    """

    @abstractmethod
    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module | Callable,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        config: dict[str, Any],
        metrics_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """Train the model.

        Returns:
            A dict with at least ``best_epoch``, ``best_metric``, and
            ``checkpoint_path``.
        """

    @abstractmethod
    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: nn.Module | Callable,
        config: dict[str, Any],
        metrics_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """Run validation.

        Returns:
            A dict with ``loss`` and any additional metrics.
        """

    @abstractmethod
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: dict[str, Any],
        path: str,
    ) -> None:
        """Persist a checkpoint to *path*."""

    @abstractmethod
    def load_checkpoint(
        self,
        path: str,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        """Restore from *path*.

        Returns:
            A dict with ``epoch``, ``metrics``, and any extra state.
        """
