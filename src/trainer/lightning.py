from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from .base import TrainerBackend

logger = logging.getLogger(__name__)


class _LitWrapper(pl.LightningModule):
    """Thin Lightning wrapper around an arbitrary :class:`nn.Module`."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module | Callable,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        metrics_fn: Callable | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.metrics_fn = metrics_fn

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        if self.metrics_fn is not None:
            metrics = self.metrics_fn(outputs, targets)
            for key, value in metrics.items():
                self.log(f"val_{key}", value, prog_bar=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    def configure_optimizers(self) -> dict[str, Any]:
        config: dict[str, Any] = {"optimizer": self._optimizer}
        if self._scheduler is not None:
            config["lr_scheduler"] = {
                "scheduler": self._scheduler,
                "monitor": "val_loss",
            }
        return config


class LightningTrainerBackend(TrainerBackend):
    """PyTorch Lightning :class:`pl.Trainer` backend."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._extra_kwargs = kwargs

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
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
        max_epochs: int = config.get("max_epochs", 100)

        # -- Checkpointing config -----------------------------------------
        ckpt_cfg = config.get("checkpointing", {})
        monitor = ckpt_cfg.get("monitor", "val_loss")
        mode = ckpt_cfg.get("mode", "min")
        save_dir = ckpt_cfg.get("dirpath", "checkpoints")

        checkpoint_cb = ModelCheckpoint(
            dirpath=save_dir,
            monitor=monitor,
            mode=mode,
            save_top_k=ckpt_cfg.get("save_top_k", 1),
            filename="best-{epoch:02d}-{val_loss:.4f}",
        )

        callbacks: list[pl.Callback] = [checkpoint_cb, LearningRateMonitor()]

        # -- Early stopping ------------------------------------------------
        es_cfg = config.get("early_stopping", {})
        patience = es_cfg.get("patience", 0)
        if patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor=monitor,
                    mode=mode,
                    patience=patience,
                    verbose=True,
                )
            )

        # -- Logger --------------------------------------------------------
        pl_logger = config.get("logger", True)

        # -- Build Trainer -------------------------------------------------
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices="auto",
            callbacks=callbacks,
            logger=pl_logger,
            **self._extra_kwargs,
        )

        lit_model = _LitWrapper(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics_fn=metrics_fn,
        )

        trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_path = checkpoint_cb.best_model_path
        best_score = checkpoint_cb.best_model_score
        best_epoch = -1
        if best_path:
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            best_epoch = ckpt.get("epoch", -1)

        return {
            "best_epoch": best_epoch,
            "best_metric": best_score.item() if best_score is not None else None,
            "checkpoint_path": best_path,
        }

    # ------------------------------------------------------------------
    # validate
    # ------------------------------------------------------------------
    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: nn.Module | Callable,
        config: dict[str, Any],
        metrics_fn: Callable | None = None,
    ) -> dict[str, Any]:
        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            **self._extra_kwargs,
        )

        lit_model = _LitWrapper(
            model=model,
            loss_fn=loss_fn,
            optimizer=torch.optim.Adam(model.parameters()),  # dummy, unused
            scheduler=None,
            metrics_fn=metrics_fn,
        )

        results = trainer.validate(lit_model, dataloaders=val_loader)
        if results:
            return results[0]
        return {"loss": float("inf")}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: dict[str, Any],
        path: str,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )

    def load_checkpoint(
        self,
        path: str,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return {
            "epoch": ckpt.get("epoch", 0),
            "metrics": ckpt.get("metrics", {}),
        }
