from __future__ import annotations

import logging
import sys
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import tqdm
from lightning.fabric import Fabric

from .base import TrainerBackend

logger = logging.getLogger(__name__)


def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module | Callable,
    optimizer: optim.Optimizer,
    fabric: Fabric,
    grad_clip: float | None = None,
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0
    num_elements = 0
    num_batches = len(train_loader)

    if fabric.global_rank == 0:
        pbar = tqdm.tqdm(total=num_batches, desc="train", file=sys.stdout)

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        fabric.backward(loss)

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = targets.size(0) if hasattr(targets, "size") else 1
        total_loss += loss.item() * batch_size
        num_elements += batch_size

        if fabric.global_rank == 0:
            pbar.set_postfix(loss=f"{loss.item():.5f}")
            pbar.update()

    if fabric.global_rank == 0:
        pbar.close()

    # Aggregate across devices
    fabric.barrier()
    gathered_loss = fabric.all_gather(torch.tensor(total_loss, device=fabric.device))
    gathered_n = fabric.all_gather(torch.tensor(num_elements, device=fabric.device, dtype=torch.float))
    avg_loss = gathered_loss.sum().item() / max(gathered_n.sum().item(), 1)
    return avg_loss


def _validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module | Callable,
    fabric: Fabric,
    metrics_fn: Callable | None = None,
) -> tuple[float, dict[str, Any]]:
    """Run one validation epoch.  Returns ``(avg_loss, metrics_dict)``."""
    model.eval()
    total_loss = 0.0
    num_elements = 0
    num_batches = len(val_loader)
    all_metrics: dict[str, float] = {}

    if fabric.global_rank == 0:
        pbar = tqdm.tqdm(total=num_batches, desc="val", file=sys.stdout)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(val_loader):
            inputs, targets = batch
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            batch_size = targets.size(0) if hasattr(targets, "size") else 1
            total_loss += loss.item() * batch_size
            num_elements += batch_size

            if metrics_fn is not None:
                batch_metrics = metrics_fn(outputs, targets)
                for key, value in batch_metrics.items():
                    val = value.item() if isinstance(value, torch.Tensor) else float(value)
                    all_metrics[key] = all_metrics.get(key, 0.0) + val * batch_size

            if fabric.global_rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.5f}")
                pbar.update()

    if fabric.global_rank == 0:
        pbar.close()

    # Aggregate across devices
    fabric.barrier()
    gathered_loss = fabric.all_gather(torch.tensor(total_loss, device=fabric.device))
    gathered_n = fabric.all_gather(torch.tensor(num_elements, device=fabric.device, dtype=torch.float))
    total_n = max(gathered_n.sum().item(), 1)
    avg_loss = gathered_loss.sum().item() / total_n

    avg_metrics: dict[str, Any] = {}
    for key, value in all_metrics.items():
        gathered_v = fabric.all_gather(torch.tensor(value, device=fabric.device))
        avg_metrics[key] = gathered_v.sum().item() / total_n

    return avg_loss, avg_metrics


class FabricTrainerBackend(TrainerBackend):
    """Lightning Fabric manual-loop backend.

    This mirrors the training pattern used in the parent project's
    ``src/train.py`` + ``src/training/train_val.py``.
    """

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
        precision: str = config.get("precision", "32-true")
        grad_clip: float | None = config.get("grad_clip", None)

        # -- Checkpointing config -----------------------------------------
        ckpt_cfg = config.get("checkpointing", {})
        monitor = ckpt_cfg.get("monitor", "val_loss")
        mode = ckpt_cfg.get("mode", "min")
        save_dir = ckpt_cfg.get("dirpath", "checkpoints")

        # -- Early stopping ------------------------------------------------
        es_cfg = config.get("early_stopping", {})
        patience = es_cfg.get("patience", 0)

        # -- Build Fabric --------------------------------------------------
        fabric = Fabric(
            accelerator="auto",
            devices="auto",
            precision=precision,
            **self._extra_kwargs,
        )
        fabric.launch()

        model, optimizer = fabric.setup(model, optimizer)
        train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

        # -- Training loop -------------------------------------------------
        best_metric: float | None = None
        best_epoch: int = -1
        best_path: str = ""
        patience_counter: int = 0

        import os
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(max_epochs):
            if fabric.global_rank == 0:
                logger.info(f"Epoch {epoch + 1}/{max_epochs}")

            train_loss = _train_one_epoch(
                model, train_loader, loss_fn, optimizer, fabric, grad_clip
            )
            val_loss, val_metrics = _validate_one_epoch(
                model, val_loader, loss_fn, fabric, metrics_fn
            )

            # Determine the metric to track
            if monitor == "val_loss":
                current_metric = val_loss
            else:
                metric_key = monitor.replace("val_", "")
                current_metric = val_metrics.get(metric_key, val_loss)

            # Step scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_metric)
                else:
                    scheduler.step()

            if fabric.global_rank == 0:
                logger.info(
                    f"  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
                    f"metrics={val_metrics}"
                )

            # -- Best tracking & checkpointing -----------------------------
            is_better = False
            if best_metric is None:
                is_better = True
            elif mode == "min" and current_metric < best_metric:
                is_better = True
            elif mode == "max" and current_metric > best_metric:
                is_better = True

            if is_better:
                best_metric = current_metric
                best_epoch = epoch
                patience_counter = 0
                best_path = os.path.join(
                    save_dir, f"best-epoch={epoch:02d}-val_loss={val_loss:.4f}.pt"
                )
                if fabric.global_rank == 0:
                    # Unwrap DDP / FSDP wrapper if present
                    raw_model = model.module if hasattr(model, "module") else model
                    self.save_checkpoint(
                        raw_model,
                        optimizer,
                        epoch,
                        {"val_loss": val_loss, **val_metrics},
                        best_path,
                    )
                    logger.info(f"  Saved best checkpoint: {best_path}")
            else:
                patience_counter += 1

            # -- Early stopping --------------------------------------------
            if patience > 0 and patience_counter >= patience:
                if fabric.global_rank == 0:
                    logger.info(
                        f"Early stopping triggered after {patience} epochs "
                        f"without improvement."
                    )
                break

        return {
            "best_epoch": best_epoch,
            "best_metric": best_metric,
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
        precision: str = config.get("precision", "32-true")
        fabric = Fabric(
            accelerator="auto",
            devices="auto",
            precision=precision,
            **self._extra_kwargs,
        )
        fabric.launch()

        model = fabric.setup(model)
        val_loader = fabric.setup_dataloaders(val_loader)

        val_loss, val_metrics = _validate_one_epoch(
            model, val_loader, loss_fn, fabric, metrics_fn
        )
        return {"loss": val_loss, **val_metrics}

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
