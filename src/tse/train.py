from __future__ import annotations

"""TSE training entry point.

Usage::

    python -m src.tse.train --config configs/tse/orange_pi.yaml [--data_dir ./BinauralCuratedDataset]
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.soundscape_dataset import SoundscapeDataset
from src.metrics.tse import compute_tse_metrics
from src.tse.loss import MultiResoFuseLoss
from src.tse.model import TFGridNet
from src.trainer import create_trainer

logger = logging.getLogger(__name__)


def _build_datasets(cfg: dict, data_dir: str | None = None):
    """Create train and validation SoundscapeDataset instances."""
    d = cfg["data"]

    fg_dir = d["fg_dir"]
    noise_dir = d["noise_dir"]
    hrtf_list = d["hrtf_list"]

    if data_dir is not None:
        # Re-root relative paths under data_dir
        if not Path(fg_dir).is_absolute():
            fg_dir = str(Path(data_dir) / Path(fg_dir).relative_to(Path(fg_dir).parts[0])
                         if len(Path(fg_dir).parts) > 1 else Path(data_dir) / fg_dir)
        if not Path(noise_dir).is_absolute():
            noise_dir = str(Path(data_dir) / Path(noise_dir).relative_to(Path(noise_dir).parts[0])
                           if len(Path(noise_dir).parts) > 1 else Path(data_dir) / noise_dir)
        if not Path(hrtf_list).is_absolute():
            hrtf_list = str(Path(data_dir) / Path(hrtf_list).relative_to(Path(hrtf_list).parts[0])
                           if len(Path(hrtf_list).parts) > 1 else Path(data_dir) / hrtf_list)

    common = dict(
        fg_dir=fg_dir,
        noise_dir=noise_dir,
        hrtf_list=hrtf_list,
        sr=d.get("sr", 16000),
        duration=d.get("duration", 5),
        num_fg_range=tuple(d.get("num_fg_range", [1, 1])),
        num_bg_range=tuple(d.get("num_bg_range", [0, 0])),
        num_noise_range=tuple(d.get("num_noise_range", [1, 1])),
        snr_range_fg=tuple(d.get("snr_range_fg", [5, 15])),
        snr_range_bg=tuple(d.get("snr_range_bg", [0, 10])),
        hrtf_type=d.get("hrtf_type", "CIPIC"),
        samples_per_epoch=d.get("samples_per_epoch", 20000),
        task="tse",
    )

    train_ds = SoundscapeDataset(split="train", **common)
    val_ds = SoundscapeDataset(split="val", **common)
    return train_ds, val_ds


def _build_model(cfg: dict) -> TFGridNet:
    m = cfg["model"]
    return TFGridNet(
        stft_chunk_size=m.get("stft_chunk_size", 96),
        stft_pad_size=m.get("stft_pad_size", 64),
        stft_back_pad=m.get("stft_back_pad", 96),
        num_input_channels=m.get("num_input_channels", 2),
        num_output_channels=m.get("num_output_channels", 1),
        num_layers=m.get("num_layers", 6),
        latent_dim=m.get("latent_dim", 32),
        hidden_channels=m.get("hidden_channels", 64),
        speaker_dim=m.get("speaker_dim", 20),
        bidirectional=m.get("bidirectional", False),
        film_preset=m.get("film_preset", "all_except_first"),
        use_first_ln=m.get("use_first_ln", False),
        embedding_type=m.get("embedding_type", "embedding"),
        embedding_dim=m.get("embedding_dim", 0),
        embedding_activation=m.get("embedding_activation", ""),
        embedding_init=m.get("embedding_init", ""),
    )


def _build_loss(cfg: dict) -> MultiResoFuseLoss:
    lc = cfg.get("loss", {})
    return MultiResoFuseLoss(l1_ratio=lc.get("l1_ratio", 10))


def _metrics_fn(
    est: torch.Tensor, gt: torch.Tensor, mix: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return compute_tse_metrics(est, gt, mix, metrics=("si_sdri", "snri"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train TSE model")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data root")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Data
    train_ds, val_ds = _build_datasets(cfg, data_dir=args.data_dir)
    d = cfg["data"]
    train_loader = DataLoader(
        train_ds,
        batch_size=d.get("batch_size", 8),
        shuffle=True,
        num_workers=d.get("num_workers", 8),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=d.get("batch_size", 8),
        shuffle=False,
        num_workers=d.get("num_workers", 8),
        pin_memory=True,
    )

    # Model
    model = _build_model(cfg)
    logger.info(
        "Model parameters: %.2fM",
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    # Optimizer & scheduler
    tc = cfg.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc.get("lr", 1e-3),
        weight_decay=tc.get("weight_decay", 0.0),
    )
    sp = tc.get("scheduler_params", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=sp.get("factor", 0.5),
        patience=sp.get("patience", 5),
    )

    # Loss
    loss_fn = _build_loss(cfg)

    # Trainer
    trainer = create_trainer(tc.get("backend", "lightning"))
    trainer.fit(
        model, train_loader, val_loader, loss_fn, optimizer, scheduler, cfg, _metrics_fn,
    )


if __name__ == "__main__":
    main()
