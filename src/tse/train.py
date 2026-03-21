from __future__ import annotations

"""TSE training entry point.

Usage::

    python -m src.tse.train --config configs/tse/orange_pi.yaml [--data_dir /path/to/BinauralCuratedDataset]
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.soundscape_dataset import SoundscapeDataset
from src.tse.loss import MultiResoFuseLoss
from src.tse.net import Net
from src.trainer import create_trainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_datasets(cfg: dict, data_dir: str | None = None):
    """Create train and validation SoundscapeDataset instances.

    The config paths (``fg_dir``, ``noise_dir``) are expected to be base
    directories *without* the split suffix.  This function appends
    ``/train`` or ``/val`` automatically.  Similarly, ``hrtf_list`` should
    point to the *train* list; the val list is derived by replacing
    ``train_hrtf`` with ``val_hrtf``.
    """
    d = cfg["data"]
    m = cfg.get("model", {})

    fg_dir_base = d["fg_dir"]
    noise_dir_base = d["noise_dir"]
    hrtf_list_base = d["hrtf_list"]

    # Re-root relative paths under --data_dir when given.
    if data_dir is not None:
        if not Path(fg_dir_base).is_absolute():
            fg_dir_base = str(Path(data_dir) / fg_dir_base)
        if not Path(noise_dir_base).is_absolute():
            noise_dir_base = str(Path(data_dir) / noise_dir_base)
        if not Path(hrtf_list_base).is_absolute():
            hrtf_list_base = str(Path(data_dir) / hrtf_list_base)

    common = dict(
        sr=d.get("sr", 16000),
        duration=d.get("duration", 5),
        num_fg_range=tuple(d.get("num_fg_range", [1, 1])),
        num_bg_range=tuple(d.get("num_bg_range", [0, 0])),
        num_noise_range=tuple(d.get("num_noise_range", [1, 1])),
        snr_range_fg=tuple(d.get("snr_range_fg", [5, 15])),
        snr_range_bg=tuple(d.get("snr_range_bg", [0, 10])),
        hrtf_type=d.get("hrtf_type", "CIPIC"),
        samples_per_epoch=d.get("samples_per_epoch", 20000),
        num_output_channels=m.get("num_output_channels", 1),
        task="tse",
    )

    train_ds = SoundscapeDataset(
        split="train",
        fg_dir=os.path.join(fg_dir_base, "train"),
        noise_dir=os.path.join(noise_dir_base, "train"),
        hrtf_list=hrtf_list_base,
        **common,
    )
    val_hrtf = hrtf_list_base.replace("train_hrtf", "val_hrtf")
    val_ds = SoundscapeDataset(
        split="val",
        fg_dir=os.path.join(fg_dir_base, "val"),
        noise_dir=os.path.join(noise_dir_base, "val"),
        hrtf_list=val_hrtf,
        **common,
    )
    return train_ds, val_ds


def _build_model(cfg: dict) -> Net:
    """Construct a :class:`Net` (TFGridNet) from the YAML config."""
    m = cfg["model"]
    return Net(
        model_name=m.get(
            "model_name",
            "src.tse.multiflim_guided_tfnet.MultiFiLMGuidedTFNet",
        ),
        block_model_name=m.get(
            "block_model_name",
            "src.tse.gridnet_block.GridNetBlock",
        ),
        block_model_params=m.get("block_model_params", {}),
        speaker_dim=m.get("speaker_dim", 20),
        stft_chunk_size=m.get("stft_chunk_size", 96),
        stft_pad_size=m.get("stft_pad_size", 64),
        stft_back_pad=m.get("stft_back_pad", 96),
        num_input_channels=m.get("num_input_channels", 2),
        num_output_channels=m.get("num_output_channels", 1),
        num_layers=m.get("num_layers", 6),
        latent_dim=m.get("latent_dim", 32),
        embedding_params=m.get("embedding_params", {}),
        film_params=m.get("film_params", {}),
        use_first_ln=m.get("use_first_ln", False),
    )


def _build_loss(cfg: dict) -> MultiResoFuseLoss:
    lc = cfg.get("loss", {})
    return MultiResoFuseLoss(l1_ratio=lc.get("l1_ratio", 10))


def _metrics_fn(
    est: torch.Tensor, gt: torch.Tensor, mix: torch.Tensor,
) -> dict[str, float]:
    """Lightweight training metrics (SI-SDRi and SNRi)."""
    from torchmetrics.functional import (
        scale_invariant_signal_distortion_ratio as si_sdr,
        signal_noise_ratio as snr_fn,
    )

    # est/gt: (B, C, T), mix: (B, M, T) — mono-downmix mix to match
    mix_mono = mix.mean(dim=1, keepdim=True).expand_as(gt)

    si_sdri = (si_sdr(est, gt) - si_sdr(mix_mono, gt)).mean()
    snri = (snr_fn(est, gt) - snr_fn(mix_mono, gt)).mean()

    return {"si_sdri": si_sdri, "snri": snri}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train TSE model")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data root")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

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
