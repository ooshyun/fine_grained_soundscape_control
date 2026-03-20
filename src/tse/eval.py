from __future__ import annotations

"""TSE evaluation entry point.

Usage::

    # From checkpoint:
    python -m src.tse.eval --checkpoint runs/tse/best.pt --config configs/tse/orange_pi.yaml

    # From HuggingFace:
    python -m src.tse.eval --pretrained ooshyun/semantic_listening --model orange_pi \\
        --config configs/tse/orange_pi.yaml
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.soundscape_dataset import SoundscapeDataset
from src.metrics.tse import compute_tse_metrics
from src.tse.model import TFGridNet, load_pretrained

logger = logging.getLogger(__name__)


def _build_test_dataset(cfg: dict, data_dir: str | None = None) -> SoundscapeDataset:
    d = cfg["data"]

    fg_dir = d["fg_dir"]
    noise_dir = d["noise_dir"]
    hrtf_list = d["hrtf_list"]

    if data_dir is not None:
        if not Path(fg_dir).is_absolute():
            fg_dir = str(Path(data_dir) / fg_dir)
        if not Path(noise_dir).is_absolute():
            noise_dir = str(Path(data_dir) / noise_dir)
        if not Path(hrtf_list).is_absolute():
            hrtf_list = str(Path(data_dir) / hrtf_list)

    return SoundscapeDataset(
        fg_dir=fg_dir,
        noise_dir=noise_dir,
        hrtf_list=hrtf_list,
        split="test",
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


def _load_model(args, cfg: dict) -> TFGridNet:
    """Load model from checkpoint or HuggingFace."""
    if args.pretrained:
        model = load_pretrained(repo_id=args.pretrained, model_name=args.model)
    elif args.checkpoint:
        m = cfg["model"]
        model = TFGridNet(
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
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    else:
        raise ValueError("Provide either --checkpoint or --pretrained")
    return model


@torch.inference_mode()
def evaluate(model: TFGridNet, loader: DataLoader, cfg: dict, device: torch.device):
    """Run inference and compute metrics over the dataset."""
    metric_names = tuple(cfg.get("evaluation", {}).get("metrics", ["si_sdri", "snri"]))
    model.to(device).eval()

    all_rows: list[dict[str, float]] = []

    for idx, (inputs, targets) in enumerate(loader):
        mixture = inputs["mixture"].to(device)
        embedding = inputs["embedding"].to(device)
        target = targets["target"].to(device)

        out = model({"mixture": mixture, "embedding": embedding})
        est = out["output"]  # [B, S, t]

        # Use first output source and first channel of mixture for metrics
        B = est.shape[0]
        for b in range(B):
            est_b = est[b, 0]             # [t]
            gt_b = target[b, 0]           # [t]
            mix_b = mixture[b, 0]         # [t]
            # Align lengths
            min_len = min(est_b.shape[-1], gt_b.shape[-1], mix_b.shape[-1])
            metrics = compute_tse_metrics(
                est_b[..., :min_len].unsqueeze(0),
                gt_b[..., :min_len].unsqueeze(0),
                mix_b[..., :min_len].unsqueeze(0),
                metrics=metric_names,
            )
            row = {k: v.item() for k, v in metrics.items()}
            row["sample_idx"] = idx * loader.batch_size + b
            all_rows.append(row)

        if (idx + 1) % 50 == 0:
            logger.info("Processed %d / %d batches", idx + 1, len(loader))

    return all_rows, metric_names


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate TSE model")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--pretrained", type=str, default=None, help="HuggingFace repo ID")
    parser.add_argument("--model", type=str, default="orange_pi", help="Model name for --pretrained")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data root")
    parser.add_argument("--output_dir", type=str, default="runs/tse/eval", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detected if omitted)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = _load_model(args, cfg)
    test_ds = _build_test_dataset(cfg, data_dir=args.data_dir)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["data"].get("batch_size", 8),
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 8),
        pin_memory=True,
    )

    logger.info("Evaluating on %d samples with device=%s", len(test_ds), device)
    rows, metric_names = evaluate(model, test_loader, cfg, device)

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_idx"] + list(metric_names))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved per-sample results to %s", csv_path)

    # Summary
    summary: dict[str, float] = {}
    for name in metric_names:
        vals = [r[name] for r in rows]
        summary[f"{name}_mean"] = sum(vals) / len(vals) if vals else 0.0
        summary[f"{name}_median"] = sorted(vals)[len(vals) // 2] if vals else 0.0
    summary["num_samples"] = len(rows)

    summary_path = out_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved metrics summary to %s", summary_path)

    for k, v in summary.items():
        logger.info("  %s: %.4f", k, v)


if __name__ == "__main__":
    main()
