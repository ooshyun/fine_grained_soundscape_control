from __future__ import annotations

"""SED evaluation entry-point.

Usage::

    python -m src.sed.eval --checkpoint runs/sed/best.pt --config configs/sed/ast_finetune.yaml
    python -m src.sed.eval --pretrained ooshyun/sound_event_detection --model finetuned_ast
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.soundscape_dataset import SoundscapeDataset
from src.metrics.sed import ClassificationMetrics
from src.sed.model import ASTModel, load_pretrained

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_test_dataset(config: dict) -> SoundscapeDataset:
    """Instantiate a :class:`SoundscapeDataset` for test evaluation."""
    data_cfg = config["data"]
    return SoundscapeDataset(
        fg_dir=data_cfg["fg_dir"],
        noise_dir=data_cfg["noise_dir"],
        hrtf_list=data_cfg["hrtf_list"],
        split="test",
        sr=data_cfg.get("sr", 16000),
        duration=data_cfg.get("duration", 5),
        bg_dir=data_cfg.get("bg_dir"),
        num_fg_range=tuple(data_cfg.get("num_fg_range", [1, 5])),
        num_bg_range=tuple(data_cfg.get("num_bg_range", [1, 3])),
        num_noise_range=tuple(data_cfg.get("num_noise_range", [1, 1])),
        snr_range_fg=tuple(data_cfg.get("snr_range_fg", [5, 15])),
        snr_range_bg=tuple(data_cfg.get("snr_range_bg", [0, 10])),
        num_total_labels=config["model"].get("num_classes", 20),
        samples_per_epoch=data_cfg.get("samples_per_epoch", 20000),
        hrtf_type=data_cfg.get("hrtf_type", "CIPIC"),
        task="sed",
    )


def _collate_fn(batch):
    """Custom collate: stack mixture waveforms and label vectors."""
    inputs_list, targets_list = zip(*batch)

    mixtures = torch.stack([inp["mixture"] for inp in inputs_list])
    labels = torch.stack([inp["labels"] for inp in inputs_list])

    return {
        "mixture": mixtures,
        "labels": labels,
    }, targets_list


def _load_model(args: argparse.Namespace, config: dict) -> ASTModel:
    """Load model from checkpoint or HuggingFace Hub."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.pretrained:
        model = load_pretrained(
            repo_id=args.pretrained,
            model_name=args.model or "finetuned_ast",
            device=device,
        )
    elif args.checkpoint:
        model_cfg = config["model"]
        model = ASTModel(
            model_name=model_cfg.get(
                "name", "MIT/ast-finetuned-audioset-10-10-0.4593"
            ),
            num_classes=model_cfg.get("num_classes", 20),
            freeze_encoder=False,
            sample_rate=model_cfg.get("sample_rate", 16000),
        )
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        logger.info("Loaded checkpoint: %s", args.checkpoint)
    else:
        raise ValueError("Provide --checkpoint or --pretrained")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SED model")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Local checkpoint")
    parser.add_argument(
        "--pretrained", type=str, default=None, help="HuggingFace repo ID"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model name within HF repo"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Override data root")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.data_dir:
        config["data"]["fg_dir"] = str(Path(args.data_dir) / "scaper_fmt")
        config["data"]["noise_dir"] = str(
            Path(args.data_dir) / "noise_scaper_fmt"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = _load_model(args, config)
    model.eval()
    logger.info(
        "Model loaded: %d parameters", model.get_total_parameters()
    )

    # Test dataset
    test_ds = _build_test_dataset(config)
    data_cfg = config["data"]
    test_loader = DataLoader(
        test_ds,
        batch_size=data_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 8),
        collate_fn=_collate_fn,
        pin_memory=True,
    )

    # Run inference
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    logger.info("Running inference on %d samples...", len(test_ds))
    with torch.no_grad():
        for batch_inputs, _ in test_loader:
            mixtures = batch_inputs["mixture"].to(device)
            labels = batch_inputs["labels"]

            # Mono downmix for AST (expects single-channel)
            if mixtures.ndim == 3:
                mixtures = mixtures.mean(dim=1)

            logits = model(mixtures)
            preds = torch.sigmoid(logits).cpu().numpy()

            all_predictions.append(preds)
            all_targets.append(labels.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Compute metrics
    eval_cfg = config.get("evaluation", {})
    threshold = eval_cfg.get("threshold", 0.5)
    per_class = eval_cfg.get("per_class", True)

    metrics_calculator = ClassificationMetrics()
    results = metrics_calculator.compute_all(
        predictions, targets, threshold=threshold, per_label=per_class
    )

    # Print results
    logger.info("=== Evaluation Results ===")
    for key in ("map", "f1", "precision", "recall", "auc_roc", "d_prime", "accuracy"):
        if key in results:
            logger.info("  %s: %.4f", key, results[key])

    # Save results
    output_dir = Path(args.output_dir or config.get("checkpointing", {}).get("save_dir", "runs/sed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    # Convert numpy types for JSON serialisation
    serialisable = {k: float(v) for k, v in results.items()}
    with open(metrics_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    predictions_path = output_dir / "raw_predictions.npz"
    np.savez(predictions_path, predictions=predictions, targets=targets)
    logger.info("Saved raw predictions to %s", predictions_path)


if __name__ == "__main__":
    main()
