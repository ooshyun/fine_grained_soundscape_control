from __future__ import annotations

"""SED evaluation entry-point.

Supports MisophoniaDataset (original) and SoundscapeDataset (new),
with optional per-class optimal thresholds for paper reproduction.

Usage examples::

    # Paper reproduction with optimal thresholds
    python -m src.sed.eval \\
        --pretrained ooshyun/sound_event_detection --model finetuned_ast \\
        --dataset misophonia --root_dataset_dir /scr \\
        --num_fg_min 1 --num_fg_max 1 \\
        --num_bg_min 1 --num_bg_max 1 \\
        --num_noise_min 1 --num_noise_max 1 \\
        --sr 16000 --duration 5 --samples 2000 \\
        --thresholds configs/sed/optimal_thresholds.json \\
        --output_dir eval_results/paper/tgt1_bg1-1
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.metrics.sed import ClassificationMetrics
from src.sed.model import ASTModel, load_pretrained

logger = logging.getLogger(__name__)

# 20 classes in alphabetical order (matching MisophoniaDataset)
CLASS_NAMES = [
    "alarm_clock", "baby_cry", "birds_chirping", "car_horn", "cat",
    "cock_a_doodle_doo", "computer_typing", "cricket", "dog", "door_knock",
    "glass_breaking", "gunshot", "hammer", "music", "ocean",
    "singing", "siren", "speech", "thunderstorm", "toilet_flush",
]


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def _build_misophonia_dataset(args, config):
    """Build a MisophoniaDataset from CLI args."""
    from src.datasets.MisophoniaDataset import MisophoniaDataset

    run_config = {}
    config_path = Path(args.run_dir) / "config.json" if args.run_dir else None
    if config_path and config_path.exists():
        with open(config_path, "r") as f:
            run_config = json.load(f)

    test_args = run_config.get("onflight_test_data_args", {})
    root = args.root_dataset_dir or run_config.get("root_dataset_dir", "/scr")

    sr = args.sr or test_args.get("sr", 16000)
    duration = args.duration or test_args.get("duration", 5)
    samples = args.samples or test_args.get("samples_per_epoch", 2000)

    fg_max = (
        args.num_fg_max
        if args.num_fg_max is not None
        else test_args.get("num_fg_sounds_max", 5)
    )

    return MisophoniaDataset(
        fg_sounds_dir=test_args.get(
            "fg_sounds_dir", "BinauralCuratedDataset/scaper_fmt/test"
        ),
        bg_sounds_dir=test_args.get(
            "bg_sounds_dir", "BinauralCuratedDataset/bg_scaper_fmt/test"
        ),
        noise_sounds_dir=test_args.get(
            "noise_sounds_dir", "BinauralCuratedDataset/noise_scaper_fmt/test"
        ),
        hrtf_list=test_args.get(
            "hrtf_list", "BinauralCuratedDataset/hrtf/CIPIC/test_hrtf.txt"
        ),
        split="test",
        sr=sr,
        duration=duration,
        hrtf_type=test_args.get("hrtf_type", "CIPIC"),
        num_total_labels=test_args.get("num_total_labels", 20),
        num_fg_sounds_min=(
            args.num_fg_min
            if args.num_fg_min is not None
            else test_args.get("num_fg_sounds_min", 1)
        ),
        num_fg_sounds_max=fg_max,
        num_bg_sounds_min=(
            args.num_bg_min
            if args.num_bg_min is not None
            else test_args.get("num_bg_sounds_min", 1)
        ),
        num_bg_sounds_max=(
            args.num_bg_max
            if args.num_bg_max is not None
            else test_args.get("num_bg_sounds_max", 3)
        ),
        num_noise_sounds_min=(
            args.num_noise_min
            if args.num_noise_min is not None
            else test_args.get("num_noise_sounds_min", 1)
        ),
        num_noise_sounds_max=(
            args.num_noise_max
            if args.num_noise_max is not None
            else test_args.get("num_noise_sounds_max", 1)
        ),
        num_output_channels=max(fg_max, test_args.get("num_output_channels", 5)),
        snr_range_fg=test_args.get("snr_range_fg", [5, 15]),
        snr_range_bg=test_args.get("snr_range_bg", [0, 10]),
        ref_db=test_args.get("ref_db", -50),
        augmentations=[],
        samples_per_epoch=samples,
        onflight_mode=1,
        root_dataset_dir=root,
    )


def _build_soundscape_dataset(config):
    """Build a SoundscapeDataset from YAML config."""
    from src.datasets.soundscape_dataset import SoundscapeDataset

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


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def _collate_misophonia(batch):
    inputs_list, targets_list = zip(*batch)
    mixtures = torch.stack([inp["mixture"] for inp in inputs_list])
    labels = torch.stack([inp["label_vector"] for inp in inputs_list])
    return {"mixture": mixtures, "labels": labels}, targets_list


def _collate_soundscape(batch):
    inputs_list, targets_list = zip(*batch)
    mixtures = torch.stack([inp["mixture"] for inp in inputs_list])
    labels = torch.stack([inp["labels"] for inp in inputs_list])
    return {"mixture": mixtures, "labels": labels}, targets_list


# ---------------------------------------------------------------------------
# Threshold loading
# ---------------------------------------------------------------------------


def _load_thresholds(path: str, num_classes: int = 20) -> np.ndarray:
    """Load per-class thresholds from JSON file.

    Returns array of shape (num_classes,) with per-class thresholds.
    """
    with open(path, "r") as f:
        data = json.load(f)

    thresholds = np.full(num_classes, 0.5)  # default

    for i, name in enumerate(CLASS_NAMES):
        if name in data:
            thresholds[i] = data[name]
        elif f"label_{i}" in data:
            thresholds[i] = data[f"label_{i}"]

    return thresholds


def _apply_per_class_threshold(
    predictions: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """Binarize predictions using per-class thresholds.

    Args:
        predictions: (N, C) score array
        thresholds: (C,) threshold array

    Returns:
        (N, C) binary array
    """
    return (predictions >= thresholds[None, :]).astype(int)


def _compute_metrics_with_thresholds(
    predictions: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    """Compute paper metrics (Accuracy, Precision, Recall, F1) with per-class thresholds."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred_binary = _apply_per_class_threshold(predictions, thresholds)
    target_binary = targets.astype(int)

    n_samples, n_classes = predictions.shape
    results = {}

    # Per-class accuracy (fraction of samples correctly classified per class)
    per_class_acc = []
    for c in range(n_classes):
        acc = accuracy_score(target_binary[:, c], pred_binary[:, c])
        per_class_acc.append(acc)
        results[f"{CLASS_NAMES[c]}_accuracy"] = acc

    results["accuracy"] = float(np.mean(per_class_acc))

    # Macro-averaged precision, recall, F1
    results["precision"] = float(
        precision_score(target_binary, pred_binary, average="macro", zero_division=0)
    )
    results["recall"] = float(
        recall_score(target_binary, pred_binary, average="macro", zero_division=0)
    )
    results["f1"] = float(
        f1_score(target_binary, pred_binary, average="macro", zero_division=0)
    )

    # mAP and AUC-ROC (threshold-independent)
    try:
        results["map"] = float(
            average_precision_score(target_binary, predictions, average="macro")
        )
    except Exception:
        pass
    try:
        results["auc_roc"] = float(
            roc_auc_score(target_binary, predictions, average="macro")
        )
    except Exception:
        pass

    # Per-class precision, recall, F1
    for c in range(n_classes):
        p = precision_score(target_binary[:, c], pred_binary[:, c], zero_division=0)
        r = recall_score(target_binary[:, c], pred_binary[:, c], zero_division=0)
        f = f1_score(target_binary[:, c], pred_binary[:, c], zero_division=0)
        results[f"{CLASS_NAMES[c]}_precision"] = float(p)
        results[f"{CLASS_NAMES[c]}_recall"] = float(r)
        results[f"{CLASS_NAMES[c]}_f1"] = float(f)

    return results


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(args, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.pretrained:
        model = load_pretrained(
            repo_id=args.pretrained,
            model_name=args.model or "finetuned_ast",
            device=device,
        )
    elif args.checkpoint:
        model_cfg = config["model"] if config else {}
        model = ASTModel(
            model_name=model_cfg.get(
                "name", "MIT/ast-finetuned-audioset-10-10-0.4593"
            ),
            num_labels=model_cfg.get("num_classes", 20),
        )
        state_dict = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        logger.info("Loaded checkpoint: %s", args.checkpoint)
    else:
        raise ValueError("Provide --checkpoint or --pretrained")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate SED model")

    # Model source
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)

    # Dataset
    parser.add_argument(
        "--dataset", type=str, choices=["misophonia", "soundscape"],
        default="misophonia",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--root_dataset_dir", type=str, default=None)
    parser.add_argument("--sr", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--samples", type=int, default=None)

    # Eval condition overrides
    parser.add_argument("--num_fg_min", type=int, default=None)
    parser.add_argument("--num_fg_max", type=int, default=None)
    parser.add_argument("--num_bg_min", type=int, default=None)
    parser.add_argument("--num_bg_max", type=int, default=None)
    parser.add_argument("--num_noise_min", type=int, default=None)
    parser.add_argument("--num_noise_max", type=int, default=None)

    # Threshold
    parser.add_argument(
        "--thresholds", type=str, default=None,
        help="Path to per-class optimal thresholds JSON",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Global threshold (used if --thresholds not set)",
    )

    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = None
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ---- Load model ----
    model = _load_model(args, config)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %d parameters", total_params)

    # ---- Build dataset ----
    if args.dataset == "misophonia":
        test_ds = _build_misophonia_dataset(args, config or {})
        collate_fn = _collate_misophonia
    else:
        if config is None:
            raise ValueError("--config required for soundscape dataset")
        test_ds = _build_soundscape_dataset(config)
        collate_fn = _collate_soundscape

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ---- Run inference ----
    all_predictions = []
    all_targets = []

    logger.info("Running inference on %d samples...", len(test_ds))
    with torch.no_grad():
        for batch_inputs, _ in test_loader:
            mixtures = batch_inputs["mixture"].to(device)
            labels = batch_inputs["labels"]

            outputs = model({"mixture": mixtures})
            logits = outputs["output"]
            preds = torch.sigmoid(logits).cpu().numpy()

            all_predictions.append(preds)
            all_targets.append(labels.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    logger.info(
        "Inference complete: predictions %s, targets %s",
        predictions.shape, targets.shape,
    )

    # ---- Compute metrics ----
    if args.thresholds:
        # Per-class optimal thresholds (paper reproduction)
        thresholds = _load_thresholds(args.thresholds, predictions.shape[1])
        logger.info("Using per-class thresholds from %s", args.thresholds)
        logger.info("  Thresholds: %s", dict(zip(CLASS_NAMES, thresholds)))
        results = _compute_metrics_with_thresholds(predictions, targets, thresholds)
    else:
        # Fixed threshold
        eval_cfg = config.get("evaluation", {}) if config else {}
        threshold = args.threshold or eval_cfg.get("threshold", 0.5)
        logger.info("Using fixed threshold: %.4f", threshold)
        metrics_calculator = ClassificationMetrics()
        results = metrics_calculator.compute_all(
            predictions, targets, threshold=threshold, per_label=True,
            class_names=CLASS_NAMES,
        )

    # Print results (paper format: Accuracy, Precision, Recall, F1)
    logger.info("=== Evaluation Results ===")
    for key in ("accuracy", "precision", "recall", "f1", "map", "auc_roc"):
        if key in results:
            logger.info("  %s: %.4f", key, results[key])

    # ---- Save results ----
    output_dir = Path(
        args.output_dir
        or (config.get("checkpointing", {}).get("save_dir", "runs/sed") if config else "eval_results/sed")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    serialisable = {k: float(v) for k, v in results.items()}
    with open(metrics_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    predictions_path = output_dir / "raw_predictions.npz"
    np.savez(predictions_path, predictions=predictions, targets=targets)
    logger.info("Saved raw predictions to %s", predictions_path)


if __name__ == "__main__":
    main()
