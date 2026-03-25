from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics")
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

"""SED evaluation entry-point.

Supports MisophoniaDataset and SoundscapeDataset,
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

import transformers
transformers.logging.set_verbosity_error()

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

# Dataset class → AudioSet class name mapping (from Classes.yaml)
_DATASET_TO_AUDIOSET = {
    "alarm_clock": "Alarm clock",
    "baby_cry": "Baby cry, infant cry",
    "birds_chirping": "Chirp, tweet",
    "car_horn": "Vehicle horn, car horn, honking",
    "cat": "Meow",
    "cock_a_doodle_doo": "Crowing, cock-a-doodle-doo",
    "computer_typing": "Computer keyboard",
    "cricket": "Cricket",
    "dog": "Bark",
    "door_knock": "Knock",
    "glass_breaking": "Shatter",
    "gunshot": "Gunshot, gunfire",
    "hammer": "Hammer",
    "music": "Melody",
    "ocean": "Waves, surf",
    "singing": "Singing",
    "siren": "Siren",
    "speech": "Speech",
    "thunderstorm": "Thunderstorm",
    "toilet_flush": "Toilet flush",
}


def _load_base_ast(device: str = "cuda"):
    """Load base AST (527-class, AudioSet pretrained).

    Returns:
        model: ASTForAudioClassification wrapper
        label_filter: list of AudioSet indices corresponding to our 20 classes
        class_names_527: full list of 527 AudioSet class names
    """
    from transformers import AutoFeatureExtractor, ASTForAudioClassification

    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    ast_model = ASTForAudioClassification.from_pretrained(model_name)
    ast_model.to(device)
    ast_model.eval()

    # Get 527 AudioSet class names
    class_names_527 = list(ast_model.config.id2label.values())

    # Build label_filter: map our 20 classes to AudioSet indices
    label_filter = []
    mapped_names = []
    for dataset_class in CLASS_NAMES:
        audioset_name = _DATASET_TO_AUDIOSET.get(dataset_class)
        if audioset_name and audioset_name in class_names_527:
            idx = class_names_527.index(audioset_name)
            label_filter.append(idx)
            mapped_names.append(dataset_class)
        else:
            logger.warning("Class %s → %s not found in AudioSet", dataset_class, audioset_name)

    logger.info("Label filter: %d/%d classes mapped to AudioSet indices", len(label_filter), len(CLASS_NAMES))
    for i, (dc, af) in enumerate(zip(mapped_names, label_filter)):
        logger.info("  %s → AudioSet[%d] (%s)", dc, af, class_names_527[af])

    return ast_model, feature_extractor, label_filter, class_names_527


def _run_base_ast_inference(
    model, feature_extractor, label_filter, dataloader, device,
):
    """Run inference with base 527-class AST + softmax + label filter.

    Pipeline:
    1. feature_extractor → AST → softmax(527)
    2. scores[:, label_filter] → 20 classes
    3. max pooling over frames (no-op for clip-level AST)
    """
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_inputs, _ in dataloader:
            mixtures = batch_inputs["mixture"].to(device)
            labels = batch_inputs["labels"]

            # Mono downmix
            if mixtures.ndim == 3:
                mixtures = mixtures.mean(dim=1)

            # Normalize
            peak = mixtures.abs().max()
            if peak > 1.0:
                mixtures = mixtures / (peak + 1e-6)

            # Feature extraction (expects numpy)
            waveform_np = mixtures.cpu().numpy()
            inputs = feature_extractor(
                waveform_np, sampling_rate=16000, return_tensors="pt"
            )
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else torch.tensor(v).to(device)
                for k, v in inputs.items()
            }

            # AST forward → softmax (527 classes)
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=-1)  # (B, 527)

            # Filter to 20 classes
            filtered = scores[:, label_filter].cpu().numpy()  # (B, 20)

            # Max pooling (no-op for clip-level)
            filtered = np.max(filtered, axis=0, keepdims=True)

            all_predictions.append(filtered)
            all_targets.append(labels.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return predictions, targets


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


def _find_optimal_thresholds(
    predictions: np.ndarray,
    targets: np.ndarray,
    method: str = "f1_max",
) -> dict:
    """Find per-class optimal thresholds from predictions.

    Args:
        predictions: (N, C) score array.
        targets: (N, C) binary label array.
        method: ``"f1_max"`` (maximize F1 per class on PR curve).

    Returns:
        Dict with per-class thresholds keyed by class name, plus ``"overall"``.
    """
    from sklearn.metrics import precision_recall_curve

    num_classes = predictions.shape[1]
    thresholds_dict: dict[str, float] = {}

    # Per-class
    for c in range(num_classes):
        precision, recall, thresholds = precision_recall_curve(
            targets[:, c], predictions[:, c]
        )
        f1_scores = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-10)
        )
        best_idx = int(np.argmax(f1_scores))
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"label_{c}"
        thresholds_dict[name] = float(thresholds[best_idx])

    # Overall (flattened)
    precision, recall, thresholds = precision_recall_curve(
        targets.ravel(), predictions.ravel()
    )
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )
    best_idx = int(np.argmax(f1_scores))
    thresholds_dict["overall"] = float(thresholds[best_idx])

    return thresholds_dict


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
    parser.add_argument(
        "--base_ast", action="store_true",
        help="Use base 527-class AST with softmax + label filter (paper eval)",
    )

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
    parser.add_argument(
        "--find_thresholds", action="store_true",
        help="Find optimal per-class thresholds on val set, then evaluate test set",
    )
    parser.add_argument(
        "--val_samples", type=int, default=None,
        help="Number of val samples for threshold search (default: same as --samples)",
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
    logging.getLogger("root").setLevel(logging.WARNING)

    config = None
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ---- Load model ----
    use_base_ast = args.base_ast
    base_ast_components = None

    if use_base_ast:
        logger.info("=== Using base 527-class AST (paper eval) ===")
        ast_model, feature_extractor, label_filter, class_names_527 = _load_base_ast(device)
        total_params = sum(p.numel() for p in ast_model.parameters())
        base_ast_components = (ast_model, feature_extractor, label_filter)
    else:
        model = _load_model(args, config)
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %d parameters", total_params)

    # ---- Helper: build dataset for a given split ----
    def _build_split_dataset(split):
        if args.dataset == "misophonia":
            if split == "test":
                return _build_misophonia_dataset(args, config or {}), _collate_misophonia
            else:
                # val split
                from src.datasets.MisophoniaDataset import MisophoniaDataset
                fg_max = args.num_fg_max if args.num_fg_max is not None else 5
                root = args.root_dataset_dir or "/scr"
                sr = args.sr or 16000
                duration = args.duration or 5
                samples = args.val_samples or args.samples or 2000
                ds = MisophoniaDataset(
                    fg_sounds_dir=f"BinauralCuratedDataset/scaper_fmt/{split}",
                    bg_sounds_dir=f"BinauralCuratedDataset/bg_scaper_fmt/{split}",
                    noise_sounds_dir=f"BinauralCuratedDataset/noise_scaper_fmt/{split}",
                    hrtf_list=f"BinauralCuratedDataset/hrtf/CIPIC/{split}_hrtf.txt",
                    split=split, sr=sr, duration=duration, hrtf_type="CIPIC",
                    num_total_labels=20,
                    num_fg_sounds_min=(args.num_fg_min if args.num_fg_min is not None else 1),
                    num_fg_sounds_max=fg_max,
                    num_bg_sounds_min=(args.num_bg_min if args.num_bg_min is not None else 1),
                    num_bg_sounds_max=(args.num_bg_max if args.num_bg_max is not None else 3),
                    num_noise_sounds_min=(args.num_noise_min if args.num_noise_min is not None else 1),
                    num_noise_sounds_max=(args.num_noise_max if args.num_noise_max is not None else 1),
                    num_output_channels=max(fg_max, 5),
                    snr_range_fg=[5, 15], snr_range_bg=[0, 10], ref_db=-50,
                    augmentations=[], samples_per_epoch=samples,
                    onflight_mode=1, root_dataset_dir=root,
                )
                return ds, _collate_misophonia
        else:
            if config is None:
                raise ValueError("--config required for soundscape dataset")
            return _build_soundscape_dataset(config), _collate_soundscape

    # ---- Helper: run inference ----
    def _run_inference(dataloader, split_name="test"):
        if use_base_ast:
            ast_m, feat_ext, lf = base_ast_components
            return _run_base_ast_inference(ast_m, feat_ext, lf, dataloader, device)
        else:
            all_preds, all_tgts = [], []
            logger.info("Running inference on %s set...", split_name)
            with torch.no_grad():
                for batch_inputs, _ in dataloader:
                    mixtures = batch_inputs["mixture"].to(device)
                    labels = batch_inputs["labels"]
                    outputs = model({"mixture": mixtures})
                    # Use softmax scores from model (not sigmoid on logits)
                    preds = outputs["scores"].cpu().numpy()
                    all_preds.append(preds)
                    all_tgts.append(labels.numpy())
            return np.concatenate(all_preds, axis=0), np.concatenate(all_tgts, axis=0)

    # ---- Find optimal thresholds on val set (if requested) ----
    found_thresholds = None
    if args.find_thresholds:
        logger.info("=== Finding optimal thresholds on validation set ===")
        val_ds, val_collate = _build_split_dataset("val")
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=val_collate, pin_memory=True,
        )
        val_predictions, val_targets = _run_inference(val_loader, "val")
        found_thresholds = _find_optimal_thresholds(val_predictions, val_targets)
        logger.info("Optimal thresholds found:")
        for name, t in found_thresholds.items():
            if name != "overall":
                logger.info("  %s: %.4f", name, t)
        logger.info("  overall: %.4f", found_thresholds["overall"])

    # ---- Build test dataset ----
    test_ds, collate_fn = _build_split_dataset("test")
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # ---- Run inference on test set ----
    predictions, targets = _run_inference(test_loader, "test")
    logger.info(
        "Inference complete: predictions %s, targets %s",
        predictions.shape, targets.shape,
    )

    # ---- Compute metrics ----
    if found_thresholds is not None:
        # Use thresholds found on val set
        thresholds = np.array([found_thresholds[n] for n in CLASS_NAMES])
        logger.info("Using thresholds found on val set")
        results = _compute_metrics_with_thresholds(predictions, targets, thresholds)
    elif args.thresholds:
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

    # Save found thresholds if applicable
    if found_thresholds is not None:
        thresh_path = output_dir / "optimal_thresholds.json"
        with open(thresh_path, "w") as f:
            json.dump(found_thresholds, f, indent=2)
        logger.info("Saved optimal thresholds to %s", thresh_path)


if __name__ == "__main__":
    main()
