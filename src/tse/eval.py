from __future__ import annotations

"""TSE evaluation entry point.

Reproduces the exact evaluation logic from the original Sementic-Listening-v2
``src/eval.py``, including:
- On-the-fly dataset from config (original MisophoniaDataset)
- Per-label inference for nO=1 models (one-hot embedding per active source)
- Direct multi-output for nO>1 models
- ``compute_metrics_tse`` with ``label_vector`` for active-channel matching

Usage::

    # From HuggingFace pretrained (recommended):
    python -m src.tse.eval --pretrained ooshyun/fine_grained_soundscape_control --model orange_pi

    # From local checkpoint:
    python -m src.tse.eval --run_dir /path/to/run_dir

    # Override data root:
    python -m src.tse.eval --pretrained ... --data_dir /scr
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers (from original src/metrics/metrics.py)
# ---------------------------------------------------------------------------

from src.metrics.tse import Metrics, compute_metrics_tse


# ---------------------------------------------------------------------------
# Model loading — mirrors original utils.load_pretrained
# ---------------------------------------------------------------------------

def _import_attr(path: str):
    """Dynamic import: 'a.b.c.Cls' -> getattr(import_module('a.b.c'), 'Cls')."""
    import importlib
    module_path, attr = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), attr)


def _load_model_from_run_dir(run_dir: str, use_last: bool = True):
    """Load model from a run directory (config.json + checkpoints/)."""
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path) as f:
        params = json.load(f)

    # Build HL module (contains model, optimizer, etc.)
    from src.tse.hl_module import PLModule

    pl_module = PLModule(
        fabric=None, **params["pl_module_args"]
    )

    # Load checkpoint
    name = "last.pt" if use_last else "best.pt"
    ckpt_path = os.path.join(run_dir, f"checkpoints/{name}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}"
        )
    pl_module.load_state(ckpt_path)
    logger.info("Loaded checkpoint %s (epoch %d)", ckpt_path, pl_module.epoch)

    return pl_module.model, params


def _load_model_from_pretrained(repo_id: str, model_name: str):
    """Load model from HuggingFace Hub."""
    from src.tse.model import load_pretrained
    model = load_pretrained(repo_id=repo_id, model_name=model_name)
    # Also need config for dataset args
    from huggingface_hub import hf_hub_download
    from src.tse.model import _MODEL_NAME_MAP
    run_dir_name = _MODEL_NAME_MAP.get(model_name, model_name)
    config_path = hf_hub_download(
        repo_id=repo_id, filename=f"{run_dir_name}/config.json",
    )
    with open(config_path) as f:
        params = json.load(f)
    return model, params


# ---------------------------------------------------------------------------
# Dataset loading — uses original MisophoniaDataset via config
# ---------------------------------------------------------------------------

def _build_dataset(params: dict, data_dir: str | None = None):
    """Build test dataset from config, matching original eval.py logic."""
    test_data_args = dict(params["onflight_test_data_args"])
    test_data_args["split"] = "test"

    if data_dir is not None:
        test_data_args["root_dataset_dir"] = data_dir
    elif "root_dataset_dir" in params:
        test_data_args["root_dataset_dir"] = params["root_dataset_dir"]

    dataset_cls_path = params.get("onflight_val_dataset",
                                   "src.datasets.MisophoniaDataset.MisophoniaDataset")
    try:
        dataset_cls = _import_attr(dataset_cls_path)
    except (ImportError, ModuleNotFoundError):
        # Fallback: try from new repo's soundscape dataset
        from src.datasets.soundscape_dataset import SoundscapeDataset
        logger.warning("Could not import %s, using SoundscapeDataset", dataset_cls_path)
        d = test_data_args
        return SoundscapeDataset(
            fg_dir=d.get("fg_sounds_dir", ""),
            noise_dir=d.get("noise_sounds_dir", ""),
            hrtf_list=d.get("hrtf_list", ""),
            split="test",
            sr=d.get("sr", 16000),
            duration=d.get("duration", 5),
            num_fg_range=(d.get("num_fg_sounds_min", 1), d.get("num_fg_sounds_max", 5)),
            num_bg_range=(d.get("num_bg_sounds_min", 1), d.get("num_bg_sounds_max", 3)),
            num_noise_range=(d.get("num_noise_sounds_min", 1), d.get("num_noise_sounds_max", 1)),
            snr_range_fg=tuple(d.get("snr_range_fg", [5, 15])),
            snr_range_bg=tuple(d.get("snr_range_bg", [0, 10])),
            hrtf_type=d.get("hrtf_type", "CIPIC"),
            samples_per_epoch=d.get("samples_per_epoch", 2000),
            task="tse",
        )

    return dataset_cls(**test_data_args)


# ---------------------------------------------------------------------------
# Evaluation loop — mirrors original src/eval.py lines 394-615
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, test_loader, params, device, output_dir):
    """Run evaluation matching the original eval.py logic exactly."""

    # Initialize metrics (same as original)
    metrics_func_names_list = [
        "snr_i",
        "snr_per_channel",
        "snr_per_channel_mix",
        "si_sdr",
        "si_sdr_i",
        "si_sdr_per_channel",
        "si_sdr_per_channel_mix",
    ]
    metrics_func_list = {name: Metrics(name) for name in metrics_func_names_list}

    model = model.to(device)
    model.eval()

    records = []
    metrics_avg = {}
    metrics_count = {}

    pbar = tqdm.tqdm(total=len(test_loader))

    for idx, (inputs, targets) in enumerate(test_loader):
        gt = targets["target"]
        sample_name = inputs.get("folder", idx)

        # Move tensors to device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        # Set up embedding from label_vector
        if "embedding" not in inputs or inputs.get("embedding") is None:
            if "label_vector" in inputs:
                lv = inputs["label_vector"]
                if isinstance(lv, np.ndarray):
                    lv = torch.from_numpy(lv).unsqueeze(0).to(device)
                inputs["embedding"] = lv
            else:
                raise ValueError("label_vector not found in inputs")

        # Convert numpy arrays to tensors
        for k in list(inputs.keys()):
            if isinstance(inputs[k], np.ndarray):
                inputs[k] = torch.from_numpy(inputs[k]).unsqueeze(0).to(device)

        # ----------------------------------------------------------
        # Inference — matches original eval.py lines 460-514
        # ----------------------------------------------------------
        label_vector = inputs["label_vector"]

        if model.nO == 1 and label_vector.sum() > model.nO:
            # Per-label inference for single-output models
            length_fg_labels = len(targets["fg_labels"])
            for i in range(length_fg_labels):
                if "None" in targets["fg_labels"][i]:
                    length_fg_labels -= 1

            output = torch.zeros(
                (1, length_fg_labels, inputs["mixture"].shape[-1])
            )
            label_vector_target_one = torch.zeros_like(label_vector)
            index_input_label_vector_one = torch.where(label_vector == 1)[1]

            inputs["original_embedding"] = inputs["embedding"]
            for src_idx, id_label in enumerate(index_input_label_vector_one):
                label_vector_target_one[0, id_label] = 1
                inputs["embedding"] = label_vector_target_one.to(device)
                outputs = model(inputs)
                output[0, src_idx] = outputs["output"].squeeze(0)
                label_vector_target_one[0, id_label] = 0
        else:
            # Multi-output model: single forward pass
            outputs = model(inputs)
            output = outputs["output"]

        # ----------------------------------------------------------
        # Compute metrics — matches original eval.py lines 571-606
        # ----------------------------------------------------------
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt).unsqueeze(0)

        output_cpu = output.cpu()
        mixture_cpu = inputs["mixture"].cpu()
        label_vector_cpu = label_vector.cpu()

        if isinstance(gt, torch.Tensor):
            metrics_result = compute_metrics_tse(
                gt=gt,
                est=output_cpu,
                mix=mixture_cpu,
                label_vector=label_vector_cpu,
                metric_func_list=list(metrics_func_list.values()),
            )

            row = {}
            row["sample"] = (
                sample_name.item()
                if isinstance(sample_name, torch.Tensor)
                else sample_name
            )

            for metric_name, metric_val in metrics_result.items():
                if isinstance(metric_val, torch.Tensor):
                    metric_val = metric_val.item()
                row[metric_name] = metric_val

                if "per_channel" in metric_name:
                    continue

                if metric_name not in metrics_avg:
                    metrics_avg[metric_name] = 0.0
                    metrics_count[metric_name] = 0

                metrics_count[metric_name] += 1
                metrics_avg[metric_name] = (
                    metrics_avg[metric_name] * (metrics_count[metric_name] - 1)
                    + metric_val
                ) / metrics_count[metric_name]

            records.append(row)

        # Update progress bar
        desc_parts = []
        for mn in ["snr_i", "si_sdr", "si_sdr_i"]:
            if mn in metrics_avg:
                desc_parts.append(f"{mn}: {metrics_avg[mn]:.4f}")
        pbar.set_description(f"Avg: {', '.join(desc_parts)}")
        pbar.update(1)

    pbar.close()

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Per-sample CSV
    import pandas as pd
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Saved per-sample results to %s", csv_path)

    # Total averages
    averages_path = os.path.join(output_dir, "metrics_total_averages.json")
    with open(averages_path, "w") as f:
        json.dump(metrics_avg, f, indent=2)
    logger.info("Saved averages to %s", averages_path)

    # Print final results
    logger.info("")
    logger.info("=" * 60)
    logger.info("  FINAL RESULTS")
    logger.info("=" * 60)
    for mn, val in sorted(metrics_avg.items()):
        logger.info("  %s: %.4f", mn, val)

    return metrics_avg, records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_model_from_config(config_path: str, checkpoint_path: str):
    """Load model from a YAML config + Lightning or raw checkpoint."""
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Build model from YAML config (same as train.py _build_model)
    from src.tse.net import Net

    m = cfg["model"]
    model = Net(
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

    # Load checkpoint (handle Lightning format)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        # Lightning checkpoint — keys are prefixed with "model."
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            new_key = k.replace("model.", "", 1) if k.startswith("model.") else k
            state_dict[new_key] = v
        model.load_state_dict(state_dict, strict=True)
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    model.eval()
    logger.info("Loaded model from config=%s, checkpoint=%s", config_path, checkpoint_path)

    # Convert YAML config to params dict for dataset building
    params = _yaml_to_eval_params(cfg)
    return model, params


def _yaml_to_eval_params(cfg: dict) -> dict:
    """Store the raw YAML config for direct SoundscapeDataset construction."""
    return {"_yaml_cfg": cfg}


def _build_dataset_from_yaml(cfg: dict, data_dir: str | None = None, num_samples: int | None = None):
    """Build test SoundscapeDataset directly from YAML config."""
    from src.datasets.soundscape_dataset import SoundscapeDataset
    from pathlib import Path

    d = cfg["data"]
    m = cfg.get("model", {})

    fg_dir = os.path.join(d["fg_dir"], "test")
    noise_dir = os.path.join(d["noise_dir"], "test")
    hrtf_list = d["hrtf_list"].replace("train_hrtf", "test_hrtf")

    if data_dir is not None:
        if not Path(fg_dir).is_absolute():
            fg_dir = str(Path(data_dir) / fg_dir)
        if not Path(noise_dir).is_absolute():
            noise_dir = str(Path(data_dir) / noise_dir)
        if not Path(hrtf_list).is_absolute():
            hrtf_list = str(Path(data_dir) / hrtf_list)

    samples = num_samples if num_samples is not None else d.get("samples_per_epoch", 2000)

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
        samples_per_epoch=samples,
        num_output_channels=m.get("num_output_channels", 1),
        task="tse",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate TSE model")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Path to model run directory (config.json + checkpoints/)")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="HuggingFace repo ID")
    parser.add_argument("--model", type=str, default="orange_pi",
                        help="Model name for --pretrained")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config path (use with --checkpoint)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (Lightning or raw)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override dataset root (e.g., /scr)")
    parser.add_argument("--output_dir", type=str, default="runs/tse/eval",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Override number of eval samples")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Sample rate")
    parser.add_argument("--use_last", action="store_true", default=True,
                        help="Use last.pt instead of best.pt")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load model
    if args.config and args.checkpoint:
        model, params = _load_model_from_config(args.config, args.checkpoint)
    elif args.pretrained:
        model, params = _load_model_from_pretrained(args.pretrained, args.model)
    elif args.run_dir:
        model, params = _load_model_from_run_dir(args.run_dir, use_last=args.use_last)
    else:
        raise ValueError("Provide --config/--checkpoint, --pretrained, or --run_dir")

    logger.info("Model: nI=%d, nO=%d", model.nI, model.nO)

    # Load dataset
    if "_yaml_cfg" in params:
        test_dataset = _build_dataset_from_yaml(
            params["_yaml_cfg"], data_dir=args.data_dir, num_samples=args.num_samples,
        )
    else:
        if args.num_samples is not None:
            params["onflight_test_data_args"]["samples_per_epoch"] = args.num_samples
        test_dataset = _build_dataset(params, data_dir=args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0,
    )
    logger.info("Test dataset: %d samples", len(test_dataset))

    # Run evaluation
    metrics_avg, records = evaluate(model, test_loader, params, device, args.output_dir)


if __name__ == "__main__":
    main()
