#!/usr/bin/env python3
"""Build noise_scaper_fmt from an extracted BinauralCuratedDataset.

This script is independent from the main pipeline. It takes an existing
BinauralCuratedDataset directory (from the public tar at
https://semantichearing.cs.washington.edu/BinauralCuratedDataset.tar)
and creates the noise_scaper_fmt/ directory that the tar does not include.

The noise_scaper_fmt/ contains symlinks to TAU Urban Acoustic Scenes audio,
organized by scene label and split.

Usage:
    # After downloading and extracting the public tar:
    #   wget https://semantichearing.cs.washington.edu/BinauralCuratedDataset.tar
    #   tar xf BinauralCuratedDataset.tar
    python scripts/build_noise_scaper_fmt.py --data_dir /path/to/BinauralCuratedDataset

    # Or with a separate TAU directory:
    python scripts/build_noise_scaper_fmt.py --data_dir /path/to/BinauralCuratedDataset \
        --tau_dir /path/to/TAU-acoustic-sounds
"""
from __future__ import annotations

import argparse
import glob
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def curate_samples(samples: list[str], is_test: bool = False) -> pd.DataFrame:
    """Parse TAU filenames into (fname, label, id) DataFrame."""
    processed = pd.DataFrame(
        {"fname": [os.path.basename(x).split(".")[0] for x in samples]}
    )
    if not is_test:
        # dev: scene-city-timestamp-id.wav → label = scene-city
        processed["label"] = processed["fname"].apply(
            lambda x: x.split("-")[0] + "-" + x.split("-")[1]
        )
    else:
        # eval: numeric id only
        processed["label"] = processed["fname"].apply(
            lambda x: x.split(".")[0]
        )
    processed["id"] = processed["fname"].apply(
        lambda x: "-".join(x.split("-")[2:]).split(".")[0]
    )
    return processed[["fname", "label", "id"]]


def build_noise_scaper_fmt(data_dir: str, tau_dir: str | None = None) -> None:
    """Build noise_scaper_fmt/ with symlinks to TAU audio."""
    random.seed(0)
    np.random.seed(0)

    # Find TAU data
    if tau_dir is None:
        tau_dir = os.path.join(data_dir, "TAU-acoustic-sounds")

    if not os.path.isdir(tau_dir):
        raise FileNotFoundError(
            f"TAU directory not found: {tau_dir}. "
            "Expected TAU-acoustic-sounds/ inside data_dir."
        )

    # Discover samples
    dev_samples = sorted(
        glob.glob(os.path.join(
            tau_dir, "TAU-urban-acoustic-scenes-2019-development", "audio", "*.wav"
        ))
    )
    eval_samples = sorted(
        glob.glob(os.path.join(
            tau_dir, "TAU-urban-acoustic-scenes-2019-evaluation", "audio", "*.wav"
        ))
    )

    print(f"TAU dev: {len(dev_samples)} files")
    print(f"TAU eval: {len(eval_samples)} files")

    if not dev_samples:
        raise FileNotFoundError(
            f"No dev audio found in {tau_dir}/TAU-urban-acoustic-scenes-2019-development/audio/"
        )

    # Curate dev → train + val (90:10 per label)
    dev_curated = curate_samples(dev_samples)

    train_frames: list[pd.DataFrame] = []
    val_frames: list[pd.DataFrame] = []
    for label in sorted(dev_curated["label"].unique()):
        subset = dev_curated[dev_curated["label"] == label]
        if len(subset) <= 1:
            continue
        tr, va = train_test_split(subset, test_size=0.1)
        train_frames.append(tr)
        val_frames.append(va)

    train_samples = pd.concat(train_frames)
    val_samples = pd.concat(val_frames)
    test_samples = curate_samples(eval_samples, is_test=True)

    # Add relative paths
    train_samples = train_samples.copy()
    val_samples = val_samples.copy()
    test_samples = test_samples.copy()

    train_samples["fname"] = train_samples["fname"].apply(
        lambda x: f"TAU-urban-acoustic-scenes-2019-development/audio/{x}.wav"
    )
    val_samples["fname"] = val_samples["fname"].apply(
        lambda x: f"TAU-urban-acoustic-scenes-2019-development/audio/{x}.wav"
    )
    test_samples["fname"] = test_samples["fname"].apply(
        lambda x: f"TAU-urban-acoustic-scenes-2019-evaluation/audio/{x}.wav"
    )

    # Keep common labels between train and val
    common_labels = list(
        set(train_samples["label"].unique())
        & set(val_samples["label"].unique())
    )
    train_samples = train_samples[train_samples["label"].isin(common_labels)]
    val_samples = val_samples[val_samples["label"].isin(common_labels)]

    # Write CSVs (if not already present)
    csv_dir = os.path.join(data_dir, "TAU-acoustic-sounds")
    os.makedirs(csv_dir, exist_ok=True)
    cols = ["label", "fname", "id"]
    for name, df in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        csv_path = os.path.join(csv_dir, f"{name}.csv")
        if not os.path.exists(csv_path):
            df[cols].to_csv(csv_path, index=False)
            print(f"  Wrote {csv_path} ({len(df)} rows)")
        else:
            print(f"  [skip] {csv_path} already exists")

    # Create symlinks in noise_scaper_fmt
    dataset_name = "TAU-acoustic-sounds"
    symlink_dir = os.path.join(data_dir, "noise_scaper_fmt")

    for split_name, split_df in [
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    ]:
        for _, row in tqdm(
            split_df.iterrows(),
            total=len(split_df),
            desc=f"noise_scaper_fmt/{split_name}",
        ):
            label_str = str(row["label"])
            dest_dir = os.path.join(symlink_dir, split_name, label_str)
            os.makedirs(dest_dir, exist_ok=True)

            # Relative symlink: ../../../TAU-acoustic-sounds/{fname}
            src = os.path.join("..", "..", "..", dataset_name, row["fname"])
            fname = dataset_name.lower() + "_" + os.path.basename(row["fname"])
            dest = os.path.join(dest_dir, fname)

            if not os.path.lexists(dest):
                os.symlink(src, dest)

    print(f"\n✓ noise_scaper_fmt built:")
    for split in ("train", "val", "test"):
        split_dir = os.path.join(symlink_dir, split)
        if os.path.isdir(split_dir):
            n_dirs = len(os.listdir(split_dir))
            print(f"  {split}: {n_dirs} scene directories")
        else:
            print(f"  {split}: not created")


def main():
    parser = argparse.ArgumentParser(
        description="Build noise_scaper_fmt from extracted BinauralCuratedDataset"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to extracted BinauralCuratedDataset/"
    )
    parser.add_argument(
        "--tau_dir", type=str, default=None,
        help="Path to TAU-acoustic-sounds/ (default: data_dir/TAU-acoustic-sounds/)"
    )
    args = parser.parse_args()

    build_noise_scaper_fmt(args.data_dir, args.tau_dir)


if __name__ == "__main__":
    main()
