#!/usr/bin/env python3
"""Build TAU CSVs + noise_scaper_fmt from an extracted BinauralCuratedDataset.

This script is independent from the main pipeline. It takes an existing
BinauralCuratedDataset directory (from the public tar at
https://semantichearing.cs.washington.edu/BinauralCuratedDataset.tar)
and creates:
  1. TAU-acoustic-sounds/{train,val,test}.csv  (if not present)
  2. noise_scaper_fmt/{train,val,test}/{scene}/  symlinks

The public tar does NOT include noise_scaper_fmt, so this script must
be run after extraction.

Usage:
    # After downloading and extracting the public tar:
    #   wget https://semantichearing.cs.washington.edu/BinauralCuratedDataset.tar
    #   tar xf BinauralCuratedDataset.tar

    # If TAU audio is inside BinauralCuratedDataset/TAU-acoustic-sounds/:
    python scripts/build_noise_scaper_fmt.py \
        --data_dir /path/to/BinauralCuratedDataset

    # If TAU audio is at a separate raw path (e.g. TAU-2019/):
    python scripts/build_noise_scaper_fmt.py \
        --data_dir /path/to/BinauralCuratedDataset \
        --tau_raw_dir /path/to/TAU-2019
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


def _find_tau_audio(data_dir: str, tau_raw_dir: str | None) -> str:
    """Find TAU audio directory. Returns path containing
    TAU-urban-acoustic-scenes-2019-{development,evaluation}/ subdirs.
    """
    # 1. Explicit raw dir
    if tau_raw_dir and os.path.isdir(tau_raw_dir):
        dev = os.path.join(tau_raw_dir, "TAU-urban-acoustic-scenes-2019-development")
        if os.path.isdir(dev):
            return tau_raw_dir

    # 2. Inside data_dir/TAU-acoustic-sounds/
    tau_in_data = os.path.join(data_dir, "TAU-acoustic-sounds")
    dev = os.path.join(tau_in_data, "TAU-urban-acoustic-scenes-2019-development")
    if os.path.isdir(dev):
        return tau_in_data

    # 3. data_dir itself
    dev = os.path.join(data_dir, "TAU-urban-acoustic-scenes-2019-development")
    if os.path.isdir(dev):
        return data_dir

    raise FileNotFoundError(
        f"Cannot find TAU audio. Looked in:\n"
        f"  1. --tau_raw_dir: {tau_raw_dir}\n"
        f"  2. {tau_in_data}/TAU-urban-acoustic-scenes-2019-development/\n"
        f"  3. {data_dir}/TAU-urban-acoustic-scenes-2019-development/\n"
        f"Provide --tau_raw_dir pointing to the directory containing "
        f"TAU-urban-acoustic-scenes-2019-development/"
    )


def _ensure_tau_links(data_dir: str, tau_audio_dir: str) -> None:
    """Ensure TAU-acoustic-sounds/ inside data_dir has symlinks to
    TAU-urban-acoustic-scenes-2019-{development,evaluation}/ audio.
    This is needed so that noise_scaper_fmt symlinks resolve correctly.
    """
    tau_dest = os.path.join(data_dir, "TAU-acoustic-sounds")
    os.makedirs(tau_dest, exist_ok=True)

    for subdir in (
        "TAU-urban-acoustic-scenes-2019-development",
        "TAU-urban-acoustic-scenes-2019-evaluation",
    ):
        src = os.path.join(tau_audio_dir, subdir)
        dst = os.path.join(tau_dest, subdir)
        if os.path.isdir(src) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
            print(f"  Linked {dst} -> {src}")


def build_noise_scaper_fmt(
    data_dir: str, tau_raw_dir: str | None = None
) -> None:
    """Build TAU CSVs + noise_scaper_fmt/."""
    random.seed(0)
    np.random.seed(0)

    # Step 0: Find TAU audio and ensure symlinks
    tau_audio_dir = _find_tau_audio(data_dir, tau_raw_dir)
    print(f"TAU audio dir: {tau_audio_dir}")
    _ensure_tau_links(data_dir, tau_audio_dir)

    # Step 1: Discover samples
    dev_samples = sorted(glob.glob(os.path.join(
        tau_audio_dir, "TAU-urban-acoustic-scenes-2019-development", "audio", "*.wav"
    )))
    eval_samples = sorted(glob.glob(os.path.join(
        tau_audio_dir, "TAU-urban-acoustic-scenes-2019-evaluation", "audio", "*.wav"
    )))
    print(f"TAU dev: {len(dev_samples)} files")
    print(f"TAU eval: {len(eval_samples)} files")

    if not dev_samples:
        raise FileNotFoundError(
            f"No dev audio in {tau_audio_dir}/"
            "TAU-urban-acoustic-scenes-2019-development/audio/"
        )

    # Step 2: Build or load CSVs
    csv_dir = os.path.join(data_dir, "TAU-acoustic-sounds")
    os.makedirs(csv_dir, exist_ok=True)

    train_csv = os.path.join(csv_dir, "train.csv")
    val_csv = os.path.join(csv_dir, "val.csv")
    test_csv = os.path.join(csv_dir, "test.csv")

    if os.path.exists(train_csv) and os.path.getsize(train_csv) > 50:
        # Use existing CSVs
        print(f"  [use existing] {train_csv}")
        train_samples = pd.read_csv(train_csv)
        val_samples = pd.read_csv(val_csv)
        test_samples = pd.read_csv(test_csv)
    else:
        # Generate CSVs from raw audio
        print("  Generating TAU CSVs from raw audio...")
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

        # Keep common labels
        common_labels = list(
            set(train_samples["label"].unique())
            & set(val_samples["label"].unique())
        )
        train_samples = train_samples[train_samples["label"].isin(common_labels)]
        val_samples = val_samples[val_samples["label"].isin(common_labels)]

        # Write CSVs
        cols = ["label", "fname", "id"]
        for name, df in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
            csv_path = os.path.join(csv_dir, f"{name}.csv")
            df[cols].to_csv(csv_path, index=False)
            print(f"  Wrote {csv_path} ({len(df)} rows)")

    print(f"  TAU splits: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    # Step 3: Create noise_scaper_fmt symlinks
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
        description="Build TAU CSVs + noise_scaper_fmt from BinauralCuratedDataset"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to extracted BinauralCuratedDataset/"
    )
    parser.add_argument(
        "--tau_raw_dir", type=str, default=None,
        help="Path to raw TAU data (e.g. TAU-2019/) containing "
             "TAU-urban-acoustic-scenes-2019-{development,evaluation}/"
    )
    args = parser.parse_args()

    build_noise_scaper_fmt(args.data_dir, args.tau_raw_dir)


if __name__ == "__main__":
    main()
