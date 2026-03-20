from __future__ import annotations

import glob
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TAUCollector:
    """Collect and curate TAU Urban Acoustic Scenes into train/val/test.

    Expected input layout::

        raw_dir/TAU-urban-acoustic-scenes-2019-development/audio/*.wav
        raw_dir/TAU-urban-acoustic-scenes-2019-evaluation/audio/*.wav

    Filename convention: ``{scene}-{city}-{timestamp}-{id}.wav``

    Output: symlinks in ``output_dir/noise_scaper_fmt/{train,val,test}/{scene}/``

    Also writes ``output_dir/TAU-acoustic-sounds/{train,val,test}.csv`` as
    intermediate manifests.
    """

    def __init__(self) -> None:
        pass

    def _curate_samples(
        self, samples: list[str], is_test: bool = False
    ) -> pd.DataFrame:
        processed = pd.DataFrame(
            {"fname": [os.path.basename(x).split(".")[0] for x in samples]}
        )
        if not is_test:
            processed["label"] = processed["fname"].apply(
                lambda x: x.split("-")[0] + "-" + x.split("-")[1]
            )
        else:
            processed["label"] = processed["fname"].apply(
                lambda x: x.split(".")[0]
            )
        processed["id"] = processed["fname"].apply(
            lambda x: "-".join(x.split("-")[2:]).split(".")[0]
        )
        return processed[["fname", "label", "id"]]

    def collect(self, raw_dir: str, output_dir: str) -> None:
        tau_root = raw_dir  # raw_dir should contain both TAU directories

        # Discover samples
        dev_samples = sorted(
            glob.glob(
                os.path.join(
                    tau_root,
                    "TAU-urban-acoustic-scenes-2019-development",
                    "audio",
                    "*.wav",
                )
            )
        )
        eval_samples = sorted(
            glob.glob(
                os.path.join(
                    tau_root,
                    "TAU-urban-acoustic-scenes-2019-evaluation",
                    "audio",
                    "*.wav",
                )
            )
        )

        # Curate dev → train + val (90:10 per label)
        dev_curated = self._curate_samples(dev_samples)

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
        test_samples = self._curate_samples(eval_samples, is_test=True)

        # Add relative paths to audio
        train_src = "TAU-urban-acoustic-scenes-2019-development/audio"
        val_src = "TAU-urban-acoustic-scenes-2019-development/audio"
        test_src = "TAU-urban-acoustic-scenes-2019-evaluation/audio"

        train_samples = train_samples.copy()
        val_samples = val_samples.copy()
        test_samples = test_samples.copy()

        train_samples["fname"] = train_samples["fname"].apply(
            lambda x: os.path.join(train_src, f"{x}.wav")
        )
        val_samples["fname"] = val_samples["fname"].apply(
            lambda x: os.path.join(val_src, f"{x}.wav")
        )
        test_samples["fname"] = test_samples["fname"].apply(
            lambda x: os.path.join(test_src, f"{x}.wav")
        )

        # Keep common labels between train and val
        common_labels = list(
            set(train_samples["label"].unique())
            & set(val_samples["label"].unique())
        )
        train_samples = train_samples[train_samples["label"].isin(common_labels)]
        val_samples = val_samples[val_samples["label"].isin(common_labels)]

        # Write intermediate CSVs
        dataset_name = "TAU-acoustic-sounds"
        csv_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(csv_dir, exist_ok=True)

        cols = ["label", "fname", "id"]
        train_samples[cols].to_csv(
            os.path.join(csv_dir, "train.csv"), index=False
        )
        val_samples[cols].to_csv(
            os.path.join(csv_dir, "val.csv"), index=False
        )
        test_samples[cols].to_csv(
            os.path.join(csv_dir, "test.csv"), index=False
        )

        # Create symlinks in noise_scaper_fmt
        symlink_dir = os.path.join(output_dir, "noise_scaper_fmt")

        for split_name, split_df in [
            ("train", train_samples),
            ("val", val_samples),
            ("test", test_samples),
        ]:
            for _, row in tqdm(
                split_df.iterrows(),
                total=len(split_df),
                desc=f"TAU {split_name}",
            ):
                label_str = (
                    row["label"]
                    if isinstance(row["label"], str)
                    else str(row["label"])
                )
                dest_dir = os.path.join(symlink_dir, split_name, label_str)
                os.makedirs(dest_dir, exist_ok=True)

                src = os.path.join(
                    "..", "..", "..", dataset_name, row["fname"]
                )
                fname = (
                    dataset_name.lower()
                    + "_"
                    + os.path.basename(row["fname"])
                )
                dest = os.path.join(dest_dir, fname)

                if os.path.exists(dest):
                    continue
                os.symlink(src, dest)

        logger.info(
            "TAU: train=%d  val=%d  test=%d",
            len(train_samples),
            len(val_samples),
            len(test_samples),
        )
