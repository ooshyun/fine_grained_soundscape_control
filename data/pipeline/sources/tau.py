from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .base import BaseSource

logger = logging.getLogger(__name__)


class TAUSource(BaseSource):
    """Collect and curate TAU Urban Acoustic Scenes into train/val/test.

    Expected input layout::

        raw_dir/TAU-2019/TAU-urban-acoustic-scenes-2019-development/audio/*.wav
        raw_dir/TAU-2019/TAU-urban-acoustic-scenes-2019-evaluation/audio/*.wav

    Filename convention: ``{scene}-{city}-{timestamp}-{id}.wav``

    Output:
        - ``curated_dir/TAU-acoustic-sounds/{train,val,test}.csv``
        - Symlinks in ``curated_dir/noise_scaper_fmt/{split}/{scene}/``
    """

    name = "TAU-2019"
    key = "tau"
    ZENODO_BASE = "https://zenodo.org/records/2589280/files"

    def print_download_guide(self) -> None:
        print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  TAU Urban Acoustic Scenes 2019                         │
  │                                                         │
  │  License: Tampere Univ. custom (non-commercial)         │
  │  Size: ~20 GB (10 audio zips + meta)                    │
  │                                                         │
  │  Download from: {self.ZENODO_BASE}                      │
  │  Files: audio.1.zip ~ audio.10.zip + meta.zip           │
  │                                                         │
  │  Extract all to:                                        │
  │    <manual_dir>/TAU-2019/                               │
  │                                                         │
  │  Then re-run with --manual_dir <path>                   │
  └─────────────────────────────────────────────────────────┘""")

    def download(self, raw_dir: Path) -> None:
        import ssl
        import urllib.request
        import zipfile

        ssl._create_default_https_context = ssl._create_unverified_context
        out = raw_dir / self.name
        if (out / ".done").exists():
            print(f"  [skip] {self.name} already downloaded")
            return
        out.mkdir(parents=True, exist_ok=True)
        files = [
            f"TAU-urban-acoustic-scenes-2019-development.audio.{i}.zip"
            for i in range(1, 11)
        ] + ["TAU-urban-acoustic-scenes-2019-development.meta.zip"]
        for fname in files:
            url = f"{self.ZENODO_BASE}/{fname}?download=1"
            zip_path = out / fname
            if not zip_path.exists():
                print(f"  Downloading {fname} ...")
                urllib.request.urlretrieve(url, str(zip_path))
            print(f"  Extracting {fname} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out)
            zip_path.unlink()
        (out / ".done").touch()
        print(f"  ✓ {self.name} downloaded")

    @staticmethod
    def _curate_samples(
        samples: list[str], is_test: bool = False,
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

    def collect(self, raw_dir: Path, curated_dir: Path) -> None:
        tau_root = str(raw_dir / self.name)

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
            tr, va = train_test_split(subset, test_size=0.1, random_state=42)
            train_frames.append(tr)
            val_frames.append(va)

        train_samples = pd.concat(train_frames) if train_frames else pd.DataFrame(columns=["fname", "label", "id"])
        val_samples = pd.concat(val_frames) if val_frames else pd.DataFrame(columns=["fname", "label", "id"])
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
        csv_dir = curated_dir / dataset_name
        csv_dir.mkdir(parents=True, exist_ok=True)

        cols = ["fname", "label", "id"]
        train_samples[cols].to_csv(csv_dir / "train.csv", index=False)
        val_samples[cols].to_csv(csv_dir / "val.csv", index=False)
        test_samples[cols].to_csv(csv_dir / "test.csv", index=False)

        # Create symlinks in noise_scaper_fmt
        symlink_dir = curated_dir / "noise_scaper_fmt"

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
                dest_dir = symlink_dir / split_name / label_str
                dest_dir.mkdir(parents=True, exist_ok=True)

                src = os.path.join(
                    "..", "..", "..", dataset_name, row["fname"]
                )
                fname = (
                    dataset_name.lower()
                    + "_"
                    + os.path.basename(row["fname"])
                )
                dest = dest_dir / fname

                if dest.exists():
                    continue
                os.symlink(src, str(dest))

        logger.info(
            "TAU: train=%d  val=%d  test=%d",
            len(train_samples),
            len(val_samples),
            len(test_samples),
        )
