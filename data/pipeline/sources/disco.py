from __future__ import annotations

import glob
import logging
import os
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .base import BaseSource

logger = logging.getLogger(__name__)

# Maps DISCO-noise folder names to AudioSet label names.
# ``None`` means no AudioSet equivalent — skipped.
DISCO_TO_AUDIOSET: dict[str, str | None] = {
    "baby": "Baby cry, infant cry",
    "blender": "Blender",
    "dishwasher": None,
    "electric_shaver_toothbrush": "Toothbrush",
    "fan": "Mechanical fan",
    "frying": "Frying (food)",
    "printer": "Printer",
    "vacuum_cleaner": "Vacuum cleaner",
    "washing_machine": None,
    "water": "Water",
}


class DISCOSource(BaseSource):
    name = "disco_noises"
    key = "disco"

    def __init__(self, ontology):
        self.ontology = ontology

    # Zenodo raw layout: DISCO/{train,test}/{label}/*.wav
    # Expected layout:   disco_noises/{train,test}/{label}/*.wav
    ZENODO_RAW_NAMES = ("DISCO",)

    def download(self, raw_dir: Path) -> None:
        import shutil

        out = raw_dir / self.name
        if (out / ".done").exists():
            print(f"  [skip] {self.name} already downloaded")
            return

        # Check for Zenodo raw layout (DISCO/{train,test}/)
        for zenodo_name in self.ZENODO_RAW_NAMES:
            zenodo_dir = raw_dir / zenodo_name
            if zenodo_dir.exists() and (zenodo_dir / "train").is_dir():
                print(f"  Found Zenodo raw: {zenodo_dir}")
                out.mkdir(parents=True, exist_ok=True)
                for split in ("train", "test"):
                    src = zenodo_dir / split
                    dst = out / split
                    if src.exists() and not dst.exists():
                        dst.symlink_to(src.resolve())
                (out / ".done").touch()
                print(f"  ✓ {self.name} linked from {zenodo_name}/")
                return

        # Fallback: download from HF
        print("  Loading from HF: ooshyun/fine_grained_soundscape_control (disco) ...")
        from huggingface_hub import snapshot_download

        tmp = raw_dir / "_hf_download"
        snapshot_download(
            repo_id="ooshyun/fine_grained_soundscape_control",
            repo_type="dataset",
            allow_patterns="disco/**",
            local_dir=str(tmp),
        )
        src = tmp / "disco"
        if src.exists():
            if out.exists():
                shutil.rmtree(out)
            shutil.move(str(src), str(out))
        shutil.rmtree(tmp, ignore_errors=True)
        (out / ".done").touch()
        print(f"  ✓ {self.name} downloaded")

    def collect(self, raw_dir: Path, curated_dir: Path) -> None:
        dataset_dir = raw_dir / self.name
        out_dir = curated_dir / self.name

        # Gather all files across train/ and test/ subdirectories
        files_by_label: dict[str, list[str]] = {}
        for split in ("train", "test"):
            split_dir = dataset_dir / split
            if not split_dir.is_dir():
                continue
            for label_dir in split_dir.iterdir():
                if not label_dir.is_dir():
                    continue
                label = label_dir.name
                for f in label_dir.glob("*"):
                    if f.is_file():
                        files_by_label.setdefault(label, []).append(str(f))

        train_records: list[dict] = []
        val_records: list[dict] = []
        test_records: list[dict] = []

        for label, file_list in files_by_label.items():
            audioset_label = DISCO_TO_AUDIOSET.get(label)
            if audioset_label is None:
                continue

            _id = self.ontology.get_id_from_name(audioset_label)

            # 67:33 train:test split
            train_files, test_files = train_test_split(
                file_list, test_size=0.33, random_state=42,
            )

            # 90:10 train:val split
            random.shuffle(train_files)
            val_split = int(round(0.1 * len(train_files)))
            val_files = train_files[:val_split]
            train_files = train_files[val_split:]

            for fname in train_files:
                train_records.append(
                    dict(
                        fname=os.path.relpath(fname, dataset_dir),
                        label=audioset_label,
                        id=_id,
                    )
                )
            for fname in test_files:
                test_records.append(
                    dict(
                        fname=os.path.relpath(fname, dataset_dir),
                        label=audioset_label,
                        id=_id,
                    )
                )
            for fname in val_files:
                val_records.append(
                    dict(
                        fname=os.path.relpath(fname, dataset_dir),
                        label=audioset_label,
                        id=_id,
                    )
                )

        train_df = pd.DataFrame.from_records(train_records)
        val_df = pd.DataFrame.from_records(val_records)
        test_df = pd.DataFrame.from_records(test_records)

        self._write_csvs(out_dir, train_df, val_df, test_df)

        logger.info(
            "DISCO: train=%d  val=%d  test=%d",
            len(train_records),
            len(val_records),
            len(test_records),
        )
