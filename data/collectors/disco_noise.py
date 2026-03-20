from __future__ import annotations

import glob
import logging
import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split

from data.collectors.ontology import Ontology

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


class DISCOCollector:
    """Collect and curate DISCO-noise labels into train/val/test CSV splits.

    Expected input layout::

        raw_dir/disco_noises/
            train/{label}/*.wav
            test/{label}/*.wav

    Output::

        output_dir/disco_noises/{train,val,test}.csv  (columns: fname, label, id)
    """

    def __init__(self, ontology: Ontology) -> None:
        self.ontology = ontology

    def collect(self, raw_dir: str, output_dir: str) -> None:
        dataset_dir = os.path.join(raw_dir, "disco_noises")
        out_dir = os.path.join(output_dir, "disco_noises")
        os.makedirs(out_dir, exist_ok=True)

        # Gather all files across train/ and test/ subdirectories
        files_by_label: dict[str, list[str]] = {}
        for split in ("train", "test"):
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for label in os.listdir(split_dir):
                label_dir = os.path.join(split_dir, label)
                if not os.path.isdir(label_dir):
                    continue
                for f in glob.glob(os.path.join(label_dir, "*")):
                    files_by_label.setdefault(label, []).append(f)

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
                file_list, test_size=0.33
            )

            # 90:10 train:val split
            random.shuffle(train_files)
            val_split = int(round(0.1 * len(train_files)))
            val_files = train_files[:val_split]
            train_files = train_files[val_split:]

            for fname in train_files:
                train_records.append(
                    dict(
                        id=_id,
                        label=audioset_label,
                        fname=os.path.relpath(fname, dataset_dir),
                    )
                )
            for fname in test_files:
                test_records.append(
                    dict(
                        id=_id,
                        label=audioset_label,
                        fname=os.path.relpath(fname, dataset_dir),
                    )
                )
            for fname in val_files:
                val_records.append(
                    dict(
                        id=_id,
                        label=audioset_label,
                        fname=os.path.relpath(fname, dataset_dir),
                    )
                )

        for name, records in [
            ("train", train_records),
            ("val", val_records),
            ("test", test_records),
        ]:
            df = pd.DataFrame.from_records(records)
            df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

        logger.info(
            "DISCO: train=%d  val=%d  test=%d",
            len(train_records),
            len(val_records),
            len(test_records),
        )
