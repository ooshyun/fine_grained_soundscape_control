"""Prepare stage — consolidate curated CSVs into Scaper-format directories.

Reads per-dataset ``{train,val,test}.csv`` files produced by the collect
stage, creates foreground/background symlinks in Scaper layout, copies CIPIC
HRTF files, and writes ``start_times.csv`` with silence-trimming metadata.
"""
from __future__ import annotations

import glob
import logging
import os
import random
import shutil
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from .ontology import Ontology
from .silence import trim_silence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# id2classname construction
# ---------------------------------------------------------------------------


def build_id2classname(
    class_map_path: str | Path,
    ontology: Ontology,
) -> dict[str, str]:
    """Build a mapping from AudioSet IDs to dataset class names.

    Uses ``class_map.yaml`` and the Ontology to expand each named class
    into its complete subtree so that all descendant IDs map to the
    top-level class name.
    """
    with open(class_map_path) as f:
        class_data: dict[str, list[str]] = yaml.safe_load(f)

    id2classname: dict[str, str] = {}
    for class_name, elements in class_data.items():
        for element_name in elements:
            node_id = ontology.get_id_from_name(element_name)
            if node_id is None:
                logger.warning(
                    "Element %r not found in ontology — skipping", element_name
                )
                continue
            for cid in ontology.get_subtree(node_id):
                id2classname[cid] = class_name

    return id2classname


# ---------------------------------------------------------------------------
# Background validation
# ---------------------------------------------------------------------------


def is_valid_background(
    sample_id: str,
    ontology: Ontology,
    id2classname: dict[str, str],
) -> bool:
    """Return ``True`` if *sample_id* qualifies as a background label.

    A label is valid background iff it:
    - Is NOT already a foreground label (in id2classname)
    - Is NOT in the Music subtree
    - Is NOT in the Human voice subtree
    - Is NOT an ancestor or descendant of any foreground label
    """
    # Already a foreground label
    if sample_id in id2classname:
        return False

    # Exclude Music subtree
    if ontology.is_reachable(ontology.MUSIC, sample_id):
        return False

    # Exclude Human voice subtree
    human_voice_id = ontology.get_id_from_name("Human voice")
    if human_voice_id and ontology.is_reachable(human_voice_id, sample_id):
        return False

    # Exclude ancestors/descendants of any foreground label
    foreground_ids = list(id2classname.keys())
    for fg_id in foreground_ids:
        if ontology.is_reachable(sample_id, fg_id) or ontology.is_reachable(
            fg_id, sample_id
        ):
            return False

    return True


# ---------------------------------------------------------------------------
# Scaper-format writer
# ---------------------------------------------------------------------------


def write_scaper_source(
    dataset_name: str,
    split: str,
    raw_dir: Path,
    fg_dir: Path,
    bg_dir: Path,
    id2classname: dict[str, str],
    ontology: Ontology,
    all_samples: list[dict],
    csv_path: Path,
) -> None:
    """Read a curated CSV and create foreground/background symlinks.

    Parameters
    ----------
    dataset_name:
        Dataset identifier (e.g. ``"FSD50K"``).
    split:
        One of ``"train"``, ``"val"``, ``"test"``.
    raw_dir:
        Root directory containing raw dataset files.
    fg_dir:
        Base foreground directory (``scaper_fmt/``).
    bg_dir:
        Base background directory (``bg_scaper_fmt/``).
    id2classname:
        AudioSet ID to class name mapping.
    ontology:
        Loaded Ontology instance.
    all_samples:
        Accumulator list for start_times records.
    csv_path:
        Path to the ``{split}.csv`` file.
    """
    if not csv_path.exists():
        logger.warning("Missing CSV: %s — skipping", csv_path)
        return

    dataset = pd.read_csv(csv_path)
    fg_out_dir = fg_dir / split
    bg_out_dir = bg_dir / split

    logger.info("Consolidating %s/%s ...", dataset_name, split)

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"{dataset_name}/{split}"):
        sample_id = row["id"]

        if sample_id in id2classname:
            classname = id2classname[sample_id]
            out_path = fg_out_dir / classname
        elif is_valid_background(sample_id, ontology, id2classname):
            label = row["label"]
            out_path = bg_out_dir / label
        else:
            continue

        out_path.mkdir(parents=True, exist_ok=True)

        # Relative symlink: ../../..<dataset_name>/<fname>
        src = os.path.join("..", "..", "..", dataset_name, row["fname"])
        fname = dataset_name.lower() + "_" + os.path.basename(row["fname"])
        dest = out_path / fname

        # Trim silence and record metadata
        raw_audio_path = raw_dir / dataset_name / row["fname"]
        start_sample, first_silence, end_sample = trim_silence(str(raw_audio_path))
        assert start_sample < end_sample

        all_samples.append(
            {
                "fname": str(dest),
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "first_silence": first_silence,
            }
        )

        if not dest.exists():
            os.symlink(src, str(dest))


# ---------------------------------------------------------------------------
# HRTF handling
# ---------------------------------------------------------------------------


def prepare_hrtf(raw_dir: Path, output_dir: Path) -> None:
    """Copy CIPIC SOFA files and generate train/val/test split lists."""
    cipic_src = raw_dir / "cipic-hrtf-database"
    cipic_dst = output_dir / "hrtf" / "CIPIC"
    cipic_dst.mkdir(parents=True, exist_ok=True)

    sofa_files = sorted(
        glob.glob(str(cipic_src / "**" / "*.sofa"), recursive=True)
    )

    if not sofa_files:
        logger.warning("No CIPIC SOFA files found in %s", cipic_src)
        return

    # Copy files
    copied: list[str] = []
    for src in sofa_files:
        dst = cipic_dst / os.path.basename(src)
        if not dst.exists():
            shutil.copy2(src, str(dst))
        copied.append(os.path.basename(src))

    # 80:10:10 split (random.seed(0) set in run_prepare)
    random.shuffle(copied)
    n = len(copied)
    n_train = int(round(0.8 * n))
    n_val = int(round(0.1 * n))

    train_files = copied[:n_train]
    val_files = copied[n_train : n_train + n_val]
    test_files = copied[n_train + n_val :]

    hrtf_dir = output_dir / "hrtf"
    for name, file_list in [
        ("train_hrtf.txt", train_files),
        ("val_hrtf.txt", val_files),
        ("test_hrtf.txt", test_files),
    ]:
        with open(hrtf_dir / name, "w") as f:
            f.write("\n".join(file_list) + "\n")

    logger.info(
        "HRTF: %d SOFA files → train=%d  val=%d  test=%d",
        n,
        len(train_files),
        len(val_files),
        len(test_files),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_prepare(
    curated_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    ontology: Ontology,
    data_dir: Path,
) -> None:
    """Prepare final training data from curated CSVs.

    Parameters
    ----------
    curated_dir:
        Directory with per-dataset ``{train,val,test}.csv`` files.
    raw_dir:
        Root of raw dataset files (for audio access and HRTF).
    output_dir:
        Top-level output directory.
    ontology:
        Loaded AudioSet Ontology instance.
    data_dir:
        Path to ``data/`` directory (contains ``class_map.yaml``).
    """
    import numpy as np

    random.seed(0)
    np.random.seed(0)

    # Step 1: Build id → classname mapping
    logger.info("=== Prepare Step 1: Building id2classname mapping ===")
    class_map_path = data_dir / "class_map.yaml"
    id2classname = build_id2classname(class_map_path, ontology)
    logger.info(
        "Mapped %d AudioSet IDs to %d class names",
        len(id2classname),
        len(set(id2classname.values())),
    )

    # Step 2: Consolidate into Scaper format
    logger.info("=== Prepare Step 2: Consolidating into Scaper format ===")
    fg_dir = output_dir / "scaper_fmt"
    bg_dir = output_dir / "bg_scaper_fmt"

    datasets = ["FSD50K", "ESC-50", "musdb18", "disco_noises"]
    splits = ["train", "val", "test"]
    all_samples: list[dict] = []

    for dataset_name in datasets:
        for split in splits:
            csv_path = curated_dir / dataset_name / f"{split}.csv"
            write_scaper_source(
                dataset_name=dataset_name,
                split=split,
                raw_dir=raw_dir,
                fg_dir=fg_dir,
                bg_dir=bg_dir,
                id2classname=id2classname,
                ontology=ontology,
                all_samples=all_samples,
                csv_path=csv_path,
            )

    # Step 3: HRTF preparation
    logger.info("=== Prepare Step 3: HRTF preparation ===")
    prepare_hrtf(raw_dir, output_dir)

    # Step 4: Save start_times.csv
    if all_samples:
        df = pd.DataFrame.from_records(all_samples)
        df.to_csv(output_dir / "start_times.csv", index=False)
        logger.info("Saved start_times.csv with %d entries", len(df))

    logger.info("=== Prepare stage complete ===")
