"""Data preparation pipeline — consolidates multiple audio datasets into
a unified Scaper-format directory structure.

Usage::

    python data/prepare.py --raw_dir ./raw_datasets --output_dir ./BinauralCuratedDataset

Steps:
1. Run each per-dataset collector to produce ``{train,val,test}.csv`` files.
2. Load ``class_map.yaml`` + ``ontology.json`` to build an AudioSet-ID →
   classname mapping.
3. For every sample in every dataset CSV, create symlinks:
   - Foreground: ``scaper_fmt/{split}/{classname}/``
   - Background: ``bg_scaper_fmt/{split}/{label}/``
4. Copy CIPIC HRTF SOFA files and generate split lists.
5. Save ``start_times.csv`` with silence-trimming information.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import shutil
import sys
import typing

# Ensure repo root is on sys.path so 'data.collectors' resolves
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import librosa
import numpy as np
import pandas as pd
import yaml
from scipy.io.wavfile import read as wavread
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

from data.collectors.ontology import Ontology
from data.collectors.fsd50k import FSD50KCollector
from data.collectors.esc50 import ESC50Collector
from data.collectors.musdb18 import MUSDB18Collector
from data.collectors.disco_noise import DISCOCollector
from data.collectors.tau import TAUCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _pcm2float(sig: np.ndarray, dtype: str = "float32") -> np.ndarray:
    """Convert PCM integer signal to float in [-1, 1]."""
    sig = np.asarray(sig)
    if sig.dtype.kind not in "iu":
        raise TypeError("'sig' must be an array of integers")
    dt = np.dtype(dtype)
    if dt.kind != "f":
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dt) - offset) / abs_max


def trim_silence(path: str) -> tuple[int, int, int]:
    """Return ``(start_sample, first_silence, end_sample)`` for an audio file.

    Uses ``librosa.effects.trim`` at 40 dB and a 1-second sliding-window
    power threshold to locate the first silence boundary.
    """
    sr, data = wavread(path)
    if len(data.shape) > 1:
        data = np.sum(data, axis=1)

    if data.dtype != np.float32:
        data = _pcm2float(data)

    start, end = librosa.effects.trim(data, top_db=40)[1]
    data = data[start:end]

    window_size = int(round(1 * 44100))
    avg_power = uniform_filter1d(data ** 2, size=window_size, mode="constant")
    threshold = 0.1 * avg_power.max()

    mask = avg_power < threshold
    if mask.any():
        first_silence = int(np.argmax(mask))
    else:
        first_silence = int(end)

    return int(start), first_silence, int(end)


def is_valid_background(
    label_id: str,
    foreground_ids: list[str],
    ontology: Ontology,
) -> bool:
    """A label is valid background iff it is NOT in an excluded subtree
    (Music, Human voice) and is neither an ancestor nor child of any
    foreground label.
    """
    excluded_subtrees = [
        ontology.MUSIC,
        ontology.get_id_from_name("Human voice"),
    ]
    for subtree_root in excluded_subtrees:
        if ontology.is_reachable(subtree_root, label_id):
            return False

    for fg_id in foreground_ids:
        if ontology.is_reachable(label_id, fg_id) or ontology.is_reachable(
            fg_id, label_id
        ):
            return False

    return True


# ---------------------------------------------------------------------------
# Scaper-format writer
# ---------------------------------------------------------------------------


def write_scaper_source(
    dataset_name: str,
    dataset_type: str,
    base_dir: str,
    fg_dest_dir: str,
    bg_dest_dir: str,
    id2classname: dict[str, str],
    ontology: Ontology,
    all_samples: list[dict],
    dry_run: bool = False,
) -> None:
    dataset_path = os.path.join(base_dir, dataset_name)
    fg_out_dir = os.path.join(fg_dest_dir, dataset_type)
    bg_out_dir = os.path.join(bg_dest_dir, dataset_type)

    csv_path = os.path.join(dataset_path, f"{dataset_type}.csv")
    if not os.path.exists(csv_path):
        logger.warning("Missing CSV: %s — skipping", csv_path)
        return

    dataset = pd.read_csv(csv_path)

    logger.info("Consolidating %s/%s ...", dataset_name, dataset_type)

    foreground_ids = list(id2classname.keys())

    for _, sample_data in tqdm(dataset.iterrows(), total=len(dataset)):
        sample_id = sample_data["id"]

        if sample_id in id2classname:
            out_dir = fg_out_dir
            classname = id2classname[sample_id]
        elif is_valid_background(sample_id, foreground_ids, ontology):
            out_dir = bg_out_dir
            classname = ontology.get_label(sample_id)
        else:
            continue

        out_path = os.path.join(out_dir, classname)
        os.makedirs(out_path, exist_ok=True)

        # Relative symlink: ../../..<dataset_name>/<fname>
        src = os.path.join("..", "..", "..", dataset_name, sample_data["fname"])
        fname = dataset_name.lower() + "_" + os.path.basename(sample_data["fname"])
        dest = os.path.join(out_path, fname)

        if dry_run:
            logger.debug("Would symlink %s → %s", src, dest)
            continue

        start_sample, first_silence, end_sample = trim_silence(
            os.path.join(dataset_path, sample_data["fname"])
        )
        assert start_sample < end_sample
        all_samples.append(
            {
                "fname": dest,
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "first_silence": first_silence,
            }
        )
        if not os.path.exists(dest):
            os.symlink(src, dest)


# ---------------------------------------------------------------------------
# HRTF handling
# ---------------------------------------------------------------------------


def prepare_hrtf(raw_dir: str, output_dir: str) -> None:
    """Copy CIPIC SOFA files and generate train/val/test split lists."""
    cipic_src = os.path.join(raw_dir, "CIPIC-HRTF", "CIPIC_SOFA")
    cipic_dst = os.path.join(output_dir, "hrtf", "CIPIC")
    os.makedirs(cipic_dst, exist_ok=True)

    sofa_files = sorted(glob.glob(os.path.join(cipic_src, "**", "*.sofa"), recursive=True))

    if not sofa_files:
        logger.warning("No CIPIC SOFA files found in %s", cipic_src)
        return

    # Copy files
    copied: list[str] = []
    for src in sofa_files:
        dst = os.path.join(cipic_dst, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        copied.append(os.path.basename(src))

    # 80:10:10 split
    random.shuffle(copied)
    n = len(copied)
    n_train = int(round(0.8 * n))
    n_val = int(round(0.1 * n))

    train_files = copied[:n_train]
    val_files = copied[n_train : n_train + n_val]
    test_files = copied[n_train + n_val :]

    hrtf_dir = os.path.join(output_dir, "hrtf")
    for name, file_list in [
        ("train_hrtf.txt", train_files),
        ("val_hrtf.txt", val_files),
        ("test_hrtf.txt", test_files),
    ]:
        with open(os.path.join(hrtf_dir, name), "w") as f:
            f.write("\n".join(file_list) + "\n")

    logger.info(
        "HRTF: %d SOFA files → train=%d  val=%d  test=%d",
        n,
        len(train_files),
        len(val_files),
        len(test_files),
    )


# ---------------------------------------------------------------------------
# id2classname construction
# ---------------------------------------------------------------------------


def _get_subtree(node_id: str, ontology_dict: dict[str, dict]) -> list[str]:
    subtree = [node_id]
    for child_id in ontology_dict[node_id]["child_ids"]:
        subtree.extend(_get_subtree(child_id, ontology_dict))
    return subtree


def build_id2classname(
    class_map_path: str, ontology_json_path: str
) -> dict[str, str]:
    """Build a mapping from AudioSet IDs to our dataset class names.

    Uses the class map YAML (``class_map.yaml``) and the full AudioSet
    ontology to expand each named class into its complete subtree.
    """
    with open(class_map_path) as f:
        class_data: dict[str, list[str]] = yaml.safe_load(f)

    with open(ontology_json_path) as f:
        ontology_list = json.load(f)

    ontology_dict = {}
    for item in ontology_list:
        ontology_dict[item["id"]] = item
        ontology_dict[item["name"]] = item

    id2classname: dict[str, str] = {}
    for class_name, elements in class_data.items():
        for element_name in elements:
            class_id = ontology_dict[element_name]["id"]
            class_ids = _get_subtree(class_id, ontology_dict)
            for cid in class_ids:
                id2classname[cid] = class_name

    return id2classname


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate audio datasets into Scaper format."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Root directory containing raw datasets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the consolidated dataset.",
    )
    parser.add_argument(
        "--class_map",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "class_map.yaml"),
        help="Path to class_map.yaml.",
    )
    parser.add_argument(
        "--ontology",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "ontology.json"),
        help="Path to ontology.json.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print actions without writing files.",
    )
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    ontology = Ontology(args.ontology)

    # ------------------------------------------------------------------
    # Step 1: Run per-dataset collectors
    # ------------------------------------------------------------------
    logger.info("=== Step 1: Running dataset collectors ===")

    # Each raw_dir sub-path matches the actual download directory layout:
    #   raw_dir/FSD50K/{FSD50K.metadata,FSD50K.dev_audio,...}
    #   raw_dir/ESC-50/ESC-50-master/{audio,meta}
    #   raw_dir/musdb18/{train,test}
    #   raw_dir/DISCO/{train,test}/{label}
    #   raw_dir/TAU-2019/TAU-urban-acoustic-scenes-2019-development/{audio,meta.csv}
    fsd50k_dir = os.path.join(args.raw_dir, "FSD50K")
    esc50_dir = os.path.join(args.raw_dir, "ESC-50", "ESC-50-master")
    musdb18_dir = args.raw_dir  # collector internally appends "musdb18"
    disco_dir = args.raw_dir  # collector internally appends "disco_noises" → remapped to DISCO
    tau_dir = os.path.join(args.raw_dir, "TAU-2019")

    FSD50KCollector(fsd50k_dir, ontology).collect(fsd50k_dir, args.output_dir)
    ESC50Collector(ontology).collect(esc50_dir, args.output_dir)
    try:
        MUSDB18Collector(ontology).collect(musdb18_dir, args.output_dir)
    except Exception as exc:
        logger.warning(
            "MUSDB18 collection failed (ffmpeg may be unavailable): %s", exc
        )
    DISCOCollector(ontology).collect(disco_dir, args.output_dir)
    TAUCollector().collect(tau_dir, args.output_dir)

    # ------------------------------------------------------------------
    # Step 2: Build id → classname mapping
    # ------------------------------------------------------------------
    logger.info("=== Step 2: Building id2classname mapping ===")
    id2classname = build_id2classname(args.class_map, args.ontology)
    logger.info("Mapped %d AudioSet IDs to %d class names",
                len(id2classname), len(set(id2classname.values())))

    # ------------------------------------------------------------------
    # Step 3: Consolidate into Scaper format
    # ------------------------------------------------------------------
    logger.info("=== Step 3: Consolidating into Scaper format ===")

    fg_output_dir = os.path.join(args.output_dir, "scaper_fmt")
    bg_output_dir = os.path.join(args.output_dir, "bg_scaper_fmt")

    datasets = ["FSD50K", "ESC-50", "musdb18", "disco_noises"]
    dataset_types = ["train", "val", "test"]
    all_samples: list[dict] = []

    for dataset_name in datasets:
        for dataset_type in dataset_types:
            write_scaper_source(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                base_dir=args.output_dir,
                fg_dest_dir=fg_output_dir,
                bg_dest_dir=bg_output_dir,
                id2classname=id2classname,
                ontology=ontology,
                all_samples=all_samples,
                dry_run=args.dry_run,
            )

    # ------------------------------------------------------------------
    # Step 4: HRTF preparation
    # ------------------------------------------------------------------
    logger.info("=== Step 4: HRTF preparation ===")
    prepare_hrtf(args.raw_dir, args.output_dir)

    # ------------------------------------------------------------------
    # Step 5: Save start_times.csv
    # ------------------------------------------------------------------
    if all_samples:
        df = pd.DataFrame.from_records(all_samples)
        df.to_csv(os.path.join(args.output_dir, "start_times.csv"), index=False)
        logger.info("Saved start_times.csv with %d entries", len(df))

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
