from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .base import BaseSource

logger = logging.getLogger(__name__)


class FSD50KSource(BaseSource):
    name = "FSD50K"
    key = "fsd50k"

    def __init__(self, ontology):
        self.ontology = ontology
        self.pp_pnp_ratings: dict | None = None

    def download(self, raw_dir: Path) -> None:
        """Download FSD50K.

        Supports two modes:
        1. If raw data already exists (Zenodo layout), skip.
        2. Otherwise, download from HF mirror Fhrozen/FSD50k.
        """
        out = raw_dir / self.name
        # Check for Zenodo layout
        if (out / "FSD50K.metadata" / "pp_pnp_ratings_FSD50K.json").exists():
            print(f"  [skip] {self.name} already exists (Zenodo layout)")
            return
        if (out / ".done").exists():
            print(f"  [skip] {self.name} already downloaded")
            return
        print("  Loading from HF: Fhrozen/FSD50k ...")
        import soundfile as sf
        from datasets import load_dataset

        ds = load_dataset("Fhrozen/FSD50k")
        rows = []
        for split_name in ds:
            audio_dir = out / f"FSD50K.{split_name}_audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            for row in ds[split_name]:
                fname = f"{row['filename']}.wav"
                audio_path = audio_dir / fname
                if not audio_path.exists():
                    audio = row["audio"]
                    sf.write(str(audio_path), audio["array"], audio["sampling_rate"])
                rows.append({
                    "filename": row["filename"],
                    "split": split_name,
                    "labels": row.get("labels", ""),
                    "mids": row.get("mids", ""),
                    "fname": f"FSD50K.{split_name}_audio/{fname}",
                })
        pd.DataFrame(rows).to_csv(out / "metadata.csv", index=False)
        (out / ".done").touch()
        print(f"  ✓ {self.name} downloaded ({len(rows)} samples)")

    # ------------------------------------------------------------------
    # Quality filtering (for Zenodo layout with pp_pnp_ratings)
    # ------------------------------------------------------------------

    def _is_pp_sample(self, fname: str) -> bool:
        """Return True if every label has >=2 positive and 0 negative/uncertain."""
        if self.pp_pnp_ratings is None:
            return True
        label_ratings = self.pp_pnp_ratings.get(fname, {})
        for node_id in label_ratings:
            ratings = label_ratings[node_id]
            counts = {1.0: 0, 0.5: 0, 0: 0, -1: 0}
            for r in ratings:
                counts[r] += 1
            if counts[0.0] > 0 or counts[-1] > 0 or counts[1.0] < 2:
                return False
        return True

    def _curate_samples(self, samples: pd.DataFrame) -> pd.DataFrame:
        samples = samples.dropna().copy()
        samples["fname"] = samples["fname"].apply(str)
        samples["mids"] = samples["mids"].apply(lambda x: x.split(","))
        samples["labels"] = samples["labels"].apply(lambda x: x.split(","))

        # Quality filter
        samples["pp_sample"] = samples.apply(
            lambda x: self._is_pp_sample(x["fname"]), axis=1
        )
        samples = samples[samples["pp_sample"]]

        # Single-label only
        samples = samples[samples["mids"].apply(lambda x: len(x) == 1)]

        samples["id"] = samples["mids"].apply(lambda x: x[0])
        samples["label"] = samples["id"].apply(self.ontology.get_label)
        return samples

    # ------------------------------------------------------------------
    # Collect
    # ------------------------------------------------------------------

    def collect(self, raw_dir: Path, curated_dir: Path) -> None:
        dataset_dir = raw_dir / self.name
        out_dir = curated_dir / self.name

        # Detect layout: Zenodo (has collection CSVs) or HF (has metadata.csv)
        zenodo_meta = dataset_dir / "FSD50K.metadata" / "collection" / "collection_dev.csv"
        hf_meta = dataset_dir / "metadata.csv"

        if zenodo_meta.exists():
            self._collect_zenodo(dataset_dir, out_dir)
        elif hf_meta.exists():
            self._collect_hf(dataset_dir, out_dir)
        else:
            raise FileNotFoundError(
                f"Cannot find FSD50K metadata in {dataset_dir}. "
                "Expected either Zenodo layout (FSD50K.metadata/collection/) "
                "or HF layout (metadata.csv)."
            )

    def _collect_zenodo(self, dataset_dir: Path, out_dir: Path) -> None:
        """Collect from original Zenodo layout with quality filtering."""
        # Load pp_pnp ratings
        ratings_path = dataset_dir / "FSD50K.metadata" / "pp_pnp_ratings_FSD50K.json"
        with open(ratings_path) as f:
            self.pp_pnp_ratings = json.load(f)

        dev_samples = pd.read_csv(
            dataset_dir / "FSD50K.metadata" / "collection" / "collection_dev.csv"
        )
        eval_samples = pd.read_csv(
            dataset_dir / "FSD50K.metadata" / "collection" / "collection_eval.csv"
        )

        dev_curated = self._curate_samples(dev_samples)
        test_samples = self._curate_samples(eval_samples)

        # Stratified 90:10 train:val split per label
        train_frames: list[pd.DataFrame] = []
        val_frames: list[pd.DataFrame] = []
        for label in sorted(dev_curated["label"].unique()):
            subset = dev_curated[dev_curated["label"] == label]
            if len(subset) <= 1:
                continue
            tr, va = train_test_split(subset, test_size=0.1, random_state=42)
            train_frames.append(tr)
            val_frames.append(va)

        if not train_frames:
            raise ValueError("No training samples after filtering")
        train_samples = pd.concat(train_frames)
        val_samples = pd.concat(val_frames)

        # Relative file paths
        for df, src_dir in [
            (train_samples, "FSD50K.dev_audio"),
            (val_samples, "FSD50K.dev_audio"),
            (test_samples, "FSD50K.eval_audio"),
        ]:
            df["fname"] = df["fname"].apply(
                lambda x, d=src_dir: f"{d}/{x}.wav"
            )

        # Common label constraint
        common_labels = (
            set(train_samples["label"].unique())
            & set(val_samples["label"].unique())
            & set(test_samples["label"].unique())
        )
        train_samples = train_samples[train_samples["label"].isin(common_labels)]
        val_samples = val_samples[val_samples["label"].isin(common_labels)]
        test_samples = test_samples[test_samples["label"].isin(common_labels)]

        cols = ["fname", "label", "id"]
        self._write_csvs(out_dir, train_samples[cols], val_samples[cols], test_samples[cols])
        print(
            f"  FSD50K (Zenodo): train={len(train_samples)}  "
            f"val={len(val_samples)}  test={len(test_samples)}"
        )

    def _collect_hf(self, dataset_dir: Path, out_dir: Path) -> None:
        """Collect from HF-downloaded metadata.csv (no quality filtering)."""
        meta = pd.read_csv(dataset_dir / "metadata.csv")
        meta = meta.dropna(subset=["mids"]).copy()
        meta["mids"] = meta["mids"].astype(str)

        # Single-label filter
        meta = meta[~meta["mids"].str.contains(",", na=False)]

        meta["id"] = meta["mids"].str.strip()
        meta["label"] = meta["id"].apply(self.ontology.get_label)

        dev = meta[meta["split"] == "dev"].copy()
        test_samples = meta[meta["split"] == "eval"].copy()

        train_frames: list[pd.DataFrame] = []
        val_frames: list[pd.DataFrame] = []
        for label in sorted(dev["label"].unique()):
            subset = dev[dev["label"] == label]
            if len(subset) <= 1:
                continue
            tr, va = train_test_split(subset, test_size=0.1, random_state=42)
            train_frames.append(tr)
            val_frames.append(va)

        if not train_frames:
            raise ValueError("No training samples after filtering")
        train_samples = pd.concat(train_frames)
        val_samples = pd.concat(val_frames)

        common_labels = (
            set(train_samples["label"].unique())
            & set(val_samples["label"].unique())
            & set(test_samples["label"].unique())
        )
        train_samples = train_samples[train_samples["label"].isin(common_labels)]
        val_samples = val_samples[val_samples["label"].isin(common_labels)]
        test_samples = test_samples[test_samples["label"].isin(common_labels)]

        cols = ["fname", "label", "id"]
        self._write_csvs(out_dir, train_samples[cols], val_samples[cols], test_samples[cols])
        print(
            f"  FSD50K (HF): train={len(train_samples)}  "
            f"val={len(val_samples)}  test={len(test_samples)}"
        )
