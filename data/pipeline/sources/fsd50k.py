from __future__ import annotations
from pathlib import Path
import pandas as pd
import soundfile as sf
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from .base import BaseSource


class FSD50KSource(BaseSource):
    name = "FSD50K"
    key = "fsd50k"

    def __init__(self, ontology):
        self.ontology = ontology

    def download(self, raw_dir: Path) -> None:
        out = raw_dir / self.name
        if (out / ".done").exists():
            print(f"  [skip] {self.name} already downloaded")
            return
        print("  Loading from HF: Fhrozen/FSD50k ...")
        ds = load_dataset("Fhrozen/FSD50k", trust_remote_code=True)
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

    def collect(self, raw_dir: Path, curated_dir: Path) -> None:
        dataset_dir = raw_dir / self.name
        out_dir = curated_dir / self.name

        meta = pd.read_csv(dataset_dir / "metadata.csv")
        meta = meta.dropna(subset=["mids"]).copy()
        meta["mids"] = meta["mids"].astype(str)

        # Single-label filter: keep rows where mids has exactly 1 AudioSet ID
        meta = meta[~meta["mids"].str.contains(",", na=False)]

        # Label resolution
        meta["id"] = meta["mids"].str.strip()
        meta["label"] = meta["id"].apply(self.ontology.get_label)

        # Split: dev → train/val (90:10 stratified), eval → test
        dev = meta[meta["split"] == "dev"].copy()
        test_samples = meta[meta["split"] == "eval"].copy()

        train_frames: list[pd.DataFrame] = []
        val_frames: list[pd.DataFrame] = []
        for label in sorted(dev["label"].unique()):
            subset = dev[dev["label"] == label]
            if len(subset) <= 1:
                continue
            tr, va = train_test_split(
                subset, test_size=0.1, random_state=42,
            )
            train_frames.append(tr)
            val_frames.append(va)

        if not train_frames:
            raise ValueError("No training samples after filtering")
        train_samples = pd.concat(train_frames)
        val_samples = pd.concat(val_frames)

        # Common label constraint: only keep labels in ALL 3 splits
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
            f"  FSD50K: train={len(train_samples)}  "
            f"val={len(val_samples)}  test={len(test_samples)}"
        )
