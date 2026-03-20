from __future__ import annotations
from pathlib import Path
import pandas as pd
import soundfile as sf
from datasets import load_dataset
from .base import BaseSource


class ESC50Source(BaseSource):
    name = "ESC-50"
    key = "esc50"

    def __init__(self, ontology):
        self.ontology = ontology

    def download(self, raw_dir: Path) -> None:
        out = raw_dir / self.name
        if (out / ".done").exists():
            print(f"  [skip] {self.name} already downloaded")
            return
        print("  Loading from HF: ashraq/esc50 ...")
        ds = load_dataset("ashraq/esc50")
        audio_dir = out / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for row in ds["train"]:  # ESC-50 on HF is single split
            fname = row["filename"]
            audio_path = audio_dir / fname
            if not audio_path.exists():
                audio = row["audio"]
                sf.write(str(audio_path), audio["array"], audio["sampling_rate"])
            rows.append({
                "filename": fname,
                "fold": row["fold"],
                "target": row["target"],
                "category": row["category"],
                "esc10": row.get("esc10", False),
            })
        meta_dir = out / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(meta_dir / "esc50.csv", index=False)
        (out / ".done").touch()
        print(f"  ✓ {self.name} downloaded ({len(rows)} samples)")

    def collect(self, raw_dir: Path, curated_dir: Path) -> None:
        raise NotImplementedError
