from __future__ import annotations
from pathlib import Path
import pandas as pd
import soundfile as sf
from datasets import load_dataset
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
        raise NotImplementedError
