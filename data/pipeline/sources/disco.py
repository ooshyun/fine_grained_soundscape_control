from __future__ import annotations
from pathlib import Path
import shutil
from .base import BaseSource


class DISCOSource(BaseSource):
    name = "disco_noises"
    key = "disco"

    def __init__(self, ontology):
        self.ontology = ontology

    def download(self, raw_dir: Path) -> None:
        out = raw_dir / self.name
        if (out / ".done").exists():
            print(f"  [skip] {self.name} already downloaded")
            return
        print("  Loading from HF: ooshyun/soundscape-control-data (disco) ...")
        from huggingface_hub import snapshot_download
        tmp = raw_dir / "_hf_download"
        snapshot_download(
            repo_id="ooshyun/soundscape-control-data",
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
        raise NotImplementedError
