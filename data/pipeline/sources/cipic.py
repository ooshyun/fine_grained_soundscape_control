from __future__ import annotations
from pathlib import Path
import shutil
from .base import BaseSource


class CIPICSource(BaseSource):
    name = "cipic-hrtf-database"
    key = "cipic"

    def download(self, raw_dir: Path) -> None:
        out = raw_dir / self.name
        if (out / ".done").exists():
            print(f"  [skip] {self.name} already downloaded")
            return
        print("  Loading from HF: ooshyun/fine_grained_soundscape_control (cipic) ...")
        from huggingface_hub import snapshot_download
        tmp = raw_dir / "_hf_download_cipic"
        snapshot_download(
            repo_id="ooshyun/fine_grained_soundscape_control",
            repo_type="dataset",
            allow_patterns="cipic_hrtf/**",
            local_dir=str(tmp),
        )
        src = tmp / "cipic_hrtf"
        if src.exists():
            if out.exists():
                shutil.rmtree(out)
            shutil.move(str(src), str(out))
        shutil.rmtree(tmp, ignore_errors=True)
        (out / ".done").touch()
        print(f"  ✓ {self.name} downloaded")

    def collect(self, raw_dir: Path, curated_dir: Path) -> None:
        """CIPIC has no collect stage — HRTF prep is done in prepare stage."""
        print(f"  [skip] {self.name} — handled in prepare stage (HRTF)")
