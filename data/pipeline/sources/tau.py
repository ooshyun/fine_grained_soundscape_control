from __future__ import annotations
from pathlib import Path
from .base import BaseSource


class TAUSource(BaseSource):
    name = "TAU-2019"
    key = "tau"
    ZENODO_BASE = "https://zenodo.org/records/2589280/files"

    def print_download_guide(self) -> None:
        print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  TAU Urban Acoustic Scenes 2019                         │
  │                                                         │
  │  License: Tampere Univ. custom (non-commercial)         │
  │  Size: ~20 GB (10 audio zips + meta)                    │
  │                                                         │
  │  Download from: {self.ZENODO_BASE}                      │
  │  Files: audio.1.zip ~ audio.10.zip + meta.zip           │
  │                                                         │
  │  Extract all to:                                        │
  │    <manual_dir>/TAU-2019/                               │
  │                                                         │
  │  Then re-run with --manual_dir <path>                   │
  └─────────────────────────────────────────────────────────┘""")

    def download(self, raw_dir: Path) -> None:
        import ssl, urllib.request, zipfile
        ssl._create_default_https_context = ssl._create_unverified_context
        out = raw_dir / self.name
        if (out / ".done").exists():
            print(f"  [skip] {self.name} already downloaded")
            return
        out.mkdir(parents=True, exist_ok=True)
        files = [
            f"TAU-urban-acoustic-scenes-2019-development.audio.{i}.zip"
            for i in range(1, 11)
        ] + ["TAU-urban-acoustic-scenes-2019-development.meta.zip"]
        for fname in files:
            url = f"{self.ZENODO_BASE}/{fname}?download=1"
            zip_path = out / fname
            if not zip_path.exists():
                print(f"  Downloading {fname} ...")
                urllib.request.urlretrieve(url, str(zip_path))
            print(f"  Extracting {fname} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out)
            zip_path.unlink()
        (out / ".done").touch()
        print(f"  ✓ {self.name} downloaded")

    def collect(self, raw_dir: Path, curated_dir: Path) -> None:
        raise NotImplementedError
