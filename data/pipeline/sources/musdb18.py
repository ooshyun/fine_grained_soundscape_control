from __future__ import annotations
from pathlib import Path
from .base import BaseSource


class MUSDB18Source(BaseSource):
    name = "musdb18"
    key = "musdb18"
    ZENODO_URL = "https://zenodo.org/records/1117372/files/musdb18.zip?download=1"

    def __init__(self, ontology):
        self.ontology = ontology

    def print_download_guide(self) -> None:
        print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  musdb18 — Manual Download Required                     │
  │                                                         │
  │  License: Academic/non-commercial only                  │
  │  URL: {self.ZENODO_URL}  │
  │                                                         │
  │  Download and extract to:                               │
  │    <manual_dir>/musdb18/                                │
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
        zip_path = out / "musdb18.zip"
        if not zip_path.exists():
            print(f"  Downloading from Zenodo ...")
            urllib.request.urlretrieve(self.ZENODO_URL, str(zip_path))
        print(f"  Extracting ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out)
        zip_path.unlink()
        (out / ".done").touch()
        print(f"  ✓ {self.name} downloaded")

    def collect(self, raw_dir: Path, curated_dir: Path) -> None:
        raise NotImplementedError
