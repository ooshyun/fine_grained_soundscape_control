from __future__ import annotations
from pathlib import Path
from .base import BaseSource


class TAUSource(BaseSource):
    name = "TAU-2019"
    key = "tau"

    def download(self, raw_dir: Path) -> None:
        raise NotImplementedError

    def collect(self, raw_dir: Path, curated_dir: Path) -> None:
        raise NotImplementedError
