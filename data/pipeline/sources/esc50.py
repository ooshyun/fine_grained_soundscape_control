from __future__ import annotations
from pathlib import Path
from .base import BaseSource


class ESC50Source(BaseSource):
    name = "ESC-50"
    key = "esc50"

    def __init__(self, ontology) -> None:
        self.ontology = ontology

    def download(self, raw_dir: Path) -> None:
        raise NotImplementedError

    def collect(self, raw_dir: Path, curated_dir: Path) -> None:
        raise NotImplementedError
