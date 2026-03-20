from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class BaseSource(ABC):
    """Common interface for all dataset sources."""

    name: str  # e.g. "FSD50K"
    key: str   # e.g. "fsd50k"

    @abstractmethod
    def download(self, raw_dir: Path) -> None:
        """Download raw dataset files into raw_dir/{self.name}/."""

    @abstractmethod
    def collect(self, raw_dir: Path, curated_dir: Path) -> None:
        """Produce {train,val,test}.csv in curated_dir/{self.name}/."""

    def print_download_guide(self) -> None:
        """Print manual download instructions. Override for Zenodo datasets."""
        print(f"  No manual download needed for {self.name}.")

    def _write_csvs(
        self,
        out_dir: Path,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, df in [("train", train), ("val", val), ("test", test)]:
            df.to_csv(out_dir / f"{split}.csv", index=False)
            print(f"  {split}: {len(df)} samples")
