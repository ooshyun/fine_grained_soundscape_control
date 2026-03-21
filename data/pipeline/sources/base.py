from __future__ import annotations
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class BaseSource(ABC):
    """Common interface for all dataset sources."""

    name: str  # e.g. "FSD50K"
    key: str   # e.g. "fsd50k"

    # Mapping from source name to reference CSV directory name.
    # Override in subclass if the reference CSV dir differs from self.name.
    ref_csv_dir: str | None = None

    @abstractmethod
    def download(self, raw_dir: Path) -> None:
        """Download raw dataset files into raw_dir/{self.name}/."""

    @abstractmethod
    def collect(self, raw_dir: Path, curated_dir: Path) -> None:
        """Produce {train,val,test}.csv in curated_dir/{self.name}/."""

    def try_use_reference_csvs(
        self, curated_dir: Path, reference_dir: Path | None
    ) -> bool:
        """Copy reference CSVs if available. Returns True if copied."""
        if reference_dir is None:
            return False
        csv_dir_name = self.ref_csv_dir or self.name
        ref = reference_dir / csv_dir_name
        if not ref.exists():
            return False
        csvs = [ref / f"{s}.csv" for s in ("train", "val", "test")]
        if not all(c.exists() for c in csvs):
            return False
        out_dir = curated_dir / self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for c in csvs:
            shutil.copy2(c, out_dir / c.name)
        for s in ("train", "val", "test"):
            df = pd.read_csv(out_dir / f"{s}.csv")
            print(f"  {s}: {len(df)} samples (reference)")
        return True

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
