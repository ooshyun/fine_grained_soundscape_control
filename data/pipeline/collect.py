from __future__ import annotations
from pathlib import Path
from .sources.base import BaseSource


def run_collect(
    sources: dict[str, BaseSource],
    raw_dir: Path,
    curated_dir: Path,
) -> None:
    curated_dir.mkdir(parents=True, exist_ok=True)
    for key, source in sources.items():
        src_dir = raw_dir / source.name
        if not src_dir.exists():
            print(f"  [skip] {source.name} not found in {raw_dir}")
            continue
        print(f"\n  Collecting {source.name} ...")
        source.collect(raw_dir, curated_dir)
    print("\nCollect stage complete.")
