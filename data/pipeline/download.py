from __future__ import annotations
from pathlib import Path
from .sources.base import BaseSource

ZENODO_DATASETS = {"musdb18", "tau"}


def run_download(
    sources: dict[str, BaseSource],
    raw_dir: Path,
    manual_dir: Path | None = None,
    dry_run: bool = False,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    for key, source in sources.items():
        print(f"\n{'='*60}")
        print(f"  [{source.key}] {source.name}")
        print(f"{'='*60}")

        if key in ZENODO_DATASETS:
            if manual_dir and (manual_dir / source.name).exists():
                print(f"  Found in manual_dir: {manual_dir / source.name}")
                dst = raw_dir / source.name
                if not dst.exists():
                    dst.symlink_to((manual_dir / source.name).resolve())
                continue
            else:
                print(f"  ⚠ {source.name} requires manual download.")
                source.print_download_guide()
                continue

        if dry_run:
            print(f"  [dry-run] Would download {source.name}")
            continue

        source.download(raw_dir)
    print("\nDownload stage complete.")
