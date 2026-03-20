#!/usr/bin/env python3
from __future__ import annotations
"""Download script for public audio datasets used in the project.

Supported datasets:
  - FSD50K (Freesound Dataset 50K)
  - ESC-50 (Environmental Sound Classification)
  - musdb18 (Music Source Separation)
  - DISCO (Domestic Indoor Sound Collection of Noises)
  - TAU-2019 (TAU Urban Acoustic Scenes 2019)
  - CIPIC HRTF (CIPIC Head-Related Transfer Function Database)

Usage:
  python data/download.py --output_dir ./raw_datasets
  python data/download.py --output_dir ./raw_datasets --datasets fsd50k,esc50
  python data/download.py --dry-run
"""

import argparse
import os
import ssl
import urllib.request
import zipfile
from pathlib import Path

# Workaround for SSL certificate issues on HPC clusters
ssl._create_default_https_context = ssl._create_unverified_context

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "fsd50k": {
        "name": "FSD50K",
        "files": [
            # dev_audio is a 6-part split zip (~18GB total)
            {"url": "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.zip?download=1", "filename": "FSD50K.dev_audio.zip", "no_extract": True},
            {"url": "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z01?download=1", "filename": "FSD50K.dev_audio.z01", "no_extract": True},
            {"url": "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z02?download=1", "filename": "FSD50K.dev_audio.z02", "no_extract": True},
            {"url": "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z03?download=1", "filename": "FSD50K.dev_audio.z03", "no_extract": True},
            {"url": "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z04?download=1", "filename": "FSD50K.dev_audio.z04", "no_extract": True},
            {"url": "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z05?download=1", "filename": "FSD50K.dev_audio.z05", "no_extract": True},
            # eval_audio is a 2-part split zip (~6.3GB total)
            {"url": "https://zenodo.org/records/4060432/files/FSD50K.eval_audio.zip?download=1", "filename": "FSD50K.eval_audio.zip", "no_extract": True},
            {"url": "https://zenodo.org/records/4060432/files/FSD50K.eval_audio.z01?download=1", "filename": "FSD50K.eval_audio.z01", "no_extract": True},
            # metadata + ground_truth + doc (regular zips)
            {"url": "https://zenodo.org/records/4060432/files/FSD50K.metadata.zip?download=1", "filename": "FSD50K.metadata.zip"},
            {"url": "https://zenodo.org/records/4060432/files/FSD50K.ground_truth.zip?download=1", "filename": "FSD50K.ground_truth.zip"},
            {"url": "https://zenodo.org/records/4060432/files/FSD50K.doc.zip?download=1", "filename": "FSD50K.doc.zip"},
        ],
        # Post-download: merge split zips and extract
        "post_download": "fsd50k_merge",
    },
    "esc50": {
        "name": "ESC-50",
        "files": [
            {
                "url": "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip",
                "filename": "ESC-50-master.zip",
            },
        ],
    },
    "musdb18": {
        "name": "musdb18",
        "files": [
            {
                "url": "https://zenodo.org/records/1117372/files/musdb18.zip?download=1",
                "filename": "musdb18.zip",
            },
        ],
    },
    "disco": {
        "name": "DISCO",
        "source": "huggingface",
        "hf_repo": "ooshyun/fine-grained-soundscape",
        "hf_path": "disco",
        "files": [],
    },
    "tau": {
        "name": "TAU-2019",
        "files": [
            {
                "url": f"https://zenodo.org/records/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.{i}.zip?download=1",
                "filename": f"TAU-urban-acoustic-scenes-2019-development.audio.{i}.zip",
            }
            for i in range(1, 11)
        ]
        + [
            {
                "url": "https://zenodo.org/records/2589280/files/TAU-urban-acoustic-scenes-2019-development.meta.zip?download=1",
                "filename": "TAU-urban-acoustic-scenes-2019-development.meta.zip",
            },
        ],
    },
    "cipic": {
        "name": "CIPIC-HRTF",
        "source": "huggingface",
        "hf_repo": "ooshyun/fine-grained-soundscape",
        "hf_path": "cipic_hrtf",
        "files": [],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DownloadProgressBar(tqdm):
    """tqdm wrapper that hooks into urllib reporthook."""

    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int = -1):
        if total_size > 0:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def _remote_file_size(url: str) -> int | None:
    """Return Content-Length from a HEAD request, or None if unavailable."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def _download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _DownloadProgressBar(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=dest.name
    ) as pbar:
        urllib.request.urlretrieve(url, str(dest), reporthook=pbar.update_to)


def _extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract a zip archive into *extract_dir*.

    Falls back to the system ``unzip`` command when Python's ``zipfile``
    cannot handle the archive (e.g. multi-disk / ZIP64 spans).
    """
    import subprocess

    print(f"  Extracting {zip_path.name} -> {extract_dir}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        print(f"  Python zipfile failed, falling back to system unzip...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["unzip", "-o", "-q", str(zip_path), "-d", str(extract_dir)],
            check=True,
        )


def _merge_split_zip(dataset_dir: Path, base_name: str) -> None:
    """Merge split zip parts and extract. E.g. FSD50K.dev_audio.zip + .z01-.z05"""
    import subprocess

    zip_main = dataset_dir / f"{base_name}.zip"
    if not zip_main.exists():
        print(f"  [skip merge] {base_name}.zip not found")
        return

    # Check if already extracted (look for the expected directory)
    expected_dir = dataset_dir / base_name.replace(".", "_").replace("FSD50K_", "FSD50K.")
    # Actually just check for any extracted content
    print(f"  Merging split zip: {base_name}...")
    combined = dataset_dir / f"{base_name}_combined.zip"
    if not combined.exists():
        subprocess.run(
            ["zip", "-s", "0", str(zip_main), "--out", str(combined)],
            cwd=str(dataset_dir),
            check=True,
        )
    print(f"  Extracting {combined.name}...")
    subprocess.run(
        ["unzip", "-o", "-q", str(combined), "-d", str(dataset_dir)],
        check=True,
    )
    # Clean up
    combined.unlink(missing_ok=True)
    print(f"  Merge + extract done for {base_name}")


def _download_from_huggingface(
    repo_id: str, hf_path: str, dataset_dir: Path,
) -> None:
    """Download a subfolder from a HuggingFace dataset repo."""
    from huggingface_hub import snapshot_download

    print(f"  Downloading from HuggingFace: {repo_id}/{hf_path}")
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=f"{hf_path}/**",
        local_dir=str(dataset_dir.parent),
    )
    # snapshot_download puts files under local_dir/hf_path/
    # Move to dataset_dir if needed
    src = Path(local_dir) / hf_path
    if src.exists() and src != dataset_dir:
        # If dataset_dir already has the hf_path name, it's fine
        if not dataset_dir.exists():
            src.rename(dataset_dir)
        else:
            import shutil
            # Merge into dataset_dir
            for item in src.iterdir():
                dest = dataset_dir / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.move(str(item), str(dest))
                else:
                    shutil.move(str(item), str(dest))
    print(f"  Done: {dataset_dir}")


def _sizeof_fmt(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown size"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def dry_run(selected: list[str]) -> None:
    """Print URLs and estimated sizes without downloading."""
    for key in selected:
        info = DATASETS[key]
        print(f"\n{'='*60}")
        print(f"  {info['name']}")
        print(f"{'='*60}")
        if info.get("source") == "huggingface":
            print(f"  [HuggingFace] {info['hf_repo']} / {info['hf_path']}")
            continue
        for f in info["files"]:
            size = _remote_file_size(f["url"])
            print(f"  {f['filename']:60s}  {_sizeof_fmt(size)}")
            print(f"    {f['url']}")


def download_datasets(
    selected: list[str],
    output_dir: Path,
    keep_zips: bool = False,
) -> None:
    """Download and extract selected datasets."""
    for key in selected:
        info = DATASETS[key]
        dataset_dir = output_dir / info["name"]
        dataset_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Downloading {info['name']} -> {dataset_dir}")
        print(f"{'='*60}")

        # HuggingFace dataset download
        if info.get("source") == "huggingface":
            _download_from_huggingface(
                info["hf_repo"], info["hf_path"], dataset_dir,
            )
            continue

        for f in info["files"]:
            dest = dataset_dir / f["filename"]

            try:
                # Resume support: skip if file exists with matching size
                if dest.exists():
                    expected = _remote_file_size(f["url"])
                    if expected is not None and dest.stat().st_size == expected:
                        print(f"  [skip] {f['filename']} already downloaded")
                    elif expected is None:
                        print(f"  [skip] {f['filename']} exists (size unverified)")
                    else:
                        _download_file(f["url"], dest)
                else:
                    _download_file(f["url"], dest)

                # Extract (unless marked no_extract for split zips)
                if dest.suffix == ".zip" and not f.get("no_extract", False):
                    _extract_zip(dest, dataset_dir)
                    if not keep_zips:
                        dest.unlink()
                        print(f"  Removed {f['filename']}")
            except Exception as exc:
                print(f"  [ERROR] {f['filename']}: {exc}")
                print(f"  Continuing with next file...")

        # Post-download hooks (e.g. merge split zips)
        post = info.get("post_download")
        if post == "fsd50k_merge":
            try:
                _merge_split_zip(dataset_dir, "FSD50K.dev_audio")
                _merge_split_zip(dataset_dir, "FSD50K.eval_audio")
            except Exception as exc:
                print(f"  [ERROR] FSD50K merge: {exc}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public audio datasets for training."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./raw_datasets"),
        help="Root directory to store downloaded datasets (default: ./raw_datasets)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Comma-separated list of datasets to download. "
            f"Options: {','.join(DATASETS.keys())}. Default: all."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs and expected sizes without downloading.",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep zip files after extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.datasets:
        selected = [d.strip().lower() for d in args.datasets.split(",")]
        unknown = [d for d in selected if d not in DATASETS]
        if unknown:
            raise ValueError(
                f"Unknown dataset(s): {unknown}. "
                f"Choose from: {list(DATASETS.keys())}"
            )
    else:
        selected = list(DATASETS.keys())

    if args.dry_run:
        dry_run(selected)
    else:
        download_datasets(selected, args.output_dir, keep_zips=args.keep_zips)


if __name__ == "__main__":
    main()
