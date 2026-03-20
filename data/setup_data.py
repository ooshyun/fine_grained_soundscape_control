#!/usr/bin/env python3
"""Unified dataset setup: download -> collect -> prepare."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup datasets for training.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated: fsd50k,esc50,disco,cipic,musdb18,tau. Default: all")
    parser.add_argument("--stage", choices=["download", "collect", "prepare", "all"],
                        default="all")
    parser.add_argument("--manual_dir", type=Path, default=None,
                        help="Path to manually downloaded datasets (musdb18, TAU-2019)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ds = [d.strip() for d in args.datasets.split(",")] if args.datasets else None

    from pipeline import run
    run(args.output_dir, args.stage, ds, args.manual_dir, args.dry_run)


if __name__ == "__main__":
    main()
