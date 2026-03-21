from __future__ import annotations
import random
from pathlib import Path

import numpy as np

from .ontology import Ontology
from .sources import register_sources

STAGES = ("download", "collect", "prepare")


def run(
    output_dir: Path,
    stage: str = "all",
    datasets: list[str] | None = None,
    manual_dir: Path | None = None,
    reference_dir: Path | None = None,
    dry_run: bool = False,
) -> None:
    # Match original prepare.py seed for reproducible splits
    random.seed(0)
    np.random.seed(0)

    data_dir = Path(__file__).parent.parent  # data/
    ontology = Ontology(str(data_dir / "ontology.json"))
    sources = register_sources(ontology)

    if datasets:
        sources = {k: v for k, v in sources.items() if k in datasets}

    raw_dir = output_dir / "raw"
    curated_dir = output_dir / "curated"

    run_stages = STAGES if stage == "all" else (stage,)

    for s in run_stages:
        if s == "download":
            from .download import run_download
            run_download(sources, raw_dir, manual_dir, dry_run)
        elif s == "collect":
            from .collect import run_collect
            run_collect(sources, raw_dir, curated_dir, reference_dir)
        elif s == "prepare":
            from .prepare import run_prepare
            run_prepare(curated_dir, raw_dir, output_dir, ontology, data_dir)
