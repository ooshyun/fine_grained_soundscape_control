from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.collectors.ontology import Ontology

logger = logging.getLogger(__name__)


class FSD50KCollector:
    """Collect and curate FSD50K labels into train/val/test CSV splits.

    Expected input layout::

        raw_dir/FSD50K/
            FSD50K.metadata/
                pp_pnp_ratings_FSD50K.json
                collection/collection_dev.csv
                collection/collection_eval.csv
            FSD50K.dev_audio/
            FSD50K.eval_audio/

    Output::

        output_dir/FSD50K/{train,val,test}.csv   (columns: fname, label, id)
    """

    def __init__(self, root_dir: str, ontology: Ontology) -> None:
        self.root_dir = root_dir
        self.ontology = ontology

        ratings_path = os.path.join(
            root_dir, "FSD50K.metadata", "pp_pnp_ratings_FSD50K.json"
        )
        with open(ratings_path) as f:
            self.pp_pnp_ratings: dict = json.load(f)

    # ------------------------------------------------------------------
    # Quality filtering
    # ------------------------------------------------------------------

    def _is_pp_sample(self, fname: str) -> bool:
        """Return True if every label of *fname* has >=2 positive ratings
        and zero negative/uncertain ratings."""
        label_ratings = self.pp_pnp_ratings[fname]

        for node_id in label_ratings:
            ratings = label_ratings[node_id]
            counts = {1.0: 0, 0.5: 0, 0: 0, -1: 0}
            for r in ratings:
                counts[r] += 1

            if counts[0.0] > 0 or counts[-1] > 0 or counts[1.0] < 2:
                return False
        return True

    # ------------------------------------------------------------------
    # Curation
    # ------------------------------------------------------------------

    def _curate_samples(self, samples: pd.DataFrame) -> pd.DataFrame:
        samples = samples.dropna().copy()
        samples["fname"] = samples["fname"].apply(str)
        samples["mids"] = samples["mids"].apply(lambda x: x.split(","))
        samples["labels"] = samples["labels"].apply(lambda x: x.split(","))

        # Keep only samples where all labels have sufficient positive ratings
        samples["pp_sample"] = samples.apply(
            lambda x: self._is_pp_sample(x["fname"]), axis=1
        )
        samples = samples[samples["pp_sample"]]

        # Single-label only
        samples = samples[samples["mids"].apply(lambda x: len(x) == 1)]

        samples["id"] = samples["mids"].apply(lambda x: x[0])
        samples["label"] = samples["id"].apply(
            lambda x: self.ontology.get_label(x)
        )
        return samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self, raw_dir: str, output_dir: str) -> None:
        """Run the full collection pipeline and write CSVs.

        *raw_dir* should point directly to the ``FSD50K/`` directory that
        contains ``FSD50K.metadata/``, ``FSD50K.dev_audio/``, etc.
        """
        dataset_dir = raw_dir  # already points to FSD50K/
        out_dir = os.path.join(output_dir, "FSD50K")
        os.makedirs(out_dir, exist_ok=True)

        # Load dev / eval splits
        dev_samples = pd.read_csv(
            os.path.join(
                dataset_dir, "FSD50K.metadata", "collection", "collection_dev.csv"
            )
        )
        eval_samples = pd.read_csv(
            os.path.join(
                dataset_dir, "FSD50K.metadata", "collection", "collection_eval.csv"
            )
        )

        # Re-point ontology + ratings to the actual dataset location
        self.root_dir = dataset_dir
        ratings_path = os.path.join(
            dataset_dir, "FSD50K.metadata", "pp_pnp_ratings_FSD50K.json"
        )
        with open(ratings_path) as f:
            self.pp_pnp_ratings = json.load(f)

        # Curate dev → train + val (90:10 per label)
        dev_curated = self._curate_samples(dev_samples)

        train_frames: list[pd.DataFrame] = []
        val_frames: list[pd.DataFrame] = []
        for label in sorted(dev_curated["label"].unique()):
            subset = dev_curated[dev_curated["label"] == label]
            if len(subset) <= 1:
                continue
            tr, va = train_test_split(subset, test_size=0.1)
            train_frames.append(tr)
            val_frames.append(va)

        train_samples = pd.concat(train_frames)
        val_samples = pd.concat(val_frames)
        test_samples = self._curate_samples(eval_samples)

        # Attach relative file paths
        for df, src_dir in [
            (train_samples, "FSD50K.dev_audio"),
            (val_samples, "FSD50K.dev_audio"),
            (test_samples, "FSD50K.eval_audio"),
        ]:
            df["fname"] = df["fname"].apply(
                lambda x, d=src_dir: os.path.join(d, f"{x}.wav")
            )

        # Keep only labels present in all three splits
        common_labels = list(
            set(train_samples["label"].unique())
            & set(val_samples["label"].unique())
            & set(test_samples["label"].unique())
        )
        train_samples = train_samples[train_samples["label"].isin(common_labels)]
        val_samples = val_samples[val_samples["label"].isin(common_labels)]
        test_samples = test_samples[test_samples["label"].isin(common_labels)]

        cols = ["fname", "label", "id"]
        train_samples[cols].to_csv(os.path.join(out_dir, "train.csv"), index=False)
        val_samples[cols].to_csv(os.path.join(out_dir, "val.csv"), index=False)
        test_samples[cols].to_csv(os.path.join(out_dir, "test.csv"), index=False)

        logger.info(
            "FSD50K: train=%d  val=%d  test=%d",
            len(train_samples),
            len(val_samples),
            len(test_samples),
        )
