"""Tests for the collect stage — FSD50K and ESC-50 collectors."""
from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from data.pipeline.sources.fsd50k import FSD50KSource
from data.pipeline.sources.esc50 import ESC50Source, ESC50_TO_AUDIOSET
from data.pipeline.collect import run_collect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeOntology:
    """Minimal ontology stub for testing — maps IDs to labels deterministically."""
    _id_to_label = {
        "/m/001": "Bark",
        "/m/002": "Meow",
        "/m/003": "Rain",
        "/m/004": "Wind",
    }
    _label_to_id = {v: k for k, v in _id_to_label.items()}

    def get_label(self, mid: str) -> str:
        return self._id_to_label[mid]

    def get_id_from_name(self, name: str) -> str | None:
        return self._label_to_id.get(name)


def _make_ontology_mock():
    return _FakeOntology()


# ---------------------------------------------------------------------------
# FSD50K tests
# ---------------------------------------------------------------------------

class TestFSD50KCollect:
    """FSD50K collector tests."""

    def _setup_fsd50k(self, tmp_path: Path) -> tuple[Path, Path]:
        raw_dir = tmp_path / "raw"
        curated_dir = tmp_path / "curated"
        fsd_dir = raw_dir / "FSD50K"
        fsd_dir.mkdir(parents=True)
        return raw_dir, curated_dir

    def test_single_label_filter(self, tmp_path: Path):
        """Multi-label rows must be excluded; single-label rows kept."""
        raw_dir, curated_dir = self._setup_fsd50k(tmp_path)

        # Build metadata: mix of single- and multi-label rows
        # Need enough samples per label for stratified split
        rows = []
        # Single-label: /m/001 (Bark) — 20 dev + 5 eval
        for i in range(20):
            rows.append({"filename": f"bark_dev_{i}", "split": "dev",
                         "labels": "Bark", "mids": "/m/001",
                         "fname": f"FSD50K.dev_audio/bark_dev_{i}.wav"})
        for i in range(5):
            rows.append({"filename": f"bark_eval_{i}", "split": "eval",
                         "labels": "Bark", "mids": "/m/001",
                         "fname": f"FSD50K.eval_audio/bark_eval_{i}.wav"})

        # Single-label: /m/002 (Meow) — 20 dev + 5 eval
        for i in range(20):
            rows.append({"filename": f"meow_dev_{i}", "split": "dev",
                         "labels": "Meow", "mids": "/m/002",
                         "fname": f"FSD50K.dev_audio/meow_dev_{i}.wav"})
        for i in range(5):
            rows.append({"filename": f"meow_eval_{i}", "split": "eval",
                         "labels": "Meow", "mids": "/m/002",
                         "fname": f"FSD50K.eval_audio/meow_eval_{i}.wav"})

        # Multi-label: should be filtered out
        rows.append({"filename": "multi_1", "split": "dev",
                      "labels": "Bark,Meow", "mids": "/m/001,/m/002",
                      "fname": "FSD50K.dev_audio/multi_1.wav"})
        rows.append({"filename": "multi_2", "split": "eval",
                      "labels": "Bark,Meow", "mids": "/m/001,/m/002",
                      "fname": "FSD50K.eval_audio/multi_2.wav"})

        pd.DataFrame(rows).to_csv(
            raw_dir / "FSD50K" / "metadata.csv", index=False
        )

        source = FSD50KSource(ontology=_make_ontology_mock())
        source.collect(raw_dir, curated_dir)

        out_dir = curated_dir / "FSD50K"
        for split in ("train", "val", "test"):
            df = pd.read_csv(out_dir / f"{split}.csv")
            # No multi-label IDs should appear
            assert not df["id"].str.contains(",").any()

    def test_common_label_constraint(self, tmp_path: Path):
        """Labels only in some splits are removed from all splits."""
        raw_dir, curated_dir = self._setup_fsd50k(tmp_path)

        rows = []
        # Bark: present in both dev and eval → should survive
        for i in range(20):
            rows.append({"filename": f"bark_d_{i}", "split": "dev",
                         "labels": "Bark", "mids": "/m/001",
                         "fname": f"FSD50K.dev_audio/bark_d_{i}.wav"})
        for i in range(5):
            rows.append({"filename": f"bark_e_{i}", "split": "eval",
                         "labels": "Bark", "mids": "/m/001",
                         "fname": f"FSD50K.eval_audio/bark_e_{i}.wav"})

        # Meow: present in dev and eval → should survive
        for i in range(20):
            rows.append({"filename": f"meow_d_{i}", "split": "dev",
                         "labels": "Meow", "mids": "/m/002",
                         "fname": f"FSD50K.dev_audio/meow_d_{i}.wav"})
        for i in range(5):
            rows.append({"filename": f"meow_e_{i}", "split": "eval",
                         "labels": "Meow", "mids": "/m/002",
                         "fname": f"FSD50K.eval_audio/meow_e_{i}.wav"})

        # Rain: only in dev, NOT in eval → should be removed
        for i in range(20):
            rows.append({"filename": f"rain_d_{i}", "split": "dev",
                         "labels": "Rain", "mids": "/m/003",
                         "fname": f"FSD50K.dev_audio/rain_d_{i}.wav"})

        pd.DataFrame(rows).to_csv(
            raw_dir / "FSD50K" / "metadata.csv", index=False
        )

        source = FSD50KSource(ontology=_make_ontology_mock())
        source.collect(raw_dir, curated_dir)

        out_dir = curated_dir / "FSD50K"
        for split in ("train", "val", "test"):
            df = pd.read_csv(out_dir / f"{split}.csv")
            # Rain should not appear (missing from test split)
            assert "Rain" not in df["label"].values

    def test_csv_columns(self, tmp_path: Path):
        """Output CSVs must have columns: fname, label, id."""
        raw_dir, curated_dir = self._setup_fsd50k(tmp_path)

        rows = []
        for i in range(20):
            rows.append({"filename": f"bark_d_{i}", "split": "dev",
                         "labels": "Bark", "mids": "/m/001",
                         "fname": f"FSD50K.dev_audio/bark_d_{i}.wav"})
        for i in range(5):
            rows.append({"filename": f"bark_e_{i}", "split": "eval",
                         "labels": "Bark", "mids": "/m/001",
                         "fname": f"FSD50K.eval_audio/bark_e_{i}.wav"})
        for i in range(20):
            rows.append({"filename": f"meow_d_{i}", "split": "dev",
                         "labels": "Meow", "mids": "/m/002",
                         "fname": f"FSD50K.dev_audio/meow_d_{i}.wav"})
        for i in range(5):
            rows.append({"filename": f"meow_e_{i}", "split": "eval",
                         "labels": "Meow", "mids": "/m/002",
                         "fname": f"FSD50K.eval_audio/meow_e_{i}.wav"})

        pd.DataFrame(rows).to_csv(
            raw_dir / "FSD50K" / "metadata.csv", index=False
        )

        source = FSD50KSource(ontology=_make_ontology_mock())
        source.collect(raw_dir, curated_dir)

        out_dir = curated_dir / "FSD50K"
        for split in ("train", "val", "test"):
            df = pd.read_csv(out_dir / f"{split}.csv")
            assert list(df.columns) == ["fname", "label", "id"]


# ---------------------------------------------------------------------------
# ESC-50 tests
# ---------------------------------------------------------------------------

class TestESC50Collect:
    """ESC-50 collector tests."""

    def _setup_esc50(self, tmp_path: Path) -> tuple[Path, Path]:
        raw_dir = tmp_path / "raw"
        curated_dir = tmp_path / "curated"
        esc_dir = raw_dir / "ESC-50" / "meta"
        esc_dir.mkdir(parents=True)
        return raw_dir, curated_dir

    def test_fold_split(self, tmp_path: Path):
        """Folds 1-3 → train, fold 4 → val, fold 5 → test."""
        raw_dir, curated_dir = self._setup_esc50(tmp_path)

        rows = []
        for fold in range(1, 6):
            for i in range(4):
                rows.append({
                    "filename": f"fold{fold}_{i}.wav",
                    "fold": fold,
                    "target": 0,
                    "category": "dog",
                    "esc10": True,
                })
        pd.DataFrame(rows).to_csv(
            raw_dir / "ESC-50" / "meta" / "esc50.csv", index=False
        )

        source = ESC50Source(ontology=_make_ontology_mock())
        source.collect(raw_dir, curated_dir)

        out_dir = curated_dir / "ESC-50"
        train = pd.read_csv(out_dir / "train.csv")
        val = pd.read_csv(out_dir / "val.csv")
        test = pd.read_csv(out_dir / "test.csv")

        # 3 folds × 4 samples = 12 train, 1 fold × 4 = 4 val, 1 fold × 4 = 4 test
        assert len(train) == 12
        assert len(val) == 4
        assert len(test) == 4

    def test_none_mapping_skipped(self, tmp_path: Path):
        """Categories mapping to None in ESC50_TO_AUDIOSET are excluded."""
        raw_dir, curated_dir = self._setup_esc50(tmp_path)

        rows = [
            {"filename": "a.wav", "fold": 1, "target": 0, "category": "dog", "esc10": True},
            # drinking_sipping maps to None
            {"filename": "b.wav", "fold": 1, "target": 1, "category": "drinking_sipping", "esc10": False},
        ]
        pd.DataFrame(rows).to_csv(
            raw_dir / "ESC-50" / "meta" / "esc50.csv", index=False
        )

        source = ESC50Source(ontology=_make_ontology_mock())
        source.collect(raw_dir, curated_dir)

        train = pd.read_csv(curated_dir / "ESC-50" / "train.csv")
        assert len(train) == 1
        assert train.iloc[0]["label"] == "Bark"

    def test_relative_paths(self, tmp_path: Path):
        """Output fname should be audio/{filename}."""
        raw_dir, curated_dir = self._setup_esc50(tmp_path)

        rows = [
            {"filename": "1-100032-A-0.wav", "fold": 1, "target": 0,
             "category": "dog", "esc10": True},
        ]
        pd.DataFrame(rows).to_csv(
            raw_dir / "ESC-50" / "meta" / "esc50.csv", index=False
        )

        source = ESC50Source(ontology=_make_ontology_mock())
        source.collect(raw_dir, curated_dir)

        train = pd.read_csv(curated_dir / "ESC-50" / "train.csv")
        assert train.iloc[0]["fname"] == "audio/1-100032-A-0.wav"

    def test_csv_columns(self, tmp_path: Path):
        """Output CSVs must have columns: fname, label, id."""
        raw_dir, curated_dir = self._setup_esc50(tmp_path)

        rows = []
        for fold in range(1, 6):
            rows.append({
                "filename": f"f{fold}.wav", "fold": fold,
                "target": 0, "category": "dog", "esc10": True,
            })
        pd.DataFrame(rows).to_csv(
            raw_dir / "ESC-50" / "meta" / "esc50.csv", index=False
        )

        source = ESC50Source(ontology=_make_ontology_mock())
        source.collect(raw_dir, curated_dir)

        out_dir = curated_dir / "ESC-50"
        for split in ("train", "val", "test"):
            df = pd.read_csv(out_dir / f"{split}.csv")
            assert list(df.columns) == ["fname", "label", "id"]


# ---------------------------------------------------------------------------
# run_collect dispatcher tests
# ---------------------------------------------------------------------------

class TestRunCollect:
    """Tests for the collect dispatcher."""

    def test_skips_missing_sources(self, tmp_path: Path, capsys):
        """Sources whose raw dir doesn't exist are skipped."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        curated_dir = tmp_path / "curated"

        mock_source = MagicMock(spec=["name", "collect"])
        mock_source.name = "NonExistent"
        run_collect({"nope": mock_source}, raw_dir, curated_dir)

        captured = capsys.readouterr()
        assert "[skip]" in captured.out
        mock_source.collect.assert_not_called()

    def test_calls_collect_when_exists(self, tmp_path: Path):
        """Sources whose raw dir exists get their collect() called."""
        raw_dir = tmp_path / "raw"
        curated_dir = tmp_path / "curated"
        (raw_dir / "MyDataset").mkdir(parents=True)

        mock_source = MagicMock(spec=["name", "collect"])
        mock_source.name = "MyDataset"
        run_collect({"ds": mock_source}, raw_dir, curated_dir)

        mock_source.collect.assert_called_once_with(raw_dir, curated_dir)
