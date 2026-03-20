"""Tests for the collect stage — FSD50K, ESC-50, DISCO, musdb18, TAU collectors."""
from __future__ import annotations

import os

import pandas as pd
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from data.pipeline.sources.fsd50k import FSD50KSource
from data.pipeline.sources.esc50 import ESC50Source, ESC50_TO_AUDIOSET
from data.pipeline.sources.disco import DISCOSource, DISCO_TO_AUDIOSET
from data.pipeline.sources.musdb18 import MUSDB18Source
from data.pipeline.sources.tau import TAUSource
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
        "/m/baby": "Baby cry, infant cry",
        "/m/blend": "Blender",
        "/m/tooth": "Toothbrush",
        "/m/fan": "Mechanical fan",
        "/m/fry": "Frying (food)",
        "/m/print": "Printer",
        "/m/vac": "Vacuum cleaner",
        "/m/water": "Water",
        "/m/sing": "Singing",
        "/m/melody": "Melody",
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
# DISCO tests
# ---------------------------------------------------------------------------

class TestDISCOCollect:
    """DISCO collector tests."""

    def _setup_disco(self, tmp_path: Path, labels=None) -> tuple[Path, Path]:
        raw_dir = tmp_path / "raw"
        curated_dir = tmp_path / "curated"
        if labels is None:
            labels = ["baby", "blender", "dishwasher"]
        disco_dir = raw_dir / "disco_noises"
        for split in ("train", "test"):
            for label in labels:
                label_dir = disco_dir / split / label
                label_dir.mkdir(parents=True, exist_ok=True)
                for i in range(10):
                    (label_dir / f"{label}_{split}_{i}.wav").touch()
        return raw_dir, curated_dir

    def test_audioset_mapping(self, tmp_path: Path):
        """DISCO_TO_AUDIOSET: mapped labels appear, None labels are skipped."""
        raw_dir, curated_dir = self._setup_disco(tmp_path)

        source = DISCOSource(ontology=_make_ontology_mock())
        source.collect(raw_dir, curated_dir)

        out_dir = curated_dir / "disco_noises"
        train = pd.read_csv(out_dir / "train.csv")

        # "baby" maps to "Baby cry, infant cry" → should appear
        assert "Baby cry, infant cry" in train["label"].values
        # "blender" maps to "Blender" → should appear
        assert "Blender" in train["label"].values
        # "dishwasher" maps to None → should NOT appear
        assert "dishwasher" not in train["label"].values

    def test_three_way_split(self, tmp_path: Path):
        """DISCO produces train/val/test and total count is preserved."""
        raw_dir, curated_dir = self._setup_disco(
            tmp_path, labels=["baby", "blender"]
        )

        source = DISCOSource(ontology=_make_ontology_mock())
        source.collect(raw_dir, curated_dir)

        out_dir = curated_dir / "disco_noises"
        train = pd.read_csv(out_dir / "train.csv")
        val = pd.read_csv(out_dir / "val.csv")
        test = pd.read_csv(out_dir / "test.csv")

        # All three splits should be non-empty
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

        # Total should equal number of valid label files
        # 2 labels (baby, blender) × 2 splits × 10 files = 40
        total = len(train) + len(val) + len(test)
        assert total == 40

    def test_csv_columns(self, tmp_path: Path):
        """Output CSVs must have columns: fname, label, id."""
        raw_dir, curated_dir = self._setup_disco(tmp_path, labels=["baby"])

        source = DISCOSource(ontology=_make_ontology_mock())
        source.collect(raw_dir, curated_dir)

        out_dir = curated_dir / "disco_noises"
        for split in ("train", "val", "test"):
            df = pd.read_csv(out_dir / f"{split}.csv")
            assert list(df.columns) == ["fname", "label", "id"]

    def test_relative_paths(self, tmp_path: Path):
        """fname should be relative like train/baby/file.wav."""
        raw_dir, curated_dir = self._setup_disco(tmp_path, labels=["baby"])

        source = DISCOSource(ontology=_make_ontology_mock())
        source.collect(raw_dir, curated_dir)

        out_dir = curated_dir / "disco_noises"
        all_dfs = []
        for split in ("train", "val", "test"):
            all_dfs.append(pd.read_csv(out_dir / f"{split}.csv"))
        combined = pd.concat(all_dfs)

        # All fnames should start with "train/" or "test/" (original DISCO splits)
        for fname in combined["fname"]:
            assert fname.startswith("train/") or fname.startswith("test/")


# ---------------------------------------------------------------------------
# TAU tests
# ---------------------------------------------------------------------------

class TestTAUCollect:
    """TAU collector tests."""

    def _setup_tau(self, tmp_path: Path) -> tuple[Path, Path]:
        raw_dir = tmp_path / "raw"
        curated_dir = tmp_path / "curated"
        dev_dir = (
            raw_dir / "TAU-2019"
            / "TAU-urban-acoustic-scenes-2019-development"
            / "audio"
        )
        dev_dir.mkdir(parents=True, exist_ok=True)
        # Create files with TAU naming: scene-city-timestamp-id.wav
        scenes = ["airport-barcelona", "bus-barcelona", "metro-barcelona"]
        for scene in scenes:
            for i in range(20):
                (dev_dir / f"{scene}-{i}-0.wav").touch()
        return raw_dir, curated_dir

    def test_label_parsing(self, tmp_path: Path):
        """Labels are parsed as first two dash-separated parts of filename."""
        raw_dir, curated_dir = self._setup_tau(tmp_path)

        source = TAUSource()
        source.collect(raw_dir, curated_dir)

        csv_dir = curated_dir / "TAU-acoustic-sounds"
        train = pd.read_csv(csv_dir / "train.csv")

        labels = set(train["label"].unique())
        # All labels should be scene-city format
        for label in labels:
            parts = label.split("-")
            assert len(parts) == 2, f"Expected scene-city format, got {label}"

        # Specific labels
        assert "airport-barcelona" in labels
        assert "bus-barcelona" in labels

    def test_stratified_split(self, tmp_path: Path):
        """90:10 stratified split: both train and val have same labels."""
        raw_dir, curated_dir = self._setup_tau(tmp_path)

        source = TAUSource()
        source.collect(raw_dir, curated_dir)

        csv_dir = curated_dir / "TAU-acoustic-sounds"
        train = pd.read_csv(csv_dir / "train.csv")
        val = pd.read_csv(csv_dir / "val.csv")

        train_labels = set(train["label"].unique())
        val_labels = set(val["label"].unique())

        # Common label constraint: train and val should have same labels
        assert train_labels == val_labels

        # Both splits non-empty
        assert len(train) > 0
        assert len(val) > 0

    def test_symlinks_created(self, tmp_path: Path):
        """Symlinks should be created in noise_scaper_fmt directory."""
        raw_dir, curated_dir = self._setup_tau(tmp_path)

        source = TAUSource()
        source.collect(raw_dir, curated_dir)

        symlink_dir = curated_dir / "noise_scaper_fmt"
        assert symlink_dir.exists()
        # At least train directory should have scene subdirs
        train_dir = symlink_dir / "train"
        assert train_dir.exists()
        scene_dirs = list(train_dir.iterdir())
        assert len(scene_dirs) > 0

    def test_csv_columns(self, tmp_path: Path):
        """Output CSVs must have columns: fname, label, id."""
        raw_dir, curated_dir = self._setup_tau(tmp_path)

        source = TAUSource()
        source.collect(raw_dir, curated_dir)

        csv_dir = curated_dir / "TAU-acoustic-sounds"
        for split in ("train", "val"):
            df = pd.read_csv(csv_dir / f"{split}.csv")
            assert list(df.columns) == ["fname", "label", "id"]


# ---------------------------------------------------------------------------
# musdb18 tests
# ---------------------------------------------------------------------------

class TestMUSDB18Collect:
    """musdb18 collector tests (basic, no actual STEMS extraction)."""

    def test_label_names(self):
        """musdb18 uses 'Singing' and 'Melody' as label names."""
        ontology = _make_ontology_mock()
        source = MUSDB18Source(ontology=ontology)
        # Verify the ontology can resolve both expected labels
        assert ontology.get_id_from_name("Singing") is not None
        assert ontology.get_id_from_name("Melody") is not None

    def test_write_csv_from_preextracted(self, tmp_path: Path):
        """_write_csv generates correct CSV from pre-extracted audio."""
        raw_dir = tmp_path / "raw"
        curated_dir = tmp_path / "curated"

        # Create fake pre-extracted audio structure
        dataset_dir = raw_dir / "musdb18"
        for split in ("train", "val", "test"):
            vocals_dir = dataset_dir / "audio" / split / "vocals"
            instr_dir = dataset_dir / "audio" / split / "instrumental"
            vocals_dir.mkdir(parents=True, exist_ok=True)
            instr_dir.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                (vocals_dir / f"song_{i}_v_{i}.wav").touch()
                (instr_dir / f"song_{i}_i_{i}.wav").touch()

        curated_dir.mkdir(parents=True, exist_ok=True)
        out_dir = curated_dir / "musdb18"
        out_dir.mkdir(parents=True, exist_ok=True)

        ontology = _make_ontology_mock()
        source = MUSDB18Source(ontology=ontology)

        df = source._write_csv(str(dataset_dir), "train", str(out_dir))

        assert len(df) == 6  # 3 vocals + 3 instrumental
        assert set(df["label"].unique()) == {"Singing", "Melody"}
        assert list(df.columns) == ["fname", "label", "id"]

        # Verify CSV was written
        assert (out_dir / "train.csv").exists()


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
