"""Tests for the prepare pipeline stage."""
from __future__ import annotations

import os
import random
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# build_id2classname
# ---------------------------------------------------------------------------


def test_build_id2classname():
    """Verify class_map.yaml → id2classname expansion works with real ontology."""
    from data.pipeline.ontology import Ontology
    from data.pipeline.prepare import build_id2classname

    ontology = Ontology(str(DATA_DIR / "ontology.json"))
    id2classname = build_id2classname(DATA_DIR / "class_map.yaml", ontology)

    # Must have entries
    assert len(id2classname) > 0

    # Check that known class names appear
    class_names = set(id2classname.values())
    assert "speech" in class_names
    assert "dog" in class_names
    assert "cat" in class_names

    # Speech class: "/m/09x0r" is the AudioSet ID for "Speech"
    speech_id = ontology.get_id_from_name("Speech")
    assert speech_id is not None
    assert id2classname[speech_id] == "speech"

    # Subtree expansion: children of Speech should also map to "speech"
    subtree = ontology.get_subtree(speech_id)
    for child_id in subtree:
        assert id2classname[child_id] == "speech"


def test_build_id2classname_covers_all_classes():
    """Every class in class_map.yaml should appear in id2classname values."""
    import yaml
    from data.pipeline.ontology import Ontology
    from data.pipeline.prepare import build_id2classname

    ontology = Ontology(str(DATA_DIR / "ontology.json"))
    id2classname = build_id2classname(DATA_DIR / "class_map.yaml", ontology)

    with open(DATA_DIR / "class_map.yaml") as f:
        class_data = yaml.safe_load(f)

    mapped_classes = set(id2classname.values())
    for class_name in class_data:
        assert class_name in mapped_classes, f"{class_name} not found in id2classname"


# ---------------------------------------------------------------------------
# is_valid_background
# ---------------------------------------------------------------------------


def test_is_valid_background_excludes_foreground():
    """A foreground ID should be rejected."""
    from data.pipeline.ontology import Ontology
    from data.pipeline.prepare import build_id2classname, is_valid_background

    ontology = Ontology(str(DATA_DIR / "ontology.json"))
    id2classname = build_id2classname(DATA_DIR / "class_map.yaml", ontology)

    # Pick a foreground ID
    fg_id = next(iter(id2classname))
    assert not is_valid_background(fg_id, ontology, id2classname)


def test_is_valid_background_excludes_music():
    """Music subtree labels should be excluded."""
    from data.pipeline.ontology import Ontology
    from data.pipeline.prepare import build_id2classname, is_valid_background

    ontology = Ontology(str(DATA_DIR / "ontology.json"))
    id2classname = build_id2classname(DATA_DIR / "class_map.yaml", ontology)

    # Music root itself
    assert not is_valid_background(ontology.MUSIC, ontology, id2classname)

    # A child of Music (e.g. "Musical instrument")
    music_children = ontology.ontology[ontology.MUSIC]["child_ids"]
    if music_children:
        child = music_children[0]
        # Skip if it happens to be foreground
        if child not in id2classname:
            assert not is_valid_background(child, ontology, id2classname)


def test_is_valid_background_excludes_human_voice():
    """Human voice subtree labels should be excluded."""
    from data.pipeline.ontology import Ontology
    from data.pipeline.prepare import build_id2classname, is_valid_background

    ontology = Ontology(str(DATA_DIR / "ontology.json"))
    id2classname = build_id2classname(DATA_DIR / "class_map.yaml", ontology)

    human_voice_id = ontology.get_id_from_name("Human voice")
    assert human_voice_id is not None

    # Human voice itself (may or may not be foreground, but should fail
    # either the foreground check or the human voice exclusion)
    assert not is_valid_background(human_voice_id, ontology, id2classname)


def test_is_valid_background_excludes_fg_ancestor():
    """An ancestor of a foreground label should be excluded."""
    from data.pipeline.ontology import Ontology
    from data.pipeline.prepare import build_id2classname, is_valid_background

    ontology = Ontology(str(DATA_DIR / "ontology.json"))
    id2classname = build_id2classname(DATA_DIR / "class_map.yaml", ontology)

    # Get an ancestor of a foreground ID (go up one level)
    fg_id = next(iter(id2classname))
    parent_id = ontology.ontology[fg_id].get("parent_id")
    if parent_id and parent_id not in id2classname and parent_id != ontology.ROOT:
        assert not is_valid_background(parent_id, ontology, id2classname)


# ---------------------------------------------------------------------------
# trim_silence
# ---------------------------------------------------------------------------


def test_trim_silence_returns_tuple():
    """trim_silence should return a 3-tuple of ints."""
    import soundfile as sf
    from data.pipeline.silence import trim_silence

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sr = 16000
        # Generate a signal: silence + tone + silence
        silence = np.zeros(sr, dtype=np.float32)
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        audio = np.concatenate([silence, tone, silence])
        sf.write(tmp_path, audio, sr)

        result = trim_silence(tmp_path)
        assert isinstance(result, tuple)
        assert len(result) == 3
        start, first_silence, end = result
        assert isinstance(start, int)
        assert isinstance(first_silence, int)
        assert isinstance(end, int)
        assert start < end
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# prepare_hrtf
# ---------------------------------------------------------------------------


def test_prepare_hrtf():
    """HRTF preparation should produce train/val/test txt files."""
    from data.pipeline.prepare import prepare_hrtf

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        output_dir = Path(tmpdir) / "output"
        cipic_src = raw_dir / "cipic-hrtf-database" / "subjects"
        cipic_src.mkdir(parents=True)

        # Create dummy .sofa files
        for i in range(20):
            (cipic_src / f"subject_{i:03d}.sofa").write_bytes(b"dummy")

        random.seed(0)
        prepare_hrtf(raw_dir, output_dir)

        hrtf_dir = output_dir / "hrtf"
        cipic_dst = hrtf_dir / "CIPIC"

        # Check SOFA files were copied
        assert cipic_dst.exists()
        copied = list(cipic_dst.glob("*.sofa"))
        assert len(copied) == 20

        # Check split files exist
        for name in ["train_hrtf.txt", "val_hrtf.txt", "test_hrtf.txt"]:
            fpath = hrtf_dir / name
            assert fpath.exists(), f"{name} not found"
            lines = fpath.read_text().strip().splitlines()
            assert len(lines) > 0

        # Check split sizes are reasonable (80:10:10 of 20)
        train_lines = (hrtf_dir / "train_hrtf.txt").read_text().strip().splitlines()
        val_lines = (hrtf_dir / "val_hrtf.txt").read_text().strip().splitlines()
        test_lines = (hrtf_dir / "test_hrtf.txt").read_text().strip().splitlines()

        total = len(train_lines) + len(val_lines) + len(test_lines)
        assert total == 20
        assert len(train_lines) == 16  # round(0.8 * 20)
        assert len(val_lines) == 2     # round(0.1 * 20)
        assert len(test_lines) == 2    # remainder
