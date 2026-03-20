import pytest
from unittest.mock import MagicMock
from data.pipeline.sources.musdb18 import MUSDB18Source
from data.pipeline.sources.tau import TAUSource


def test_musdb18_prints_guide(capsys):
    src = MUSDB18Source(ontology=MagicMock())
    src.print_download_guide()
    captured = capsys.readouterr()
    assert "zenodo" in captured.out.lower()
    assert "musdb18" in captured.out


def test_tau_prints_guide(capsys):
    src = TAUSource()
    src.print_download_guide()
    captured = capsys.readouterr()
    assert "Tampere" in captured.out
    assert "TAU" in captured.out


def test_skip_if_done(tmp_path):
    src = MUSDB18Source(ontology=MagicMock())
    raw_dir = tmp_path
    out = raw_dir / "musdb18"
    out.mkdir()
    (out / ".done").touch()
    # Should not raise or attempt download
    src.download(raw_dir)
