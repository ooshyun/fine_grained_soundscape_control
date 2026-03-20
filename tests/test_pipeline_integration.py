import subprocess
import sys
from pathlib import Path

SETUP_SCRIPT = str(Path(__file__).parent.parent / "data" / "setup_data.py")


def test_dry_run():
    """Verify CLI runs in dry-run mode without errors."""
    result = subprocess.run(
        [sys.executable, SETUP_SCRIPT, "--output_dir", "/tmp/test_pipeline_data", "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Download stage complete" in result.stdout


def test_stage_select():
    """Verify --stage flag works."""
    result = subprocess.run(
        [sys.executable, SETUP_SCRIPT, "--output_dir", "/tmp/test_pipeline_data",
         "--stage", "download", "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"


def test_dataset_filter():
    """Verify --datasets flag filters correctly."""
    result = subprocess.run(
        [sys.executable, SETUP_SCRIPT, "--output_dir", "/tmp/test_pipeline_data",
         "--datasets", "disco,cipic", "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    # Should only show disco and cipic, not FSD50K
    assert "FSD50K" not in result.stdout


def test_help_flag():
    """Verify --help works."""
    result = subprocess.run(
        [sys.executable, SETUP_SCRIPT, "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--output_dir" in result.stdout
    assert "--datasets" in result.stdout
    assert "--stage" in result.stdout
    assert "--manual_dir" in result.stdout
