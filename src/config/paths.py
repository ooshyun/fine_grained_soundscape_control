"""Dataset path configuration for Semantic Listening."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Get the project root directory.

    This function locates the project root by finding the directory
    containing the pyproject.toml file, starting from this file's location
    and walking up the directory tree.

    Returns:
        Path to the project root directory

    Raises:
        RuntimeError: If project root cannot be found
    """
    current = Path(__file__).resolve()

    # Walk up the directory tree looking for pyproject.toml
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError(
        "Could not find project root (no pyproject.toml found). "
        f"Started search from {current}"
    )


# Cache the project root to avoid repeated file system operations
_PROJECT_ROOT = None


def get_cached_project_root() -> Path:
    """
    Get cached project root directory.

    This caches the result of get_project_root() to avoid repeated
    file system operations.

    Returns:
        Path to the project root directory
    """
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = get_project_root()
    return _PROJECT_ROOT


@dataclass(frozen=True)
class DatasetPaths:
    """
    Configuration for dataset paths.

    This centralizes all hardcoded dataset paths to make the project
    easily configurable for different environments.

    Attributes:
        dataset_root: Root directory where datasets are extracted
        dataset_dir: Directory containing the dataset archive
        dataset_name: Name of the dataset archive file
    """

    dataset_root: str = "/scr"
    dataset_dir: str = (
        "/mmfs1/gscratch/intelligentsystems/common_datasets/"
        "SemanticListening/"
    )
    dataset_name: str = "BinauralCuratedDataset.tar"

    @property
    def dataset_archive_path(self) -> str:
        """Full path to the dataset archive file."""
        return os.path.join(self.dataset_dir, self.dataset_name)

    @property
    def extracted_dataset_path(self) -> str:
        """Path where the dataset is extracted."""
        dataset_name_without_ext = self.dataset_name.split(".")[0]
        return os.path.join(self.dataset_root, dataset_name_without_ext)

    @property
    def scaper_fmt_path(self) -> str:
        """Path to scaper formatted data."""
        return os.path.join(self.extracted_dataset_path, "scaper_fmt")

    @property
    def bg_scaper_fmt_path(self) -> str:
        """Path to background scaper formatted data."""
        return os.path.join(self.extracted_dataset_path, "bg_scaper_fmt")

    @property
    def noise_scaper_fmt_path(self) -> str:
        """Path to noise scaper formatted data."""
        return os.path.join(self.extracted_dataset_path, "noise_scaper_fmt")

    def validate_archive_exists(self) -> bool:
        """Check if the dataset archive file exists."""
        return os.path.exists(self.dataset_archive_path)

    def validate_extracted_dataset(self) -> bool:
        """Check if the extracted dataset exists."""
        return os.path.exists(self.extracted_dataset_path)

    def validate_dataset_structure(self) -> tuple[bool, list[str]]:
        """
        Validate that all required subdirectories exist.

        Returns:
            Tuple of (is_valid, list_of_missing_paths)
        """
        required_paths = [
            self.scaper_fmt_path,
            self.bg_scaper_fmt_path,
            self.noise_scaper_fmt_path,
        ]

        missing_paths = [
            path for path in required_paths if not os.path.exists(path)
        ]

        return (len(missing_paths) == 0, missing_paths)


def get_default_dataset_paths() -> DatasetPaths:
    """
    Get default dataset paths configuration.

    This can be extended to load from environment variables or config files.

    Returns:
        DatasetPaths instance with default values
    """
    # Check for environment variable overrides
    dataset_root = os.getenv("DATASET_ROOT", "/scr")
    dataset_dir = os.getenv(
        "DATASET_DIR",
        "/mmfs1/gscratch/intelligentsystems/common_datasets/"
        "SemanticListening/",
    )
    dataset_name = os.getenv("DATASET_NAME", "BinauralCuratedDataset.tar")

    return DatasetPaths(
        dataset_root=dataset_root,
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
    )


def get_dataset_root() -> str:
    """
    Get the dataset root directory.

    This function returns the root directory where datasets are stored,
    with the following priority:
    1. DATASET_ROOT environment variable
    2. Default value "/scr"

    The dataset root is where dataset directories like BinauralCuratedDataset
    are located.

    Returns:
        Dataset root directory path

    Example:
        >>> # With environment variable set
        >>> os.environ['DATASET_ROOT'] = '/mnt/sda1/tmp'
        >>> get_dataset_root()
        '/mnt/sda1/tmp'

        >>> # Without environment variable
        >>> get_dataset_root()
        '/scr'
    """
    return os.getenv("DATASET_ROOT", "/scr")


def get_split_paths(
    config: dict, dataset_root: str, split: str
) -> dict[str, str]:
    """
    Get foreground and noise paths for a specific data split.

    Args:
        config: Configuration dictionary
        dataset_root: Root directory of dataset
        split: Data split name ('train', 'val', or 'test')

    Returns:
        Dictionary with 'fg_sounds_dir' and 'noise_sounds_dir' keys
    """
    split_config = config[f"onflight_{split}_data_args"]

    return {
        "fg_sounds_dir": os.path.join(
            dataset_root, split_config["fg_sounds_dir"]
        ),
        "noise_sounds_dir": os.path.join(
            dataset_root, split_config["noise_sounds_dir"]
        ),
    }
