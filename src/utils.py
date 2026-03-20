"""Minimal utility functions for standalone operation.

Provides the subset of src.utils from the parent Sementic-Listening-v2 repo
that is needed by this project's datasets and augmentations code.
"""

import importlib
import json
import os

import librosa
import soundfile as sf


def import_attr(import_path: str):
    """Dynamically import a class/function from a dotted path string.

    Example:
        >>> cls = import_attr("src.datasets.augmentations.PitchAugmentation.PitchAugmentation")
    """
    module, attr = import_path.rsplit(".", 1)
    return getattr(importlib.import_module(module), attr)


def read_audio_file(file_path: str, sr: int):
    """Read an audio file and return as numpy array.

    Returns:
        numpy array of shape (n_channels, n_samples)
    """
    return librosa.core.load(file_path, mono=False, sr=sr)[0]


def write_audio_file(file_path: str, data, sr: int, subtype: str = "PCM_16"):
    """Write audio data to a file.

    Args:
        file_path: Output file path.
        data: Audio signal (n_channels x n_samples).
        sr: Sampling rate.
        subtype: Audio subtype for soundfile.
    """
    sf.write(file_path, data.T, sr, subtype)


def read_json(path: str) -> dict:
    """Read a JSON file."""
    with open(path, "rb") as f:
        return json.load(f)
