"""Silence-trimming utility for audio files.

Identifies the leading/trailing silence boundaries and the first
internal silence region using a sliding-window power threshold.
"""
from __future__ import annotations

import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d


def trim_silence(path: str) -> tuple[int, int, int]:
    """Return ``(start_sample, first_silence, end_sample)`` for an audio file.

    Uses ``librosa.effects.trim`` at 40 dB and a 1-second sliding-window
    power threshold to locate the first silence boundary.
    """
    data, sr = librosa.load(path, sr=None, mono=True)
    _, (start, end) = librosa.effects.trim(data, top_db=40)
    trimmed = data[start:end]

    window_size = int(1 * sr)
    if len(trimmed) < window_size:
        return int(start), int(end), int(end)

    avg_power = uniform_filter1d(trimmed ** 2, size=window_size)
    threshold = 0.1 * avg_power.max() if avg_power.max() > 0 else 0
    mask = avg_power < threshold
    first_silence = int(np.argmax(mask)) + start if mask.any() else int(end)

    return int(start), first_silence, int(end)
