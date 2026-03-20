from __future__ import annotations

import logging
import os
import random

import numpy as np
import sofa
import torchaudio
import torch
from scipy.signal import convolve

logger = logging.getLogger(__name__)


class CIPICSimulator:
    """CIPIC HRTF simulator using SOFA files for binaural rendering.

    Reads a text file listing SOFA file paths and spatialises monaural sources
    by convolving with randomly selected HRIRs.
    """

    # Default face-to-face measurement index for the CIPIC database.
    FACE_TO_FACE_IDX = 608

    def __init__(self, sofa_file_list: str, sr: int) -> None:
        sofa_dir = os.path.dirname(sofa_file_list)
        with open(sofa_file_list, "r") as f:
            subjects = f.read().strip().split("\n")
        self.sofa_files = [os.path.join(sofa_dir, s) for s in subjects if s]
        self.sr = sr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _convolve(self, src: np.ndarray, hrtf: sofa.Database, ir_idx: int) -> np.ndarray:
        """Convolve a mono source with a binaural IR and return (2, T)."""
        idx_map = {"M": ir_idx}
        ir_sr = hrtf.Data.SamplingRate.get_values(indices=idx_map).item()
        rir = hrtf.Data.IR.get_values(indices=idx_map).astype(np.float32)

        # Resample the IR when its sample-rate differs from the target.
        if int(ir_sr) != self.sr:
            rir = torchaudio.functional.resample(
                torch.from_numpy(rir), int(ir_sr), self.sr
            ).numpy()

        src_l = convolve(src, rir[0])[: len(src)]
        src_r = convolve(src, rir[1])[: len(src)]
        return np.stack([src_l, src_r], axis=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self,
        sources: list[np.ndarray],
        noise: list[np.ndarray],
        seed: int,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Spatialise *sources* and *noise* with a random CIPIC HRIR.

        Parameters
        ----------
        sources:
            List of mono waveforms (1-D ``np.ndarray``).
        noise:
            Mono noise waveform (1-D ``np.ndarray``).  When passed as a list
            only the first element is used (kept for API compat).
        seed:
            Seed that makes the SOFA file and position choices reproducible.

        Returns
        -------
        (binaural_sources, binaural_noise)
            ``binaural_sources`` is a list of ``(2, T)`` arrays.
            ``binaural_noise`` is a single ``(2, T)`` array.
        """
        if isinstance(noise, list):
            noise = noise[0]

        rng = random.Random(seed)
        hrtf = None
        max_trials = 100
        for _ in range(max_trials):
            try:
                sofa_file = rng.choice(self.sofa_files)
                hrtf = sofa.Database.open(sofa_file)
                logger.debug("Opened SOFA file: %s", sofa_file)
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("Error opening SOFA file: %s", exc)
                hrtf = None
        if hrtf is None:
            raise ValueError(
                f"Failed to open any SOFA file after {max_trials} trials"
            )

        try:
            n_positions = hrtf.Dimensions.M
            bi_srcs: list[np.ndarray] = []
            for src in sources:
                pos = rng.choice(range(n_positions))
                bi_srcs.append(self._convolve(src, hrtf, pos))

            bi_noise = self._convolve(noise, hrtf, rng.choice(range(n_positions)))
            return bi_srcs, bi_noise
        finally:
            # CRITICAL: close the SOFA file handle to prevent FD leaks.
            if hrtf is not None:
                try:
                    hrtf.close()
                except Exception as close_err:  # noqa: BLE001
                    logger.warning("Failed to close SOFA file: %s", close_err)
