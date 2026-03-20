from __future__ import annotations

"""Unified soundscape dataset for TSE and SED tasks.

Synthesises binaural mixtures on-the-fly by sampling foreground, background,
and noise sources, normalising loudness, spatialising via HRTF simulation,
and mixing.
"""

import logging
import math
import os
import re
import time
import traceback as tb

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.datasets.augmentations import AudioAugmentations
from src.datasets.hrtf import CIPICSimulator

logger = logging.getLogger(__name__)

# Optional fast resampling via torchaudio (falls back to librosa).
try:
    import torchaudio

    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    _TORCHAUDIO_AVAILABLE = False

try:
    import librosa

    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _glob_re(pattern: str, strings):
    """Filter *strings* by a regex *pattern*."""
    return list(filter(re.compile(pattern).match, strings))


def _get_lufs(audio: np.ndarray, sample_rate: int) -> float:
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    if audio.ndim > 1:
        audio = audio.flatten()
    meter = pyln.Meter(sample_rate)
    return meter.integrated_loudness(audio)


def _normalize_to_lufs(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float,
    max_iterations: int = 100,
    tolerance: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Iteratively normalise *audio* to *target_lufs*."""
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    elif not isinstance(audio, np.ndarray):
        audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.flatten()

    current_lufs = _get_lufs(audio, sample_rate)
    normalized = pyln.normalize.loudness(audio, current_lufs, target_lufs)

    for _ in range(max_iterations):
        error = abs(current_lufs - target_lufs)
        if error < tolerance:
            return normalized, current_lufs
        current_lufs = _get_lufs(normalized, sample_rate)
        normalized = pyln.normalize.loudness(normalized, current_lufs, target_lufs)

    return normalized, current_lufs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SoundscapeDataset(Dataset):
    """On-the-fly binaural soundscape synthesis for TSE or SED.

    Parameters
    ----------
    fg_dir : str
        Root directory with sub-folders per foreground sound class.
    noise_dir : str
        Root directory with sub-folders per noise class.
    hrtf_list : str
        Path to a text file listing SOFA files for HRTF simulation.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    bg_dir : str | None
        Optional background sounds directory (same layout as *fg_dir*).
    task : str
        ``"tse"`` for target sound extraction, ``"sed"`` for sound event
        detection.
    augmentations : list[dict] | None
        Augmentation configs passed to :class:`AudioAugmentations`.
    """

    def __init__(
        self,
        fg_dir: str,
        noise_dir: str,
        hrtf_list: str,
        split: str,
        sr: int = 16000,
        duration: int = 5,
        bg_dir: str | None = None,
        num_fg_range: tuple[int, int] = (1, 1),
        num_bg_range: tuple[int, int] = (0, 0),
        num_noise_range: tuple[int, int] = (1, 1),
        snr_range_fg: tuple[float, float] = (5, 15),
        snr_range_bg: tuple[float, float] = (0, 10),
        ref_db: float = -50,
        num_output_channels: int = 2,
        num_total_labels: int = 20,
        samples_per_epoch: int = 20000,
        augmentations: list[dict] | None = None,
        hrtf_type: str = "CIPIC",
        task: str = "tse",
    ) -> None:
        super().__init__()
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        assert task in ("tse", "sed"), f"Invalid task: {task}"

        self.fg_dir = fg_dir
        self.noise_dir = noise_dir
        self.bg_dir = bg_dir
        self.hrtf_list = hrtf_list
        self.split = split
        self.sr = sr
        self.duration = duration
        self.num_fg_range = num_fg_range
        self.num_bg_range = num_bg_range
        self.num_noise_range = num_noise_range
        self.snr_range_fg = snr_range_fg
        self.snr_range_bg = snr_range_bg
        self.ref_db = ref_db
        self.num_output_channels = num_output_channels
        self.num_total_labels = num_total_labels
        self.samples_per_epoch = samples_per_epoch
        self.task = task
        self.pwr_threshold = -40
        self.lufs_tolerance = 0.1

        # Augmentations
        self.perturbations = AudioAugmentations(augmentations or [])

        # HRTF simulator
        if hrtf_type == "CIPIC":
            self.hrtf_simulator = CIPICSimulator(self.hrtf_list, sr)
        else:
            raise NotImplementedError(f"Unsupported hrtf_type: {hrtf_type}")

        # Sound class inventories
        self.fg_sounds = sorted(os.listdir(self.fg_dir))
        if len(self.fg_sounds) > num_total_labels:
            self.fg_sounds = self.fg_sounds[:num_total_labels]
        elif len(self.fg_sounds) < num_total_labels:
            raise ValueError(
                f"Not enough FG sounds in {self.fg_dir}. "
                f"Expected >= {num_total_labels}, found {len(self.fg_sounds)}."
            )

        self.bg_sounds = sorted(os.listdir(self.bg_dir)) if self.bg_dir else []
        self.noise_sounds = sorted(os.listdir(self.noise_dir))

        logger.info("FG sounds (%d): %s", len(self.fg_sounds), self.fg_sounds)
        logger.info("BG sounds (%d): %s", len(self.bg_sounds), self.bg_sounds)
        logger.info("Noise sounds (%d): %s", len(self.noise_sounds), self.noise_sounds)

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.samples_per_epoch

    # ------------------------------------------------------------------
    # Audio I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_audio(sndfile: sf.SoundFile, num_frames: int) -> np.ndarray:
        """Read audio and convert to mono float32."""
        if sndfile.name.endswith(".flac"):
            audio = sndfile.read(frames=num_frames, dtype="int32")
            audio = (audio / (2 ** 31 - 1)).astype(np.float32)
        else:
            audio = sndfile.read(frames=num_frames, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio

    def sample_snippet(
        self,
        path: str,
        duration: float,
        sr: int,
        rng: np.random.RandomState,
    ) -> np.ndarray | None:
        """Load a random snippet of *duration* seconds from *path*.

        Returns ``None`` on any I/O failure so callers can retry.
        """
        if not os.path.exists(path):
            logger.warning("File does not exist: %s", path)
            return None

        try:
            with sf.SoundFile(path) as f:
                file_sr = f.samplerate
                num_frames = f.frames
                if file_sr <= 0 or num_frames <= 0:
                    return None

                total_frames = math.ceil(duration * file_sr)

                if total_frames > num_frames:
                    audio = self._read_audio(f, num_frames)
                    remain = total_frames - num_frames
                    pad_front = rng.randint(0, remain + 1)
                    audio = np.pad(audio, (pad_front, remain - pad_front))
                else:
                    start = rng.randint(0, num_frames - total_frames + 1)
                    f.seek(start)
                    audio = self._read_audio(f, total_frames)

                # Resample
                if file_sr != sr:
                    audio = self._resample(audio, file_sr, sr)

                tgt_samples = int(sr * duration)
                audio = audio[:tgt_samples]
                return audio

        except Exception as exc:
            logger.warning("Error reading %s: %s", path, exc)
            return None

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample with torchaudio; fall back to librosa."""
        if _TORCHAUDIO_AVAILABLE:
            try:
                t = torch.from_numpy(audio).float().unsqueeze(0)
                resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
                return resampler(t).squeeze(0).numpy()
            except Exception:
                pass
        if _LIBROSA_AVAILABLE:
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        raise RuntimeError("Neither torchaudio nor librosa available for resampling")

    # ------------------------------------------------------------------
    # Random source sampling
    # ------------------------------------------------------------------

    def _get_random_snippet(
        self,
        src_dir: str,
        label: str,
        rng: np.random.RandomState,
        require_power: bool = True,
    ) -> np.ndarray | None:
        """Sample a non-silent audio snippet from *src_dir/label*."""
        folder = os.path.join(src_dir, label)
        files = _glob_re(r".*\.[wav|flac]", os.listdir(folder))
        files = [os.path.join(folder, x) for x in files]
        if not files:
            logger.warning("No audio files in %s", folder)
            return None

        pwr_threshold = self.pwr_threshold
        for outer in range(100):
            path = rng.choice(files)
            for _ in range(10):
                audio = self.sample_snippet(path, self.duration, self.sr, rng)
                if audio is None:
                    break
                if not require_power:
                    return audio
                pwr_dB = 10 * np.log10(np.mean(audio ** 2) + 1e-9)
                if pwr_dB > pwr_threshold:
                    return audio
            if outer > 0 and outer % 100 == 0:
                pwr_threshold -= 1

        logger.warning("Failed to sample valid audio from %s", folder)
        return None

    # ------------------------------------------------------------------
    # Scene synthesis
    # ------------------------------------------------------------------

    def create_scene(self, rng: np.random.RandomState):
        """Synthesise one binaural scene.

        Returns
        -------
        mixture, target, label_vector, fg_labels, n_fg
        """
        n_fg = rng.randint(self.num_fg_range[0], self.num_fg_range[1] + 1)
        n_bg = rng.randint(self.num_bg_range[0], self.num_bg_range[1] + 1) if self.bg_sounds else 0
        n_noise = rng.randint(self.num_noise_range[0], self.num_noise_range[1] + 1)

        # --- Foreground ---
        fg_audios: list[np.ndarray] = []
        fg_labels: list[str] = []
        fg_src_ids: list[int] = []
        label_vector = torch.zeros(len(self.fg_sounds))

        while len(fg_labels) < n_fg:
            src_id = rng.randint(0, len(self.fg_sounds))
            if src_id in fg_src_ids:
                continue
            label = self.fg_sounds[src_id]
            audio = self._get_random_snippet(self.fg_dir, label, rng, require_power=True)
            if audio is not None:
                fg_audios.append(audio)
                fg_labels.append(label)
                fg_src_ids.append(src_id)
                label_vector[src_id] = 1

        # --- Background ---
        bg_audios: list[np.ndarray] = []
        if self.bg_sounds and n_bg > 0:
            for _ in range(n_bg * 10):
                if len(bg_audios) >= n_bg:
                    break
                label = rng.choice(self.bg_sounds)
                audio = self._get_random_snippet(self.bg_dir, label, rng, require_power=True)
                if audio is not None:
                    bg_audios.append(audio)

        # --- Noise ---
        noise_audio: np.ndarray | None = None
        if n_noise > 0:
            for _ in range(n_noise * 10):
                if noise_audio is not None:
                    break
                label = rng.choice(self.noise_sounds)
                noise_audio = self._get_random_snippet(self.noise_dir, label, rng, require_power=False)
        if noise_audio is None:
            noise_audio = np.zeros(int(self.sr * self.duration), dtype=np.float32)

        # --- LUFS normalisation ---
        fg_snr = rng.uniform(self.snr_range_fg[0], self.snr_range_fg[1])
        bg_snr = rng.uniform(self.snr_range_bg[0], self.snr_range_bg[1])
        fg_lufs = self.ref_db + fg_snr
        bg_lufs = self.ref_db + bg_snr

        for i, a in enumerate(fg_audios):
            fg_audios[i], _ = _normalize_to_lufs(a, self.sr, fg_lufs, tolerance=self.lufs_tolerance)
        for i, a in enumerate(bg_audios):
            bg_audios[i], _ = _normalize_to_lufs(a, self.sr, bg_lufs, tolerance=self.lufs_tolerance)
        if n_noise > 0:
            noise_audio, _ = _normalize_to_lufs(noise_audio, self.sr, self.ref_db, tolerance=self.lufs_tolerance)

        # --- HRTF spatialisation ---
        seed = int(rng.randint(1, 1_000_000))
        all_mono = fg_audios + bg_audios
        bi_srcs, bi_noise = self.hrtf_simulator.simulate(all_mono, noise_audio, seed)

        # --- Mix ---
        mixture = sum(bi_srcs) + bi_noise  # (2, T)
        mixture = torch.from_numpy(np.asarray(mixture)).float()

        # --- Ground truth target ---
        gt = torch.zeros((self.num_output_channels, mixture.shape[-1]))
        n_fg_actual = len(fg_labels)
        for i in range(n_fg_actual):
            bi_fg = bi_srcs[i]  # (2, T)
            if self.task == "tse":
                # Mono downmix of the binaural foreground source.
                mono = np.mean(bi_fg, axis=0)
                gt[i] = torch.from_numpy(mono).float()
            else:
                # SED: sum binaural fg into binaural target.
                gt += torch.from_numpy(np.asarray(bi_fg)).float()

        return mixture, gt, label_vector, fg_labels, n_fg_actual

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        try:
            if self.split == "train":
                seed = idx + np.random.randint(1_000_000)
            else:
                seed = idx
            rng = np.random.RandomState(seed)

            mixture, target, label_vector, fg_labels, n_fg = self.create_scene(rng)

            # Augmentations (train only)
            if self.split == "train":
                mixture, target = self.perturbations.apply(mixture, target, rng)

            # Peak normalise
            peak = torch.abs(mixture).max()
            if peak > 1:
                mixture /= peak
                target /= peak

            # Build return dicts
            if self.task == "tse":
                inputs = {
                    "mixture": mixture,
                    "label_vector": label_vector,
                    "embedding": int(torch.argmax(label_vector).item()),
                }
            else:
                inputs = {
                    "mixture": mixture,
                    "labels": label_vector,  # multi-hot
                }

            targets = {
                "target": target,
                "fg_labels": fg_labels,
                "num_fg_labels": n_fg,
            }

            return inputs, targets

        except Exception as exc:
            logger.error("Error in __getitem__[%d]: %s\n%s", idx, exc, tb.format_exc())
            raise
