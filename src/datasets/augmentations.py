from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from numpy import integer, newaxis
from numpy import sum as npsum
from numpy.fft import irfft, rfftfreq
from numpy.random import Generator, RandomState, default_rng
from numpy import sqrt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: coloured noise generation (Timmer & Koenig 1995)
# ---------------------------------------------------------------------------

def _get_normal_distribution(random_state):
    if isinstance(random_state, (integer, int)) or random_state is None:
        random_state = default_rng(random_state)
        return random_state.normal
    if isinstance(random_state, (Generator, RandomState)):
        return random_state.normal
    raise ValueError(
        "random_state must be int, numpy.random.Generator, "
        "numpy.random.RandomState, or None"
    )


def powerlaw_psd_gaussian(exponent: float, size, fmin: float = 0, random_state=None):
    """Generate Gaussian (1/f)**beta noise normalised to unit variance."""
    try:
        size = list(size)
    except TypeError:
        size = [size]

    samples = size[-1]
    f = rfftfreq(samples)

    if not (0 <= fmin <= 0.5):
        raise ValueError("fmin must be chosen between 0 and 0.5.")
    fmin = max(fmin, 1.0 / samples)

    s_scale = f.copy()
    ix = npsum(s_scale < fmin)
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale ** (-exponent / 2.0)

    w = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2.0
    sigma = 2 * sqrt(npsum(w ** 2)) / samples

    size[-1] = len(f)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]

    normal_dist = _get_normal_distribution(random_state)
    sr = normal_dist(scale=s_scale, size=size)
    si = normal_dist(scale=s_scale, size=size)

    if not (samples % 2):
        si[..., -1] = 0
        sr[..., -1] *= sqrt(2)
    si[..., 0] = 0
    sr[..., 0] *= sqrt(2)

    s = sr + 1j * si
    y = irfft(s, n=samples, axis=-1) / sigma
    return y


# ---------------------------------------------------------------------------
# Individual augmentations
# ---------------------------------------------------------------------------

class SpeedAugmentation:
    """Time-stretch via SoX speed + rate effects."""

    def __init__(self, min_speed: float = 0.9, max_speed: float = 1.1, sample_rate: int = 16000):
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.sample_rate = sample_rate

    def __call__(self, mixture: torch.Tensor, target: torch.Tensor, rng: np.random.RandomState):
        T = mixture.shape[-1]
        speed_factor = rng.uniform(self.min_speed, self.max_speed)
        sox_effects = [["speed", str(speed_factor)], ["rate", str(self.sample_rate)]]

        mixture, _ = torchaudio.sox_effects.apply_effects_tensor(mixture, self.sample_rate, sox_effects)
        target, _ = torchaudio.sox_effects.apply_effects_tensor(target, self.sample_rate, sox_effects)

        if mixture.shape[-1] > T:
            mixture = mixture[..., :T]
            target = target[..., :T]
        else:
            mixture = F.pad(mixture, (0, T - mixture.shape[-1]))
            target = F.pad(target, (0, T - target.shape[-1]))

        return mixture, target


class PitchAugmentation:
    """Pitch-shift via SoX pitch + rate effects."""

    def __init__(self, min_pitch_shift: float = -200, max_pitch_shift: float = 200, sample_rate: int = 16000):
        self.pitch_shift_lims = [min_pitch_shift, max_pitch_shift]
        self.sample_rate = sample_rate

    def _apply(self, audio: torch.Tensor, rng: np.random.RandomState) -> torch.Tensor:
        pitch_factor = rng.uniform(self.pitch_shift_lims[0], self.pitch_shift_lims[1])
        sox_effects = [["pitch", str(pitch_factor)], ["rate", str(self.sample_rate)]]
        old_T = audio.shape[-1]
        audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, self.sample_rate, sox_effects)
        if audio.shape[-1] >= old_T:
            audio = audio[..., :old_T]
        else:
            audio = F.pad(audio, (old_T - audio.shape[-1], 0))
        return audio

    def __call__(self, mixture: torch.Tensor, target: torch.Tensor, rng: np.random.RandomState):
        T = mixture.shape[-1]
        mixture = self._apply(mixture, rng)
        target = self._apply(target, rng)

        if mixture.shape[-1] > T:
            mixture = mixture[..., :T]
            target = target[..., :T]
        else:
            mixture = F.pad(mixture, (0, T - mixture.shape[-1]))
            target = F.pad(target, (0, T - target.shape[-1]))

        return mixture, target


class ChannelGainAugmentation:
    """Apply random per-channel dB gain."""

    def __init__(self, max_channel_gain_db: float = 6.0, reference_channels: list[int] | None = None, unique: bool = False):
        self.max_channel_gain = max_channel_gain_db
        self.unique = unique
        self.reference_channels = reference_channels if reference_channels is not None else [0, 1]

    def __call__(self, mixture: torch.Tensor, target: torch.Tensor, rng: np.random.RandomState):
        C = mixture.shape[0]

        def _random_gain():
            gain_db = 2 * (rng.random() - 0.5) * self.max_channel_gain
            return 10 ** (gain_db / 20)

        if self.unique:
            g = _random_gain()
            gains = [g] * C
        else:
            gains = [_random_gain() for _ in range(C)]

        gt_ch = 0
        for i in range(C):
            mixture[i] = mixture[i] * gains[i]
            if i in self.reference_channels:
                target[gt_ch] = target[gt_ch] * gains[i]
                gt_ch += 1

        return mixture, target


class SampleShiftAugmentation:
    """Circular shift each channel by a random number of samples."""

    def __init__(self, max_shift: int = 10, reference_channels: list[int] | None = None, unique: bool = False):
        self.max_shift = max_shift
        self.unique = unique
        self.reference_channels = reference_channels if reference_channels is not None else [0, 1]

    def __call__(self, mixture: torch.Tensor, target: torch.Tensor, rng: np.random.RandomState):
        C = mixture.shape[0]

        if self.unique:
            s = int(rng.randint(-self.max_shift, self.max_shift + 1))
            shifts = [s] * C
        else:
            shifts = [int(rng.randint(-self.max_shift, self.max_shift + 1)) for _ in range(C)]

        gt_ch = 0
        for i in range(C):
            mixture[i] = torch.roll(mixture[i], shifts[i], dims=-1)
            if i in self.reference_channels:
                target[gt_ch] = torch.roll(target[gt_ch], shifts[i], dims=-1)
                gt_ch += 1

        return mixture, target


class FrequencyMaskingAugmentation:
    """Zero-out random frequency bins via STFT -> mask -> iSTFT."""

    def __init__(
        self,
        min_freq_masks: int = 1,
        max_freq_masks: int = 10,
        unique: bool = False,
        nfft: int = 4096,
        reference_channels: list[int] | None = None,
    ):
        self.min_freq_masks = min_freq_masks
        self.max_freq_masks = max_freq_masks
        self.unique = unique
        self.nfft = nfft
        self.reference_channels = reference_channels if reference_channels is not None else [0, 1]

    def __call__(self, mixture: torch.Tensor, target: torch.Tensor, rng: np.random.RandomState):
        C = mixture.shape[0]
        T = mixture.shape[-1]
        N = self.nfft // 2 + 1
        window = torch.hamming_window(self.nfft, device=mixture.device)

        if self.unique:
            n_masks = rng.randint(self.min_freq_masks, self.max_freq_masks + 1)
            freq_idxs = rng.permutation(N)[:n_masks]
            freqs = [freq_idxs] * C
        else:
            freqs = []
            for _ in range(C):
                n_masks = rng.randint(self.min_freq_masks, self.max_freq_masks + 1)
                freqs.append(rng.permutation(N)[:n_masks])

        gt_ch = 0
        for i in range(C):
            mask = freqs[i]
            S = torch.stft(mixture[i], n_fft=self.nfft, return_complex=True, window=window)
            S[mask] = 0
            mixture[i] = torch.istft(S, n_fft=self.nfft, length=T, window=window)

            if i in self.reference_channels:
                S_gt = torch.stft(target[gt_ch], n_fft=self.nfft, return_complex=True, window=window)
                S_gt[mask] = 0
                target[gt_ch] = torch.istft(S_gt, n_fft=self.nfft, length=T, window=window)
                gt_ch += 1

        return mixture, target


class WhitePinkBrownAugmentation:
    """Add a mix of white, pink, and brown noise to the mixture."""

    def __init__(
        self,
        max_white_level: float = 1e-3,
        max_pink_level: float = 5e-3,
        max_brown_level: float = 5e-3,
    ):
        self.max_white_level = max_white_level
        self.max_pink_level = max_pink_level
        self.max_brown_level = max_brown_level

    def __call__(self, mixture: torch.Tensor, target: torch.Tensor, rng: np.random.RandomState):
        shape = mixture.shape

        # White noise
        wn_level = self.max_white_level * rng.rand()
        wn = wn_level * torch.from_numpy(rng.normal(0, 1, size=shape)).float()

        # Pink noise (1/f)
        pn_level = self.max_pink_level * rng.rand()
        pn = pn_level * torch.from_numpy(powerlaw_psd_gaussian(1, shape, random_state=0)).float()

        # Brown noise (1/f^2)
        bn_level = self.max_brown_level * rng.rand()
        bn = bn_level * torch.from_numpy(powerlaw_psd_gaussian(2, shape, random_state=0)).float()

        mixture = mixture + (wn + pn + bn).to(mixture.device)
        return mixture, target


# ---------------------------------------------------------------------------
# Registry for string-based lookup
# ---------------------------------------------------------------------------

_AUGMENTATION_REGISTRY: dict[str, type] = {
    "SpeedAugmentation": SpeedAugmentation,
    "PitchAugmentation": PitchAugmentation,
    "ChannelGainAugmentation": ChannelGainAugmentation,
    "SampleShiftAugmentation": SampleShiftAugmentation,
    "FrequencyMaskingAugmentation": FrequencyMaskingAugmentation,
    "WhitePinkBrownAugmentation": WhitePinkBrownAugmentation,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AudioAugmentations:
    """Apply a configurable sequence of audio augmentations.

    Parameters
    ----------
    configs:
        List of dicts, each with keys ``"type"`` (str), ``"prob"`` (float),
        and optionally ``"params"`` (dict).
    """

    def __init__(self, configs: list[dict]) -> None:
        self.augmentations: list[tuple[float, object]] = []
        for cfg in configs:
            aug_type = cfg["type"]
            prob = cfg.get("prob", 1.0)
            params = cfg.get("params", {})

            # Look up in local registry first, fall back to dotted import path.
            cls = _AUGMENTATION_REGISTRY.get(aug_type)
            if cls is None:
                # Support fully-qualified class names for extensibility.
                module_path, class_name = aug_type.rsplit(".", 1)
                import importlib
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)

            self.augmentations.append((prob, cls(**params)))

    def __len__(self) -> int:
        return len(self.augmentations)

    def apply(
        self,
        mixture: torch.Tensor,
        target: torch.Tensor,
        rng: np.random.RandomState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for prob, aug in self.augmentations:
            if rng.random() < prob:
                mixture, target = aug(mixture, target, rng)
        return mixture, target

    # Alias kept for backward-compat with callers using the old name.
    apply_random_augmentations = apply
