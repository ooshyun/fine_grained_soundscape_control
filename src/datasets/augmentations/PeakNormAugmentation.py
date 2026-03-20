import torch
import numpy as np


class PeakNormAugmentation:
    def __init__(self, min_scale, max_scale):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, audio_data, gt_audio, rng: np.random.RandomState):
        peak = torch.abs(audio_data).max()

        scale = rng.rand() * (self.max_scale - self.min_scale) + self.min_scale
        scale = scale / (peak + 1e-6)

        augmented_audio_data = audio_data * scale
        augmented_gt_audio = gt_audio * scale

        return augmented_audio_data, augmented_gt_audio
