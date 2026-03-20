import torch
import torchaudio
import torch.nn.functional as F
import numpy as np


class PitchAugmentation:
    def __init__(self, min_pitch_shift, max_pitch_shift, sample_rate=16000):
        self.pitch_shift_lims = [min_pitch_shift, max_pitch_shift]

        self.sample_rate = sample_rate

    def apply_pitch_shift(self, audio: np.ndarray, rng: np.random.RandomState):
        """
        Augments the pitch of the audio
        """
        pitch_factor = rng.uniform(self.pitch_shift_lims[0], self.pitch_shift_lims[1])
        # print("PITCH SHIFT", pitch_factor)
        sox_effects = [
            ["pitch", str(pitch_factor)],
            ["rate", str(self.sample_rate)],
        ]
        old_T = audio.shape[-1]
        audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio, self.sample_rate, sox_effects
        )
        if audio.shape[-1] >= old_T:
            audio = audio[..., :old_T]
        else:
            audio = torch.nn.functional.pad(audio, (old_T - audio.shape[-1], 0))
        return audio

    def __call__(self, audio_data, gt_audio, rng: np.random.RandomState):
        T = audio_data.shape[-1]

        transformed_audio = self.apply_pitch_shift(audio_data, rng)
        augmented_gt_audio = self.apply_pitch_shift(gt_audio, rng)

        # Adjust size so it is the same as the original size
        if transformed_audio.shape[-1] > T:
            transformed_audio = transformed_audio[..., :T]
            augmented_gt_audio = augmented_gt_audio[..., :T]
        else:
            transformed_audio = F.pad(
                transformed_audio, (0, T - transformed_audio.shape[-1])
            )
            augmented_gt_audio = F.pad(
                augmented_gt_audio, (0, T - augmented_gt_audio.shape[-1])
            )

        assert transformed_audio.shape[-1] == T
        assert augmented_gt_audio.shape[-1] == T

        return transformed_audio, augmented_gt_audio
