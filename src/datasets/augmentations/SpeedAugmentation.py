import torch
import torchaudio
import torch.nn.functional as F
import numpy as np


class SpeedAugmentation:
    def __init__(self, min_speed, max_speed, sample_rate=16000):
        self.min_speed = min_speed
        self.max_speed = max_speed

        self.sample_rate = sample_rate

    def __call__(self, audio_data, gt_audio, rng: np.random.RandomState):
        T = audio_data.shape[-1]
        speed_factor = rng.uniform(self.min_speed, self.max_speed)

        # is_zero = torch.abs(gt_audio[0]).max() == 0

        # change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_data, self.sample_rate, sox_effects
        )

        augmented_gt_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            gt_audio, self.sample_rate, sox_effects
        )

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

        # print(sox_effects, transformed_audio.shape, augmented_gt_audio.shape, is_zero, torch.abs(augmented_gt_audio[0]).max())

        # if (not is_zero) and (torch.abs(augmented_gt_audio).max() == 0):
        #     import src.utils as utils
        #     import pickle as pkl
        #     print("FOUND ERROR")
        #     utils.write_audio_file('tests/dbg_orig.wav', gt_audio.numpy(), 24000)
        #     utils.write_audio_file('tests/dbg.wav', augmented_gt_audio.numpy(), 24000)
        #     with open('tests/params_dbg.pkl', 'wb') as f:
        #         pkl.dump(dict(speed=speed_factor, rate=self.sample_rate), f)

        assert transformed_audio.shape[-1] == T
        assert augmented_gt_audio.shape[-1] == T

        return transformed_audio, augmented_gt_audio
