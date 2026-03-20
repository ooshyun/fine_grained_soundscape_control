import torch
import numpy as np


class FrequencyMaskingAugmentation:
    def __init__(
        self,
        min_freq_masks,
        max_freq_masks,
        unique=False,
        nfft=4096,
        reference_channels=[0, 1],
    ):
        self.min_freq_masks = min_freq_masks
        self.max_freq_masks = max_freq_masks

        self.unique = unique
        self.nfft = nfft
        self.reference_channels = reference_channels

    def __call__(self, audio_data, gt_audio, rng: np.random.RandomState):
        C = audio_data.shape[0]
        T = audio_data.shape[-1]
        N = self.nfft // 2 + 1

        self.window = torch.hamming_window(self.nfft, device=audio_data.device)

        # Get shifts for each channel
        if self.unique:
            unique_nfreqs = rng.randint(self.min_freq_masks, self.max_freq_masks + 1)
            freqs = rng.permutation(N)[:unique_nfreqs]
            freqs = [freqs] * C
        else:
            freqs = []
            for i in range(C):
                n_masks = rng.randint(self.min_freq_masks, self.max_freq_masks + 1)
                freqs.append(rng.permutation(N)[:n_masks])

        augmented_audio_data = audio_data
        augmented_gt_audio = gt_audio

        gt_channel = 0
        for i in range(C):
            mask = freqs[i]

            S = torch.stft(
                augmented_audio_data[i],
                n_fft=self.nfft,
                return_complex=True,
                window=self.window,
            )
            S[mask] = 0
            augmented_audio_data[i] = torch.istft(
                S, n_fft=self.nfft, length=T, window=self.window
            )

            if i in self.reference_channels:
                S = torch.stft(
                    augmented_gt_audio[gt_channel],
                    n_fft=self.nfft,
                    return_complex=True,
                    window=self.window,
                )
                S[mask] = 0
                augmented_gt_audio[gt_channel] = torch.istft(
                    S, n_fft=self.nfft, length=T, window=self.window
                )

                gt_channel += 1

        return augmented_audio_data, augmented_gt_audio
