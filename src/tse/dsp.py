import torch
import torch.nn as nn
import numpy as np


def mod_pad(x, chunk_size, pad):
    # Mod pad the input to perform integer number of
    # inferences
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = nn.functional.pad(x, (0, mod))
    x = nn.functional.pad(x, pad)

    return x, mod


def get_perfect_synthesis_window(analysis_win, lookback, chunksize, lookahead):
    nfft = lookback + chunksize + lookahead  # 256, BSS_MLP_1comp_100p
    assert analysis_win.shape[0] == nfft

    oWS = (
        chunksize + lookahead
    )  # TODO Check. 96 + 64 = 160, BSS_MLP_1comp_100p -> 10 sec

    synthesis_win = torch.zeros(oWS)
    A = synthesis_win.shape[0]
    B = chunksize
    N = nfft
    if (A % B) == 0:
        synthesis_win = torch.zeros(oWS)
        for i in range(A):
            num = analysis_win[N - A + i]

            denom = 0
            for k in range(A // B):
                denom += analysis_win[N - A + (i % B) + k * B] ** 2

            synthesis_win[i] = num / denom
    elif lookahead < chunksize:
        synthesis_win = torch.zeros(oWS)
        for i in range(A):
            if i >= lookahead and i < oWS - lookahead:
                # Non-overlapping region
                synthesis_win[i] = 1 / analysis_win[lookback + i]
            else:
                # Overlapping region
                num = analysis_win[lookback + i]

                denom = 0
                for k in range(A // B + 1):
                    denom += analysis_win[lookback + (i % B) + k * B] ** 2
                synthesis_win[i] = num / denom
    else:
        print(
            "This case is not handled for perfect window synthesis.\
               Either let front pad be a multiple of chunk size, or make chunk size > front pad."
        )
        raise NotImplementedError

    return synthesis_win


class DualWindowTF(nn.Module):
    def __init__(
        self, stft_chunk_size, stft_back_pad, stft_front_pad, window_type="hann"
    ):
        super(DualWindowTF, self).__init__()

        self.stft_chunk_size = stft_chunk_size  # 96, BSS_MLP_1comp_100p
        self.stft_front_pad = stft_front_pad  # 64, BSS_MLP_1comp_100p
        self.stft_back_pad = stft_back_pad  # 96, BSS_MLP_1comp_100p

        self.nfft = (
            stft_back_pad + stft_chunk_size + stft_front_pad
        )  # 256, BSS_MLP_1comp_100p

        self.nfreqs = self.nfft // 2 + 1  # 129, BSS_MLP_1comp_100p

        # Construct synthesis/analysis windows
        if window_type == "hann":
            window_fn = lambda x: np.hanning(x)
        elif window_type == "rect":
            window_fn = lambda x: np.ones(x)
        else:
            raise ValueError("Invalid window type!")

        # Construct analysis and synthesis windows
        self.analysis_window = torch.from_numpy(window_fn(self.nfft)).float()
        self.synthesis_window = get_perfect_synthesis_window(
            self.analysis_window, stft_back_pad, stft_chunk_size, stft_front_pad
        )

        # Inverse STFT buffer shape
        self.istft_lookback = (
            1 + (self.synthesis_window.shape[0] - 1) // self.stft_chunk_size
        )
        # 1 + (153 - 1) // 96 = 2, BSS_MLP_1comp_100p

    def init_buffers(self, batch_size, num_channels, device):
        return torch.zeros(
            batch_size * num_channels,
            self.istft_lookback,
            self.synthesis_window.shape[0],
            device=device,
        )

    def stft(self, audio_td: torch.Tensor, pad=True):
        """
        audio_td: Audio in the time-domain (B, C, t)
        Returns (B, RC, T, F), where R is real/imaginary
        """

        # Pad to get an evenly-divisible number of time-bins
        pad_amount = 0

        pad_size = (self.stft_back_pad, self.stft_front_pad)
        if pad:
            pad_size = (self.stft_back_pad, self.stft_front_pad)
            audio_td, pad_amount = mod_pad(
                audio_td, chunk_size=self.stft_chunk_size, pad=pad_size
            )

        B, M, T = audio_td.shape
        audio_td = audio_td.reshape(B * M, T)

        if audio_td.device != self.analysis_window.device:
            self.analysis_window = self.analysis_window.to(audio_td.device)

        X = torch.stft(
            audio_td,
            n_fft=self.nfft,
            hop_length=self.stft_chunk_size,
            win_length=self.nfft,
            window=self.analysis_window,
            center=False,
            normalized=False,
            return_complex=True,
        )
        X = torch.view_as_real(X)  # [B*M, F, T, 2]
        BM, _F, T, C = X.shape

        X = X.reshape(B, M, _F, T, C)  # [B, M, F, T, 2]
        X = X.permute(0, 4, 1, 3, 2)  # [B, 2, M, T, F]

        # (shoh10): Add this line
        X = X.flatten(1, 2)
        # X = X.reshape(B, 2*M, T, _F) # [B, 2M, T, F]

        return X, pad_amount

    def istft(self, audio_tfd: torch.Tensor, pad_amount=0, istft_buf=None):
        """
        audio_tfd: Audio in the time-frequency domain (B, R * C, T, F)
        pad_amount: Amount of padding that was added during STFT phase
        """
        if istft_buf is None:
            batch_size = audio_tfd.shape[0]
            num_channels = audio_tfd.shape[1] // 2
            istft_buf = self.init_buffers(batch_size, num_channels, audio_tfd.device)

        B, RC, T, F = audio_tfd.shape
        audio_tfd = audio_tfd.reshape(B, 2, RC // 2, T, F)
        audio_tfd = audio_tfd[:, 0] + 1j * audio_tfd[:, 1]  # [B, C, T, F]
        audio_tfd = audio_tfd.flatten(0, 1)  # [B * C, T, F]

        x = torch.fft.irfft(audio_tfd, dim=-1)  # [B * C, T, nfft]
        # oWS = chunksize + lookahead = 96 + 64 = 160
        # generally window
        # | lookback | chunksize | lookahead |
        # oWs use
        # | chunksize | lookahead |
        # So it does not get an effect from next frame
        # Then how do we get the synthesis window?
        oWS = self.synthesis_window.shape[0]
        x = x[..., -oWS:]  # [B * C, T, oWS]

        # Copy synthesis window to same device
        if self.synthesis_window.device != x.device:
            self.synthesis_window = self.synthesis_window.to(x.device)

        # Apply synthesis window
        x = x * self.synthesis_window.reshape(1, 1, -1)

        x = torch.cat([istft_buf, x], dim=1)  # Concat along T
        istft_buf = x[:, -istft_buf.shape[1] :]  # Update buffer

        # Get full signal
        # TODO(shoh10): Check this and make scratch istft function
        x = x.transpose(1, 2)  # [B * C, oWS, T]
        x = nn.functional.fold(
            x,
            output_size=(
                self.stft_chunk_size * x.shape[-1] + (oWS - self.stft_chunk_size),
                1,
            ),
            kernel_size=(oWS, 1),
            stride=(self.stft_chunk_size, 1),
        )  # [BS, 1, t, 1]
        x = x.squeeze(-1)

        x = x[
            ..., -T * self.stft_chunk_size - self.stft_front_pad : -self.stft_front_pad
        ]  # [BS, 1, t]
        x = x.reshape(B, RC // 2, x.shape[-1])  # [B, S, t]

        # Remove padding added during STFT
        if pad_amount > 0:
            x = x[..., :-pad_amount]

        return x, istft_buf


def ILD(x1, x2, tol=1e-6):
    # x - B, T
    # ILD - B, F, T
    ILD = torch.log10(torch.div(x1.abs() + tol, x2.abs() + tol))
    return ILD


def IPD(x1, x2, tol=1e-6):
    # x - B, T
    # ILD - B, F, T
    IPD = torch.angle(x1) - torch.angle(x2)
    IPD_cos = torch.cos(IPD)
    IPD_sin = torch.sin(IPD)
    IPD_map = torch.cat((IPD_sin, IPD_cos), dim=1)
    return IPD_map


def IPD_ONNX(real1, imag1, real2, imag2, norm, norm_ref, tol=1e-6):
    B, _, f, T = real2.shape

    real2 = real2.repeat((1, real1.shape[1], 1, 1))  # .reshape(B*(M-1), 1, f, T)
    imag2 = imag2.repeat((1, imag1.shape[1], 1, 1))  # .reshape(B*(M-1), 1, f, T)

    IPD_cos = (real1 * real2 + imag1 * imag2) / (norm * norm_ref + tol)
    IPD_sin = (real2 * imag1 - imag2 * real1) / (norm * norm_ref + tol)

    IPD_cos = IPD_cos.reshape(-1, 1, f, T)
    IPD_sin = IPD_sin.reshape(-1, 1, f, T)

    IPD_map = torch.cat((IPD_sin, IPD_cos), dim=1)

    IPD_map = IPD_map.reshape(B, 2 * imag1.shape[1], f, T)

    return IPD_map


def MC_features_ONNX(reals, imags, eps=1e-6):
    # Input: [B, M, F, T] or [B, M, T, F]
    r2, r1 = torch.split(reals, [1, reals.shape[1] - 1], dim=1)
    i2, i1 = torch.split(imags, [1, reals.shape[1] - 1], dim=1)

    # Compute magnitude
    norm = torch.sqrt(torch.square(reals) + torch.square(imags))
    norm_ref, norm = torch.split(norm, [1, norm.shape[1] - 1], dim=1)

    # Compute ILD
    ILD_m = torch.log10(torch.div(norm + eps, norm_ref + eps))

    # Compute IPD
    IPD_m = IPD_ONNX(r1, i1, r2, i2, norm, norm_ref)

    out = torch.cat([ILD_m, IPD_m], dim=1)  # [B, 3M-3, f, T]

    return out


def MC_features_ONNX_np(reals, imags, eps=1e-6):
    # Input: [M, F] or [M, F]
    # returns [3M-3, F]
    with torch.no_grad():
        reals = torch.from_numpy(reals).unsqueeze(1).unsqueeze(0)
        imags = torch.from_numpy(imags).unsqueeze(1).unsqueeze(0)

        feats = MC_features_ONNX(reals, imags).squeeze(0).squeeze(1)

        return feats.numpy()
