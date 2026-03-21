import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
from src.tse.dsp import get_perfect_synthesis_window


def _import_attr(name):
    """Dynamically import a class/function from a dotted path string."""
    module_path, _, attr_name = name.rpartition(".")
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def mod_pad(x, chunk_size, pad):
    # Mod pad the input to perform integer number of
    # inferences
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)

    return x, mod


# A TF-domain network guided by an embedding vector
class Net(nn.Module):
    def __init__(
        self,
        model_name,
        block_model_name,
        block_model_params,
        speaker_dim=256,
        stft_chunk_size=64,
        stft_pad_size=32,
        stft_back_pad=32,
        num_input_channels=1,
        num_output_channels=1,
        num_layers=6,
        latent_dim=48,
        embedding_params={},
        film_params={},
        use_first_ln=False,  # For TF-GridNet
    ):
        super(Net, self).__init__()

        # num input/output channels
        self.nI = num_input_channels
        self.nO = num_output_channels

        # num channels to the Delayed TF-network
        num_separator_inputs = self.nI * 2

        self.stft_chunk_size = stft_chunk_size
        self.stft_pad_size = stft_pad_size
        self.stft_back_pad = stft_back_pad

        # Input conv to convert input audio to a latent representation
        self.nfft = stft_back_pad + stft_chunk_size + stft_pad_size

        self.nfreqs = self.nfft // 2 + 1

        # Construct synthesis/analysis windows (rect)
        window_fn = lambda x: np.ones(x)
        self.analysis_window = torch.from_numpy(window_fn(self.nfft)).float()
        self.synthesis_window = get_perfect_synthesis_window(
            self.analysis_window, stft_back_pad, stft_chunk_size, stft_pad_size
        )

        # Number of chunks to use for overlap & add
        self.istft_lookback = (
            1 + (self.synthesis_window.shape[0] - 1) // self.stft_chunk_size
        )

        # TF-Network
        self.tfgridnet = _import_attr(model_name)(
            block_model_name,
            block_model_params,
            spk_dim=speaker_dim,
            n_fft=self.nfft,
            num_inputs=num_separator_inputs,
            latent_dim=latent_dim,
            n_srcs=num_output_channels,
            n_layers=num_layers,
            use_first_ln=use_first_ln,
            embedding_params=embedding_params,
            film_params=film_params,
        )

    def init_buffers(self, batch_size, device):
        buffers = {}

        buffers["tfnet_bufs"] = self.tfgridnet.init_buffers(batch_size, device)
        buffers["istft_buf"] = torch.zeros(
            batch_size * self.nO,
            self.synthesis_window.shape[0],
            self.istft_lookback,
            device=device,
        )

        return buffers

    def extract_features(self, x):
        """
        x: (B, M, T)
        returns: (B, C*M, T, F)
        """
        # torch.Size([1, 2, 220500])
        B, M, T = x.shape

        x = x.reshape(B * M, T)
        x = torch.stft(
            x,
            n_fft=self.nfft,
            hop_length=self.stft_chunk_size,
            win_length=self.nfft,
            window=self.analysis_window.to(x.device),
            center=False,
            normalized=False,
            return_complex=True,
        )
        x = torch.view_as_real(x)  # [B*M, F, T, 2]
        BM, _F, T, C = x.shape

        x = x.reshape(B, M, _F, T, C)  # [B, M, F, T, 2]

        x = x.permute(0, 4, 1, 3, 2)  # [B, 2, M. T, F]
        x = x.reshape(B, C * M, T, _F)  # [B, 2M, T, F]

        return x

    def synthesis(self, x, input_state):
        """
        x: (B, S, T, C*F)
        returns: (B, S, t)
        """
        istft_buf = input_state["istft_buf"]

        x = x.transpose(2, 3)  # [B, S, CF, T]

        B, S, CF, T = x.shape
        X = x.reshape(B * S, CF, T)
        X = X.reshape(B * S, 2, -1, T).permute(0, 2, 3, 1)  # [BS, F, T, C]
        X = X[..., 0] + 1j * X[..., 1]

        x = torch.fft.irfft(X, dim=1)  # [BS, iW, T]
        x = x[:, -self.synthesis_window.shape[0] :]  # [BS, oW, T]

        # Apply synthesis window
        x = x * self.synthesis_window.unsqueeze(0).unsqueeze(-1).to(x.device)

        oW = self.synthesis_window.shape[0]

        # Concatenate blocks from previous IFFTs
        x = torch.cat([istft_buf, x], dim=-1)
        istft_buf = x[..., -istft_buf.shape[1] :]  # Update buffer

        # Get full signal
        x = F.fold(
            x,
            output_size=(
                self.stft_chunk_size * x.shape[-1] + (oW - self.stft_chunk_size),
                1,
            ),
            kernel_size=(oW, 1),
            stride=(self.stft_chunk_size, 1),
        )  # [BS, 1, t]

        # Drop samples from previous chunks and from pad
        if self.stft_pad_size > 0:
            x = x[
                :, :, -T * self.stft_chunk_size - self.stft_pad_size : -self.stft_pad_size
            ]
        else:
            x = x[:, :, -T * self.stft_chunk_size :]
        x = x.reshape(B, S, -1)  # [B, S, t]

        input_state["istft_buf"] = istft_buf

        return x, input_state

    def predict(self, x, embedding, input_state, pad=True):
        """
        B: batch
        M: mic
        t: time step (time-domain)
        x: (B, M, t)
        R: real or imaginary
        S: n_src
        """

        mod = 0
        if pad:
            pad_size = (self.stft_back_pad, self.stft_pad_size)
            x, mod = mod_pad(x, chunk_size=self.stft_chunk_size, pad=pad_size)

        # Time-domain to TF-domain
        x = self.extract_features(x)  # [B, RM, T, F]

        x, input_state["tfnet_bufs"] = self.tfgridnet(
            x, embedding, input_state["tfnet_bufs"]
        )

        # TF-domain to time-domain
        x, next_state = self.synthesis(x, input_state)  # [B, S * M, t], S: n_src

        if mod != 0:
            x = x[:, :, :-mod]

        return x, next_state

    def forward(self, inputs, input_state=None, pad=True):
        x = inputs["mixture"]
        # Use label_vector (float one-hot) as embedding when available,
        # matching the original training code (semhearing_hl_module.py).
        embedding = inputs.get("label_vector", inputs["embedding"])

        # Ensure embedding is (B, D) float for FiLM layers.
        if isinstance(embedding, torch.Tensor):
            if embedding.dtype in (torch.long, torch.int, torch.int32):
                embedding = torch.nn.functional.one_hot(
                    embedding.long(), num_classes=self.tfgridnet.embedding_dim
                ).float()
            elif embedding.dtype != torch.float32:
                embedding = embedding.float()
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)

        # Create empty state if it is not passed
        if input_state is None:
            input_state = self.init_buffers(x.shape[0], x.device)

        B, M, t = x.shape
        x, next_state = self.predict(x, embedding, input_state, pad)

        return {"output": x, "next_state": next_state}
