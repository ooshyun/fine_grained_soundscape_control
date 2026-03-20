from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------

def _mod_pad(x: torch.Tensor, chunk_size: int, pad: tuple[int, int]):
    """Pad *x* so its last dimension is divisible by *chunk_size*."""
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)
    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)
    return x, mod


def _get_perfect_synthesis_window(
    analysis_win: torch.Tensor,
    lookback: int,
    chunksize: int,
    lookahead: int,
) -> torch.Tensor:
    """Compute the perfect-reconstruction synthesis window."""
    nfft = lookback + chunksize + lookahead
    assert analysis_win.shape[0] == nfft

    oWS = chunksize + lookahead
    synthesis_win = torch.zeros(oWS)
    A = oWS
    B = chunksize
    N = nfft

    if (A % B) == 0:
        for i in range(A):
            num = analysis_win[N - A + i]
            denom = 0.0
            for k in range(A // B):
                denom += analysis_win[N - A + (i % B) + k * B] ** 2
            synthesis_win[i] = num / denom
    elif lookahead < chunksize:
        for i in range(A):
            if lookahead <= i < oWS - lookahead:
                synthesis_win[i] = 1.0 / analysis_win[lookback + i]
            else:
                num = analysis_win[lookback + i]
                denom = 0.0
                for k in range(A // B + 1):
                    denom += analysis_win[lookback + (i % B) + k * B] ** 2
                synthesis_win[i] = num / denom
    else:
        raise NotImplementedError(
            "This case is not handled for perfect window synthesis. "
            "Either let front pad be a multiple of chunk size, "
            "or make chunk size > front pad."
        )
    return synthesis_win


# ---------------------------------------------------------------------------
# FiLM (Feature-wise Linear Modulation)
# ---------------------------------------------------------------------------

class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer."""

    def __init__(self, input_channels: int, embedding_channels: int) -> None:
        super().__init__()
        self.a = nn.Linear(embedding_channels, input_channels)
        self.b = nn.Linear(embedding_channels, input_channels)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x: [B, C, *], emb: [B, D]
        a = self.a(emb)
        b = self.b(emb)
        while len(a.shape) != len(x.shape):
            a = a.unsqueeze(-1)
            b = b.unsqueeze(-1)
        return x * a + b


# ---------------------------------------------------------------------------
# GridNetBlock
# ---------------------------------------------------------------------------

class GridNetBlock(nn.Module):
    """Time-frequency processing block with intra-frame and inter-frame LSTMs.

    Intra-frame: bidirectional LSTM across frequency bins.
    Inter-frame: causal (unidirectional) LSTM across time frames.
    Both branches use residual connections and LayerNorm.
    """

    def __init__(
        self,
        latent_dim: int,
        n_freqs: int,
        hidden_channels: int = 32,
        freq_compression: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        time_domain_bidirectional = bidirectional

        self.H = hidden_channels
        self.n_freqs = n_freqs

        # Frequency compression / decompression
        self.compress_frequencies = freq_compression > 1
        if self.compress_frequencies:
            self.freq_compress = nn.Conv1d(
                latent_dim, latent_dim,
                kernel_size=freq_compression, stride=freq_compression,
            )
            self.act = nn.ReLU()
            self.freq_decompress = nn.ConvTranspose1d(
                latent_dim, latent_dim,
                kernel_size=freq_compression, stride=freq_compression,
                output_padding=n_freqs - (n_freqs // freq_compression) * freq_compression,
            )

        # Intra-frame: bidirectional LSTM across frequency bins
        self.intra_norm = nn.LayerNorm(latent_dim)
        self.intra_seq2seq = nn.LSTM(
            latent_dim, hidden_channels, 1, batch_first=True, bidirectional=True,
        )
        self.intra_linear = nn.Linear(2 * hidden_channels, latent_dim)

        # Inter-frame: causal LSTM across time frames
        self.inter_norm = nn.LayerNorm(latent_dim)
        self.inter_rnn = nn.LSTM(
            latent_dim, hidden_channels, 1,
            batch_first=True, bidirectional=time_domain_bidirectional,
        )
        self.inter_linear = nn.Linear(
            hidden_channels * (int(time_domain_bidirectional) + 1), latent_dim,
        )

        self.edge = False

    def init_buffers(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        c0 = torch.zeros(1, batch_size * self.n_freqs, self.H, device=device)
        h0 = torch.zeros(1, batch_size * self.n_freqs, self.H, device=device)
        return {"c0": c0, "h0": h0}

    def forward(
        self, x: torch.Tensor, init_state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            x: [B, T, Q, C] — batch, time, frequency, channels.
            init_state: LSTM hidden/cell states for inter-frame RNN.

        Returns:
            (output [B, T, Q, C], updated_state).
        """
        B, T, Q, C = x.shape

        if init_state is None:
            init_state = self.init_buffers(B, x.device)

        # --- Intra-frame (frequency) ---
        input_ = x
        intra_rnn = x.reshape(B * T, Q, C)

        if self.compress_frequencies:
            intra_rnn = self.freq_compress(intra_rnn.transpose(1, 2))
            intra_rnn = self.act(intra_rnn).transpose(1, 2)

        intra_rnn = self.intra_norm(intra_rnn)
        intra_rnn, _ = self.intra_seq2seq(intra_rnn)
        intra_rnn = self.intra_linear(intra_rnn)

        if self.compress_frequencies:
            intra_rnn = self.freq_decompress(intra_rnn.transpose(1, 2))
            intra_rnn = intra_rnn.transpose(1, 2)

        intra_rnn = intra_rnn.view(B, T, Q, C)
        intra_rnn = intra_rnn + input_  # residual

        # --- Inter-frame (time) ---
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)

        h0 = init_state["h0"]
        c0 = init_state["c0"]

        inter_rnn = inter_rnn.transpose(1, 2).reshape(B * Q, T, C)
        self.inter_rnn.flatten_parameters()
        inter_rnn, (h0, c0) = self.inter_rnn(inter_rnn, (h0, c0))
        inter_rnn = inter_rnn.view(B, Q, T, self.H).transpose(1, 2)

        init_state["h0"] = h0
        init_state["c0"] = c0

        inter_rnn = self.inter_linear(inter_rnn)
        inter_rnn = inter_rnn + input_  # residual

        return inter_rnn, init_state


# ---------------------------------------------------------------------------
# MultiFiLMGuidedTFNet
# ---------------------------------------------------------------------------

class MultiFiLMGuidedTFNet(nn.Module):
    """Conv2d projection -> N GridNetBlocks (+ optional FiLM) -> DeConv2d.

    FiLM layers are placed at ``film_positions`` to condition on a speaker
    embedding.
    """

    def __init__(
        self,
        *,
        spk_dim: int = 256,
        latent_dim: int = 48,
        n_srcs: int = 2,
        n_fft: int = 128,
        num_inputs: int = 1,
        n_layers: int = 6,
        hidden_channels: int = 32,
        bidirectional: bool = False,
        use_first_ln: bool = False,
        embedding_type: str = "embedding",
        embedding_dim: int = 0,
        embedding_activation: str = "",
        embedding_init: str = "",
        film_preset: str = "all_except_first",
        film_positions: list[int] | None = None,
        freq_compression: int = 1,
    ) -> None:
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.num_inputs = num_inputs
        self.freq_compression = freq_compression
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
        self.n_freqs = n_freqs
        self.latent_dim = latent_dim
        self.use_first_ln = use_first_ln
        self.n_fft = n_fft

        t_ksize = 3
        self.t_ksize = t_ksize
        ks, padding = (t_ksize, t_ksize), (0, 1)

        # Project to latent space
        self.conv = nn.Conv2d(num_inputs, latent_dim, ks, padding=padding)

        if use_first_ln:
            self.ln = nn.LayerNorm(latent_dim)

        # Embedding layer
        self.embedding_dim: int
        if embedding_type == "" or embedding_dim == 0:
            self.emb_layer: nn.Module | None = None
            self.embedding_dim = spk_dim
        else:
            embedding_type = embedding_type.lower()
            if embedding_type == "linear":
                layers: list[nn.Module] = [nn.Linear(spk_dim, embedding_dim)]
                if embedding_activation == "relu":
                    layers.append(nn.ReLU())
                elif embedding_activation == "tanh":
                    layers.append(nn.Tanh())
                self.emb_layer = nn.Sequential(*layers)
            elif embedding_type == "embedding":
                self.emb_layer = nn.Embedding(spk_dim, embedding_dim)
            else:
                raise ValueError(f"Invalid embedding type: {embedding_type}")

            if embedding_init == "xavier_uniform":
                for m in (self.emb_layer.modules() if hasattr(self.emb_layer, 'modules') else [self.emb_layer]):
                    if isinstance(m, (nn.Linear, nn.Embedding)):
                        nn.init.xavier_uniform_(m.weight)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.zeros_(m.bias)

            self.embedding_dim = embedding_dim

        # FiLM positions
        if film_positions is None:
            film_positions = []
        film_positions = sorted(film_positions)

        if film_preset == "all":
            film_positions = list(range(n_layers))
        elif film_preset == "all_except_first":
            film_positions = list(range(1, n_layers))
        elif film_preset == "first":
            film_positions = [0]

        if len(film_positions) > n_layers:
            film_positions = film_positions[:n_layers]
        self.film_positions = film_positions

        self.film_layers = nn.ModuleDict({
            f"film_layer_{pos}": FiLM(latent_dim, self.embedding_dim)
            for pos in film_positions
        })

        # GridNetBlocks
        self.blocks = nn.ModuleList([
            GridNetBlock(
                n_freqs=n_freqs,
                latent_dim=latent_dim,
                hidden_channels=hidden_channels,
                freq_compression=freq_compression,
                bidirectional=bidirectional,
            )
            for _ in range(n_layers)
        ])

        # Project back to TF-domain
        self.deconv = nn.ConvTranspose2d(
            latent_dim, n_srcs * 2, ks, padding=(t_ksize - 1, 1),
        )

    def init_buffers(self, batch_size: int, device: torch.device) -> dict[str, Any]:
        conv_buf = torch.zeros(
            batch_size, self.num_inputs, self.t_ksize - 1, self.n_freqs, device=device,
        )
        deconv_buf = torch.zeros(
            batch_size, self.latent_dim, self.t_ksize - 1, self.n_freqs, device=device,
        )
        block_bufs: dict[str, Any] = {}
        for i in range(len(self.blocks)):
            block_bufs[f"buf{i}"] = self.blocks[i].init_buffers(batch_size, device)
        return {"conv_buf": conv_buf, "deconv_buf": deconv_buf, "block_bufs": block_bufs}

    def forward(
        self,
        current_input: torch.Tensor,
        embedding: torch.Tensor,
        input_state: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Args:
            current_input: [B, CM, T, F] — real/imag concatenated input.
            embedding: [B, D] or [B] (integer indices for nn.Embedding).
            input_state: Buffers from previous call (streaming).

        Returns:
            (output [B, S, T, 2F], updated_state).
        """
        n_batch, _, n_frames, n_freqs = current_input.shape
        batch = current_input

        if input_state is None:
            input_state = self.init_buffers(n_batch, current_input.device)

        conv_buf = input_state["conv_buf"]
        deconv_buf = input_state["deconv_buf"]
        gridnet_buf = input_state["block_bufs"]

        # Causal conv padding
        batch = torch.cat((conv_buf, batch), dim=2)
        conv_buf = batch[:, :, -(self.t_ksize - 1):, :]

        batch = self.conv(batch)  # [B, D, T, F]

        if self.use_first_ln:
            batch = batch.permute(0, 2, 3, 1)  # [B, T, F, D]
            batch = self.ln(batch)
            batch = batch.permute(0, 3, 1, 2)  # [B, D, T, F]

        # Apply embedding
        if self.emb_layer is not None:
            embedding = self.emb_layer(embedding)

        # Process through blocks with FiLM
        for ii in range(self.n_layers):
            if ii == 0:
                if ii in self.film_positions:
                    batch = self.film_layers[f"film_layer_{ii}"](batch, embedding)
                batch = batch.permute(0, 2, 3, 1)  # [B, T, F, D]
            else:
                if ii in self.film_positions:
                    batch = batch.permute(0, 3, 1, 2)  # [B, D, T, F]
                    batch = self.film_layers[f"film_layer_{ii}"](batch, embedding)
                    batch = batch.permute(0, 2, 3, 1)  # [B, T, F, D]

            batch, gridnet_buf[f"buf{ii}"] = self.blocks[ii](batch, gridnet_buf[f"buf{ii}"])

        batch = batch.permute(0, 3, 1, 2)  # [B, D, T, F]

        # Causal deconv padding
        batch = torch.cat((deconv_buf, batch), dim=2)
        deconv_buf = batch[:, :, -(self.t_ksize - 1):, :]

        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]
        batch = batch.view(n_batch, self.n_srcs, 2, n_frames, n_freqs)
        batch = batch.transpose(2, 3).reshape(
            n_batch, self.n_srcs, n_frames, 2 * n_freqs,
        )

        input_state["conv_buf"] = conv_buf
        input_state["deconv_buf"] = deconv_buf
        input_state["block_bufs"] = gridnet_buf

        return batch, input_state


# ---------------------------------------------------------------------------
# TFGridNet (top-level STFT wrapper)
# ---------------------------------------------------------------------------

class TFGridNet(nn.Module):
    """Target Sound Extraction model: STFT -> MultiFiLMGuidedTFNet -> iSTFT.

    This is the top-level module that wraps the separator network with
    analysis (STFT) and synthesis (iSTFT with perfect-reconstruction window).
    """

    def __init__(
        self,
        stft_chunk_size: int = 96,
        stft_pad_size: int = 64,
        stft_back_pad: int = 96,
        num_input_channels: int = 2,
        num_output_channels: int = 1,
        num_layers: int = 6,
        latent_dim: int = 32,
        hidden_channels: int = 64,
        speaker_dim: int = 20,
        bidirectional: bool = False,
        film_preset: str = "all_except_first",
        film_positions: list[int] | None = None,
        use_first_ln: bool = False,
        embedding_type: str = "embedding",
        embedding_dim: int = 0,
        embedding_activation: str = "",
        embedding_init: str = "",
        freq_compression: int = 1,
    ) -> None:
        super().__init__()

        self.nI = num_input_channels
        self.nO = num_output_channels

        num_separator_inputs = self.nI * 2  # real + imag per channel

        self.stft_chunk_size = stft_chunk_size
        self.stft_pad_size = stft_pad_size
        self.stft_back_pad = stft_back_pad
        self.nfft = stft_back_pad + stft_chunk_size + stft_pad_size
        self.nfreqs = self.nfft // 2 + 1

        # Analysis / synthesis windows (rectangular)
        self.analysis_window = torch.ones(self.nfft)
        self.synthesis_window = _get_perfect_synthesis_window(
            self.analysis_window, stft_back_pad, stft_chunk_size, stft_pad_size,
        )

        self.istft_lookback = 1 + (self.synthesis_window.shape[0] - 1) // self.stft_chunk_size

        # Separator
        self.tfgridnet = MultiFiLMGuidedTFNet(
            spk_dim=speaker_dim,
            n_fft=self.nfft,
            num_inputs=num_separator_inputs,
            latent_dim=latent_dim,
            n_srcs=num_output_channels,
            n_layers=num_layers,
            hidden_channels=hidden_channels,
            bidirectional=bidirectional,
            use_first_ln=use_first_ln,
            embedding_type=embedding_type,
            embedding_dim=embedding_dim,
            embedding_activation=embedding_activation,
            embedding_init=embedding_init,
            film_preset=film_preset,
            film_positions=film_positions,
            freq_compression=freq_compression,
        )

    def init_buffers(self, batch_size: int, device: torch.device) -> dict[str, Any]:
        buffers: dict[str, Any] = {}
        buffers["tfnet_bufs"] = self.tfgridnet.init_buffers(batch_size, device)
        buffers["istft_buf"] = torch.zeros(
            batch_size * self.nO,
            self.synthesis_window.shape[0],
            self.istft_lookback,
            device=device,
        )
        return buffers

    # ----- STFT analysis -----

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Convert time-domain audio to TF-domain representation.

        Args:
            x: [B, M, T] — batch, microphones, time samples.

        Returns:
            [B, 2M, T_frames, F] — real/imag concatenated.
        """
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
        BM, _F, Tf, C = x.shape
        x = x.reshape(B, M, _F, Tf, C)     # [B, M, F, T, 2]
        x = x.permute(0, 4, 1, 3, 2)       # [B, 2, M, T, F]
        x = x.reshape(B, C * M, Tf, _F)     # [B, 2M, T, F]
        return x

    # ----- iSTFT synthesis -----

    def synthesis(
        self, x: torch.Tensor, input_state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Convert TF-domain back to time-domain via iSTFT with overlap-add.

        Args:
            x: [B, S, T, 2F] — separator output.
            input_state: Contains ``istft_buf`` for streaming OLA.

        Returns:
            (audio [B, S, t], updated_state).
        """
        istft_buf = input_state["istft_buf"]

        x = x.transpose(2, 3)  # [B, S, 2F, T]
        B, S, CF, T = x.shape
        X = x.reshape(B * S, CF, T)
        X = X.reshape(B * S, 2, -1, T).permute(0, 2, 3, 1)  # [BS, F, T, 2]
        X = X[..., 0] + 1j * X[..., 1]

        x = torch.fft.irfft(X, dim=1)  # [BS, nfft, T]
        oW = self.synthesis_window.shape[0]
        x = x[:, -oW:]  # [BS, oW, T]

        # Apply synthesis window
        x = x * self.synthesis_window.unsqueeze(0).unsqueeze(-1).to(x.device)

        # Overlap-add with buffer
        x = torch.cat([istft_buf, x], dim=-1)
        istft_buf = x[..., -istft_buf.shape[-1]:]

        x = F.fold(
            x,
            output_size=(self.stft_chunk_size * x.shape[-1] + (oW - self.stft_chunk_size), 1),
            kernel_size=(oW, 1),
            stride=(self.stft_chunk_size, 1),
        )

        # Crop to valid region
        if self.stft_pad_size > 0:
            x = x[:, :, -T * self.stft_chunk_size - self.stft_pad_size:-self.stft_pad_size]
        else:
            x = x[:, :, -T * self.stft_chunk_size:]
        x = x.reshape(B, S, -1)

        input_state["istft_buf"] = istft_buf
        return x, input_state

    # ----- Forward -----

    def forward(
        self,
        inputs_dict: dict[str, torch.Tensor],
        input_state: dict[str, Any] | None = None,
        pad: bool = True,
    ) -> dict[str, Any]:
        """
        Args:
            inputs_dict: Must contain ``"mixture"`` [B, M, t] and
                ``"embedding"`` [B] (int indices) or [B, D] (continuous).
            input_state: Streaming buffers (``None`` for offline).
            pad: Whether to mod-pad the input.

        Returns:
            ``{"output": [B, S, t], "next_state": ...}``
        """
        x = inputs_dict["mixture"]
        embedding = inputs_dict["embedding"]

        if input_state is None:
            input_state = self.init_buffers(x.shape[0], x.device)

        mod = 0
        if pad:
            pad_size = (self.stft_back_pad, self.stft_pad_size)
            x, mod = _mod_pad(x, chunk_size=self.stft_chunk_size, pad=pad_size)

        # Analysis
        x = self.extract_features(x)  # [B, 2M, T, F]

        # Separator
        x, input_state["tfnet_bufs"] = self.tfgridnet(
            x, embedding, input_state["tfnet_bufs"],
        )

        # Synthesis
        x, next_state = self.synthesis(x, input_state)

        if mod != 0:
            x = x[:, :, :-mod]

        return {"output": x, "next_state": next_state}


# ---------------------------------------------------------------------------
# Pretrained model loading
# ---------------------------------------------------------------------------

_MODEL_NAME_MAP = {
    "orange_pi": "tfgridnet_large_snr_ctl_v2_1ch_1spk_1out_20000samples_20sounds_16000sr_96chunk_film_all_except_first_onflight",
    "raspberry_pi": "tfgridnet_small_snr_ctl_v2_1ch_1spk_1out_20000samples_20sounds_16000sr_96chunk_film_all_except_first_onflight",
    "neuralaid": "tfmlpnet_snr_ctl_v2_1ch_1spk_1out_20000samples_20sounds_16000sr_96chunk_film_all_except_first_onflight",
}


def load_pretrained(
    repo_id: str = "ooshyun/semantic_listening",
    model_name: str = "orange_pi",
) -> TFGridNet:
    """Download and instantiate a pretrained :class:`TFGridNet` from HuggingFace.

    Args:
        repo_id: HuggingFace Hub repository ID.
        model_name: One of ``"orange_pi"``, ``"raspberry_pi"``, ``"neuralaid"``.

    Returns:
        A :class:`TFGridNet` with pretrained weights loaded.
    """
    import json

    from huggingface_hub import hf_hub_download

    if model_name not in _MODEL_NAME_MAP:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Available: {list(_MODEL_NAME_MAP.keys())}"
        )

    run_dir = _MODEL_NAME_MAP[model_name]

    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{run_dir}/config.json",
    )
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{run_dir}/checkpoints/best.pt",
    )

    with open(config_path) as f:
        config = json.load(f)

    # Build model from config — handle nested pl_module_args.model_params
    if "pl_module_args" in config and "model_params" in config["pl_module_args"]:
        mp = config["pl_module_args"]["model_params"]
        block_params = mp.get("block_model_params", {})
        emb_params = mp.get("embedding_params", {})
        film_params = mp.get("film_params", {})
    elif "model" in config:
        mp = config["model"]
        block_params = mp
        emb_params = mp
        film_params = mp
    else:
        mp = config
        block_params = config
        emb_params = config
        film_params = config

    model = TFGridNet(
        stft_chunk_size=mp.get("stft_chunk_size", 96),
        stft_pad_size=mp.get("stft_pad_size", 64),
        stft_back_pad=mp.get("stft_back_pad", 96),
        num_input_channels=mp.get("num_input_channels", 2),
        num_output_channels=mp.get("num_output_channels", 1),
        num_layers=mp.get("num_layers", 6),
        latent_dim=mp.get("latent_dim", 32),
        hidden_channels=block_params.get("hidden_channels", 64),
        speaker_dim=mp.get("speaker_dim", 20),
        bidirectional=block_params.get("bidirectional", False),
        film_preset=film_params.get("film_preset", "all_except_first"),
        use_first_ln=mp.get("use_first_ln", False),
        embedding_type=emb_params.get("embedding_type", ""),
        embedding_dim=emb_params.get("embedding_dim", 0),
        embedding_activation=emb_params.get("embedding_activation", ""),
        embedding_init=emb_params.get("embedding_init", ""),
        freq_compression=block_params.get("freq_compression", 1),
    )

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    # Handle wrapped state dicts (e.g. {"model": ...})
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    logger.info("Loaded pretrained model '%s' from %s", model_name, repo_id)
    return model
