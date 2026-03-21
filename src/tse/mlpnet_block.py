import torch
import torch.nn as nn
from src.tse.batched_lstm import BatchedLSTM
# import src.utils as utils  # debug only


class MLPStack(nn.Module):
    def __init__(self, n_freqs, n_channels, num_layers=2, use_norm=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_norm = use_norm
        for i in range(num_layers):
            layer = nn.ModuleDict()

            if self.use_norm:
                layer["bn1"] = nn.BatchNorm1d(num_features=n_channels)
                layer["bn2"] = nn.BatchNorm1d(num_features=n_channels)

            layer["linear1a"] = nn.Conv1d(n_freqs, n_freqs, 1)
            layer["act1"] = nn.ReLU()
            layer["linear1b"] = nn.Conv1d(n_freqs, n_freqs, 1)

            layer["linear2a"] = nn.Linear(n_channels, n_channels)
            layer["act2"] = nn.ReLU()
            layer["linear2b"] = nn.Linear(n_channels, n_channels)

            self.layers.append(layer)

    def forward(self, x):
        """
        x: (B, F, C)
        """

        for layer in self.layers:
            input_ = x

            x = layer["linear1a"](x)  # [B, F, C]

            # Frequency-wise
            if self.use_norm:
                x = layer["bn1"](x.transpose(1, 2)).transpose(1, 2)  # [B, F, C]

            x = layer["act1"](x)  # [B, F, C]
            x = layer["linear1b"](x)  # [B, F, C]

            # Residual
            x = x + input_

            input_ = x  # [B, F, C]

            x = layer["linear2a"](x)  # [B, F, C]
            # Channel-wise
            if self.use_norm:
                x = layer["bn2"](x.transpose(1, 2)).transpose(1, 2)  # Needs [B, C, F]

            x = layer["act2"](x)  # [B, F, C]
            x = layer["linear2b"](x)  # [B, F, C]

            # Residual
            x = x + input_  # [B, F, C]

        return x, None


class MLPBlock(nn.Module):
    def __init__(
        self,
        latent_dim,
        n_freqs,
        hidden_channels=32,
        freq_compression=1,
        mlp_reps=2,
        use_norm=False,
    ):
        super().__init__()
        time_domain_bidirectional = False  # Causal

        if hidden_channels < 0:
            hidden_channels = latent_dim

        self.H = hidden_channels
        self.n_freqs = n_freqs

        self.compress_frequencies = freq_compression > 1

        # Frequency compression/decompression
        if self.compress_frequencies:
            self.freq_compress = nn.Conv1d(
                in_channels=latent_dim,
                out_channels=latent_dim,
                kernel_size=freq_compression,
                stride=freq_compression,
            )
            self.act = nn.ReLU()

            self.freq_decompress = nn.ConvTranspose1d(
                in_channels=latent_dim,
                out_channels=latent_dim,
                kernel_size=freq_compression,
                stride=freq_compression,
                output_padding=n_freqs
                - (n_freqs // freq_compression) * freq_compression,
            )

        # Intra-frame processing
        self.intra_seq2seq = MLPStack(
            n_freqs // freq_compression,
            latent_dim,
            num_layers=mlp_reps,
            use_norm=use_norm,
        )
        self.intra_linear = nn.Linear(
            latent_dim,
            latent_dim,
        )

        # Time-domain LSTM
        self.inter_rnn = nn.LSTM(
            latent_dim,
            hidden_channels,
            1,
            batch_first=True,
            bidirectional=time_domain_bidirectional,
        )
        self.inter_linear = nn.Linear(
            hidden_channels * (time_domain_bidirectional + 1), latent_dim
        )

        # Edge mode
        self.edge = False

    def init_buffers(self, batch_size, device):
        ctx_buf = {}

        if not self.edge:
            c0 = torch.zeros((1, batch_size * self.n_freqs, self.H), device=device)
            h0 = torch.zeros((1, batch_size * self.n_freqs, self.H), device=device)
        else:
            c0 = torch.zeros((1, self.H, batch_size * self.n_freqs), device=device)
            h0 = torch.zeros((1, self.H, batch_size * self.n_freqs), device=device)

        ctx_buf["c0"] = c0
        ctx_buf["h0"] = h0

        return ctx_buf

    def forward(self, x, init_state=None, debugDir=None):
        """GridNetBlock Forward.

        Args:
            x: [B, T, Q, C]
            out: [B, T, Q, C]
        """

        if init_state is None:
            init_state = self.init_buffers(x.shape[0], Q.device)

        B, T, Q, C = x.shape

        # Store input for residual connection
        input_ = x

        intra_rnn = x.reshape(B * T, Q, C)  # [B * T, Q, C]

        # Compress frequencies before intra-frame
        if self.compress_frequencies:
            intra_rnn = self.freq_compress(
                intra_rnn.transpose(1, 2)
            )  # [BT, C, K] K = Q // stride
            intra_rnn = self.act(intra_rnn).transpose(1, 2)  # [BT, K, C]

        # Intra-frame processing
        intra_rnn, _ = self.intra_seq2seq(intra_rnn)  # [BT, *, H]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, *, C]

        # Decompress frequencies
        if self.compress_frequencies:
            intra_rnn = self.freq_decompress(intra_rnn.transpose(1, 2))  # [BT, C, Q]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, Q, C]

        intra_rnn = intra_rnn.view(B, T, Q, C)  # [B, T, Q, C]
        intra_rnn = intra_rnn + input_  # [B, T, Q, C]
        out = intra_rnn

        if debugDir is not None:
            utils.save_tensor(debugDir[1], f"IntraRNNOut{debugDir[0]}.pt", intra_rnn)

        # Inter-frame processing
        input_ = intra_rnn  # [B, T, Q, C]

        inter_rnn = input_  # [B, T, Q, C]

        h0 = init_state["h0"]
        c0 = init_state["c0"]

        # Inter frame processing with state updates
        if not self.edge:
            inter_rnn = inter_rnn.transpose(1, 2).reshape(B * Q, T, C)  # [BQ, T, C]

            self.inter_rnn.flatten_parameters()
            inter_rnn, (h0, c0) = self.inter_rnn(inter_rnn, (h0, c0))  # [BQ, -1, H]
            inter_rnn = inter_rnn.reshape([B, Q, T, self.H]).transpose(
                1, 2
            )  # [B, T, Q, C]
        else:
            assert T == 1, f"In edge mode, there must be only 1 frame. Found {T}"
            inter_rnn = inter_rnn.squeeze(1)  # [B, Q, H]
            inter_rnn, (h0, c0) = self.inter_rnn(inter_rnn, (h0, c0))  # [B, Q, H]
            inter_rnn = inter_rnn.unsqueeze(1)  # [B, T, Q, H]

        init_state["h0"] = h0
        init_state["c0"] = c0

        inter_rnn = self.inter_linear(inter_rnn)  # [*, C]

        if debugDir is not None:
            utils.save_tensor(debugDir[1], f"InterRNNOut{debugDir[0]}.pt", inter_rnn)

        inter_rnn = inter_rnn + input_  # [B, T, Q, C]

        out = inter_rnn

        return out, init_state

    def edge_mode(self):
        state_dict = self.inter_rnn.state_dict()
        self.inter_rnn = BatchedLSTM(
            self.inter_rnn.input_size, self.inter_rnn.hidden_size
        )
        self.inter_rnn.set_weights(state_dict)

        self.edge = True
