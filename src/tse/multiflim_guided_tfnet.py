import torch
import torch.nn as nn
from src.tse.film import FiLM

import logging

# Logging is configured via setup_logging in entry point scripts
logger = logging.getLogger(__name__)


def _import_attr(name):
    """Dynamically import a class/function from a dotted path string."""
    module_path, _, attr_name = name.rpartition(".")
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


# A TF-domain network guided by an embedding vector with restructured FiLM placement
class MultiFiLMGuidedTFNet(nn.Module):
    def __init__(
        self,
        block_model_name,
        block_model_params,
        spk_dim=256,
        latent_dim=48,
        n_srcs=2,
        n_fft=128,
        num_inputs=1,
        n_layers=6,
        use_first_ln=False,  # For TF-GridNet
        embedding_params={},
        film_params={},
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.num_inputs = num_inputs
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
        embedding_dim, embedding_type, embedding_activation, embedding_init = (
            embedding_params["embedding_dim"],
            embedding_params["embedding_type"],
            embedding_params["embedding_activation"],
            embedding_params["embedding_init"],
        )

        if embedding_type == "" or embedding_dim == 0:
            self.embedding = None
            self.embedding_type = None
            self.embedding_dim = spk_dim
        else:
            logging.info(
                f"Use embedding layer with type: {embedding_type} and dimension: {embedding_dim}"
            )
            embedding_type = embedding_type.lower()
            if embedding_type == "linear":
                if embedding_activation == "relu":
                    self.embedding = nn.Sequential(
                        nn.Linear(spk_dim, embedding_dim), nn.ReLU()
                    )
                elif embedding_activation == "tanh":
                    self.embedding = nn.Sequential(
                        nn.Linear(spk_dim, embedding_dim), nn.Tanh()
                    )
                else:
                    logging.warning(
                        f"Invalid embedding activation: {embedding_activation}, using no activation"
                    )
                    self.embedding = nn.Sequential(
                        nn.Linear(spk_dim, embedding_dim),
                    )
            elif embedding_type == "embedding":
                self.embedding = nn.Embedding(spk_dim, embedding_dim)
            else:
                raise ValueError(f"Invalid embedding type: {embedding_type}")

            if embedding_type is not None and embedding_init == "xavier_uniform":
                if isinstance(self.embedding, nn.Linear):
                    nn.init.xavier_uniform_(self.embedding.weight)
                    if self.embedding.bias is not None:
                        nn.init.zeros_(self.embedding.bias)
                elif isinstance(self.embedding, nn.Embedding):
                    nn.init.xavier_uniform_(self.embedding.weight)

            self.embedding_type = embedding_type
            self.embedding_dim = embedding_dim

        # FiLM Layers
        film_positions, film_preset = (
            film_params["film_positions"],
            film_params["film_preset"],
        )

        film_positions = sorted(film_positions)

        if film_preset == "all":
            film_positions = list(range(n_layers))
        elif film_preset == "all_except_first":
            film_positions = list(range(1, n_layers))
        elif film_preset == "first":
            film_positions = list(range(0, 1))

        if len(film_positions) > n_layers:
            film_positions = film_positions[:n_layers]

        self.film_positions = film_positions

        # Create FiLM layers for each block
        self.film_layers = nn.ModuleDict(
            {
                f"film_layer_{pos}": FiLM(
                    input_channels=latent_dim, embedding_channels=self.embedding_dim
                )
                for pos in film_positions
            }
        )

        logger.debug(f"film_preset: {film_preset}")
        logger.debug(f"film_positions: {film_positions}")
        logger.debug(f"film_layers: {self.film_layers}")

        # Process through a stack of blocks
        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                _import_attr(block_model_name)(
                    n_freqs=n_freqs, latent_dim=latent_dim, **block_model_params
                )
            )

        # Project back to TF-Domain
        self.deconv = nn.ConvTranspose2d(
            latent_dim, n_srcs * 2, ks, padding=(self.t_ksize - 1, 1)
        )

    def init_buffers(self, batch_size, device):
        conv_buf = torch.zeros(
            batch_size, self.num_inputs, self.t_ksize - 1, self.n_freqs, device=device
        )

        deconv_buf = torch.zeros(
            batch_size, self.latent_dim, self.t_ksize - 1, self.n_freqs, device=device
        )

        block_buffers = {}
        for i in range(len(self.blocks)):
            block_buffers[f"buf{i}"] = self.blocks[i].init_buffers(batch_size, device)

        return dict(conv_buf=conv_buf, deconv_buf=deconv_buf, block_bufs=block_buffers)

    def forward(
        self, current_input: torch.Tensor, embedding: torch.Tensor, input_state
    ) -> torch.Tensor:
        """
        B: batch, M: mic, F: freq bin, C: real/imag, T: time frame
        D: dimension of the embedding vector
        current_input: (B, CM, T, F)
        embedding: (B, D)
        output: (B, S, T, C*F)
        """
        n_batch, _, n_frames, n_freqs = current_input.shape
        batch = current_input

        if input_state is None:
            input_state = self.init_buffers(
                current_input.shape[0], current_input.device
            )

        conv_buf = input_state["conv_buf"]
        deconv_buf = input_state["deconv_buf"]
        gridnet_buf = input_state["block_bufs"]

        batch = torch.cat((conv_buf, batch), dim=2)

        conv_buf = batch[:, :, -(self.t_ksize - 1) :, :]  # Update conv state

        batch = self.conv(batch)  # [B, D, T, F]

        if self.use_first_ln:
            # LayerNorm expects the last dimension to be latent_dim
            # Permute to [B, T, F, D] for layer norm, then back to [B, D, T, F]
            batch = batch.permute(0, 2, 3, 1)  # [B, T, F, D]
            batch = self.ln(batch)
            batch = batch.permute(0, 3, 1, 2)  # [B, D, T, F]

        # Apply embedding with one-hot encoding previously used
        if self.embedding is not None:
            logging.debug(f"embedding: {embedding}")
            logging.debug(f"embedding: {embedding.shape}")
            logging.debug(f"embedding: {embedding.dtype}")
            embedding = self.embedding(embedding)  # [B, Embedding_dim]

        logger.debug(f"After embedding: {embedding}")
        logger.debug(f"After embedding.shape: {embedding.shape}")

        # Process through blocks with FiLM applied at specified positions
        for ii in range(self.n_layers):
            # Apply FiLM at the specified position for this block
            if ii == 0:
                if ii in self.film_positions:
                    batch = self.film_layers[f"film_layer_{ii}"](
                        batch, embedding
                    )  # [B, D, T, F]
                batch = batch.permute(0, 2, 3, 1)  # [B, T, F, D]
            else:
                if ii in self.film_positions:
                    batch = batch.permute(0, 3, 1, 2)  # [B, D, T, F]
                    batch = self.film_layers[f"film_layer_{ii}"](
                        batch, embedding
                    )  # [B, D, T, F]
                    batch = batch.permute(0, 2, 3, 1)  # [B, T, F, D]

            # Process through the block [B, T, F, D] -> [B, T, F, D]
            batch, gridnet_buf[f"buf{ii}"] = self.blocks[ii](
                batch, gridnet_buf[f"buf{ii}"]
            )  # [B, T, F, D]

        batch = batch.permute(0, 3, 1, 2)  # [B, D, T, F]

        batch = torch.cat((deconv_buf, batch), dim=2)

        deconv_buf = batch[:, :, -(self.t_ksize - 1) :, :]  # Update deconv state

        batch = self.deconv(batch)  # [B, n_srcs*C, T, F]
        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        batch = batch.transpose(2, 3).reshape(
            n_batch, self.n_srcs, n_frames, 2 * n_freqs
        )  # [B, S, T, F]

        input_state["conv_buf"] = conv_buf
        input_state["deconv_buf"] = deconv_buf
        input_state["block_bufs"] = gridnet_buf

        return batch, input_state

    def edge_mode(self):
        for i in range(len(self.blocks)):
            self.blocks[i].edge_mode()


if __name__ == "__main__":
    pass
