from __future__ import annotations

"""TSE model wrapper.

Re-exports the original model classes from the parent project and provides
a ``load_pretrained()`` helper for checkpoint loading.
"""

import logging
from typing import Any

import torch

# Re-export original classes for direct use
from src.tse.net import Net
from src.tse.multiflim_guided_tfnet import MultiFiLMGuidedTFNet
from src.tse.gridnet_block import GridNetBlock
from src.tse.film import FiLM
from src.tse.dsp import get_perfect_synthesis_window, mod_pad
from src.tse.loss import MultiResoFuseLoss

logger = logging.getLogger(__name__)


# Alias for convenience — Net is the top-level STFT wrapper (TFGridNet)
TFGridNet = Net


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
) -> Net:
    """Download and instantiate a pretrained :class:`Net` (TFGridNet) from HuggingFace.

    Args:
        repo_id: HuggingFace Hub repository ID.
        model_name: One of ``"orange_pi"``, ``"raspberry_pi"``, ``"neuralaid"``.

    Returns:
        A :class:`Net` with pretrained weights loaded.
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
    elif "model" in config:
        mp = config["model"]
    else:
        mp = config

    # The original Net class takes model_name (dotted path) and block_model_name
    # For pretrained loading, use the original import paths within this package
    model_name_dotted = mp.get("model_name", "src.tse.multiflim_guided_tfnet.MultiFiLMGuidedTFNet")
    block_model_name = mp.get("block_model_name", "src.tse.gridnet_block.GridNetBlock")
    block_model_params = mp.get("block_model_params", {})
    embedding_params = mp.get("embedding_params", {
        "embedding_dim": 0,
        "embedding_type": "",
        "embedding_activation": "",
        "embedding_init": "",
    })
    film_params = mp.get("film_params", {
        "film_positions": [],
        "film_preset": "all_except_first",
    })

    model = Net(
        model_name=model_name_dotted,
        block_model_name=block_model_name,
        block_model_params=block_model_params,
        speaker_dim=mp.get("speaker_dim", 20),
        stft_chunk_size=mp.get("stft_chunk_size", 96),
        stft_pad_size=mp.get("stft_pad_size", 64),
        stft_back_pad=mp.get("stft_back_pad", 96),
        num_input_channels=mp.get("num_input_channels", 1),
        num_output_channels=mp.get("num_output_channels", 1),
        num_layers=mp.get("num_layers", 6),
        latent_dim=mp.get("latent_dim", 32),
        embedding_params=embedding_params,
        film_params=film_params,
        use_first_ln=mp.get("use_first_ln", False),
    )

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    # Handle wrapped state dicts (e.g. {"model": ...})
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    logger.info("Loaded pretrained model '%s' from %s", model_name, repo_id)
    return model
