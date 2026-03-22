from __future__ import annotations

"""AST-based Sound Event Detection model.

Provides a ``load_pretrained()`` helper for checkpoint loading from
HuggingFace Hub.
"""

import json
import logging
from typing import Optional, Union

import torch

from src.sed.ast_hf import ASTHuggingFace

logger = logging.getLogger(__name__)

# Re-export for convenience
ASTModel = ASTHuggingFace

# ---------------------------------------------------------------------------
# Pre-trained model loading
# ---------------------------------------------------------------------------

# Mapping from friendly name to HuggingFace Hub subfolder
_PRETRAINED_MAP: dict[str, str] = {
    "finetuned_ast": "sed_ast_snr_ctl_v2_16k",
}

# Baseline models that don't need custom weights (use HF pretrained directly)
_BASELINE_MODELS: dict[str, dict] = {
    "ast_pretrained": {
        "hf_model": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "num_labels": 527,
        "description": "Pretrained AST on AudioSet (527 classes, no fine-tuning)",
    },
    "yamnet": {
        "loader": "yamnet",
        "description": "Google YAMNet on AudioSet (521 classes, TF Hub)",
    },
}


def load_pretrained(
    repo_id: str = "ooshyun/sound_event_detection",
    model_name: str = "finetuned_ast",
    device: Optional[Union[str, torch.device]] = None,
) -> ASTHuggingFace:
    """Download and instantiate a pre-trained :class:`ASTHuggingFace`.

    The original ``ASTHuggingFace`` overrides ``state_dict()`` and
    ``load_state_dict()`` to delegate to ``self.ast_model``, so loading
    is simply ``model.load_state_dict(state["model"], strict=True)``.

    Args:
        repo_id: Hugging Face Hub repository identifier.
        model_name: Friendly name mapped to a sub-folder in the repo
            (see :data:`_PRETRAINED_MAP`).
        device: Device to place the model on.

    Returns:
        An :class:`ASTHuggingFace` with loaded weights.
    """
    # Handle baseline models (no custom weights, use HF pretrained directly)
    if model_name in _BASELINE_MODELS:
        info = _BASELINE_MODELS[model_name]
        logger.info("Loading baseline model: %s (%s)", model_name, info["description"])

        if info.get("loader") == "yamnet":
            from src.sed.yamnet import YAMNetModel
            model = YAMNetModel(device=device)
        else:
            model = ASTHuggingFace(
                model_name=info["hf_model"],
                num_labels=info["num_labels"],
            )

        if device is not None:
            model = model.to(device)
        model.eval()
        return model

    from huggingface_hub import hf_hub_download

    subfolder = _PRETRAINED_MAP.get(model_name, model_name)

    # Download config
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{subfolder}/config.json",
    )
    with open(config_path, "r") as f:
        config = json.load(f)

    # Download weights
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{subfolder}/checkpoints/best.pt",
    )

    # Instantiate model from config (handle nested pl_module_args format)
    model_params = config
    if "pl_module_args" in config:
        model_params = config["pl_module_args"].get("model_params", config)
    hf_model_name = model_params.get(
        "model_name", "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    num_classes = model_params.get("num_labels", model_params.get("num_classes", 20))
    sample_rate = config.get("pl_module_args", config).get("sr", 16000)

    model = ASTHuggingFace(
        model_name=hf_model_name,
        num_labels=num_classes,
    )

    # Load state dict — ASTHuggingFace.load_state_dict delegates to ast_model
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    # Handle nested state dicts
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict, strict=True)

    if device is not None:
        model = model.to(device)

    model.eval()
    logger.info(
        "Loaded pre-trained %s from %s/%s", model_name, repo_id, subfolder
    )
    return model
