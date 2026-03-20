from __future__ import annotations

"""AST-based Sound Event Detection model.

Wraps the Hugging Face Audio Spectrogram Transformer (AST) pre-trained on
AudioSet and replaces the classification head for fine-tuning on a custom
label set.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ASTModel(nn.Module):
    """Audio Spectrogram Transformer for multi-label sound event detection.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier (default: AudioSet-pretrained AST).
    num_classes : int
        Number of target classes for the custom classification head.
    freeze_encoder : bool
        If *True*, freeze all parameters except the classifier head.
    sample_rate : int
        Expected input audio sample rate in Hz.
    """

    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_classes: int = 20,
        freeze_encoder: bool = True,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        from transformers import ASTForAudioClassification, AutoFeatureExtractor

        self.sample_rate = sample_rate
        self.num_classes = num_classes

        # Load pre-trained AST and feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = ASTForAudioClassification.from_pretrained(model_name)

        # Store original AudioSet class names from model config
        self.audioset_class_names: list[str] = list(
            self.model.config.id2label.values()
        )

        # Replace classifier head for our label set
        hidden_size = self.model.config.hidden_size
        self.model.classifier = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.model.classifier.weight)
        nn.init.zeros_(self.model.classifier.bias)

        if freeze_encoder:
            self.freeze_encoder()

        logger.info(
            "ASTModel initialised: %s, %d classes, encoder %s",
            model_name,
            num_classes,
            "frozen" if freeze_encoder else "trainable",
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run a forward pass on raw audio waveforms.

        Args:
            waveform: ``(B, T)`` raw audio at :attr:`sample_rate`.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        # Feature extractor expects numpy arrays
        if waveform.is_cuda:
            wav_np = waveform.detach().cpu().numpy()
        else:
            wav_np = waveform.detach().numpy()

        inputs = self.feature_extractor(
            [w for w in wav_np],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        # Move inputs to same device as model
        device = next(self.model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        outputs = self.model(**inputs)
        return outputs.logits  # (B, num_classes)

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> None:
        """Freeze all parameters except the classifier head."""
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        logger.info("Froze encoder layers (classifier head remains trainable)")

    def unfreeze_encoder(self) -> None:
        """Unfreeze all model layers for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("Unfroze all model layers")

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self, waveform: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference and return scores + embeddings.

        Args:
            waveform: ``(B, T)`` or ``(T,)`` raw audio.

        Returns:
            scores: Sigmoid probabilities, shape ``(B, num_classes)``.
            embeddings: CLS token embeddings from the last hidden state,
                shape ``(B, hidden_size)``.
        """
        was_training = self.training
        self.eval()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.is_cuda:
            wav_np = waveform.cpu().numpy()
        else:
            wav_np = waveform.numpy()

        inputs = self.feature_extractor(
            [w for w in wav_np],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        device = next(self.model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        outputs = self.model(**inputs, output_hidden_states=True)
        scores = torch.sigmoid(outputs.logits)
        embeddings = outputs.hidden_states[-1][:, 0, :]  # CLS token

        if was_training:
            self.train()

        return scores, embeddings

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_trainable_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_parameters(self) -> int:
        """Return the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Pre-trained model loading
# ---------------------------------------------------------------------------

# Mapping from friendly name to HuggingFace Hub subfolder
_PRETRAINED_MAP: dict[str, str] = {
    "finetuned_ast": "sed_ast_snr_ctl_v2_16k",
}


def load_pretrained(
    repo_id: str = "ooshyun/sound_event_detection",
    model_name: str = "finetuned_ast",
    device: Optional[Union[str, torch.device]] = None,
) -> ASTModel:
    """Download and instantiate a pre-trained :class:`ASTModel`.

    Args:
        repo_id: Hugging Face Hub repository identifier.
        model_name: Friendly name mapped to a sub-folder in the repo
            (see :data:`_PRETRAINED_MAP`).
        device: Device to place the model on.

    Returns:
        An :class:`ASTModel` with loaded weights.
    """
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
        filename=f"{subfolder}/best.pt",
    )

    # Instantiate model from config
    hf_model_name = config.get(
        "model_name", "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    num_classes = config.get("num_classes", 20)
    sample_rate = config.get("sample_rate", 16000)

    model = ASTModel(
        model_name=hf_model_name,
        num_classes=num_classes,
        freeze_encoder=False,
        sample_rate=sample_rate,
    )

    # Load state dict
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    # Handle nested state dicts (e.g. {"model_state_dict": ...})
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)

    if device is not None:
        model = model.to(device)

    model.eval()
    logger.info(
        "Loaded pre-trained %s from %s/%s", model_name, repo_id, subfolder
    )
    return model
