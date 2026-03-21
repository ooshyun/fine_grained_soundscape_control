"""YAMNet wrapper for SED evaluation.

Uses TensorFlow Hub's YAMNet model for inference only.
Wraps the TF model to provide a PyTorch-compatible interface
matching ASTHuggingFace's predict() signature.

Requires: tensorflow, tensorflow-hub

Usage:
    model = YAMNetModel(device="cuda")
    outputs = model.predict(waveform_16k)  # (batch, samples)
    # outputs["scores"]: (batch, num_classes) softmax probabilities
    # outputs["output"]: same as scores
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# YAMNet class map (521 AudioSet classes)
_YAMNET_CLASS_MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"


class YAMNetModel(nn.Module):
    """YAMNet wrapper with PyTorch-compatible interface.

    Internally uses TensorFlow Hub for inference, but exposes
    the same predict() interface as ASTHuggingFace.
    """

    SAMPLE_RATE = 16000
    NUM_CLASSES = 521

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError:
            raise ImportError(
                "YAMNet requires tensorflow and tensorflow-hub. "
                "Install with: pip install tensorflow tensorflow-hub"
            )

        # Suppress TF warnings
        tf.get_logger().setLevel("ERROR")

        self._device = device or torch.device("cpu")
        logger.info("Loading YAMNet from TensorFlow Hub...")
        self._yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        logger.info("YAMNet loaded (521 classes)")

        # Load class names from ontology.json (AudioSet)
        ontology_path = Path(__file__).parent.parent.parent / "data" / "ontology.json"
        if ontology_path.exists():
            with open(ontology_path) as f:
                ontology = json.load(f)
            self.class_names = [item["name"] for item in ontology][:self.NUM_CLASSES]
            self.ids = [item["id"] for item in ontology][:self.NUM_CLASSES]
        else:
            self.class_names = [f"class_{i}" for i in range(self.NUM_CLASSES)]
            self.ids = [f"/m/{i:04x}" for i in range(self.NUM_CLASSES)]

        self.num_classes = self.NUM_CLASSES

        # Dummy parameter so .to(device) works
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def to(self, device=None, *args, **kwargs):
        if device is not None:
            self._device = torch.device(device) if isinstance(device, str) else device
        return super().to(device, *args, **kwargs)

    def forward(self, inputs: dict) -> dict:
        """Forward pass matching ASTHuggingFace interface."""
        return self.predict(inputs["mixture"])

    def predict(self, audio: torch.Tensor) -> dict:
        """Run YAMNet inference.

        Args:
            audio: (batch, channels, samples) or (batch, samples) at 16kHz

        Returns:
            dict with "scores" (softmax probs) and "output" (same)
        """
        import tensorflow as tf

        # Handle input shape
        if audio.dim() == 3:
            # (batch, channels, samples) → mono
            audio = audio.mean(dim=1)
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0)

        batch_scores = []
        for i in range(audio.shape[0]):
            waveform = audio[i].detach().cpu().numpy().astype(np.float32)
            # YAMNet expects mono float32 waveform at 16kHz
            scores, embeddings, spectrogram = self._yamnet(waveform)
            # scores shape: (num_frames, 521) — average over frames
            avg_scores = tf.reduce_mean(scores, axis=0).numpy()
            batch_scores.append(avg_scores)

        scores_np = np.stack(batch_scores, axis=0)  # (batch, 521)
        scores_tensor = torch.from_numpy(scores_np).to(self._device)

        return {
            "scores": scores_tensor,
            "output": scores_tensor,
        }

    def eval(self):
        """No-op (TF model is always in eval mode)."""
        return self
