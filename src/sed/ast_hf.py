"""
YAMNet Model - Hugging Face Implementation

Uses Hugging Face Transformers for audio classification.
This provides a clean PyTorch implementation with pre-trained weights.
"""

import json
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union, List
import logging
from pathlib import Path
import urllib.request

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None

logger = logging.getLogger(__name__)


class ASTHuggingFace(nn.Module):
    """
    YAMNet implementation using Hugging Face transformers.

    Uses audio classification models from Hugging Face that are compatible
    with AudioSet or similar to YAMNet architecture.
    """

    SAMPLE_RATE = 16000
    NUM_CLASSES = 527
    CLASS_MAP_PATH = "data/ontology.json"

    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        device: Optional[Union[str, torch.device]] = None,
        num_labels: int = 527,
        unfreeze_layers: List[str] = ["classifier"],
    ):
        """
        Initialize audio classification model.

        Args:
            model_name: Hugging Face model name
                Options:
                - "MIT/ast-finetuned-audioset-10-10-0.4593" (Audio Spectrogram Transformer on AudioSet)
                - Custom YAMNet if available
            device: Device to run on
        """
        super().__init__()
        from transformers import (
            AutoFeatureExtractor,
            ASTForAudioClassification,
            AutoModelForAudioClassification,
        )

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {self.device}")

        # Load feature extractor and model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name
        )
        self.ast_model = (
            ASTForAudioClassification.from_pretrained(model_name)
            if num_labels == self.NUM_CLASSES
            else AutoModelForAudioClassification.from_pretrained(
                model_name, num_labels=num_labels, ignore_mismatched_sizes=True
            )
        )
        self.ast_model.to(self.device)
        self.ast_model.eval()

        # Get class labels from model config
        self.class_names = list(self.ast_model.config.id2label.values())
        self.num_classes = len(self.class_names)
        self.unfreeze_layers = unfreeze_layers
        self.frozen_all = False

        # Load class names from ontology.json
        logger.info(f"model config: \n{self.ast_model.config}\n")
        with open(self.CLASS_MAP_PATH, "r") as f:
            ontology = json.load(f)
        ids = [None] * len(self.class_names)
        for index, class_name in enumerate(self.class_names):
            for class_ in ontology:
                if class_["name"] == class_name:
                    ids[index] = class_["id"]
                    break
        self.ids = ids

        assert len(self.ids) == len(
            self.class_names
        ), f"Number of IDs and class names do not match: {len(self.ids)} != {len(self.class_names)}"

        logger.info(f"Model loaded successfully ({self.num_classes} classes)")

        # Print the each node in model and node name
        for name, param in self.ast_model.named_parameters():
            logger.info(f"name: {name}, shape: {param.shape}")
            logger.info(f"-" * 50)

        # self.freeze_model()

    def freeze_model(self):
        """Freeze the model."""
        for param in self.ast_model.parameters():
            param.requires_grad = False
        self.unfreeze_model_layers(self.unfreeze_layers)
        self.frozen_all = True

        # Verify that some parameters are trainable
        trainable_params = sum(
            1 for p in self.ast_model.parameters() if p.requires_grad
        )
        total_params = sum(1 for _ in self.ast_model.parameters())
        logger.info(
            f"Frozen model: {trainable_params}/{total_params} parameters are trainable"
        )

        return self

    def unfreeze_model_layers(self, layers: List[str]):
        """Unfreeze the model layers."""
        for name, param in self.ast_model.named_parameters():
            # Check if the parameter name starts with any of the layer prefixes
            if any(name.startswith(layer) for layer in layers):
                logger.info(f"Unfreezing layer: {name}")
                param.requires_grad = True
        return self

    def unfreeze_model(self):
        """Unfreeze the model."""
        for param in self.ast_model.parameters():
            param.requires_grad = True
        self.frozen_all = False
        return self

    def compile(self):
        """Compile the model."""
        # self.ast_model = torch.compile(self.ast_model) # It only give logits
        return self

    def predict(
        self, waveform: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run inference on audio waveform.

        Args:
            waveform: Audio waveform, shape (num_samples,), 16kHz mono

        Returns:
            logits: Class logits, shape (batch_size, num_classes)
            scores: Class probabilities, shape (batch_size, num_classes)
            embeddings: Feature embeddings (if available), shape (batch_size, embedding_dim)
        """
        # Convert waveform to CPU numpy array for feature extractor
        # Feature extractor expects numpy array, not CUDA tensor
        if isinstance(waveform, torch.Tensor):
            waveform_cpu = waveform.detach().cpu().numpy()
        else:
            waveform_cpu = np.asarray(waveform)

        # Extract features (feature extractor expects CPU numpy array)
        inputs = self.feature_extractor(
            waveform_cpu, sampling_rate=self.SAMPLE_RATE, return_tensors="pt"
        )

        # Move inputs to device
        inputs = {
            k: (
                v.to(self.device)
                if isinstance(v, torch.Tensor)
                else torch.tensor(v).to(self.device)
            )
            for k, v in inputs.items()
        }

        # Run model
        outputs = self.ast_model(**inputs, output_hidden_states=True)
        logits = outputs.logits  # Raw logits for loss computation
        scores = torch.softmax(logits, dim=-1)  # Probabilities
        embeddings = outputs.hidden_states[-1][:, 0, :]  # CLS token embeddings
        return logits, scores, embeddings

    def predict_aggregated(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        aggregation: str = "mean",
    ) -> np.ndarray:
        """
        Run inference (already aggregated for clip-level models).

        Args:
            waveform: Audio waveform
            aggregation: Ignored for clip-level models

        Returns:
            Class probabilities, shape (num_classes,)
        """
        _, scores, _ = self.predict(waveform)
        return scores

    def forward(self, inputs):
        """
        Forward pass that converts PyTorch inputs to TensorFlow and back.

        Args:
            inputs: Dictionary with 'mixture' tensor and optionally 'embedding'

        Returns:
            Dictionary with 'output' key containing model predictions
        """
        # Extract mixture audio
        mixture = inputs.get("mixture")
        if mixture is None:
            raise ValueError("Inputs must contain 'mixture' key")

        # Normalize to [-1, 1] if needed
        if mixture.abs().max() > 1.0:
            mixture = mixture / (mixture.abs().max() + 1e-6)

        # Multichannel to mono
        if mixture.ndim > 2:
            mixture = mixture.mean(dim=1)

        try:
            logits, scores, embeddings = self.predict(mixture)
        except Exception as e:
            logger.error(f"Error running AST: {e}")
            # Fallback - retry prediction
            logits, scores, embeddings = self.predict(mixture)

        # Return in expected format
        # Return logits as "output" for loss computation (BCE loss expects logits)
        return {
            "output": logits,  # Class logits for loss computation
            "scores": scores,  # Class probabilities (softmax)
            "embeddings": embeddings,  # Feature embeddings
        }

    def get_top_classes(
        self, scores: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top-k predicted classes."""
        if scores.ndim > 1:
            scores = scores.flatten()

        top_indices = np.argsort(scores)[-top_k:][::-1]

        return [(self.class_names[i], float(scores[i])) for i in top_indices]

    def class_name_to_index(self, class_name: str) -> Optional[int]:
        """Get class index from name."""
        try:
            if class_name in self.class_names:
                return self.class_names.index(class_name)
            else:
                return None
        except ValueError:
            return None

    def index_to_idx(self, index: int) -> Optional[int]:
        """Get index from class index."""
        try:
            return self.ids[index]
        except ValueError:
            return None

    def index_to_class_name(self, index: int) -> str:
        """Get class name from index."""
        if 0 <= index < len(self.class_names):
            return self.class_names[index]
        return f"unknown_{index}"

    def get_flops(self):
        """Get FLOPS of the model (for 10 seconds of audio)."""
        if FlopCountAnalysis is not None:
            duration = 10.0
            waveform = torch.randn(
                1, int(self.SAMPLE_RATE * duration), dtype=torch.float32
            )
            inputs = self.feature_extractor(
                waveform, sampling_rate=self.SAMPLE_RATE, return_tensors="pt"
            )
            analysis_feature_extractor = FlopCountAnalysis(
                self.feature_extractor, inputs=waveform
            )
            analysis_model = FlopCountAnalysis(self.ast_model, inputs=inputs)
            return analysis_feature_extractor.total() + analysis_model.total()
        else:
            return 0

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Return the state dictionary of the model."""
        return self.ast_model.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

    def load_state_dict(self, state_dict, strict=True):
        """Load state dictionary into the model."""
        return self.ast_model.load_state_dict(state_dict, strict=strict)

    def train(self, mode=True):
        """Set the model to training mode."""
        super().train(mode)
        self.ast_model.train(mode)
        return self

    def eval(self):
        """Set the model to evaluation mode."""
        super().eval()
        self.ast_model.eval()
        return self
