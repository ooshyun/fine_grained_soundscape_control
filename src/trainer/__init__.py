from __future__ import annotations

from .base import TrainerBackend


def create_trainer(backend: str = "lightning", **kwargs) -> TrainerBackend:
    """Factory function to create a trainer backend.

    Args:
        backend: Either ``"lightning"`` (PyTorch Lightning Trainer) or
            ``"fabric"`` (Lightning Fabric manual loop).
        **kwargs: Forwarded to the backend constructor.

    Returns:
        A :class:`TrainerBackend` instance.
    """
    if backend == "lightning":
        from .lightning import LightningTrainerBackend

        return LightningTrainerBackend(**kwargs)
    elif backend == "fabric":
        from .fabric import FabricTrainerBackend

        return FabricTrainerBackend(**kwargs)
    raise ValueError(f"Unknown backend: {backend}")


__all__ = ["TrainerBackend", "create_trainer"]
