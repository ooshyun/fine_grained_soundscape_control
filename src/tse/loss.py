from __future__ import annotations

import torch
import torch.nn as nn

import auraloss.freq


class MultiResoFuseLoss(nn.Module):
    """Multi-resolution STFT loss fused with an optional L1 term.

    Combines :class:`auraloss.freq.MultiResolutionSTFTLoss` with a weighted
    L1 loss for time-domain supervision.

    Args:
        l1_ratio: Weight for the L1 component.  Set to ``0`` to disable.
        **stft_kwargs: Forwarded to ``MultiResolutionSTFTLoss``.
    """

    def __init__(self, l1_ratio: float = 10, **stft_kwargs) -> None:
        super().__init__()
        self.l1_ratio = l1_ratio
        self.stft_loss = auraloss.freq.MultiResolutionSTFTLoss(**stft_kwargs)
        self.l1 = nn.L1Loss()

    def forward(self, est: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            est: Estimated signal, shape ``(B, C, T)``.
            gt: Ground-truth signal, shape ``(B, C, T)``.

        Returns:
            Scalar loss.
        """
        loss = self.stft_loss(est, gt)
        if self.l1_ratio > 0:
            loss = loss + self.l1_ratio * self.l1(est, gt)
        return loss
