from __future__ import annotations

import torch


def si_sdr(est: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Scale-Invariant Signal-to-Distortion Ratio.

    Args:
        est: Estimated signal, shape (*, T).
        gt: Ground-truth signal, shape (*, T).

    Returns:
        SI-SDR in dB, shape (*,) -- one scalar per sample.
    """
    # s_target = <est, gt> / <gt, gt> * gt
    dot = torch.sum(est * gt, dim=-1, keepdim=True)
    s_target = dot / (torch.sum(gt * gt, dim=-1, keepdim=True) + 1e-8) * gt

    e_noise = est - s_target
    si_sdr_val = 10 * torch.log10(
        torch.sum(s_target**2, dim=-1)
        / (torch.sum(e_noise**2, dim=-1) + 1e-8)
        + 1e-8
    )
    return si_sdr_val


def si_sdri(
    est: torch.Tensor, gt: torch.Tensor, mix: torch.Tensor
) -> torch.Tensor:
    """SI-SDR improvement: si_sdr(est, gt) - si_sdr(mix, gt).

    Args:
        est: Estimated signal, shape (*, T).
        gt: Ground-truth signal, shape (*, T).
        mix: Mixture signal, shape (*, T).

    Returns:
        SI-SDRi in dB, shape (*,).
    """
    return si_sdr(est, gt) - si_sdr(mix, gt)


def snr(est: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Signal-to-Noise Ratio.

    SNR = 10 * log10(||gt||^2 / ||gt - est||^2)

    Args:
        est: Estimated signal, shape (*, T).
        gt: Ground-truth signal, shape (*, T).

    Returns:
        SNR in dB, shape (*,).
    """
    noise = gt - est
    snr_val = 10 * torch.log10(
        torch.sum(gt**2, dim=-1)
        / (torch.sum(noise**2, dim=-1) + 1e-8)
        + 1e-8
    )
    return snr_val


def snri(
    est: torch.Tensor, gt: torch.Tensor, mix: torch.Tensor
) -> torch.Tensor:
    """SNR improvement: snr(est, gt) - snr(mix, gt).

    Args:
        est: Estimated signal, shape (*, T).
        gt: Ground-truth signal, shape (*, T).
        mix: Mixture signal, shape (*, T).

    Returns:
        SNRi in dB, shape (*,).
    """
    return snr(est, gt) - snr(mix, gt)


_METRIC_FN = {
    "si_sdr": lambda est, gt, mix: si_sdr(est, gt),
    "si_sdri": si_sdri,
    "snr": lambda est, gt, mix: snr(est, gt),
    "snri": snri,
}


def compute_tse_metrics(
    est: torch.Tensor,
    gt: torch.Tensor,
    mix: torch.Tensor,
    metrics: tuple[str, ...] = ("si_sdri", "snri"),
) -> dict[str, torch.Tensor]:
    """Compute requested TSE metrics.

    Args:
        est: Estimated signal, shape (*, T).
        gt: Ground-truth signal, shape (*, T).
        mix: Mixture signal, shape (*, T).
        metrics: Tuple of metric names to compute.

    Returns:
        Dictionary mapping metric name to its value.
    """
    results: dict[str, torch.Tensor] = {}
    for name in metrics:
        if name not in _METRIC_FN:
            raise ValueError(
                f"Unknown metric: {name}. Available: {list(_METRIC_FN.keys())}"
            )
        results[name] = _METRIC_FN[name](est, gt, mix)
    return results
