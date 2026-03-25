import warnings
import torch
import torch.nn as nn

from torchaudio.functional import resample

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics")
    from torchmetrics.functional import (
        scale_invariant_signal_distortion_ratio as si_sdr,
        scale_invariant_signal_noise_ratio as si_snr,
        signal_noise_ratio as snr,
    )

from torchmetrics.functional.audio.stoi import (
    short_time_objective_intelligibility as STOI,
)
from torchmetrics.functional.audio.pesq import (
    perceptual_evaluation_speech_quality as PESQ,
)
import numpy as np

import typing

import logging

# Logging is configured via setup_logging in entry point scripts

logger = logging.getLogger(__name__)

import os
import random
import string


def generate_random_string_random(length):
    """Generates a random string of specified length using the random module."""
    # Define the pool of characters to choose from (e.g., letters and digits)
    characters = string.ascii_letters + string.digits

    # Use random.choice to pick characters and join them
    random_string = "".join(random.choice(characters) for _ in range(length))
    return random_string


def compute_decay(est, mix):
    """
    [*, C, T]
    """
    types = type(est)
    assert type(mix) == types, "All arrays must be the same type"
    if types == np.ndarray:
        est, mix = torch.from_numpy(est), torch.from_numpy(mix)

    # Ensure that, no matter what, we do not modify the original arrays
    est = est.clone()
    mix = mix.clone()

    P_est = 10 * torch.log10(torch.sum(est**2, dim=-1))  # [*, C]
    P_mix = 10 * torch.log10(torch.sum(mix**2, dim=-1))

    return (P_mix - P_est).mean(dim=-1)  # [*]


def compute_metrics_tse(
    gt: torch.Tensor,
    est: torch.Tensor,
    mix: torch.Tensor,
    label_vector: torch.Tensor,
    metric_func_list: list[typing.Callable],
):
    result = {}
    logger.debug(
        f"compute_metrics_tse: gt.shape: {gt.shape}, \
        est.shape: {est.shape}, \
        mix.shape: {mix.shape}, \
        label_vector.shape: {label_vector.shape}"
    )
    output_channels_match_label_dim = (
        True if gt.shape[1] == label_vector.shape[1] else False
    )
    for metric in metric_func_list:
        flag_per_channel_metrics = (
            True
            if metric.name == "si_sdr_per_channel"
            or metric.name == "si_sdr_per_channel_mix"
            or metric.name == "snr_per_channel"
            or metric.name == "snr_per_channel_mix"
            else False
        )
        # Skip per-channel metrics during training to avoid CUDA memory issues
        logger.debug(f"Computing metrics: {metric.name}")
        try:
            B, C, T = gt.shape
            # Collect metric results for each batch
            metric_val_list = (
                torch.zeros(B, device=gt.device) if not flag_per_channel_metrics else []
            )
            logger.debug(f"label_vector: {label_vector}")
            if not output_channels_match_label_dim:
                num_labels = label_vector.sum(dim=1, keepdim=True).int()
                logger.debug(
                    f"Computing metrics for output channels not matching label dimensions..."
                )
                logger.debug(f"num_labels: {num_labels}")
                for idx_batch in range(B):
                    index = torch.tensor([idx_batch]).to(gt.device)
                    gt_batch = torch.index_select(gt, dim=0, index=index)
                    est_batch = torch.index_select(est, dim=0, index=index)
                    mix_batch = torch.index_select(mix, dim=0, index=index)
                    logger.debug(
                        f"gt_batch.shape: {gt_batch.shape}, est_batch.shape: {est_batch.shape}, mix_batch.shape: {mix_batch.shape}"
                    )
                    mix_mono = torch.mean(mix_batch, dim=1)
                    mix_mono = mix_mono.unsqueeze(1).expand(1, C, T)
                    logger.debug(
                        f"est_batch.shape: {est_batch.flatten(0, 1).unsqueeze(1).shape}, \
                    gt_batch.shape: {gt_batch.flatten(0, 1).unsqueeze(1).shape}, \
                    mix_mono.shape: {mix_mono.flatten(0, 1).unsqueeze(1).shape}"
                    )

                    metric_result = metric(
                        est=est_batch.flatten(0, 1).unsqueeze(1),  # (B*C, 1, T)
                        gt=gt_batch.flatten(0, 1).unsqueeze(1),  # (B*C, 1, T)
                        mix=mix_mono.flatten(0, 1).unsqueeze(1),  # (B*C, 1, T)
                    )
                    num_labels_batch = num_labels[idx_batch].item()

                    logger.debug(
                        f"Metric {metric.name} result: {metric_result}, num_labels_batch: {num_labels_batch}"
                    )  # (B*C, 1)
                    logger.debug(f"metric_result.shape: {metric_result.shape}")
                    logger.debug(
                        f"idx_batch * num_labels_batch: {idx_batch * num_labels_batch}"
                    )
                    logger.debug(
                        f"(idx_batch + 1) * num_labels_batch: {(idx_batch + 1) * num_labels_batch}"
                    )
                    if flag_per_channel_metrics:
                        # Use torch operations instead of numpy to avoid CUDA memory issues
                        _metric_name = metric.name.split("_per_channel")[0]
                        logger.debug(
                            f"Generating metric values per channel for batch {idx_batch}"
                        )
                        for channel in range(num_labels_batch):
                            metric_val_list.append(
                                {
                                    "batch": idx_batch,
                                    "channel": channel,
                                    _metric_name: metric_result[channel].item(),
                                }
                            )
                    else:
                        logger.debug(f"num_labels[idx_batch]: {num_labels[idx_batch]}")
                        metric_val_list[idx_batch] = (
                            metric_result[:num_labels_batch].mean().item()
                        )
                        logger.debug(
                            f"Metric value: Batch {idx_batch}: {metric_result}"
                        )
            else:
                logger.debug(
                    f"Computing metrics for output channels matching label dimensions..."
                )
                for idx_batch in range(B):
                    index = torch.tensor([idx_batch]).to(gt.device)
                    label_vector_batch = torch.index_select(
                        label_vector, dim=0, index=index
                    )
                    gt_batch = torch.index_select(gt, dim=0, index=index)
                    est_batch = torch.index_select(est, dim=0, index=index)
                    mix_batch = torch.index_select(mix, dim=0, index=index)
                    logger.debug(
                        f"gt_batch.shape: {gt_batch.shape}, est_batch.shape: {est_batch.shape}, mix_batch.shape: {mix_batch.shape}"
                    )
                    mix_mono = torch.mean(mix_batch, dim=1)
                    mix_mono_batch = mix_mono.unsqueeze(1).expand(1, C, T).reshape(C, T)

                    nonzero_classes = label_vector_batch.flatten(0, 1) > 0

                    logger.debug(f"One hot encoding: {label_vector_batch}")
                    logger.debug(f"Non-zero classes: {nonzero_classes}")
                    logger.debug(
                        f"flatten est: {est_batch.flatten(0, 1)[nonzero_classes].unsqueeze(1).shape}"
                    )
                    logger.debug(
                        f"flatten gt: {gt_batch.flatten(0, 1)[nonzero_classes].unsqueeze(1).shape}"
                    )
                    logger.debug(
                        f"flatten mix: {mix_mono_batch[nonzero_classes].unsqueeze(1).shape}"
                    )

                    metric_result = metric(
                        est=est_batch.flatten(0, 1)[nonzero_classes].unsqueeze(1),
                        gt=gt_batch.flatten(0, 1)[nonzero_classes].unsqueeze(1),
                        mix=mix_mono_batch[nonzero_classes].unsqueeze(1),
                    )
                    logger.debug(f"Metric result: {metric_result}")

                    # shoh: only for evaluating per-channel metrics in test, don't use fabric in this case
                    if flag_per_channel_metrics:
                        # Use torch operations instead of numpy to avoid CUDA memory issues
                        nonzero_classes_idx = torch.where(nonzero_classes)[0]
                        _metric_name = metric.name.split("_per_channel")[0]
                        for ch in range(metric_result.shape[0]):
                            metric_val_list.append(
                                {
                                    "batch": idx_batch,
                                    "channel": nonzero_classes_idx[ch].item(),
                                    _metric_name: metric_result[ch].item(),
                                }
                            )
                    else:
                        # Handle multi-channel results by taking the mean across channels
                        metric_result = metric_result.mean()
                        metric_val_list[idx_batch] = metric_result
                        logger.debug(
                            f"Metric value: Batch {idx_batch}: {metric_result}"
                        )

            result[metric.name] = (
                metric_val_list
                if type(metric_val_list) is list
                else metric_val_list.mean()
            )
            logger.debug(f"Metric {metric.name} result: {result[metric.name]}")
            logger.debug(f"Result: {result}")

        except Exception as e:
            logger.error(f"[ERROR] Metric {metric.name} failed: {e}")
            logger.error(
                f"gt.shape: {gt.shape}, est.shape: {est.shape}, mix.shape: {mix.shape}"
            )

    return result


class Metrics(nn.Module):
    def __init__(self, name, fs=16000, **kwargs) -> None:
        super().__init__()
        self.fs = fs
        self.func = None
        self.name = name
        if name == "snr" or name == "snr_per_channel":
            self.func = lambda est, gt, mix: snr(preds=est, target=gt)
        elif name == "snr_per_channel_mix":
            self.func = lambda est, gt, mix: snr(preds=mix, target=gt)
        elif name == "snr_i":
            self.func = lambda est, gt, mix: snr(preds=est, target=gt) - snr(
                preds=mix, target=gt
            )
        elif name == "si_snr":
            self.func = lambda est, gt, mix: si_snr(preds=est, target=gt)
        elif name == "si_snr_i":
            self.func = lambda est, gt, mix: si_snr(preds=est, target=gt) - si_snr(
                preds=mix, target=gt
            )
        elif name == "si_sdr" or name == "si_sdr_per_channel":
            self.func = lambda est, gt, mix: si_sdr(preds=est, target=gt)
        elif name == "si_sdr_per_channel_mix":
            self.func = lambda est, gt, mix: si_sdr(preds=mix, target=gt)
        elif name == "si_sdr_i":
            self.func = lambda est, gt, mix: si_sdr(preds=est, target=gt) - si_sdr(
                preds=mix, target=gt
            )
        elif name == "STOI":
            self.func = lambda est, gt, mix: STOI(preds=est, target=gt, fs=fs)
        elif name == "WBPESQ":
            fs_new = 16000
            resample_fn = lambda x, y, z: x
            if fs_new != fs:
                resample_fn = resample
            self.func = lambda est, gt, mix: PESQ(
                preds=resample_fn(est, fs, fs_new),
                target=resample_fn(gt, fs, fs_new),
                fs=fs_new,
                mode="wb",
            )
        elif name == "NBPESQ":
            fs_new = 8000
            resample_fn = lambda x, y, z: x
            if fs_new != fs:
                resample_fn = resample
            self.func = lambda est, gt, mix: PESQ(
                preds=resample_fn(est, fs, fs_new),
                target=resample_fn(gt, fs, fs_new),
                fs=fs_new,
                mode="nb",
            )
        elif name == "Multi_Reso_L1":
            from src.tse.loss import MultiResoFuseLoss
            mult_ireso_loss = MultiResoFuseLoss(**kwargs)
            self.func = lambda est, gt, mix: mult_ireso_loss(est=est, gt=gt)
        else:
            raise NotImplementedError(f"Metric {name} not implemented!")

    def forward(self, est, gt, mix=None) -> torch.Tensor:
        """
        input: (*, C, T)
        output: (*)
        """
        types = type(est)
        assert type(gt) == types and (
            type(mix) == types or mix is None
        ), "All arrays must be the same type"
        if types == np.ndarray:
            est, gt = torch.from_numpy(est), torch.from_numpy(gt)
            if mix is not None:
                mix = torch.from_numpy(mix)

        # Ensure that, no matter what, we do not modify the original arrays
        est = est.clone()
        gt = gt.clone()

        if mix is not None:
            mix = mix.clone()

        per_channel_metrics = self.func(est=est, gt=gt, mix=mix)  # [*, C]

        if self.name == "PLCPALoss":
            return (
                per_channel_metrics[0].mean(dim=-1),
                per_channel_metrics[1].mean(dim=-1),
                per_channel_metrics[2].mean(dim=-1),
            )
        else:
            return per_channel_metrics.mean(dim=-1)  # [*]
