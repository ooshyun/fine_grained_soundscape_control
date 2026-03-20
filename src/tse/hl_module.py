import os
import time
from numpy import mean

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from src.metrics.tse import Metrics, compute_metrics_tse
from src.metrics.tse import compute_decay

import logging

# Logging is configured in train.py via setup_logging
logger = logging.getLogger(__name__)


def _import_attr(name):
    """Dynamically import a class/function from a dotted path string."""
    module_path, _, attr_name = name.rpartition(".")
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


class PLModule(object):
    import lightning_fabric as L

    def __init__(
        self,
        fabric: L.Fabric,
        model=None,
        model_params=None,
        sr=None,
        optimizer=None,
        optimizer_params=None,
        scheduler=None,
        scheduler_params=None,
        loss=None,
        loss_params=None,
        metrics=[],
        init_ckpt=None,
        grad_clip=None,
        use_dp=True,
        val_log_interval=10,  # Unused, only kept for compatibility TODO: Remove
        samples_per_speaker_number=3,
    ):
        self.fabric = fabric
        self.model_name = model
        if "SemanticHearing" in model: # waveformer model
            self.model = _import_attr(model)(**model_params["waveformer_params"])
        else:
            if "use_first_ln" not in model_params.keys():
                model_params["use_first_ln"] = False
            self.model = _import_attr(model)(**model_params)
        self.use_dp = use_dp
        # Note: With Lightning Fabric DDP, we don't need DataParallel
        # The fabric.setup() call will handle distributed training

        self.sr = sr

        # debug_overflow = DebugUnderflowOverflow(self.model)
        # Log a val sample every this many intervals
        # self.val_log_interval = val_log_interval
        self.samples_per_speaker_number = samples_per_speaker_number
        if "num_output_channels" in model_params.keys():
            if "SemanticHearing" in model:
                self.model_output_channels = model_params["waveformer_params"]["out_channels"]
            else:
                self.model_output_channels = model_params["num_output_channels"]
        else:
            raise ValueError("num_output_channels is not found in config file")

        self.use_label_vector_indexing = (
            False
            if "embedding_type" not in model_params.keys()
            or self.model.tfgridnet.embedding_type is None
            or self.model.tfgridnet.embedding_type == "linear"
            else True
        )

        # Initialize metrics
        self.metrics = [Metrics(metric) for metric in metrics]

        # Metric values
        self.metric_values = {}

        # Dataset statistics
        self.statistics = {}

        # Assine metric to monitor, and how to judge different models based on it
        # i.e. How do we define the best model (Here, we minimize val loss)
        self.monitor = "val/loss"
        self.monitor_mode = "min"

        # Mode, either train or val
        self.mode = None

        self.val_samples = {}
        self.train_samples = {}

        # Initialize loss function
        self.loss_fn = _import_attr(loss)(**loss_params)

        # Initaize weights if checkpoint is provided
        # Warning: This will only load the weights of the module
        # called "model" in this class
        if init_ckpt is not None:
            print("Loading state from fabric...")
            state = self.fabric.load(init_ckpt)["model"]
            self.model.load_state_dict(state)

        # Initialize optimizer
        self.optimizer = _import_attr(optimizer)(
            self.model.parameters(), **optimizer_params
        )
        self.optim_name = optimizer
        self.opt_params = optimizer_params

        # Grad clip
        self.grad_clip = grad_clip

        if self.grad_clip is not None:
            print(f"USING GRAD CLIP: {self.grad_clip}")
        else:
            print("ERROR! NOT USING GRAD CLIP" * 100)

        # Initialize scheduler
        self.scheduler = self.init_scheduler(scheduler, scheduler_params)
        self.scheduler_name = scheduler
        self.scheduler_params = scheduler_params

        self.epoch = 0

    def load_state(self, path):
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "current_epoch": self.epoch,
            "metric_values": self.metric_values,
            "statistics": self.statistics,
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler

        self.fabric.load(path, state)
        self.epoch = state["current_epoch"]
        self.metric_values = state["metric_values"]
        self.statistics = state["statistics"]

    def dump_state(self, path):
        state = dict(
            model=self.model,
            optimizer=self.optimizer,
            current_epoch=self.epoch,
            metric_values=self.metric_values,
            statistics=self.statistics,
        )

        if self.scheduler is not None:
            state["scheduler"] = self.scheduler
        self.fabric.save(path, state)

    def get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def on_epoch_start(self):
        pass

    def get_avg_metric_at_epoch(self, metric, epoch=None):
        if epoch is None:
            epoch = self.epoch

        return (
            self.metric_values[epoch][metric]["epoch"]
            / self.metric_values[epoch][metric]["num_elements"]
        )

    def gather_metrics(self):
        for metric in self.metric_values[self.epoch]:
            vals = self.fabric.all_gather(
                self.metric_values[self.epoch][metric]["epoch"]
            )
            nums = self.fabric.all_gather(
                self.metric_values[self.epoch][metric]["num_elements"]
            )

            self.metric_values[self.epoch][metric]["epoch"] = torch.sum(vals)
            self.metric_values[self.epoch][metric]["num_elements"] = torch.sum(nums)

    def on_epoch_end(self, best_path, wandb_run=None):
        assert self.epoch + 1 == len(
            self.metric_values
        ), "Current epoch must be equal to length of metrics (0-indexed)"

        # Gather metrics from multiple processes
        self.gather_metrics()

        monitor_metric_last = self.get_avg_metric_at_epoch(self.monitor)
        # Go over all epochs
        save = True
        for epoch in range(len(self.metric_values) - 1):
            monitor_metric_at_epoch = self.get_avg_metric_at_epoch(self.monitor, epoch)

            if self.monitor_mode == "max":
                # If there is any model with monitor larger than current, then
                # this is not the best model
                if monitor_metric_last < monitor_metric_at_epoch:
                    save = False
                    break

            if self.monitor_mode == "min":
                # If there is any model with monitor smaller than current, then
                # this is not the best model
                if monitor_metric_last > monitor_metric_at_epoch:
                    save = False
                    break

        # If this is best, save it
        if save:
            if self.fabric.global_rank == 0:
                print("Current checkpoint is the best! Saving it...")
            self.dump_state(best_path)

        val_loss = self.get_avg_metric_at_epoch("val/loss")
        val_si_sdr_i = self.get_avg_metric_at_epoch("val/si_sdr_i")

        should_print = self.fabric.global_rank == 0 if self.fabric is not None else True
        should_print = True
        if should_print:
            print(f"Val loss: {val_loss:.02f}")
            print(f"Val SI-SDRi: {val_si_sdr_i:.02f}dB")

        if self.scheduler is not None:
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                # Get last metric
                self.scheduler.step(metrics=monitor_metric_last)
            else:
                self.scheduler.step()

        logger.debug(f"Metric values: {self.metric_values[self.epoch]}")
        for metric in self.metric_values[self.epoch]:
            logger.debug(
                f"Logging metric: {self.epoch} - {metric} = {self.get_avg_metric_at_epoch(metric)}"
            )

        if wandb_run is not None:
            if self.fabric.global_rank == 0:
                import wandb

                # Log stuff on wandb
                wandb_run.log(
                    {"lr-Adam": self.get_current_lr()},
                    commit=False,
                    step=self.epoch + 1,
                )

                logger.debug(f"Metric values: {self.metric_values[self.epoch]}")
                for metric in self.metric_values[self.epoch]:
                    logger.debug(
                        f"Logging metric: {self.epoch} - {metric} = {self.get_avg_metric_at_epoch(metric)}"
                    )
                    wandb_run.log(
                        {metric: self.get_avg_metric_at_epoch(metric)},
                        commit=False,
                        step=self.epoch + 1,
                    )

                for statistic in self.statistics:
                    if not self.statistics[statistic]["logged"]:
                        data = self.statistics[statistic]["data"]
                        reduction = self.statistics[statistic]["reduction"]
                        if reduction == "mean":
                            val = mean(data)
                        elif reduction == "sum":
                            val = sum(data)
                        elif reduction == "histogram":
                            data = [[d] for d in data]
                            table = wandb.Table(data=data, columns=[statistic])
                            val = wandb.plot.histogram(
                                table, statistic, title=statistic
                            )
                        else:
                            assert 0, f"Unknown reduction {reduction}."
                        wandb_run.log({statistic: val}, commit=False)
                        self.statistics[statistic]["logged"] = True

                wandb_run.log({"epoch": self.epoch}, commit=True, step=self.epoch + 1)

        self.train_samples.clear()
        self.val_samples.clear()

        self.epoch += 1

    def log_statistic(self, name, value, reduction="mean"):
        if name not in self.statistics:
            self.statistics[name] = dict(logged=False, data=[], reduction=reduction)

        self.statistics[name]["data"].append(value)

    def log_metric(
        self,
        name,
        value,
        file_name=None,
        batch_size=1,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        sync_dist=True,
    ):
        """
        Logs a metric
        value must be the AVERAGE value across the batch
        Must provide batch size for accurate average computation
        """

        epoch_str = self.epoch
        if epoch_str not in self.metric_values:
            self.metric_values[epoch_str] = {}

        if name not in self.metric_values[epoch_str]:
            self.metric_values[epoch_str][name] = dict(
                step=None,
                epoch=None,
                step_file_name=None,
            )

        if type(value) == torch.Tensor:
            value = value.item()

        if on_step:
            if self.metric_values[epoch_str][name]["step"] is None:
                self.metric_values[epoch_str][name]["step"] = []
            self.metric_values[epoch_str][name]["step"].append(value)

            logger.debug(f"Logging metric: ")
            logger.debug(f"Metric {self.metric_values[epoch_str][name]}")

        if on_epoch:
            if self.metric_values[epoch_str][name]["epoch"] is None:
                self.metric_values[epoch_str][name]["epoch"] = 0
                self.metric_values[epoch_str][name]["num_elements"] = 0

            self.metric_values[epoch_str][name]["epoch"] += value * batch_size
            self.metric_values[epoch_str][name]["num_elements"] += batch_size

    def _step(self, batch, batch_idx, step="train"):
        inputs, targets = batch
        batch_size = inputs["mixture"].shape[0]

        input_dict = {
            "mixture": inputs["mixture"],
            "embedding": (
                inputs["label_vector"]
                if not self.use_label_vector_indexing
                else inputs["new_label_vector"]
            ),
        }
        outputs = self.model(input_dict)
        if "SemanticHearing" in self.model_name and \
            targets["num_target_speakers"][0] == 1 and \
            self.model_output_channels > 1:
            outputs["output"] = outputs["output"].mean(dim=1, keepdim=True) # make mono

        mix = inputs["mixture"]  # Take first channel in mixture as reference

        assert torch.isnan(mix).max() == 0, "Input mixture tensor has nan!"

        est = outputs["output"]
        gt = targets["target"]

        n_speakers = targets["num_target_speakers"]

        # Compute loss
        assert (
            est.shape == gt.shape
        ), f"est{est.shape} and gt{gt.shape} shapes should be same"

        try:
            loss = self.loss_fn(est=est, gt=gt).mean()
            logger.debug(
                f"[semhearing_hl_module] [rank {self.fabric.global_rank}] loss: {loss}"
            )

        except Exception as e:
            logger.error("stuck at loss function")
            logger.error(f"est shape: {est.shape}, gt shape: {gt.shape}")
            logger.error(f"est device: {est.device}, gt device: {gt.device}")
            logger.error(f"Current default CUDA device: {torch.cuda.current_device()}")
            raise e

        est_detached = est.detach().clone()
        with torch.no_grad():
            # Log loss
            self.log_metric(
                f"{step}/loss",
                loss.item(),
                batch_size=batch_size,
                on_step=(step == "train"),
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            # Log metricss
            COMPUTE_METRICS = True
            if COMPUTE_METRICS:
                metric_start = time.time()
                result = compute_metrics_tse(
                    gt=gt,
                    est=est_detached,
                    mix=mix,
                    label_vector=inputs["label_vector"],
                    metric_func_list=self.metrics,
                )

                for metric_name, metric_val in result.items():
                    self.log_metric(
                        f"{step}/{metric_name}",
                        metric_val,
                        file_name=inputs["folder"],  # TODO(shoh): for debugging
                        batch_size=batch_size,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                    )
            else:
                print(f"[INFO] Skipping metrics computation for batch {batch_idx}")

        fg_labels = targets["fg_labels"]
        label_vector = inputs["label_vector"]

        sample = {
            "mixture": mix,
            "output": est_detached,
            "target": gt,
            "n_tgt_speakers": n_speakers,
            "fg_labels": list(map(list, zip(*fg_labels))),
            "label_vector": label_vector,
        }

        return loss, sample

    def train(self):
        self.model.train()
        self.mode = "train"

    def eval(self):
        self.model.eval()
        self.mode = "val"

    def training_step(self, batch, batch_idx):
        loss, sample = self._step(batch, batch_idx, step="train")

        n_speakers = sample["n_tgt_speakers"]
        for i in range(n_speakers.shape[0]):
            spk_num = n_speakers[i].item()
            if spk_num not in self.train_samples:
                self.train_samples[spk_num] = []

            if len(self.train_samples[spk_num]) < 3:
                sample_at_batch = {}
                for k in sample:
                    sample_at_batch[k] = sample[k][i]
                self.train_samples[spk_num].append(sample_at_batch)

        return loss, n_speakers.shape[0]

    def validation_step(self, batch, batch_idx):
        loss, sample = self._step(batch, batch_idx, step="val")

        n_speakers = sample["n_tgt_speakers"]
        for i in range(n_speakers.shape[0]):
            spk_num = n_speakers[i].item()
            if spk_num not in self.val_samples:
                self.val_samples[spk_num] = []

            if len(self.val_samples[spk_num]) < self.samples_per_speaker_number:
                sample_at_batch = {}
                for k in sample:
                    sample_at_batch[k] = sample[k][i]
                self.val_samples[spk_num].append(sample_at_batch)

        return loss, n_speakers.shape[0]

    def reset_grad(self):
        self.optimizer.zero_grad()

    def backprop(self):
        # Gradient clipping
        if self.grad_clip is not None:
            if self.fabric is not None:
                self.fabric.clip_gradients(
                    self.model, self.optimizer, max_norm=self.grad_clip
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip
                )

        self.optimizer.step()

    def init_scheduler(self, scheduler, scheduler_params):
        if scheduler is not None:
            if scheduler == "sequential":
                schedulers = []
                milestones = []
                for scheduler_param in scheduler_params:
                    sched = _import_attr(scheduler_param["name"])(
                        self.optimizer, **scheduler_param["params"]
                    )
                    schedulers.append(sched)
                    milestones.append(scheduler_param["epochs"])

                # Cumulative sum for milestones
                for i in range(1, len(milestones)):
                    milestones[i] = milestones[i - 1] + milestones[i]

                # Remove last milestone as it is implied by num epochs
                milestones.pop()

                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer, schedulers, milestones
                )
            else:
                scheduler = _import_attr(scheduler)(
                    self.optimizer, **scheduler_params
                )

        return scheduler
