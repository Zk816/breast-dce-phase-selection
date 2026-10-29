"""Learned phase selection LightningModule (exp_phase_sel.md).

Stage 1 (``select: true``): backbone receives all K phase channels; a
straight-through Gumbel top-k selector zeroes all but k of them. theta gets 10x
the backbone LR. Per-epoch theta and the current hard selection are logged; the
final selection is written to ``results/{run}/selection.json``.

Stage 2 (``select: false``): no selector, ``in_phases`` = k, the datamodule
feeds only the chosen phases. Same backbone, loss, schedule and epochs, so
Learned-k / Fixed-k / Random-k / All-phases differ only in which phases enter.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig
from monai.metrics import DiceMetric
from monai.networks.nets import UNet

from bcaitu_mri_seg.net_modules.base import LitBase
from bcaitu_mri_seg.net_modules.phase_sel.selector import PhaseSelector
from bcaitu_mri_seg.net_modules.phase_sel.losses import segmentation_loss
from bcaitu_mri_seg.types.training import ValidationMetrics


class LitPhaseSelNet(LitBase):
    """UNet backbone with an optional learned phase-selection layer."""

    def __init__(
        self,
        *,
        in_phases: int = 4,
        select: bool = True,
        k: int = 2,
        tau_start: float = 5.0,
        tau_end: float = 0.5,
        tau_anneal_frac: float = 0.3,
        theta_lr_mult: float = 10.0,
        init_lr: float = 0.01,
        lr_scheduler_power: float = 1.0,
        max_epochs: int = 200,
        depth: int = 4,
        max_channels: int = 512,
        kernel_size: int = 3,
        num_res_units: int = 2,
        patch_size: int = 128,
    ) -> None:
        super().__init__(patch_size=patch_size)
        channels = tuple(min(32 * 2**p, max_channels) for p in range(depth))
        strides = (2,) * (depth - 1)
        self._net = UNet(
            spatial_dims=3, in_channels=in_phases, out_channels=1,
            channels=channels, strides=strides, kernel_size=kernel_size,
            up_kernel_size=kernel_size, num_res_units=num_res_units,
        )
        self._selector = PhaseSelector(in_phases, k, tau_start, tau_end, tau_anneal_frac) if select else None
        self._dice = DiceMetric(include_background=True)
        self._init_lr = init_lr
        self._lr_power = lr_scheduler_power
        self._max_epochs = max_epochs
        self._theta_lr_mult = theta_lr_mult
        self.save_hyperparameters()

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x if self._selector is None else self._selector(x))

    def get_sw_predictor(self, batch):
        return self._predict

    def forward(self, batch):
        return self._predict(batch["image"])

    def on_train_epoch_start(self) -> None:
        if self._selector is not None:
            self._selector.set_progress(self.current_epoch / max(1, self._max_epochs))
            self.log("tau", self._selector.tau, on_epoch=True)

    def training_step(self, batch, batch_idx):
        logits = self._predict(batch["image"])
        loss = segmentation_loss(logits, batch["mask"])
        dice = torch.nan_to_num(self._dice(self._to_mask(logits), batch["mask"]), nan=0.0).mean()
        self.log_dict({"train_loss": loss, "train_dice": dice},
                      on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_after_backward(self) -> None:
        if self._selector is None:
            return
        g = self._selector.theta.grad
        gn = float(g.norm()) if g is not None else 0.0
        if self.global_step == 0 and gn == 0.0:
            msg = "theta.grad is zero after the first step — straight-through relaxation is broken"
            raise RuntimeError(msg)
        self.log("theta_grad_norm", gn, on_step=True)

    def on_train_epoch_end(self) -> None:
        if self._selector is None:
            return
        theta = self._selector.theta.detach()
        self.log_dict({f"theta/{i}": float(v) for i, v in enumerate(theta)}, on_epoch=True)
        idx = self._selector.hard_indices()
        self.log_dict({f"selected/{j}": float(i) for j, i in enumerate(idx)}, on_epoch=True)

    def _compute_val_metrics(self, batch) -> ValidationMetrics:
        y_hat = self._compute_sliding_window(batch).detach()
        return ValidationMetrics(loss=segmentation_loss(y_hat, batch["mask"]),
                                 dice=self._dice(self._to_mask(y_hat), batch["mask"]))

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat = self._compute_sliding_window(batch).detach()
        dice = torch.nan_to_num(self._dice(self._to_mask(y_hat), batch["mask"]), nan=0.0).mean()
        if dataloader_idx == 1:
            self.log("heldout_dice", dice, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            return
        self.log_dict({"val_loss": segmentation_loss(y_hat, batch["mask"]), "val_dice": dice},
                      add_dataloader_idx=False, prog_bar=True)

    def on_train_end(self) -> None:
        if self._selector is None or self.trainer.logger is None:
            return
        out = Path("results", str(self.trainer.logger.name))
        out.mkdir(parents=True, exist_ok=True)
        (out / "selection.json").write_text(json.dumps({
            "run": str(self.trainer.logger.name),
            "k": self._selector.k,
            "num_phases": self.hparams.in_phases,
            "theta": self._selector.theta.detach().cpu().tolist(),
            "selected_indices": self._selector.hard_indices(),
            "pre_contrast_policy": "index 0 always selectable, never forced, never excluded",
        }, indent=2))

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        groups = [{"params": list(self._net.parameters()), "lr": self._init_lr}]
        if self._selector is not None:
            groups.append({"params": [self._selector.theta],
                           "lr": self._init_lr * self._theta_lr_mult})
        optimizer = torch.optim.Adam(groups, lr=self._init_lr)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=self._max_epochs, power=self._lr_power)
        return OptimizerLRSchedulerConfig(
            optimizer=optimizer,
            lr_scheduler=LRSchedulerConfigType(scheduler=scheduler, interval="epoch"),
        )
