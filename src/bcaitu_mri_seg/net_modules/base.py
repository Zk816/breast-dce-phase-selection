"""Base Lightning net module."""
from collections.abc import Callable
from typing import cast

import lightning as L
import monai.transforms as mt
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import Metric

from bcaitu_mri_seg.types.training import DataBatch, PredictDataBatch, ValidationMetrics


class LitBase(L.LightningModule):
    """Base Lightning net module."""

    _net: torch.nn.Module
    _loss: torch.nn.Module
    _dice: Metric

    def __init__(self, *, patch_size: int | tuple[int, int, int] = 128) -> None:
        """Initialize shared module parameters."""
        super().__init__()

        if isinstance(patch_size, int):
            self._patch_size = (patch_size, patch_size, patch_size)
        else:
            self._patch_size = patch_size

    def forward(self, batch: DataBatch | PredictDataBatch) -> torch.Tensor:
        return self._net(batch["image"])

    def training_step(self, batch: DataBatch, batch_idx: int) -> torch.Tensor:
        y_hat = self.forward(batch)
        loss = self._loss(y_hat, batch["mask"])

        dice = self._dice(self._to_mask(y_hat), batch["mask"])
        dice = torch.nan_to_num(dice, nan=0.0).mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_dice", dice, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: DataBatch, batch_idx: int) -> None:
        metrics = self._compute_val_metrics(batch)
        self.log_dict({
            "val_loss": metrics["loss"],
            "val_dice": metrics["dice"],
        })

    def test_step(self, batch: DataBatch, batch_idx: int) -> None:
        metrics = self._compute_val_metrics(batch)
        self.log_dict({
            "test_loss": metrics["loss"],
            "test_dice": metrics["dice"],
        })

    def predict_step(self, batch: PredictDataBatch, batch_idx: int) -> torch.Tensor:
        y_hat = self._compute_sliding_window(batch).detach()
        return torch.nn.functional.sigmoid(y_hat)

    def get_sw_predictor(
        self,
        batch: DataBatch | PredictDataBatch,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Provide a predictor callable for sw inference that only accepts the image."""
        return self._net

    def _compute_val_metrics(self, batch: DataBatch) -> ValidationMetrics:
        y_hat = self._compute_sliding_window(batch).detach()
        loss = self._loss(y_hat, batch["mask"])
        dice = self._dice(self._to_mask(y_hat), batch["mask"])

        return ValidationMetrics(
            loss=loss,
            dice=dice,
        )

    def _compute_sliding_window(self, batch: DataBatch | PredictDataBatch) -> torch.Tensor:
        y_hat = sliding_window_inference(
            inputs=batch["image"],
            roi_size=self._patch_size,
            sw_batch_size=4,
            predictor=self.get_sw_predictor(batch),
            overlap=0.5,
        )

        return cast("torch.Tensor", y_hat)

    @classmethod
    def _to_mask(cls, y_hat: torch.Tensor) -> torch.Tensor:
        post_transform = mt.Compose([
            mt.Activations(sigmoid=True),
            mt.AsDiscrete(threshold=0.5),
        ])

        return torch.stack([
            post_transform(pred)
            for pred in decollate_batch(y_hat)
        ])
