"""Types for the training process."""

from typing import NotRequired, Required, TypedDict

from monai.data.meta_tensor import MetaTensor
from torch import Tensor


class DataBatch(TypedDict):
    """Data batch with ground truth masks."""

    image: Required[Tensor]
    mask: Required[Tensor]
    num_phases: Required[Tensor]


class PredictDataBatch(TypedDict):
    """Data batch for inference."""

    image: Required[MetaTensor]
    num_phases: Required[Tensor]
    mask: NotRequired[Tensor]


class ValidationMetrics(TypedDict):
    """Validation metrics dict."""

    loss: Required[Tensor | float]
    dice: Required[Tensor | float]
