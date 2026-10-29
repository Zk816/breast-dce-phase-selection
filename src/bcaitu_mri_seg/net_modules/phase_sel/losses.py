from __future__ import annotations

import torch
import torch.nn.functional as F


def segmentation_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    focal_pos_weight: float = 0.75,
    gamma: float = 2.0,
    smooth: float = 1e-5,
) -> torch.Tensor:
    logits = logits.float()
    mask = mask.float()

    p = torch.sigmoid(logits)
    p_flat = p.reshape(-1)
    m_flat = mask.reshape(-1)

    tp = (p_flat * m_flat).sum()
    fp = (p_flat * (1.0 - m_flat)).sum()
    fn = ((1.0 - p_flat) * m_flat).sum()
    tversky = 1.0 - (tp + smooth) / (tp + tversky_alpha * fp + tversky_beta * fn + smooth)

    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    pt = torch.where(mask.bool(), p, 1.0 - p)
    alpha_t = torch.where(mask.bool(),
                          torch.full_like(bce, focal_pos_weight),
                          torch.full_like(bce, 1.0 - focal_pos_weight))
    focal_loss = (alpha_t * (1.0 - pt) ** gamma * bce).mean()

    return tversky + focal_loss
