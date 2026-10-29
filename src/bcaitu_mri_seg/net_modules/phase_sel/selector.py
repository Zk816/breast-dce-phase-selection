"""Learned phase selection: straight-through Gumbel top-k over DCE phases."""
from __future__ import annotations

import torch
import torch.nn as nn


def gumbel_topk_mask(theta: torch.Tensor, k: int, tau: float, training: bool) -> torch.Tensor:
    """Return a differentiable k-hot mask of shape (K,) over phase logits ``theta``."""
    if training:
        u = torch.rand_like(theta).clamp_(1e-10, 1.0)
        g = -torch.log(-torch.log(u))
        scores = theta + g
    else:
        scores = theta

    soft = torch.zeros_like(scores)
    remaining = scores.clone()
    for _ in range(k):
        s = torch.softmax(remaining / tau, dim=-1)
        soft = soft + s
        remaining = remaining + torch.log(1.0 - s + 1e-10)

    idx = torch.topk(scores, k).indices
    hard = torch.zeros_like(scores).scatter(-1, idx, 1.0)

    return hard - soft.detach() + soft


class PhaseSelector(nn.Module):
    """One learnable logit per phase; masks the phase channels of the input."""

    def __init__(
        self,
        num_phases: int,
        k: int,
        tau_start: float = 5.0,
        tau_end: float = 0.5,
        anneal_frac: float = 0.3,
    ) -> None:
        super().__init__()
        if not 1 <= k <= num_phases:
            msg = f"k={k} must satisfy 1 <= k <= num_phases={num_phases}"
            raise ValueError(msg)
        self.theta = nn.Parameter(torch.zeros(num_phases))
        self._k = k
        self._num_phases = num_phases
        self._tau_start = tau_start
        self._tau_end = tau_end
        self._anneal_frac = anneal_frac
        self.register_buffer("_tau", torch.tensor(tau_start))

    @property
    def k(self) -> int:
        return self._k

    @property
    def tau(self) -> float:
        return float(self._tau)

    def set_progress(self, frac: float) -> None:
        """Linearly anneal tau over the first ``anneal_frac`` of training."""
        p = min(1.0, max(0.0, frac / self._anneal_frac)) if self._anneal_frac > 0 else 1.0
        self._tau.fill_(self._tau_start + p * (self._tau_end - self._tau_start))

    def mask(self) -> torch.Tensor:
        m = gumbel_topk_mask(self.theta, self._k, float(self._tau), self.training)
        assert abs(float(m.detach().sum()) - self._k) < 1e-4, f"mask.sum()={float(m.sum())} != k={self._k}"
        return m

    def hard_indices(self) -> list[int]:
        return sorted(torch.topk(self.theta.detach(), self._k).indices.tolist())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mask().to(x.dtype)
        return x * m.view(1, self._num_phases, 1, 1, 1)
