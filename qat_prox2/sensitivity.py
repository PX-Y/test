from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import torch

from .param_filter import QuantParamSelector, iter_named_quant_params


from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch


@dataclass(frozen=True)
class SensitivityWeightConfig:
    """Parameters controlling the log-sigmoid sensitivity -> weight mapping."""
    w_min: float = 1.0
    w_max: float = 5.0
    eps_log: float = 1e-12
    eps_sigma: float = 1e-6
    normalize: bool = True    # mean-normalize output to 1.0


@dataclass(frozen=True)
class SensitivityStats:
    """Pooled log-space statistics over all quantizable per-entry sensitivities."""
    mu: torch.Tensor          # scalar: mean of log(s)
    sigma: torch.Tensor       # scalar: std of log(s), clamped
    w_mean: torch.Tensor      # scalar: population mean of unnormalized weights
    count: int                # total number of entries pooled

    @property
    def device(self) -> torch.device:
        return self.mu.device


def compute_sensitivity_stats(
    sensitivities: Iterable[torch.Tensor],
    cfg: SensitivityWeightConfig = SensitivityWeightConfig(),
    device: Optional[torch.device] = None,
) -> SensitivityStats:
    """
    First pass: stream through per-parameter sensitivity tensors and accumulate
    enough moments to compute mu, sigma, and w_mean of log(sensitivity).

    Only scalar accumulators are kept in memory; sensitivity tensors are
    visited one at a time and can be freed after each iteration.

    For mean-normalization of downstream weights we also need E[sigmoid((mu-z)/sigma)],
    which depends on the final mu and sigma. Since those aren't known until after
    the first pass, we need a second sub-pass (still streaming) over the same
    sensitivities to compute w_mean. Caller provides an iterable that can be
    re-iterated — or pass `normalize=False` in the config to skip it.
    """
    # --- Sub-pass 1a: accumulate sum and sum-of-squares of log(s) -------------
    # Use float64 for accumulators to avoid catastrophic cancellation when
    # summing millions of entries.
    acc_device = device
    total_sum = None
    total_sqsum = None
    total_count = 0

    # Cache the iterable if it's a generator — caller should pass a list or a
    # re-iterable source if normalization is needed.
    sens_list = list(sensitivities) if cfg.normalize else sensitivities

    for s in sens_list:
        if s is None:
            continue
        if acc_device is None:
            acc_device = s.device
            total_sum = torch.zeros((), dtype=torch.float64, device=acc_device)
            total_sqsum = torch.zeros((), dtype=torch.float64, device=acc_device)
        z = torch.log(s.detach().float() + cfg.eps_log)
        if acc_device is not None:
            z = z.to(acc_device)
        assert total_sum is not None and total_sqsum is not None
        total_sum += z.double().sum()
        total_sqsum += z.double().pow(2).sum()
        total_count += z.numel()
        del z  # explicit: free intermediate immediately

    if total_count == 0:
        raise ValueError("No sensitivity entries provided.")

    assert total_sum is not None and total_sqsum is not None

    mean = total_sum / total_count
    var = total_sqsum / total_count - mean * mean
    # var can go slightly negative due to float error; clamp then sqrt.
    std = var.clamp_min(0.0).sqrt().clamp_min(cfg.eps_sigma)

    mu = mean.float()
    sigma = std.float()

    # --- Sub-pass 1b (optional): compute population mean of unnormalized w ----
    if not cfg.normalize:
        w_mean = torch.tensor(1.0, device=mu.device)
    else:
        w_sum = torch.zeros((), dtype=torch.float64, device=mu.device)
        w_count = 0
        w_range = cfg.w_max - cfg.w_min
        for s in sens_list:
            if s is None:
                continue
            z = torch.log(s.detach().float() + cfg.eps_log)
            if device is not None:
                z = z.to(device)
            score = (mu - z) / sigma
            w = cfg.w_min + w_range * torch.sigmoid(score)
            w_sum += w.double().sum()
            w_count += w.numel()
            del z, score, w
        w_mean = (w_sum / w_count).float().clamp_min(cfg.eps_sigma)

    return SensitivityStats(mu=mu, sigma=sigma, w_mean=w_mean, count=total_count)

def compute_quant_rels(
    s: torch.Tensor,
    stats: SensitivityStats,
    cfg: SensitivityWeightConfig = SensitivityWeightConfig(),
) -> torch.Tensor:
    """
    Second pass: given one parameter's sensitivity tensor and the precomputed
    global stats, return the per-entry weight tensor (same shape as s).

    Cross-model calibration is preserved because mu/sigma/w_mean were computed
    over the full population in the first pass.
    """
    s = s.detach().float()
    zz = torch.log(s + cfg.eps_log).to(stats.mu.device)

    score = (stats.mu - zz) / stats.sigma
    w_range = cfg.w_max - cfg.w_min
    w = cfg.w_min + w_range * torch.sigmoid(score)

    if cfg.normalize:
        w = w / stats.w_mean

    return w