from __future__ import annotations

import torch


def quantize_to_fixed_grid(
    w: torch.Tensor,
    n_bits: int = 2,
    step: float = 0.06,
) -> torch.Tensor:
    """
    n_bits=2, step=0.06 时，量化集合为：
    {-0.06, 0, 0.06}
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")

    L = (2 ** (n_bits - 1)) - 1
    clip = L * step

    q = torch.round(w / step) * step
    q = torch.clamp(q, -clip, clip)

    return q


def ste_quantize_weight_with_mask(
    w: torch.Tensor,
    mask: torch.Tensor,
    n_bits: int = 2,
    step: float = 0.06,
) -> torch.Tensor:
    """
    mask=1：前向使用量化权重
    mask=0：前向使用浮点权重
    反向：梯度直接传给 FP shadow weight
    """
    q = quantize_to_fixed_grid(
        w,
        n_bits=n_bits,
        step=step,
    )

    mask = mask.to(
        device=w.device,
        dtype=w.dtype,
    )

    return w + (mask * (q - w)).detach()
