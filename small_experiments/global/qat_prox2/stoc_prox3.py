from dataclasses import dataclass
from typing import Dict, Optional
import torch



@torch.no_grad()
def stochastic_quantize_to_grid(
    w: torch.Tensor,
    n_bits: int,
    step: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Per-coordinate unbiased stochastic rounding onto
        {k*step | k in [-L, L]}, L = 2^(n_bits-1) - 1.
 
    For w_i with neighbors g_lo <= w_i <= g_hi (after clip):
        P[out_i = g_hi] = (w_i - g_lo) / step
        P[out_i = g_lo] = 1 - P[out_i = g_hi]
 
    E[stochastic_quantize_to_grid(w)] = clip(w, [-L*step, L*step]) elementwise.
    Variance is 0 on-grid and maximal at midpoints.
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
 
    L = (2 ** (n_bits - 1)) - 1
    clip = L * step
 
    w_clipped = torch.clamp(w, -clip, clip)
    u = w_clipped / step
    floor_u = torch.floor(u)
    frac = u - floor_u  # in [0, 1)
 
    if generator is not None:
        xi = torch.rand(w.shape, generator=generator, device=w.device, dtype=w.dtype)
    else:
        xi = torch.rand_like(w)
 
    round_up = (xi < frac).to(u.dtype)
    q = (floor_u + round_up) * step
    return torch.clamp(q, -clip, clip)
 
 
@torch.no_grad()
def per_coord_midpoint_temperature(
    w: torch.Tensor,
    n_bits: int,
    step: float,
    T: float,
) -> torch.Tensor:
    """
    Per-coordinate temperature T_i = T * 4 * p_i * (1 - p_i), where
    p_i = (w_i - g_lo_i) / step is the fractional offset to the upper neighbor.
 
    4*p*(1-p) in [0, 1]: vanishes on-grid (no exploration needed),
    peaks at 1 at the midpoint (maximum uncertainty about rounding direction).
    """
    L = (2 ** (n_bits - 1)) - 1
    clip = L * step
    w_clipped = torch.clamp(w, -clip, clip)
    u = w_clipped / step
    frac = u - torch.floor(u)
    return (T * 4.0) * frac * (1.0 - frac)
 
 
@torch.no_grad()
def temperature_quantize_to_grid(
    w: torch.Tensor,
    n_bits: int,
    step: float,
    T,                                  # scalar in [0,1) OR per-coord tensor in [0,1)
    *,
    q_det: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Mixture of stochastic and deterministic rounding:
 
        out_i = stochastic_round(w_i)    if zeta_i < T_i      (zeta_i ~ U[0,1] iid)
              = deterministic_round(w_i) otherwise
 
    Pass `q_det` to avoid recomputing the deterministic quantization when the
    caller already has it (the common case here).
    """
    is_tensor_T = torch.is_tensor(T)
 
    # Fast path: T is identically zero.
    if not is_tensor_T and T <= 0.0:
        return q_det if q_det is not None else (
            torch.clamp(torch.round(w / step) * step,
                        -((2 ** (n_bits - 1)) - 1) * step,
                         ((2 ** (n_bits - 1)) - 1) * step)
        )
 
    if q_det is None:
        # Inline deterministic quantize (mirrors quantize_to_grid in the user's code).
        L = (2 ** (n_bits - 1)) - 1
        clip = L * step
        q_det = torch.clamp(torch.round(w / step) * step, -clip, clip)
 
    q_stoch = stochastic_quantize_to_grid(w, n_bits, step, generator=generator)
 
    if generator is not None:
        zeta = torch.rand(w.shape, generator=generator, device=w.device, dtype=w.dtype)
    else:
        zeta = torch.rand_like(w)
 
    use_stoch = zeta < T  # broadcasts whether T is scalar or tensor
    return torch.where(use_stoch, q_stoch, q_det)
 
 
 
@torch.no_grad()
def topk_stochastic_quantize_to_grid(
    w: torch.Tensor,
    n_bits: int,
    step: float,
    top_k: int = 8,
    tau=None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample q_prox from the nearest top-k grid points.

    Quantization grid:
        {k * step | k in [-L, L]}, L = 2^(n_bits-1) - 1

    For each w_i:
        1. find nearest top_k grid points
        2. assign probability by distance:
              p_j ∝ exp(-|w_i - q_j| / tau)
        3. sample one q_j as q_prox_i

    Note:
        top_k will be clipped by the total number of grid points.
        For n_bits=2, there are only 3 grid points, so top_k=4 becomes 3.
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")

    L = (2 ** (n_bits - 1)) - 1
    n_levels = 2 * L + 1
    top_k = min(int(top_k), n_levels)

    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    clip = L * step
    w_clipped = torch.clamp(w, -clip, clip)

    # grid index:
    #   index 0 corresponds to -L * step
    #   index L corresponds to 0
    #   index 2L corresponds to +L * step
    u = w_clipped / step
    center_idx = torch.round(u).to(torch.long) + L
    center_idx = center_idx.clamp(0, n_levels - 1)

    # choose a local top-k window around center_idx
    left = (top_k - 1) // 2
    start_idx = center_idx - left
    start_idx = start_idx.clamp(0, n_levels - top_k)

    offsets = torch.arange(top_k, device=w.device, dtype=torch.long)
    cand_idx = start_idx.unsqueeze(-1) + offsets

    # convert index back to quantization values
    cand_q = (cand_idx.to(dtype=torch.float32) - float(L)) * float(step)
    cand_q = cand_q.to(device=w.device, dtype=w.dtype)

    # distances: shape = w.shape + (top_k,)
    dist = torch.abs(w_clipped.unsqueeze(-1) - cand_q).float()

    # tau controls exploration.
    # Smaller tau -> mostly nearest point.
    # Larger tau -> more probability on farther candidates.
    if tau is None:
        tau = 0.5 * step

    if torch.is_tensor(tau):
        tau_eff = tau.to(device=w.device, dtype=torch.float32).clamp_min(1e-12)
        tau_eff = tau_eff.unsqueeze(-1)
    else:
        tau_eff = max(float(tau), 1e-12)

    logits = -dist / tau_eff
    probs = torch.softmax(logits, dim=-1)

    flat_probs = probs.reshape(-1, top_k)

    if generator is not None:
        sampled_local = torch.multinomial(flat_probs, 1, generator=generator)
    else:
        sampled_local = torch.multinomial(flat_probs, 1)

    q_soft = (probs * cand_q).sum(dim=-1)
    return q_soft


@torch.no_grad()
def topk_temperature_quantize_to_grid(
    w: torch.Tensor,
    n_bits: int,
    step: float,
    T,
    *,
    top_k: int = 8,
    tau_scale: float = 0.5,
    q_det: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Mixture of deterministic nearest quantization and top-k stochastic quantization.

        out_i = top-k stochastic sample    if zeta_i < T_i
              = deterministic nearest     otherwise

    T controls how often we use stochastic top-k.
    tau controls how spread the top-k probability is.
    """

    is_tensor_T = torch.is_tensor(T)

    if not is_tensor_T and T <= 0.0:
        if q_det is not None:
            return q_det
        L = (2 ** (n_bits - 1)) - 1
        clip = L * step
        return torch.clamp(torch.round(w / step) * step, -clip, clip)

    if q_det is None:
        L = (2 ** (n_bits - 1)) - 1
        clip = L * step
        q_det = torch.clamp(torch.round(w / step) * step, -clip, clip)

    # Use T also to anneal tau:
    # early training: T large -> tau large -> more exploration
    # late training: T small -> tau small -> almost nearest point
    if is_tensor_T:
        tau = step * tau_scale * T.clamp_min(1e-4)
    else:
        tau = step * tau_scale * max(float(T), 1e-4)

    q_topk = topk_stochastic_quantize_to_grid(
        w,
        n_bits=n_bits,
        step=step,
        top_k=top_k,
        tau=tau,
        generator=generator,
    )

    if generator is not None:
        zeta = torch.rand(w.shape, generator=generator, device=w.device, dtype=w.dtype)
    else:
        zeta = torch.rand_like(w)

    use_stoch = zeta < T
    return torch.where(use_stoch, q_topk, q_det)

@torch.no_grad()
def topk_soft_quantize_to_grid(
    w: torch.Tensor,
    n_bits: int,
    step: float,
    top_k: int = 8,
    tau=None,
) -> torch.Tensor:
    """
    Soft top-k quantization.

    Instead of sampling one candidate q_j, compute

        q_soft = sum_j softmax(-|w - q_j| / tau) * q_j

    This is the "soft" version:
        stochastic top-k: pick one q_j
        soft top-k: use weighted average of top-k q_j
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")

    L = (2 ** (n_bits - 1)) - 1
    n_levels = 2 * L + 1
    top_k = min(int(top_k), n_levels)

    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    clip = L * step
    w_clipped = torch.clamp(w, -clip, clip)

    # nearest grid index
    u = w_clipped / step
    center_idx = torch.round(u).to(torch.long) + L
    center_idx = center_idx.clamp(0, n_levels - 1)

    # local top-k window around nearest grid point
    left = (top_k - 1) // 2
    start_idx = center_idx - left
    start_idx = start_idx.clamp(0, n_levels - top_k)

    offsets = torch.arange(top_k, device=w.device, dtype=torch.long)
    cand_idx = start_idx.unsqueeze(-1) + offsets

    # candidate quantization values
    cand_q = (cand_idx.to(dtype=torch.float32) - float(L)) * float(step)
    cand_q = cand_q.to(device=w.device, dtype=w.dtype)

    # distance to candidates
    dist = torch.abs(w_clipped.unsqueeze(-1) - cand_q).float()

    if tau is None:
        tau = 0.5 * step

    if torch.is_tensor(tau):
        tau_eff = tau.to(device=w.device, dtype=torch.float32).clamp_min(1e-12)
        tau_eff = tau_eff.unsqueeze(-1)
    else:
        tau_eff = max(float(tau), 1e-12)

    # soft assignment weights
    logits = -dist / tau_eff
    probs = torch.softmax(logits, dim=-1)

    # soft quantized point: weighted average
    q_soft = (probs.to(dtype=cand_q.dtype) * cand_q).sum(dim=-1)

    return q_soft.to(dtype=w.dtype)

@torch.no_grad()
def topk_soft_temperature_quantize_to_grid(
    w: torch.Tensor,
    n_bits: int,
    step: float,
    T,
    *,
    top_k: int = 4,
    tau_scale: float = 0.5,
    q_det: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Temperature version of soft top-k quantization.

    When T > 0:
        q_soft = sum_j softmax(-|w - q_j| / tau) * q_j

    Then mix with deterministic q_det:

        q_prox = T * q_soft + (1 - T) * q_det

    This is fully deterministic and smooth.
    """
    is_tensor_T = torch.is_tensor(T)

    if not is_tensor_T and T <= 0.0:
        if q_det is not None:
            return q_det
        L = (2 ** (n_bits - 1)) - 1
        clip = L * step
        return torch.clamp(torch.round(w / step) * step, -clip, clip)

    if q_det is None:
        L = (2 ** (n_bits - 1)) - 1
        clip = L * step
        q_det = torch.clamp(torch.round(w / step) * step, -clip, clip)

    # tau anneals with T:
    # early: larger T -> softer assignment
    # late: T -> 0 -> almost deterministic
    if is_tensor_T:
        T_eff = T.to(device=w.device, dtype=torch.float32).clamp(0.0, 1.0)
        tau = step * tau_scale * T_eff.clamp_min(1e-4)
    else:
        T_eff = max(min(float(T), 1.0), 0.0)
        tau = step * tau_scale * max(T_eff, 1e-4)

    q_soft = topk_soft_quantize_to_grid(
        w,
        n_bits=n_bits,
        step=step,
        top_k=top_k,
        tau=tau,
    )

    # Smooth mixture, no sampling.
    if torch.is_tensor(T_eff):
        return ((1.0 - T_eff) * q_det.float() + T_eff * q_soft.float()).to(dtype=w.dtype)
    else:
        return ((1.0 - T_eff) * q_det.float() + T_eff * q_soft.float()).to(dtype=w.dtype)



@dataclass
class StochasticProxConfig:
    """Static settings for the stochastic / soft proximal term."""
    enabled: bool = False
    per_coord: bool = False

    # For top-k stochastic/soft, usually set False first.
    rescale: bool = False

    T_max: float = 0.95

    # top-k candidates
    top_k: int = 4

    # tau = step * tau_scale * T
    tau_scale: float = 0.5

    # New:
    # "stochastic": sample one q from top-k
    # "soft": use soft weighted average of top-k q's
    mode: str = "stochastic"
