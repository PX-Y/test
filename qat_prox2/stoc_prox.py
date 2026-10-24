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
 
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
 
@dataclass
class StochasticProxConfig:
    """Static settings for the stochastic proximal term. T itself is annealed
    externally and passed per-call."""
    enabled: bool = False
    per_coord: bool = False     # T_i = T * 4 p_i (1-p_i); only undecided weights explore
    rescale: bool = True        # multiply prox by 1/(1-T) so E[prox] matches deterministic
    T_max: float = 0.95         # safety cap; 1/(1-T_max) = 20 is the largest blow-up allowed
 