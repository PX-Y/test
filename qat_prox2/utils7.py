

# ---------------------------------------------------------------------------
# V2: Decoupled-lambda variant
# ---------------------------------------------------------------------------
# Key change vs V1 (original):
#   desired = p_old - lr * task_delta - (dist_scale / max(dual_lambda, min_lambda)) * prox_term
#
# Instead of:  desired = p_old - dist_scale * prox_term - lr * (dual_lambda * task_delta)
#
# Rationale:
#   - lr controls the task gradient step size independently of lambda.
#   - dist_scale / max(dual_lambda, min_lambda) controls the prox (quantization)
#     pressure.  As dual_lambda increases (model struggling), prox weight
#     decreases ~ 1/lambda, focusing more on the task.  As dual_lambda
#     decreases (model doing well), prox strengthens, pushing harder toward
#     the quantization grid.
#   - min_lambda prevents division by zero when dual_lambda is near zero.
#     Default 0.1 means initial prox weight = 10 * dist_scale.
#
# All other logic (stochastic prox, sensitivity, quantize_mask) is unchanged.
import math
import torch
from typing import Optional, Dict, List, Tuple, cast
from .config import QATConfig
from .param_filter import QuantParamSelector, iter_named_quant_params
from .quant_ops import quantize_to_grid
from .sensitivity import SensitivityWeightConfig, compute_sensitivity_stats, compute_quant_rels
#from .stoc_prox import per_coord_midpoint_temperature, temperature_quantize_to_grid, StochasticProxConfig
from .stoc_prox3 import (
    per_coord_midpoint_temperature,
    temperature_quantize_to_grid,
    topk_temperature_quantize_to_grid,
    topk_soft_temperature_quantize_to_grid,
    StochasticProxConfig,
)
from typing import Optional
@torch.no_grad()
def prepare_theory_matched_quant_update_with_stoc_prox_and_sensitivity_v2(
    model,
    optimizer,
    *,
    qat,                                  # QATConfig
    selector,                             # QuantParamSelector
    dual_lambda: float,
    sens_cfg=None,                        # SensitivityWeightConfig
    warmup: bool = False,
    T: float = 0.0,
    sr_cfg=None,
    generator: Optional[torch.Generator] = None,
    quantize_mask: Optional[Dict[str, torch.Tensor]] = None,
    min_lambda: float = 0.1,              # floor for prox-weight denominator
) -> Optional[Dict[str, object]]:

    if sens_cfg is None:
        sens_cfg = SensitivityWeightConfig()
    if sr_cfg is None:
        sr_cfg = StochasticProxConfig()

    if not qat.enabled:
        return None

    group_map = {id(p): group for group in optimizer.param_groups for p in group["params"]}

    sr_active = (
        sr_cfg.enabled
        and not warmup
        and T > 0.0
    )
    T_eff_scalar = min(T, sr_cfg.T_max) if sr_active else 0.0

    # ---- Pass 0: collect entries -------------------------------------------
    entries = []  # (name, p, p_old, q_old, sens)
    for name, p in iter_named_quant_params(
        model,
        selector=selector,
        include_substrings=qat.include_substrings,
        exclude_substrings=qat.exclude_substrings,
    ):
        if not p.requires_grad:
            continue
        p_old = p.data.detach().clone()
        q_old = quantize_to_grid(p_old, qat.n_bits_w, qat.step_w)

        state = optimizer.state.get(p, {})
        sens = state.get("exp_avg_sq", None)
        if sens is not None and p.grad is not None:
            sens = sens.detach()
        else:
            sens = None

        entries.append((name, p, p_old, q_old, sens))

    if not entries:
        return {"prepared": [], "debug": {}}

    # ---- Pass 1: global sensitivity stats ----------------------------------
    valid_sens = [s for *_, s in entries if s is not None]
    stats = compute_sensitivity_stats(valid_sens, sens_cfg) if valid_sens else None

    # ---- Pass 2: per-parameter update --------------------------------------
    prepared = []
    debug_T_means = []

    # Compute prox weight once — same for all params since it only depends on
    # scalar dual_lambda (not per-param).
    prox_weight = qat.dist_scale / max(dual_lambda, min_lambda)

    for name, p, p_old, q_old, sens in entries:
        task_delta = torch.zeros_like(p_old, dtype=torch.float32)
        prox_term = torch.zeros_like(p_old, dtype=torch.float32)
        lr = 0.0

        if p.grad is not None:
            g = p.grad.detach().float()
            group = group_map[id(p)]
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps, wd = float(group["eps"]), float(group.get("weight_decay", 0.0))

            state = optimizer.state[p]
            exp_avg = state.get("exp_avg", torch.zeros_like(g)).detach().float()
            exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(g)).detach().float()
            step_raw = state.get("step", 0)
            step_t = int(step_raw.item() if torch.is_tensor(step_raw) else step_raw) + 1

            exp_avg_next = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1)
            exp_avg_sq_next = exp_avg_sq.mul(beta2).addcmul(g, g, value=1.0 - beta2)
            bc1 = 1.0 - beta1 ** step_t
            bc2 = 1.0 - beta2 ** step_t
            denom = exp_avg_sq_next.sqrt().div(math.sqrt(bc2)).add_(eps)

            adam_term = exp_avg_next.div(denom) / bc1
            if wd != 0.0:
                adam_term = adam_term + wd * p_old.detach().float()
            task_delta = adam_term

            # sensitivity weights (unchanged)
            if stats is not None and sens is not None:
                rels = compute_quant_rels(sens, stats, sens_cfg).to(p_old.device)
            else:
                rels = torch.ones_like(p_old, dtype=torch.float32)

            # ----- stochastic prox term (unchanged) --------------------------
            if sr_active:
                if sr_cfg.per_coord:
                    T_eff = per_coord_midpoint_temperature(
                        p_old, qat.n_bits_w, qat.step_w, T_eff_scalar
                    )
                    T_eff = T_eff.clamp(max=sr_cfg.T_max)
                else:
                    T_eff = T_eff_scalar

                mode = getattr(sr_cfg, "mode", "stochastic")

                if mode == "soft":
                    q_prox = topk_soft_temperature_quantize_to_grid(
                        p_old, qat.n_bits_w, qat.step_w,
                        T_eff,
                        top_k=getattr(sr_cfg, "top_k", 4),
                        tau_scale=getattr(sr_cfg, "tau_scale", 0.5),
                        q_det=q_old,
                    )
                elif getattr(sr_cfg, "top_k", 2) > 2:
                    q_prox = topk_temperature_quantize_to_grid(
                        p_old, qat.n_bits_w, qat.step_w,
                        T_eff,
                        top_k=sr_cfg.top_k,
                        tau_scale=getattr(sr_cfg, "tau_scale", 0.5),
                        q_det=q_old,
                        generator=generator,
                    )
                else:
                    q_prox = temperature_quantize_to_grid(
                        p_old, qat.n_bits_w, qat.step_w,
                        T_eff,
                        q_det=q_old,
                        generator=generator,
                    )

                use_rescale = sr_cfg.rescale and getattr(sr_cfg, "top_k", 2) <= 2
                if use_rescale:
                    if torch.is_tensor(T_eff):
                        prox_rescale = 1.0 / (1.0 - T_eff)
                    else:
                        prox_rescale = 1.0 / max(1.0 - T_eff, 1.0 - sr_cfg.T_max)
                else:
                    prox_rescale = 1.0

                prox_term = (p_old - q_prox).float() * rels * prox_rescale

                if torch.is_tensor(T_eff):
                    debug_T_means.append(float(T_eff.mean().item()))
                else:
                    debug_T_means.append(float(T_eff))
            else:
                prox_term = (p_old - q_old).float() * rels

        # ===== V2 UPDATE (decoupled lambda) =================================
        if warmup:
            desired = p_old - lr * task_delta
        else:
            # task step:  lr * task_delta          (lambda NOT involved)
            # prox step:  prox_weight * prox_term   (prox_weight = dist_scale / max(lambda, min_lambda))
            desired = p_old - lr * task_delta - prox_weight * prox_term

        # quantize_mask (unchanged)
        if (not warmup) and (quantize_mask is not None):
            mask = quantize_mask.get(name, None)
            if mask is not None:
                mask_bool = mask.to(dtype=torch.bool, device=p_old.device)
                desired = desired.to(dtype=p_old.dtype)
                desired = torch.where(mask_bool, p_old, desired)

        prepared.append((p, desired.to(dtype=p.dtype)))

    debug: Dict[str, object] = {"sens_stats": stats, "prox_weight": prox_weight}
    if sr_active:
        debug["sr"] = {
            "T_input": T,
            "T_capped": T_eff_scalar,
            "per_coord": sr_cfg.per_coord,
            "rescale": sr_cfg.rescale,
            "mean_T_eff": (sum(debug_T_means) / len(debug_T_means)) if debug_T_means else 0.0,
        }
    return {"prepared": prepared, "debug": debug}

@torch.no_grad()
def apply_prepared_quant_update(quant_step_pkg):
    """
    Write the precomputed quantization-aware parameter updates into the model.
    Returns debug information for logging.
    """
    if quant_step_pkg is None:
        return {}

    for p, desired in quant_step_pkg.get("prepared", []):
        p.data.copy_(desired)

    return quant_step_pkg.get("debug", {})
