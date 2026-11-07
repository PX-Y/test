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

@torch.no_grad()
def prepare_theory_matched_quant_update(
    model, optimizer, *, qat: QATConfig, selector: QuantParamSelector, dual_lambda: float
) -> Optional[Dict[str, object]]:
    if not qat.enabled: return None

    group_map = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    prepared = []

    for name, p in iter_named_quant_params(
        model, selector=selector, include_substrings=qat.include_substrings, exclude_substrings=qat.exclude_substrings
    ):
        if not p.requires_grad: continue

        p_old = p.data.detach().clone()
        q_old = quantize_to_grid(p_old, qat.n_bits_w, qat.step_w)

        prox_term = (p_old - q_old).float()

        task_delta = torch.zeros_like(p_old, dtype=torch.float32)
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
            step_t = int(state.get("step", 0).item() if torch.is_tensor(state.get("step", 0)) else state.get("step", 0)) + 1
            
            exp_avg_next = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1)
            exp_avg_sq_next = exp_avg_sq.mul(beta2).addcmul(g, g, value=1.0 - beta2)
            bias_correction1, bias_correction2 = 1.0 - beta1 ** step_t, 1.0 - beta2 ** step_t
            denom = exp_avg_sq_next.sqrt().div(math.sqrt(bias_correction2)).add_(eps)
            
            adam_term = exp_avg_next.div(denom) / bias_correction1
            
            #Weight Decay
            if wd != 0.0:
                adam_term = adam_term + wd * p_old.detach().float()

            task_delta = adam_term

       # x_{k+1} = x_k - \eta * [ (x_k - q(x_k)) + \lambda_k * Adam(...) ]
        desired = p_old - lr * (qat.dist_scale * prox_term + dual_lambda * task_delta)
        prepared.append((p, desired.to(dtype=p.dtype), p_old, q_old))
    
    # qat.dist_scale *= 100 ** (1.0 / 5000)  # slowly increase dist_scale from 1.0 to 100.0 over 5000 steps

    return {"prepared": prepared, "debug": {}}

@torch.no_grad()
def prepare_theory_matched_quant_update_with_sensitivity(
    model,
    optimizer,
    *,
    qat: QATConfig,
    selector: QuantParamSelector,
    dual_lambda: float,
    sens_cfg: SensitivityWeightConfig = SensitivityWeightConfig(),
    warmup: bool = False,
) -> Optional[Dict[str, object]]:
    if not qat.enabled:
        return None

    group_map = {id(p): group for group in optimizer.param_groups for p in group["params"]}

    # ---------------------------------------------------------------------
    # Pass 0: walk parameters once, record everything we need for pass 1 and
    # pass 2. Sensitivity tensors are *references* to optimizer state; no
    # extra allocation.
    # ---------------------------------------------------------------------
    entries = []  # (p, p_old, q_old, sens_or_None)
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
        # Use exp_avg_sq (pre-update) as sensitivity. For Adam with beta2~0.999
        # the EMA timescale is ~1000 steps, so one step of lag is a ~0.1%
        # perturbation to a quantity whose log spans many orders of magnitude.
        # Not meaningful. Using exp_avg_sq_next would require two full passes
        # over all params, which isn't worth it.
        sens = state.get("exp_avg_sq", None)
        if sens is not None and p.grad is not None:
            sens = sens.detach()  # keep dtype; conversion happens inside stats
        else:
            sens = None

        entries.append((p, p_old, q_old, sens))

    if not entries:
        return {"prepared": [], "debug": {}}

    # ---------------------------------------------------------------------
    # Pass 1: compute global log-sensitivity statistics. Only scalar
    # accumulators are allocated; sensitivity tensors are visited one at a
    # time and never held in memory simultaneously (they live in optimizer
    # state regardless).
    # ---------------------------------------------------------------------
    valid_sens = [s for *_, s in entries if s is not None]
    stats = compute_sensitivity_stats(valid_sens, sens_cfg) if valid_sens else None

    # ---------------------------------------------------------------------
    # Pass 2: per-parameter update, computing rels on-the-fly from stats.
    # ---------------------------------------------------------------------
    prepared = []
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

            # --- sensitivity-modulated prox -----------------------------------
            # rels: per-entry weight, mean-normalized to 1.
            #   high g² (sensitive) -> low rels  -> weak prox  -> protect
            #   low  g² (insensitive) -> high rels -> strong prox -> quantize
            if stats is not None and sens is not None:
                rels = compute_quant_rels(sens, stats, sens_cfg).to(p_old.device)
            else:
                rels = torch.ones_like(p_old, dtype=torch.float32)

            prox_term = (p_old - q_old).float() * rels

        # x_{k+1} = x_k - lr * [ dist_scale * prox + dual_lambda * task ]
        if warmup:
            desired = p_old - lr * task_delta  # no prox and lambda during warmup
        else:
            desired = p_old - qat.dist_scale * prox_term - lr * (dual_lambda * task_delta)
        prepared.append((p, desired.to(dtype=p.dtype), p_old, q_old))

    return {"prepared": prepared, "debug": {"sens_stats": stats}}


# ---------------------------------------------------------------------------
# Modified main function
# ---------------------------------------------------------------------------
# Changes vs. the original, all isolated to the inner loop:
#   (1) New args: `T: float`, `sr_cfg: StochasticProxConfig`, `generator`.
#   (2) `q_old` is STILL deterministic (returned unchanged for downstream use:
#       dual update sees deterministic loss, deployed checkpoint is q_det).
#   (3) A separate `q_prox` is sampled from the temperature mixture and used
#       only inside the prox term.
#   (4) Prox term is rescaled by 1/(1-T_eff) so its expectation equals the
#       original deterministic prox; eta / dist_scale / dual_lambda untouched.
#   (5) Warmup path is unchanged (no prox -> nothing to randomize).
 
@torch.no_grad()
def prepare_theory_matched_quant_update_with_stoc_prox_and_sensitivity(
    model,
    optimizer,
    *,
    qat,                                  # QATConfig
    selector,                             # QuantParamSelector
    dual_lambda: float,
    sens_cfg=None,                        # SensitivityWeightConfig
    warmup: bool = False,
    # --- new ---------------------------------------------------------------
    T: float = 0.0,
    sr_cfg: StochasticProxConfig = StochasticProxConfig(),
    generator: Optional[torch.Generator] = None,
    quantize_mask: Optional[Dict[str, torch.Tensor]] = None,  # if this is not none, only update non-quantized entries; ignored if warmup is True
    # -----------------------------------------------------------------------
) -> Optional[Dict[str, object]]:
    # These imports / sentinels assume the surrounding module already defines
    # the user's helpers; left as in the original.

    if sens_cfg is None:
        sens_cfg = SensitivityWeightConfig()
 
    if not qat.enabled:
        return None
 
    group_map = {id(p): group for group in optimizer.param_groups for p in group["params"]}
 
    # Decide once per call whether SR is active in this step.
    sr_active = (
        sr_cfg.enabled
        and not warmup
        and T > 0.0
    )
    T_eff_scalar = min(T, sr_cfg.T_max) if sr_active else 0.0
 
    # ---------------------------------------------------------------------
    # Pass 0: collect entries.  q_old stays deterministic.
    # Each entry includes the parameter name so we can consult `quantize_mask`.
    # ---------------------------------------------------------------------
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
        q_old = quantize_to_grid(p_old, qat.n_bits_w, qat.step_w)   # deterministic, unchanged

        state = optimizer.state.get(p, {})
        sens = state.get("exp_avg_sq", None)
        if sens is not None and p.grad is not None:
            sens = sens.detach()
        else:
            sens = None

        entries.append((name, p, p_old, q_old, sens))
 
    if not entries:
        return {"prepared": [], "debug": {}}
 
    # ---------------------------------------------------------------------
    # Pass 1: global sensitivity stats (unchanged).
    # ---------------------------------------------------------------------
    valid_sens = [s for *_, s in entries if s is not None]
    stats = compute_sensitivity_stats(valid_sens, sens_cfg) if valid_sens else None
 
    # ---------------------------------------------------------------------
    # Pass 2: per-parameter update.
    # ---------------------------------------------------------------------
    prepared = []
    debug_T_means = []  # for monitoring effective per-call temperature
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
 
            # rels: per-coord sensitivity weight (unchanged).
            if stats is not None and sens is not None:
                rels = compute_quant_rels(sens, stats, sens_cfg).to(p_old.device)
            else:
                rels = torch.ones_like(p_old, dtype=torch.float32)
 
            # ----- prox term: stochastic mixture + rescale ---------------------
            if sr_active:
                # Effective temperature: scalar or per-coord, capped at T_max.
                if sr_cfg.per_coord:
                    T_eff = per_coord_midpoint_temperature(
                        p_old, qat.n_bits_w, qat.step_w, T_eff_scalar
                    )
                    # Per-coord cap (already <= T_eff_scalar <= T_max, but defensive).
                    T_eff = T_eff.clamp(max=sr_cfg.T_max)
                else:
                    T_eff = T_eff_scalar  # scalar python float
 
                mode = getattr(sr_cfg, "mode", "stochastic")

                if mode == "soft":
                    q_prox = topk_soft_temperature_quantize_to_grid(
                        p_old,
                        qat.n_bits_w,
                        qat.step_w,
                        T_eff,
                        top_k=getattr(sr_cfg, "top_k", 4),
                        tau_scale=getattr(sr_cfg, "tau_scale", 0.5),
                        q_det=q_old,
                    )

                elif getattr(sr_cfg, "top_k", 2) > 2:
                    q_prox = topk_temperature_quantize_to_grid(
                        p_old,
                        qat.n_bits_w,
                        qat.step_w,
                        T_eff,
                        top_k=sr_cfg.top_k,
                        tau_scale=getattr(sr_cfg, "tau_scale", 0.5),
                        q_det=q_old,
                        generator=generator,
                    )

                else:
                    q_prox = temperature_quantize_to_grid(
                        p_old,
                        qat.n_bits_w,
                        qat.step_w,
                        T_eff,
                        q_det=q_old,
                        generator=generator,
                    )
 
                # Rescale so E[(p_old - q_prox) * rescale] = (p_old - q_old).
                use_rescale = sr_cfg.rescale and getattr(sr_cfg, "top_k", 2) <= 2

                if use_rescale:
                    if torch.is_tensor(T_eff):
                        prox_rescale = 1.0 / (1.0 - T_eff)
                    else:
                        prox_rescale = 1.0 / max(1.0 - T_eff, 1.0 - sr_cfg.T_max)
                else:
                    prox_rescale = 1.0
 
                prox_term = (p_old - q_prox).float() * rels * prox_rescale
 
                # Lightweight monitoring (mean per param tensor).
                if torch.is_tensor(T_eff):
                    debug_T_means.append(float(T_eff.mean().item()))
                else:
                    debug_T_means.append(float(T_eff))
            else:
                prox_term = (p_old - q_old).float() * rels
            # -------------------------------------------------------------------
 
        if warmup:
            desired = p_old - lr * task_delta
        else:
            desired = p_old - qat.dist_scale * prox_term - lr * (dual_lambda * task_delta)

        # If a quantize_mask is provided and we're not in warmup, only apply
        # the update to entries that are NOT quantized. The mask is expected
        # to map parameter names to boolean tensors where True indicates the
        # entry was quantized. Missing names are treated as not-quantized.
        if (not warmup) and (quantize_mask is not None):
            mask = quantize_mask.get(name, None)
            if mask is not None:
                mask_bool = mask.to(dtype=torch.bool, device=p_old.device)
                desired = desired.to(dtype=p_old.dtype)
                # where mask==True (quantized) -> keep p_old; else use desired
                desired = torch.where(mask_bool, p_old, desired)

        prepared.append((p, desired.to(dtype=p.dtype), p_old, q_old))
 
    debug: Dict[str, object] = {"sens_stats": stats}
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
def apply_prepared_quant_update(pkg: Optional[Dict[str, object]]) -> Optional[Dict[str, float]]:
    if pkg is None: return None
    prepared = cast(List[Tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor, torch.Tensor]], pkg["prepared"])
    debug = cast(Dict[str, float], pkg["debug"])

    for p, desired, _, _ in prepared:
        p.data.copy_(desired)
    return debug
