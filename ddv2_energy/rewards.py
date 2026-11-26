"""Bounded energy rewards and per-anchor group advantages.

The cost is c = w_r E_r/(s_r+E_r) + w_i E_i/(s_i+E_i).
Rewards R = PDM - gamma*c and advantages are detached; policy gradients
flow through the denoising transition log-probability objective.
"""

from typing import Dict, Optional, Tuple

import torch

from .config import EnergyRLConfig


def _validate_float(name, value, shape=None, device=None):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError("%s must be a floating-point tensor" % name)
    if value.ndim != 3 or any(size == 0 for size in value.shape):
        raise ValueError("%s must have nonempty [B,G,N] shape" % name)
    if shape is not None and value.shape != shape:
        raise ValueError("%s shape %s does not match %s" % (name, value.shape, shape))
    if device is not None and value.device != device:
        raise ValueError("%s must be on the same device as pdm" % name)
    if not torch.isfinite(value).all():
        raise ValueError("%s contains NaN or infinity" % name)


def _as_float32(name, value):
    result = value.detach().float()
    if not torch.isfinite(result).all():
        raise ValueError("%s overflows float32" % name)
    return result


def _bounded_cost(energy, scale):
    # Equivalent to E/(scale+E), using branches to avoid addition overflow.
    # Clamping also prevents a zero denominator in the unselected large-value branch.
    ratio = energy / scale
    small = ratio / (1.0 + ratio.clamp(max=1.0))
    large = 1.0 / (1.0 + scale / energy.clamp(min=scale))
    return torch.where(energy <= scale, small, large)


def _advantage_metrics(advantages):
    return {
        "energy_adv_positive_fraction": (advantages > 0).float().mean(),
        "energy_adv_zero_fraction": (advantages == 0).float().mean(),
        "energy_adv_negative_fraction": (advantages < 0).float().mean(),
        "energy_adv_mean": advantages.float().mean(),
    }


@torch.no_grad()
def energy_advantages(
    pdm: torch.Tensor,
    original: torch.Tensor,
    road: Optional[torch.Tensor],
    interaction: Optional[torch.Tensor],
    unsafe: Optional[torch.Tensor],
    cfg: EnergyRLConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return advantages [B,G,N] and scalar diagnostics.

    G is the sampling group within each anchor. In baseline or zero-strength
    mode, return the supplied upstream advantages without changing their dtype
    or values. The active branch normalizes energy-shaped rewards within each
    anchor and does not apply the upstream ground-truth reward gate.
    """
    if not isinstance(cfg, EnergyRLConfig):
        raise TypeError("cfg must be EnergyRLConfig")
    _validate_float("pdm", pdm)
    _validate_float("original", original, pdm.shape, pdm.device)
    pdm32 = _as_float32("pdm", pdm)
    if not cfg.enabled:
        # Preserve baseline values and dtype exactly when the branch is disabled.
        result = original.detach().clone()
        zero = pdm32.new_zeros(())
        metrics = {
            "energy_enabled": zero,
            "energy_road_mean": zero.clone(),
            "energy_interaction_mean": zero.clone(),
            "energy_bounded_cost_mean": zero.clone(),
            "energy_shaped_reward_mean": pdm32.mean(),
            "energy_group_std_mean": pdm32.std(dim=1, unbiased=False).mean(),
        }
        metrics.update(_advantage_metrics(result))
        return result, metrics

    if pdm.shape[1] < 2:
        raise ValueError("dual_graph RL requires G >= 2 samples within each anchor")
    for name, value in (("road", road), ("interaction", interaction)):
        _validate_float(name, value, pdm.shape, pdm.device)
        if (value < 0).any():
            raise ValueError("%s energy must be nonnegative" % name)
    if not isinstance(unsafe, torch.Tensor) or unsafe.dtype != torch.bool:
        raise TypeError("unsafe must be a bool tensor")
    if unsafe.shape != pdm.shape or unsafe.device != pdm.device:
        raise ValueError("unsafe must match pdm shape and device")

    road32 = _as_float32("road", road)
    interaction32 = _as_float32("interaction", interaction)
    cost = (
        cfg.road_weight * _bounded_cost(road32, cfg.road_scale)
        + cfg.interaction_weight * _bounded_cost(interaction32, cfg.interaction_scale)
    )
    shaped = pdm32 - cfg.strength * cost
    if not torch.isfinite(shaped).all():
        raise ValueError("energy-shaped rewards contain NaN or infinity")
    # Use population variance over groups within each anchor and scene.
    mean = shaped.mean(dim=1, keepdim=True)
    std = shaped.std(dim=1, keepdim=True, unbiased=False)
    safe_advantage = ((shaped - mean) / std.clamp_min(cfg.std_floor)).clamp(
        min=0.0, max=cfg.adv_clip
    )
    # Unsafe candidates retain negative advantages weighted by risk severity.
    # An entirely unsafe group must not produce positive advantages.
    result = torch.where(unsafe, -1.0 - cfg.strength * cost, safe_advantage)
    if not torch.isfinite(result).all():
        raise ValueError("energy advantages contain NaN or infinity")
    metrics = {
        "energy_enabled": pdm32.new_ones(()),
        "energy_road_mean": road32.mean(),
        "energy_interaction_mean": interaction32.mean(),
        "energy_bounded_cost_mean": cost.mean(),
        "energy_shaped_reward_mean": shaped.mean(),
        "energy_group_std_mean": std.mean(),
    }
    metrics.update(_advantage_metrics(result))
    return result.detach(), {key: value.detach() for key, value in metrics.items()}
