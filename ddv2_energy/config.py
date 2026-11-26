"""Configuration for training-time road and dual-graph rewards."""

from dataclasses import dataclass, fields
import math
from numbers import Integral, Real
from typing import Mapping


@dataclass(frozen=True)
class EnergyRLConfig:
    mode: str = "original"
    context_root: str = ""
    graph_checkpoint: str = ""
    strength: float = 0.1
    road_weight: float = 0.5
    interaction_weight: float = 0.5
    road_scale: float = 1.0
    interaction_scale: float = 1.0
    std_floor: float = 0.05
    adv_clip: float = 5.0
    chunk_size: int = 4
    motion_weight: float = 0.0

    def __post_init__(self):
        if self.mode not in ("original", "dual_graph"):
            raise ValueError("energy_mode must be 'original' or 'dual_graph'")
        for name in (
            "strength", "road_weight", "interaction_weight", "road_scale",
            "interaction_scale", "std_floor", "adv_clip", "motion_weight",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("energy_%s must be a real number" % name)
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("energy_%s must be finite" % name)
            # Reject configurations that cannot be represented by the float32 evaluator.
            if abs(value) > 3.4028234663852886e38:
                raise ValueError("energy_%s must be representable in float32" % name)
            object.__setattr__(self, name, value)
        for name in ("strength", "road_weight", "interaction_weight"):
            if getattr(self, name) < 0:
                raise ValueError("energy_%s must be nonnegative" % name)
        if not math.isclose(
            self.road_weight + self.interaction_weight, 1.0, rel_tol=0, abs_tol=1e-6
        ):
            raise ValueError("road_weight + interaction_weight must equal 1")
        for name in ("road_scale", "interaction_scale", "std_floor", "adv_clip"):
            if getattr(self, name) <= 0:
                raise ValueError("energy_%s must be positive" % name)
        # The motion adapter is not enabled in this release; reject nonzero weights.
        if self.motion_weight != 0:
            raise ValueError("motion energy is disabled; energy_motion_weight must be 0")
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, Integral):
            raise TypeError("energy_chunk_size must be an integer")
        if self.chunk_size < 1:
            raise ValueError("energy_chunk_size must be positive")
        # The enabled engine validates paths; baseline mode loads no reward context.
        for name in ("context_root", "graph_checkpoint"):
            if not isinstance(getattr(self, name), str):
                raise TypeError("energy_%s must be a string" % name)

    @property
    def enabled(self):
        return self.mode == "dual_graph" and self.strength > 0

    @classmethod
    def from_agent_config(cls, config):
        """Read energy_* fields; missing fields keep the baseline branch active."""
        defaults = cls()
        values = {}
        for field in fields(cls):
            key = "energy_" + field.name
            default = getattr(defaults, field.name)
            if isinstance(config, Mapping):
                values[field.name] = config.get(key, default)
            else:
                values[field.name] = getattr(config, key, default)
        return cls(**values)
