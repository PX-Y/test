from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional

@dataclass
class DualController:
    beta: float = 2.5
    dual_lr: float = 1e-4
    lambda_init: float = 1.0
    lambda_max: Optional[float] = None

    lam: float = 1.0

    def __post_init__(self):
        self.lam = float(self.lambda_init)

    def step(self, f_mean: float) -> float:
        # \lambda_{k+1} = \max\{0, \lambda_k + \alpha (f(x_{k+1})-\beta)\}
        g = float(f_mean - self.beta)
        self.lam = max(0.0, float(self.lam + self.dual_lr * g))

        if self.lambda_max is not None:
            self.lam = min(self.lam, float(self.lambda_max))
        return self.lam

@dataclass
class TemperatureController:
    # not implemented per=coordinate temperature control, just a global T that decays over time
    T_init: float = 0.5
    T_final: float = 0.0
    T_decay_steps: int = 5000
    decay_schedule: str = "linear"  # "linear" or "cosine"

    T: float = 0.5

    def __post_init__(self):
        self.T = float(self.T_init)

    def step(self, step: int) -> float:
        if step >= self.T_decay_steps:
            self.T = float(self.T_final)
        else:
            if self.decay_schedule == "linear":
                self.T = float(self.T_init + (self.T_final - self.T_init) * (step / self.T_decay_steps))
            elif self.decay_schedule == "cosine":
                self.T = float(self.T_init + (self.T_final - self.T_init) * (1 - math.cos(math.pi * step / self.T_decay_steps)) / 2)
        return self.T