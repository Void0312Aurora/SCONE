from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from scone.diagnostics import Diagnostics
from scone.engine import SymplecticCore, System
from scone.state import State


@dataclass(frozen=True)
class SymplecticEulerSeparable(SymplecticCore):
    system: System

    def step(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        mass = self.system.mass()
        momentum = mass * state.v

        grad_v = self._grad_potential(state.q)
        next_momentum = momentum - dt * grad_v
        next_v = next_momentum / mass
        next_q = state.q + dt * next_v

        next_state = State(q=next_q, v=next_v, t=state.t + dt)
        return next_state, {}

    def _grad_potential(self, q: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.system, "grad_potential"):
            raise AttributeError("System must implement grad_potential(q)")
        return getattr(self.system, "grad_potential")(q)

