from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from scone.diagnostics import Diagnostics
from scone.engine import EventLayer
from scone.state import State


@dataclass(frozen=True)
class NoOpEventLayer(EventLayer):
    def resolve(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        diagnostics: Diagnostics = {
            "contacts": {"penetration_max": torch.tensor(0.0, device=state.q.device, dtype=state.q.dtype)}
        }
        return state, diagnostics


@dataclass(frozen=True)
class BouncingBallEventLayer(EventLayer):
    mass: float
    restitution: float
    ground_height: float = 0.0

    def resolve(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        q = state.q
        v = state.v
        ground = torch.tensor(self.ground_height, device=q.device, dtype=q.dtype)
        mass = torch.tensor(self.mass, device=q.device, dtype=q.dtype)

        penetration = torch.clamp(ground - q, min=0.0)
        is_penetrating = penetration > 0
        is_approaching = v < 0
        do_bounce = is_penetrating & is_approaching

        next_q = torch.where(is_penetrating, ground, q)
        next_v = torch.where(do_bounce, -self.restitution * v, v)
        next_state = State(q=next_q, v=next_v, t=state.t)

        delta_v = next_v - v
        impulse = mass * delta_v

        # Discrete work via kinetic energy jump: ΔT = 0.5 * m * (v+^2 - v-^2) = 0.5 * (v+ + v-) * I
        d_ke = 0.5 * mass * (next_v * next_v - v * v)
        w_impulse = 0.5 * (next_v + v) * impulse

        diagnostics: Diagnostics = {
            "contacts": {
                "penetration_max": penetration.max(),
                "lambda_n": impulse,
                "complementarity_residual": torch.tensor(0.0, device=q.device, dtype=q.dtype),
            },
            "event": {
                "W_impulse": w_impulse.sum(),
                "dE_event": d_ke.sum(),
            },
        }
        return next_state, diagnostics
