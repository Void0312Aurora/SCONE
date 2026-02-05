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
    gravity: float
    restitution: float
    ground_height: float = 0.0
    q_slop: float = 1e-3
    v_sleep: float = 0.1

    def resolve(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        q = state.q
        v = state.v
        ground = torch.tensor(self.ground_height, device=q.device, dtype=q.dtype)
        mass = torch.tensor(self.mass, device=q.device, dtype=q.dtype)
        gravity = torch.tensor(self.gravity, device=q.device, dtype=q.dtype)
        q_slop = torch.tensor(self.q_slop, device=q.device, dtype=q.dtype)
        v_sleep = torch.tensor(self.v_sleep, device=q.device, dtype=q.dtype)

        penetration = torch.clamp(ground - q, min=0.0)
        is_penetrating = penetration > 0
        near_ground = q <= ground + q_slop
        is_approaching = v < 0

        do_rest = near_ground & (v.abs() <= v_sleep) & (v <= 0)
        do_bounce = is_penetrating & (v < -v_sleep)

        next_q = torch.where(is_penetrating, ground, q)
        next_v = torch.where(do_bounce, -self.restitution * v, v)
        next_v = torch.where(do_rest, torch.zeros_like(next_v), next_v)
        next_state = State(q=next_q, v=next_v, t=state.t)

        delta_v = next_v - v
        impulse = mass * delta_v

        # Discrete work via kinetic energy jump: ΔT = 0.5 * m * (v+^2 - v-^2) = 0.5 * (v+ + v-) * I
        d_ke = 0.5 * mass * (next_v * next_v - v * v)
        w_impulse = 0.5 * (next_v + v) * impulse
        d_pe = mass * gravity * (next_q - q)
        d_e_event = d_ke + d_pe

        gap = torch.clamp(next_q - ground, min=0.0)
        effective_gap = torch.clamp(gap - q_slop, min=0.0)
        complementarity_residual = torch.abs(torch.clamp(impulse, min=0.0) * effective_gap).max()

        mode = torch.zeros(q.shape, device=q.device, dtype=torch.int64)
        mode = torch.where(is_penetrating, torch.ones_like(mode) * 3, mode)
        mode = torch.where(do_rest, torch.ones_like(mode) * 2, mode)
        mode = torch.where(do_bounce, torch.ones_like(mode) * 1, mode)

        diagnostics: Diagnostics = {
            "contacts": {
                "penetration_max": penetration.max(),
                "lambda_n": impulse,
                "gap_max": gap.max(),
                "complementarity_residual": complementarity_residual,
                "mode": mode.max(),
            },
            "event": {
                "W_impulse": w_impulse.sum(),
                "W_position_correction": d_pe.sum(),
                "dE_kin": d_ke.sum(),
                "dE_pot": d_pe.sum(),
                "dE_event": d_e_event.sum(),
            },
        }
        return next_state, diagnostics
