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

        next_q, next_v = _apply_sleep_freeze(q=next_q, v=next_v, q_prev=state.q, context=context)

        next_state = State(q=next_q, v=next_v, t=state.t + dt)
        return next_state, {}

    def _grad_potential(self, q: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.system, "grad_potential"):
            raise AttributeError("System must implement grad_potential(q)")
        return getattr(self.system, "grad_potential")(q)


def _apply_sleep_freeze(
    *, q: torch.Tensor, v: torch.Tensor, q_prev: torch.Tensor, context: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    sleep_ctx = context.get("sleep", {}) if isinstance(context, dict) else {}
    if not isinstance(sleep_ctx, dict):
        return q, v

    if not bool(sleep_ctx.get("freeze_core", False)):
        return q, v

    sleeping_mask = sleep_ctx.get("sleeping_mask")
    if not isinstance(sleeping_mask, torch.Tensor):
        return q, v
    if sleeping_mask.numel() == 0 or not bool(sleeping_mask.any().item()):
        return q, v

    mask = sleeping_mask.to(device=v.device)
    while mask.ndim < v.ndim:
        mask = mask.unsqueeze(-1)

    q_frozen = torch.where(mask, q_prev, q)
    v_frozen = torch.where(mask, torch.zeros_like(v), v)
    return q_frozen, v_frozen
