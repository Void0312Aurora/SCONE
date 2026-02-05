from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from scone.diagnostics import Diagnostics
from scone.engine import DissipationInputLayer
from scone.state import State


@dataclass(frozen=True)
class LinearDampingLayer(DissipationInputLayer):
    damping: float

    def step(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        if self.damping <= 0.0:
            return state, {}
        scale = torch.exp(torch.tensor(-self.damping * dt, device=state.v.device, dtype=state.v.dtype))
        next_state = State(q=state.q, v=state.v * scale, t=state.t)
        return next_state, {}

