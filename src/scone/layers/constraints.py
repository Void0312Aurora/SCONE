from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from scone.diagnostics import Diagnostics
from scone.engine import ConstraintLayer
from scone.state import State


@dataclass(frozen=True)
class NoOpConstraintLayer(ConstraintLayer):
    def project(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        diagnostics: Diagnostics = {
            "constraints": {
                "residual_pos": torch.tensor(0.0, device=state.q.device, dtype=state.q.dtype),
                "residual_vel": torch.tensor(0.0, device=state.q.device, dtype=state.q.dtype),
                "power_error": torch.tensor(0.0, device=state.q.device, dtype=state.q.dtype),
            }
        }
        return state, diagnostics

