from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from scone.diagnostics import Diagnostics
from scone.state import State


class System:
    def energy(self, state: State) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def mass(self) -> torch.Tensor:
        raise NotImplementedError


class SymplecticCore:
    def step(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        raise NotImplementedError


class DissipationInputLayer:
    def step(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        raise NotImplementedError


class ConstraintLayer:
    def project(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        raise NotImplementedError


class EventLayer:
    def resolve(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        raise NotImplementedError


@dataclass(frozen=True)
class Engine:
    system: System
    core: SymplecticCore
    dissipation: DissipationInputLayer
    constraints: ConstraintLayer
    events: EventLayer

    def step(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        e_kin_0, e_pot_0, e_total_0 = self.system.energy(state)

        state_a, diag_a = self.core.step(state=state, dt=dt, context=context)
        e_kin_a, e_pot_a, e_total_a = self.system.energy(state_a)

        state_b, diag_b = self.dissipation.step(state=state_a, dt=dt, context=context)
        e_kin_b, e_pot_b, e_total_b = self.system.energy(state_b)

        state_c, diag_c = self.constraints.project(state=state_b, dt=dt, context=context)
        e_kin_c, e_pot_c, e_total_c = self.system.energy(state_c)

        state_d, diag_d = self.events.resolve(state=state_c, dt=dt, context=context)
        e_kin_d, e_pot_d, e_total_d = self.system.energy(state_d)

        d_e_a = e_total_a - e_total_0
        d_e_b = e_total_b - e_total_a
        d_e_c = e_total_c - e_total_b
        d_e_d = e_total_d - e_total_c

        p_in = torch.tensor(0.0, device=e_total_d.device, dtype=e_total_d.dtype)
        p_diss = torch.clamp(-d_e_b / dt, min=0.0)

        diagnostics: Diagnostics = {
            "energy": {
                "E_kin": e_kin_d,
                "E_pot": e_pot_d,
                "E_total": e_total_d,
                "dE_A": d_e_a,
                "dE_B": d_e_b,
                "dE_C": d_e_c,
                "dE_D": d_e_d,
            },
            "power": {"P_in": p_in, "P_diss": p_diss},
            "failsafe": {"triggered": False, "reason": ""},
        }

        for diag in (diag_a, diag_b, diag_c, diag_d):
            for key, value in diag.items():
                if key not in diagnostics:
                    diagnostics[key] = value
                elif isinstance(diagnostics[key], dict) and isinstance(value, dict):
                    diagnostics[key].update(value)
                else:
                    diagnostics[key] = value

        if not _all_finite_state(state_d):
            diagnostics["failsafe"] = {"triggered": True, "reason": "non_finite"}
            return state, diagnostics

        return state_d, diagnostics


def _all_finite_state(state: State) -> bool:
    return bool(torch.isfinite(state.q).all().item() and torch.isfinite(state.v).all().item())

