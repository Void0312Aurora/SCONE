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
        p_diss_signed = -d_e_b / dt
        p_diss = torch.clamp(p_diss_signed, min=0.0)

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
            "power": {"P_in": p_in, "P_diss": p_diss, "P_diss_signed": p_diss_signed},
            "failsafe": {"triggered": False, "reason": "", "soft_reasons": []},
        }

        for diag in (diag_a, diag_b, diag_c, diag_d):
            for key, value in diag.items():
                if key not in diagnostics:
                    diagnostics[key] = value
                elif isinstance(diagnostics[key], dict) and isinstance(value, dict):
                    diagnostics[key].update(value)
                else:
                    diagnostics[key] = value

        soft_reasons: list[str] = []
        if bool((p_diss_signed < -1e-12).item()):
            soft_reasons.append("dissipation_energy_increase")

        failsafe_cfg = context.get("failsafe", {}) if isinstance(context, dict) else {}
        if isinstance(failsafe_cfg, dict):
            solver_residual_soft = failsafe_cfg.get("solver_residual_soft")
            if solver_residual_soft is not None:
                solver = diagnostics.get("solver", {})
                if isinstance(solver, dict):
                    residual = solver.get("residual_max")
                    if isinstance(residual, torch.Tensor):
                        residual_value = float(residual.detach().max().cpu().item())
                    else:
                        residual_value = float(residual or 0.0)
                    if residual_value > float(solver_residual_soft):
                        soft_reasons.append("solver_residual_soft")
        diagnostics["failsafe"]["soft_reasons"] = soft_reasons

        hard_reason = _hard_failsafe_reason(
            state_before=state,
            state_after=state_d,
            energy_before=e_total_0,
            energy_after=e_total_d,
            diagnostics=diagnostics,
            context=context,
        )
        if hard_reason is not None:
            diagnostics["failsafe"]["triggered"] = True
            diagnostics["failsafe"]["reason"] = hard_reason
            return state, diagnostics

        return state_d, diagnostics


def _all_finite_state(state: State) -> bool:
    return bool(torch.isfinite(state.q).all().item() and torch.isfinite(state.v).all().item())


def _hard_failsafe_reason(
    *,
    state_before: State,
    state_after: State,
    energy_before: torch.Tensor,
    energy_after: torch.Tensor,
    diagnostics: Diagnostics,
    context: dict[str, Any],
) -> str | None:
    del state_before

    if not _all_finite_state(state_after):
        return "non_finite"

    failsafe_cfg = context.get("failsafe", {}) if isinstance(context, dict) else {}
    if not isinstance(failsafe_cfg, dict):
        failsafe_cfg = {}

    penetration_hard = failsafe_cfg.get("penetration_hard")
    if penetration_hard is not None:
        penetration_max = diagnostics.get("contacts", {}).get("penetration_max")
        if isinstance(penetration_max, torch.Tensor):
            penetration_value = float(penetration_max.detach().max().cpu().item())
        else:
            penetration_value = float(penetration_max or 0.0)
        if penetration_value > float(penetration_hard):
            return "penetration_hard"

    v_hard = failsafe_cfg.get("v_hard")
    if v_hard is not None:
        max_abs_v = float(state_after.v.detach().abs().max().cpu().item())
        if max_abs_v > float(v_hard):
            return "velocity_hard"

    energy_abs_hard = failsafe_cfg.get("energy_abs_hard")
    if energy_abs_hard is not None:
        if float(energy_after.detach().abs().cpu().item()) > float(energy_abs_hard):
            return "energy_abs_hard"

    energy_factor_hard = failsafe_cfg.get("energy_factor_hard")
    if energy_factor_hard is not None:
        base = float(energy_before.detach().abs().cpu().item())
        if base > 0.0:
            if float(energy_after.detach().abs().cpu().item()) > float(energy_factor_hard) * base:
                return "energy_factor_hard"

    solver_residual_hard = failsafe_cfg.get("solver_residual_hard")
    if solver_residual_hard is not None:
        solver = diagnostics.get("solver", {})
        if isinstance(solver, dict):
            residual = solver.get("residual_max")
            if isinstance(residual, torch.Tensor):
                residual_value = float(residual.detach().max().cpu().item())
            else:
                residual_value = float(residual or 0.0)
            if residual_value > float(solver_residual_hard):
                return "solver_residual_hard"

    return None
