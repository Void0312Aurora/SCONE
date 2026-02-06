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
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

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
            "failsafe": {
                "triggered": False,
                "reason": "",
                "soft_reasons": [],
                "alpha_blend": torch.tensor(1.0, device=e_total_d.device, dtype=e_total_d.dtype),
            },
        }

        for diag in (diag_a, diag_b, diag_c, diag_d):
            for key, value in diag.items():
                if key not in diagnostics:
                    diagnostics[key] = value
                elif isinstance(diagnostics[key], dict) and isinstance(value, dict):
                    diagnostics[key].update(value)
                else:
                    diagnostics[key] = value

        _ensure_minimum_diagnostics(
            diagnostics=diagnostics,
            reference=e_total_d,
            dt=dt,
            d_e_c=d_e_c,
            d_e_d=d_e_d,
        )
        _update_ledger_consistency(diagnostics=diagnostics, dt=dt, d_e_b=d_e_b, d_e_c=d_e_c, d_e_d=d_e_d, reference=e_total_d)

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
            ledger_b_soft = failsafe_cfg.get("ledger_balance_soft")
            if ledger_b_soft is not None:
                ledger = diagnostics.get("ledger", {})
                if isinstance(ledger, dict):
                    error_b = ledger.get("B_balance_error_abs", 0.0)
                    error_b_value = _to_float(error_b)
                    if error_b_value > float(ledger_b_soft):
                        soft_reasons.append("ledger_B_balance_soft")

            event_energy_soft = failsafe_cfg.get("event_energy_soft")
            if event_energy_soft is not None:
                ledger = diagnostics.get("ledger", {})
                if isinstance(ledger, dict):
                    error_d = ledger.get("D_energy_mismatch_abs", 0.0)
                    error_d_value = _to_float(error_d)
                    if error_d_value > float(event_energy_soft):
                        soft_reasons.append("ledger_D_energy_soft")

            constraint_power_soft = failsafe_cfg.get("constraint_power_soft")
            if constraint_power_soft is not None:
                constraints_diag = diagnostics.get("constraints", {})
                if isinstance(constraints_diag, dict):
                    power_error = constraints_diag.get("power_error", 0.0)
                    power_error_value = _to_float(power_error)
                    if power_error_value > float(constraint_power_soft):
                        soft_reasons.append("constraint_power_soft")
        diagnostics["failsafe"]["soft_reasons"] = soft_reasons

        soft_blend_applied = False
        if isinstance(failsafe_cfg, dict):
            alpha_blend = _resolve_soft_blend_alpha(failsafe_cfg=failsafe_cfg, soft_reasons=soft_reasons)
            if alpha_blend is not None:
                baseline_state = _resolve_soft_blend_baseline(
                    mode=str(failsafe_cfg.get("alpha_blend_baseline", "pre_event")),
                    state_prev=state,
                    state_b=state_b,
                    state_c=state_c,
                )
                state_d = _blend_states(learned=state_d, baseline=baseline_state, alpha=alpha_blend)
                e_kin_d, e_pot_d, e_total_d = self.system.energy(state_d)
                diagnostics["energy"]["E_kin"] = e_kin_d
                diagnostics["energy"]["E_pot"] = e_pot_d
                diagnostics["energy"]["E_total"] = e_total_d
                diagnostics["energy"]["dE_D"] = e_total_d - e_total_c
                diagnostics["failsafe"]["alpha_blend"] = torch.tensor(
                    alpha_blend, device=e_total_d.device, dtype=e_total_d.dtype
                )
                diagnostics["failsafe"]["soft_blended"] = True
                soft_blend_applied = True

                _ensure_minimum_diagnostics(
                    diagnostics=diagnostics,
                    reference=e_total_d,
                    dt=dt,
                    d_e_c=d_e_c,
                    d_e_d=diagnostics["energy"]["dE_D"],
                )
                _update_ledger_consistency(
                    diagnostics=diagnostics,
                    dt=dt,
                    d_e_b=d_e_b,
                    d_e_c=d_e_c,
                    d_e_d=diagnostics["energy"]["dE_D"],
                    reference=e_total_d,
                )

        if not soft_blend_applied:
            diagnostics["failsafe"]["soft_blended"] = False

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
            diagnostics["failsafe"]["alpha_blend"] = torch.tensor(0.0, device=e_total_d.device, dtype=e_total_d.dtype)
            return state, diagnostics

        return state_d, diagnostics


def _as_tensor(value: Any, *, reference: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=reference.device, dtype=reference.dtype)
        if tensor.numel() == 0:
            return torch.tensor(float(default), device=reference.device, dtype=reference.dtype)
        return tensor.reshape(-1)[0]
    if isinstance(value, (int, float)):
        return torch.tensor(float(value), device=reference.device, dtype=reference.dtype)
    return torch.tensor(float(default), device=reference.device, dtype=reference.dtype)


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().reshape(-1)[0].cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _ensure_minimum_diagnostics(
    *,
    diagnostics: Diagnostics,
    reference: torch.Tensor,
    dt: float,
    d_e_c: torch.Tensor,
    d_e_d: torch.Tensor,
) -> None:
    zero = torch.tensor(0.0, device=reference.device, dtype=reference.dtype)

    power = diagnostics.get("power")
    if not isinstance(power, dict):
        power = {}
        diagnostics["power"] = power
    power.setdefault("P_in", zero)
    power.setdefault("P_diss", zero)
    power.setdefault("P_diss_signed", power.get("P_diss", zero))

    constraints_diag = diagnostics.get("constraints")
    if not isinstance(constraints_diag, dict):
        constraints_diag = {}
        diagnostics["constraints"] = constraints_diag
    constraints_diag.setdefault("residual_pos", zero)
    constraints_diag.setdefault("residual_vel", zero)
    constraints_diag.setdefault("power_error", torch.abs(d_e_c) / torch.tensor(max(dt, 1e-12), device=zero.device, dtype=zero.dtype))

    event = diagnostics.get("event")
    if not isinstance(event, dict):
        event = {}
        diagnostics["event"] = event
    event.setdefault("W_impulse", zero)
    event.setdefault("W_position_correction", zero)
    event.setdefault("dE_kin", zero)
    event.setdefault("dE_pot", zero)
    event.setdefault("dE_event", d_e_d)

    contacts = diagnostics.get("contacts")
    if not isinstance(contacts, dict):
        contacts = {}
        diagnostics["contacts"] = contacts
    contacts.setdefault("penetration_max", zero)
    contacts.setdefault("complementarity_residual_max", zero)
    contacts.setdefault("items", [])

    solver = diagnostics.get("solver")
    if not isinstance(solver, dict):
        solver = {}
        diagnostics["solver"] = solver
    solver.setdefault("iters", zero)
    solver.setdefault("residual_max", zero)
    solver.setdefault("status", "na")

    failsafe = diagnostics.get("failsafe")
    if not isinstance(failsafe, dict):
        failsafe = {}
        diagnostics["failsafe"] = failsafe
    failsafe.setdefault("triggered", False)
    failsafe.setdefault("reason", "")
    failsafe.setdefault("soft_reasons", [])
    failsafe.setdefault("alpha_blend", torch.tensor(1.0, device=zero.device, dtype=zero.dtype))
    failsafe.setdefault("soft_blended", False)


def _update_ledger_consistency(
    *,
    diagnostics: Diagnostics,
    dt: float,
    d_e_b: torch.Tensor,
    d_e_c: torch.Tensor,
    d_e_d: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    power = diagnostics.get("power", {})
    if not isinstance(power, dict):
        power = {}
        diagnostics["power"] = power

    p_in = _as_tensor(power.get("P_in"), reference=reference, default=0.0)
    p_diss_signed = _as_tensor(power.get("P_diss_signed", power.get("P_diss")), reference=reference, default=0.0)

    d_e_b_expected = (p_in - p_diss_signed) * torch.tensor(dt, device=reference.device, dtype=reference.dtype)
    b_balance_error = d_e_b - d_e_b_expected

    constraints_diag = diagnostics.get("constraints", {})
    if not isinstance(constraints_diag, dict):
        constraints_diag = {}
        diagnostics["constraints"] = constraints_diag
    power_error = _as_tensor(constraints_diag.get("power_error"), reference=reference, default=0.0)

    c_power_from_energy = d_e_c / torch.tensor(max(dt, 1e-12), device=reference.device, dtype=reference.dtype)
    c_energy_proxy_error = torch.abs(c_power_from_energy) - power_error

    event = diagnostics.get("event", {})
    if not isinstance(event, dict):
        event = {}
        diagnostics["event"] = event
    d_e_event = _as_tensor(event.get("dE_event"), reference=reference, default=0.0)
    d_energy_mismatch = d_e_d - d_e_event

    diagnostics["ledger"] = {
        "B_balance_error": b_balance_error,
        "B_balance_error_abs": torch.abs(b_balance_error),
        "C_power_from_energy": c_power_from_energy,
        "C_power_error_from_energy_abs": torch.abs(c_energy_proxy_error),
        "D_energy_mismatch": d_energy_mismatch,
        "D_energy_mismatch_abs": torch.abs(d_energy_mismatch),
    }


def _resolve_soft_blend_alpha(*, failsafe_cfg: dict[str, Any], soft_reasons: list[str]) -> float | None:
    if not soft_reasons:
        return None
    alpha_cfg = failsafe_cfg.get("alpha_blend_soft")
    if alpha_cfg is None:
        return None
    if isinstance(alpha_cfg, bool):
        if not alpha_cfg:
            return None
        return 0.5
    if isinstance(alpha_cfg, (int, float)):
        alpha = float(alpha_cfg)
        return min(1.0, max(0.0, alpha))
    return None


def _resolve_soft_blend_baseline(*, mode: str, state_prev: State, state_b: State, state_c: State) -> State:
    if mode == "prev":
        return State(q=state_prev.q, v=state_prev.v, t=state_c.t)
    if mode == "pre_constraints":
        return state_b
    return state_c


def _blend_states(*, learned: State, baseline: State, alpha: float) -> State:
    a = torch.tensor(alpha, device=learned.q.device, dtype=learned.q.dtype)
    q = a * learned.q + (1.0 - a) * baseline.q
    v = a * learned.v + (1.0 - a) * baseline.v
    return State(q=q, v=v, t=learned.t)


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
