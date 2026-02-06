from dataclasses import dataclass
from typing import Any

import torch

from scone.diagnostics import Diagnostics
from scone.engine import Engine, EventLayer
from scone.layers.constraints import NoOpConstraintLayer
from scone.layers.dissipation import LinearDampingLayer
from scone.layers.events import NoOpEventLayer
from scone.layers.symplectic import SymplecticEulerSeparable
from scone.state import State
from scone.systems.toy import HarmonicOscillator1D


@dataclass(frozen=True)
class ResidualSpikeEvent(EventLayer):
    residual_value: float

    def resolve(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        del dt, context
        next_state = State(q=state.q + 1.0, v=state.v + 1.0, t=state.t)
        residual = torch.tensor(self.residual_value, device=state.q.device, dtype=state.q.dtype)
        diagnostics: Diagnostics = {
            "solver": {"residual_max": residual, "iters": torch.tensor(1.0, device=state.q.device, dtype=state.q.dtype)},
            "contacts": {
                "penetration_max": torch.tensor(0.0, device=state.q.device, dtype=state.q.dtype),
                "complementarity_residual_max": torch.tensor(0.0, device=state.q.device, dtype=state.q.dtype),
                "items": [],
            },
            "event": {
                "W_impulse": torch.tensor(0.0, device=state.q.device, dtype=state.q.dtype),
                "W_position_correction": torch.tensor(0.0, device=state.q.device, dtype=state.q.dtype),
                "dE_kin": torch.tensor(0.0, device=state.q.device, dtype=state.q.dtype),
                "dE_pot": torch.tensor(0.0, device=state.q.device, dtype=state.q.dtype),
                "dE_event": torch.tensor(0.0, device=state.q.device, dtype=state.q.dtype),
            },
        }
        return next_state, diagnostics


@dataclass(frozen=True)
class NonFiniteEvent(EventLayer):
    def resolve(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        del dt, context
        nan = torch.tensor(float("nan"), device=state.q.device, dtype=state.q.dtype)
        return State(q=state.q + nan, v=state.v, t=state.t), {}


def _build_engine(*, event: EventLayer) -> Engine:
    device = torch.device("cpu")
    dtype = torch.float64
    system = HarmonicOscillator1D(mass=1.0, stiffness=1.0, device=device, dtype=dtype)
    return Engine(
        system=system,
        core=SymplecticEulerSeparable(system=system),
        dissipation=LinearDampingLayer(damping=0.0),
        constraints=NoOpConstraintLayer(),
        events=event,
    )


def test_engine_populates_ledger_and_default_diagnostics() -> None:
    engine = _build_engine(event=NoOpEventLayer())
    state0 = State(
        q=torch.tensor([1.0], dtype=torch.float64),
        v=torch.tensor([0.0], dtype=torch.float64),
        t=0.0,
    )

    _, diag = engine.step(state=state0, dt=0.05, context={"failsafe": {}})

    assert "ledger" in diag
    assert "B_balance_error_abs" in diag["ledger"]
    assert "D_energy_mismatch_abs" in diag["ledger"]
    assert float(diag["ledger"]["B_balance_error_abs"].item()) < 1e-12
    assert float(diag["ledger"]["D_energy_mismatch_abs"].item()) < 1e-12

    assert "event" in diag
    assert "constraints" in diag
    assert "alpha_blend" in diag["failsafe"]
    assert float(diag["failsafe"]["alpha_blend"].item()) == 1.0
    assert bool(diag["failsafe"]["soft_blended"]) is False


def test_engine_soft_failsafe_alpha_blend_moves_towards_baseline() -> None:
    state0 = State(
        q=torch.tensor([0.3], dtype=torch.float64),
        v=torch.tensor([0.7], dtype=torch.float64),
        t=0.0,
    )
    event = ResidualSpikeEvent(residual_value=10.0)
    engine = _build_engine(event=event)

    state_no_blend, _ = engine.step(
        state=state0,
        dt=0.05,
        context={"failsafe": {"solver_residual_soft": 1.0}},
    )
    state_blend, diag_blend = engine.step(
        state=state0,
        dt=0.05,
        context={
            "failsafe": {
                "solver_residual_soft": 1.0,
                "alpha_blend_soft": 0.25,
                "alpha_blend_baseline": "prev",
            }
        },
    )

    dist_no_blend = float(torch.linalg.vector_norm(state_no_blend.q - state0.q).item())
    dist_blend = float(torch.linalg.vector_norm(state_blend.q - state0.q).item())
    assert dist_blend < dist_no_blend
    assert bool(diag_blend["failsafe"]["soft_blended"]) is True
    assert abs(float(diag_blend["failsafe"]["alpha_blend"].item()) - 0.25) < 1e-12


def test_engine_hard_failsafe_sets_alpha_zero_and_reverts_state() -> None:
    state0 = State(
        q=torch.tensor([0.3], dtype=torch.float64),
        v=torch.tensor([0.7], dtype=torch.float64),
        t=0.0,
    )
    engine = _build_engine(event=NonFiniteEvent())
    state_next, diag = engine.step(state=state0, dt=0.05, context={"failsafe": {}})

    assert torch.allclose(state_next.q, state0.q)
    assert torch.allclose(state_next.v, state0.v)
    assert bool(diag["failsafe"]["triggered"]) is True
    assert str(diag["failsafe"]["reason"]) == "non_finite"
    assert float(diag["failsafe"]["alpha_blend"].item()) == 0.0
