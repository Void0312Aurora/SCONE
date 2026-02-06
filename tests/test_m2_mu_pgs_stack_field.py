import torch

from scone.engine import Engine
from scone.layers.constraints import NoOpConstraintLayer
from scone.layers.dissipation import LinearDampingLayer
from scone.layers.events import DiskContactPGSEventLayer
from scone.layers.symplectic import SymplecticEulerSeparable
from scone.state import State
from scone.systems.rigid2d import Disk2D


def _build_engine(
    *,
    device: torch.device,
    dtype: torch.dtype,
    mu: float,
    mu_fn,
    body_material_ids: tuple[int, ...] | None = None,
    ground_material_id: int = 0,
) -> Engine:
    mass = 1.0
    radius = 0.5
    inertia = 0.5 * mass * radius * radius
    gravity = 9.81
    ground_height = 0.0
    system = Disk2D(
        mass=mass,
        radius=radius,
        gravity=gravity,
        ground_height=ground_height,
        inertia=inertia,
        device=device,
        dtype=dtype,
    )
    return Engine(
        system=system,
        core=SymplecticEulerSeparable(system=system),
        dissipation=LinearDampingLayer(damping=0.0),
        constraints=NoOpConstraintLayer(),
        events=DiskContactPGSEventLayer(
            mass=mass,
            inertia=inertia,
            radius=radius,
            gravity=gravity,
            friction_mu=mu,
            friction_mu_pair=mu,
            friction_mu_fn=mu_fn,
            body_material_ids=body_material_ids,
            ground_material_id=ground_material_id,
            restitution=0.0,
            ground_height=ground_height,
            contact_slop=1e-3,
            impact_velocity_min=0.2,
            pgs_iters=8,
            baumgarte_beta=0.2,
            residual_tol=1e-6,
            warm_start=False,
            sleep=None,
        ),
    )


def _initial_state(*, device: torch.device, dtype: torch.dtype) -> State:
    q0 = torch.tensor(
        [
            [-0.6, 0.5, 0.0],
            [0.6, 0.5, 0.0],
            [0.0, 1.3, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    v0 = torch.zeros((3, 3), device=device, dtype=dtype)
    v0[0, 0] = 2.5
    v0[1, 0] = -2.5
    v0[2, 0] = 0.2
    v0[2, 2] = 5.0
    return State(q=q0, v=v0, t=0.0)


def test_contact_mu_callback_matches_constant_mu_and_covers_pair_ground() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    state0 = _initial_state(device=device, dtype=dtype)
    dt = 0.01
    body_material_ids = (1, 2, 3)
    ground_material_id = 0

    const_mu = 0.4
    engine_const = _build_engine(
        device=device,
        dtype=dtype,
        mu=const_mu,
        mu_fn=None,
        body_material_ids=body_material_ids,
        ground_material_id=ground_material_id,
    )
    state_const, _ = engine_const.step(state=state0, dt=dt, context={"failsafe": {}})

    seen_is_pair: list[float] = []
    seen_material_pairs: set[tuple[int, int]] = set()

    def mu_fn(features: dict[str, torch.Tensor]) -> torch.Tensor:
        seen_is_pair.append(float(features["is_pair"].detach().cpu().item()))
        mi = int(features["material_i"].detach().cpu().item())
        mj = int(features["material_j"].detach().cpu().item())
        seen_material_pairs.add((mi, mj))
        return features["phi"].new_tensor(const_mu)

    engine_fn = _build_engine(
        device=device,
        dtype=dtype,
        mu=0.0,
        mu_fn=mu_fn,
        body_material_ids=body_material_ids,
        ground_material_id=ground_material_id,
    )
    state_fn, _ = engine_fn.step(state=state0, dt=dt, context={"failsafe": {}})

    assert torch.allclose(state_const.v, state_fn.v, atol=1e-6, rtol=1e-6)
    assert any(value < 0.5 for value in seen_is_pair)
    assert any(value > 0.5 for value in seen_is_pair)
    assert (1, 0) in seen_material_pairs
    assert (2, 0) in seen_material_pairs
    assert (1, 3) in seen_material_pairs
    assert (2, 3) in seen_material_pairs


def test_contact_mu_callback_changes_dynamics_when_mu_changes() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    state0 = _initial_state(device=device, dtype=dtype)
    dt = 0.01

    def mu_low(features: dict[str, torch.Tensor]) -> torch.Tensor:
        return features["phi"].new_tensor(0.05)

    def mu_high(features: dict[str, torch.Tensor]) -> torch.Tensor:
        return features["phi"].new_tensor(0.9)

    engine_low = _build_engine(device=device, dtype=dtype, mu=0.0, mu_fn=mu_low)
    engine_high = _build_engine(device=device, dtype=dtype, mu=0.0, mu_fn=mu_high)
    state_low, _ = engine_low.step(state=state0, dt=dt, context={"failsafe": {}})
    state_high, _ = engine_high.step(state=state0, dt=dt, context={"failsafe": {}})

    top_vx_diff = torch.abs(state_high.v[2, 0] - state_low.v[2, 0])
    top_omega_diff = torch.abs(state_high.v[2, 2] - state_low.v[2, 2])
    assert float((top_vx_diff + top_omega_diff).detach().cpu().item()) > 1e-4


def test_contact_mu_callback_material_features_change_dynamics() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    state0 = _initial_state(device=device, dtype=dtype)
    dt = 0.01

    def mu_from_material(features: dict[str, torch.Tensor]) -> torch.Tensor:
        material_i = features["material_i"]
        material_j = features["material_j"]
        return 0.05 + 0.1 * (material_i / 4.0) + 0.1 * (material_j / 4.0)

    engine_low = _build_engine(
        device=device,
        dtype=dtype,
        mu=0.0,
        mu_fn=mu_from_material,
        body_material_ids=(1, 1, 1),
        ground_material_id=0,
    )
    engine_high = _build_engine(
        device=device,
        dtype=dtype,
        mu=0.0,
        mu_fn=mu_from_material,
        body_material_ids=(3, 3, 3),
        ground_material_id=0,
    )
    state_low, _ = engine_low.step(state=state0, dt=dt, context={"failsafe": {}})
    state_high, _ = engine_high.step(state=state0, dt=dt, context={"failsafe": {}})

    top_vx_diff = torch.abs(state_high.v[2, 0] - state_low.v[2, 0])
    top_omega_diff = torch.abs(state_high.v[2, 2] - state_low.v[2, 2])
    assert float((top_vx_diff + top_omega_diff).detach().cpu().item()) > 1e-4
