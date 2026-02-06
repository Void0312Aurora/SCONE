import torch

from scone.engine import Engine
from scone.layers.constraints import NoOpConstraintLayer
from scone.layers.dissipation import LinearDampingLayer
from scone.layers.events import DiskContactPGSEventLayer
from scone.layers.symplectic import SymplecticEulerSeparable
from scone.sleep import SleepConfig, SleepManager
from scone.state import State
from scone.systems.rigid2d import Disk2D


def _build_disk_stack_engine(
    *,
    warm_start: bool,
    sleep: SleepManager | None = None,
    mass: float = 1.0,
    radius: float = 0.5,
    pgs_iters: int = 30,
) -> Engine:
    device = torch.device("cpu")
    dtype = torch.float64

    inertia = 0.5 * mass * radius * radius
    gravity = 9.81

    system = Disk2D(
        mass=mass,
        radius=radius,
        gravity=gravity,
        ground_height=0.0,
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
            friction_mu=0.6,
            restitution=0.0,
            ground_height=0.0,
            contact_slop=1e-3,
            impact_velocity_min=0.2,
            pgs_iters=pgs_iters,
            baumgarte_beta=0.2,
            residual_tol=1e-6,
            warm_start=warm_start,
            sleep=sleep,
        ),
    )


def _stack_state(*, vx_top: float = 0.0, omega_top: float = 0.0) -> State:
    device = torch.device("cpu")
    dtype = torch.float64
    v = torch.zeros((3, 3), device=device, dtype=dtype)
    v[2, 0] = float(vx_top)
    v[2, 2] = float(omega_top)
    return State(
        q=torch.tensor(
            [
                [-0.6, 0.5, 0.0],
                [0.6, 0.5, 0.0],
                [0.0, 1.3, 0.0],
            ],
            device=device,
            dtype=dtype,
        ),
        v=v,
        t=0.0,
    )


def test_disk_stack_pgs_multi_contact_supports_top_disk() -> None:
    engine = _build_disk_stack_engine(warm_start=True, sleep=None)
    state = _stack_state()

    next_state, diag = engine.step(state=state, dt=0.01, context={})

    assert diag["failsafe"]["triggered"] is False
    assert "solver" in diag
    assert "iters" in diag["solver"]
    assert "residual_max" in diag["solver"]
    assert diag["solver"]["status"] in {"converged", "max_iter"}

    contacts = diag["contacts"]["items"]
    assert isinstance(contacts, list)
    assert len(contacts) == 4
    ids = {c["id"] for c in contacts}
    assert ids == {"disk0-ground", "disk1-ground", "disk0-disk2", "disk1-disk2"}

    for c in contacts:
        for key in ["id", "body_i", "body_j", "phi", "n", "lambda_n", "lambda_t", "mode", "mode_name", "phi_next"]:
            assert key in c
    assert "mode_counts" in diag["contacts"]

    # Bottom disks are supported by ground; top disk is supported by two contacts.
    assert abs(float(next_state.v[0, 1].item())) < 1e-6
    assert abs(float(next_state.v[1, 1].item())) < 1e-6
    assert abs(float(next_state.v[2, 1].item())) < 1e-6

    # Penetration stays within a small tolerance (on the order of g*dt^2).
    assert float(diag["contacts"]["penetration_max"].item()) < 2e-3


def test_disk_stack_warm_start_on_off_regression_gap_is_bounded() -> None:
    def rollout_metrics(*, warm_start: bool) -> tuple[float, float, bool, dict[str, object]]:
        engine = _build_disk_stack_engine(warm_start=warm_start, sleep=None)
        state = _stack_state(vx_top=1.2, omega_top=4.0)
        context: dict[str, object] = {"failsafe": {}}
        max_pen = 0.0
        max_res = 0.0
        failsafe_any = False
        for _ in range(80):
            state, diag = engine.step(state=state, dt=0.01, context=context)
            max_pen = max(max_pen, float(diag["contacts"]["penetration_max"]))
            max_res = max(max_res, float(diag["solver"]["residual_max"]))
            failsafe_any = failsafe_any or bool(diag["failsafe"]["triggered"])
        return max_pen, max_res, failsafe_any, context

    warm_pen, warm_res, warm_fail, warm_context = rollout_metrics(warm_start=True)
    cold_pen, cold_res, cold_fail, cold_context = rollout_metrics(warm_start=False)

    assert warm_fail is False
    assert cold_fail is False

    assert warm_pen < 0.02
    assert cold_pen < 0.02
    assert warm_res < 1e-4
    assert cold_res < 1e-4

    assert abs(warm_pen - cold_pen) < 5e-4
    assert abs(warm_res - cold_res) < 5e-5

    assert "pgs_cache" in warm_context
    assert "pgs_cache" not in cold_context


def test_disk_stack_sleep_wake_has_no_large_penetration_spike() -> None:
    sleep = SleepManager(SleepConfig(enabled=True, v_sleep=0.05, v_wake=0.1, steps_to_sleep=10, freeze_core=False))
    engine = _build_disk_stack_engine(warm_start=True, sleep=sleep)
    state = _stack_state()
    context: dict[str, object] = {"failsafe": {}}

    sleeping_counts: list[int] = []
    penetration_max: list[float] = []
    failsafe_any = False
    wake_step = 55
    cache_keys_pre_wake: set[str] = set()
    for step_idx in range(90):
        if step_idx == wake_step:
            v = state.v.clone()
            v[2, 0] = 1.0
            state = State(q=state.q, v=v, t=state.t)

        state, diag = engine.step(state=state, dt=0.01, context=context)
        sleeping_counts.append(int(diag.get("sleep", {}).get("sleeping_count", 0)))
        penetration_max.append(float(diag["contacts"]["penetration_max"]))
        failsafe_any = failsafe_any or bool(diag["failsafe"]["triggered"])
        if step_idx == wake_step - 1:
            cache = context.get("pgs_cache")
            if isinstance(cache, dict):
                cache_keys_pre_wake = set(str(k) for k in cache.keys())

    assert failsafe_any is False

    pre_wake_sleep_max = max(sleeping_counts[:wake_step])
    post_wake_sleep_min = min(sleeping_counts[wake_step:])
    assert pre_wake_sleep_max > 0
    assert post_wake_sleep_min < pre_wake_sleep_max

    assert max(penetration_max[wake_step:]) < 0.02

    cache_after = context.get("pgs_cache")
    assert isinstance(cache_after, dict)
    assert len(cache_after) > 0
    cache_keys_after = set(str(k) for k in cache_after.keys())
    assert "disk0-ground" in cache_keys_after
    assert "disk1-ground" in cache_keys_after
    assert cache_keys_pre_wake
    assert len(cache_keys_pre_wake.intersection(cache_keys_after)) > 0


def test_disk_stack_ood_stress_solver_residual_stays_bounded() -> None:
    scenarios = (
        {"name": "dt_2x", "dt": 0.02, "mass": 1.0, "radius": 0.5},
        {"name": "mass_2x", "dt": 0.01, "mass": 2.0, "radius": 0.5},
        {"name": "radius_08x", "dt": 0.01, "mass": 1.0, "radius": 0.4},
    )

    for scenario in scenarios:
        engine = _build_disk_stack_engine(
            warm_start=True,
            sleep=None,
            mass=float(scenario["mass"]),
            radius=float(scenario["radius"]),
        )
        state = _stack_state(vx_top=1.2, omega_top=4.0)
        context: dict[str, object] = {"failsafe": {}}

        max_pen = 0.0
        max_res = 0.0
        max_iter_steps = 0
        status_steps = 0
        failsafe_any = False
        for _ in range(120):
            state, diag = engine.step(state=state, dt=float(scenario["dt"]), context=context)
            max_pen = max(max_pen, float(diag["contacts"]["penetration_max"]))
            max_res = max(max_res, float(diag["solver"]["residual_max"]))
            status = diag["solver"].get("status")
            if isinstance(status, str) and status:
                status_steps += 1
                if status == "max_iter":
                    max_iter_steps += 1
            failsafe_any = failsafe_any or bool(diag["failsafe"]["triggered"])

        max_iter_ratio = float(max_iter_steps / max(1, status_steps))
        assert failsafe_any is False, scenario["name"]
        assert max_pen < 0.03, scenario["name"]
        assert max_res < 5e-5, scenario["name"]
        assert max_iter_ratio < 0.05, scenario["name"]
