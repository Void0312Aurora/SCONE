import torch

from scone.engine import Engine
from scone.learned.mu import ScalarMu
from scone.layers.constraints import NoOpConstraintLayer
from scone.layers.dissipation import LinearDampingLayer
from scone.layers.events import DiskContactPGSEventLayer
from scone.layers.symplectic import SymplecticEulerSeparable
from scone.state import State
from scone.systems.rigid2d import Disk2D


def test_mu_identification_through_pgs_multi_contact_open_loop_bptt() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32

    mass = 1.0
    radius = 0.5
    inertia = 0.5 * mass * radius * radius
    gravity = 9.81
    ground_height = 0.0

    dt = 0.01
    rollout_steps = 3

    def build_engine(mu: float | torch.Tensor) -> Engine:
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

    def initial_state(omega_top: float) -> State:
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
        v0[2, 2] = float(omega_top)
        return State(q=q0, v=v0, t=0.0)

    mu_true = 0.6
    states0 = [initial_state(omega_top=3.0), initial_state(omega_top=-6.0)]

    targets: list[list[torch.Tensor]] = []
    with torch.no_grad():
        engine_true = build_engine(mu_true)
        for s0 in states0:
            state = s0
            traj: list[torch.Tensor] = []
            for _ in range(rollout_steps):
                state, _ = engine_true.step(state=state, dt=dt, context={"failsafe": {}})
                traj.append(state.v.detach())
            targets.append(traj)

    def rollout_loss(mu: float | torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=device, dtype=dtype)
        for s0, traj in zip(states0, targets, strict=True):
            engine = build_engine(mu)
            state = s0
            for step_idx in range(rollout_steps):
                state, _ = engine.step(state=state, dt=dt, context={"failsafe": {}})
                v_ref = traj[step_idx]
                # Fit top disk tangential motion (vx, omega).
                v_pred = state.v[2, [0, 2]]
                v_tgt = v_ref[2, [0, 2]]
                loss = loss + torch.mean((v_pred - v_tgt) ** 2)
        return loss / float(len(states0) * rollout_steps)

    with torch.no_grad():
        baseline_loss = rollout_loss(0.1)

    model = ScalarMu(init=0.1, mu_max=1.0, device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=0.3)

    for _ in range(12):
        opt.zero_grad(set_to_none=True)
        loss = rollout_loss(model())
        loss.backward()
        opt.step()

    mu_fit = float(model().detach().cpu().item())
    with torch.no_grad():
        fit_loss = rollout_loss(mu_fit)

    assert abs(mu_fit - mu_true) < abs(0.1 - mu_true)
    assert float(fit_loss.detach().cpu().item()) < 0.8 * float(baseline_loss.detach().cpu().item())
