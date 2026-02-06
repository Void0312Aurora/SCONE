import torch

from scone.engine import Engine
from scone.learned.mu import ScalarMu
from scone.layers.constraints import NoOpConstraintLayer
from scone.layers.dissipation import LinearDampingLayer
from scone.layers.events import DiskGroundContactEventLayer
from scone.layers.symplectic import SymplecticEulerSeparable
from scone.state import State
from scone.systems.rigid2d import Disk2D


def test_mu_rollout_identification_improves_multistep_loss() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64

    mass = 1.0
    radius = 0.5
    inertia = 0.5 * mass * radius * radius
    gravity = 9.81
    ground_height = 0.0
    dt = 0.01
    steps = 15
    n_traj = 16

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
            events=DiskGroundContactEventLayer(
                mass=mass,
                inertia=inertia,
                radius=radius,
                gravity=gravity,
                friction_mu=mu,
                restitution=0.0,
                ground_height=ground_height,
                contact_slop=1e-3,
                impact_velocity_min=0.2,
                sleep=None,
            ),
        )

    x = (torch.rand((n_traj,), device=device, dtype=dtype) - 0.5) * 2.0
    q0 = torch.stack([x, torch.full_like(x, radius), torch.zeros_like(x)], dim=-1)
    vx = torch.rand((n_traj,), device=device, dtype=dtype) * 1.0 + 3.0  # [3,4]
    v0 = torch.stack([vx, torch.zeros_like(vx), torch.zeros_like(vx)], dim=-1)
    state0 = State(q=q0, v=v0, t=0.0)

    mu_true = 0.6
    engine_true = build_engine(mu_true)
    state = state0
    target_v: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(steps):
            state, _ = engine_true.step(state=state, dt=dt, context={"failsafe": {}})
            target_v.append(state.v.detach())
    target_v_t = torch.stack(target_v, dim=0)  # (T,N,3)

    def rollout_loss(engine: Engine) -> torch.Tensor:
        state = state0
        loss = torch.tensor(0.0, device=device, dtype=dtype)
        for step_idx in range(steps):
            state, _ = engine.step(state=state, dt=dt, context={"failsafe": {}})
            v_pred = state.v[:, [0, 2]]
            v_ref = target_v_t[step_idx, :, [0, 2]]
            loss = loss + torch.mean((v_pred - v_ref) ** 2)
        return loss / float(steps)

    with torch.no_grad():
        baseline_loss = rollout_loss(build_engine(0.1))

    model = ScalarMu(init=0.1, mu_max=1.0, device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=0.3)

    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        mu = model()
        loss = rollout_loss(build_engine(mu))
        loss.backward()
        opt.step()

    mu_fit = float(model().detach().cpu().item())
    with torch.no_grad():
        fit_loss = rollout_loss(build_engine(mu_fit))

    assert abs(mu_fit - mu_true) < 5e-2
    assert float(fit_loss.detach().cpu().item()) < 0.1 * float(baseline_loss.detach().cpu().item())

