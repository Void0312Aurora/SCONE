from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from scone.engine import Engine
from scone.learned.mu import ScalarMu
from scone.layers.constraints import NoOpConstraintLayer
from scone.layers.dissipation import LinearDampingLayer
from scone.layers.events import DiskGroundContactEventLayer
from scone.layers.symplectic import SymplecticEulerSeparable
from scone.state import State
from scone.systems.rigid2d import Disk2D
from scone.utils.determinism import set_determinism


@dataclass(frozen=True)
class RolloutBatch:
    state0: State
    target_q: torch.Tensor  # (T, N, 3)
    target_v: torch.Tensor  # (T, N, 3)
    dt: float


def _build_engine(
    *,
    device: torch.device,
    dtype: torch.dtype,
    mass: float,
    radius: float,
    gravity: float,
    ground_height: float,
    mu: float | torch.Tensor,
    impact_velocity_min: float,
    contact_slop: float,
) -> Engine:
    inertia = 0.5 * mass * radius * radius
    system = Disk2D(
        mass=mass,
        radius=radius,
        gravity=gravity,
        ground_height=ground_height,
        inertia=inertia,
        device=device,
        dtype=dtype,
    )
    core = SymplecticEulerSeparable(system=system)
    dissipation = LinearDampingLayer(damping=0.0)
    constraints = NoOpConstraintLayer()
    events = DiskGroundContactEventLayer(
        mass=mass,
        inertia=inertia,
        radius=radius,
        gravity=gravity,
        friction_mu=mu,
        restitution=0.0,
        ground_height=ground_height,
        contact_slop=contact_slop,
        impact_velocity_min=impact_velocity_min,
        sleep=None,
    )
    return Engine(system=system, core=core, dissipation=dissipation, constraints=constraints, events=events)


def _generate_initial_state(
    *,
    device: torch.device,
    dtype: torch.dtype,
    n: int,
    radius: float,
    vmin: float,
    vmax: float,
) -> State:
    x = (torch.rand((n,), device=device, dtype=dtype) - 0.5) * 2.0
    y = torch.full((n,), float(radius), device=device, dtype=dtype)
    theta = torch.zeros((n,), device=device, dtype=dtype)
    q0 = torch.stack([x, y, theta], dim=-1)

    vx = torch.rand((n,), device=device, dtype=dtype) * (float(vmax) - float(vmin)) + float(vmin)
    vy = torch.zeros((n,), device=device, dtype=dtype)
    omega = torch.zeros((n,), device=device, dtype=dtype)
    v0 = torch.stack([vx, vy, omega], dim=-1)
    return State(q=q0, v=v0, t=0.0)


def _make_rollout_batch(
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    n_traj: int,
    steps: int,
    dt: float,
    mu_true: float,
    mass: float,
    radius: float,
    gravity: float,
    ground_height: float,
    vx_range: tuple[float, float],
    contact_slop: float,
    impact_velocity_min: float,
) -> RolloutBatch:
    set_determinism(seed=seed, deterministic=True)
    state0 = _generate_initial_state(
        device=device,
        dtype=dtype,
        n=n_traj,
        radius=radius,
        vmin=vx_range[0],
        vmax=vx_range[1],
    )
    engine_true = _build_engine(
        device=device,
        dtype=dtype,
        mass=mass,
        radius=radius,
        gravity=gravity,
        ground_height=ground_height,
        mu=float(mu_true),
        contact_slop=contact_slop,
        impact_velocity_min=impact_velocity_min,
    )

    qs: list[torch.Tensor] = []
    vs: list[torch.Tensor] = []
    state = state0
    context = {"failsafe": {}}
    with torch.no_grad():
        for _ in range(steps):
            state, _ = engine_true.step(state=state, dt=dt, context=context)
            qs.append(state.q.detach())
            vs.append(state.v.detach())

    target_q = torch.stack(qs, dim=0)
    target_v = torch.stack(vs, dim=0)
    return RolloutBatch(state0=state0, target_q=target_q, target_v=target_v, dt=dt)


def _rollout_loss(
    *,
    batch: RolloutBatch,
    engine: Engine,
    steps: int,
    w_q: float,
    w_v: float,
) -> torch.Tensor:
    state = batch.state0
    context = {"failsafe": {}}
    loss = torch.tensor(0.0, device=state.q.device, dtype=state.q.dtype)
    for step_index in range(steps):
        state, _ = engine.step(state=state, dt=batch.dt, context=context)
        q_target = batch.target_q[step_index]
        v_target = batch.target_v[step_index]

        # Focus on (x, theta) / (vx, omega): these are the components controlled by friction.
        q_pred = state.q[:, [0, 2]]
        q_ref = q_target[:, [0, 2]]
        v_pred = state.v[:, [0, 2]]
        v_ref = v_target[:, [0, 2]]
        loss = loss + w_q * torch.mean((q_pred - q_ref) ** 2) + w_v * torch.mean((v_pred - v_ref) ** 2)

    return loss / float(max(1, steps))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])

    parser.add_argument("--n-traj", type=int, default=64)
    parser.add_argument("--val-n-traj", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=25)
    parser.add_argument("--dt", type=float, default=0.01)

    parser.add_argument("--mu-true", type=float, default=0.6)
    parser.add_argument("--mu-init", type=float, default=0.1)
    parser.add_argument("--mu-max", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--train-iters", type=int, default=120)

    parser.add_argument("--vx-min", type=float, default=5.0)
    parser.add_argument("--vx-max", type=float, default=7.0)
    parser.add_argument("--val-vx-min", type=float, default=2.0)
    parser.add_argument("--val-vx-max", type=float, default=4.0)

    parser.add_argument("--w-q", type=float, default=0.1)
    parser.add_argument("--w-v", type=float, default=1.0)

    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--ground-height", type=float, default=0.0)
    parser.add_argument("--contact-slop", type=float, default=1e-3)
    parser.add_argument("--impact-velocity-min", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    batch = _make_rollout_batch(
        device=device,
        dtype=dtype,
        seed=int(args.seed),
        n_traj=int(args.n_traj),
        steps=int(args.rollout_steps),
        dt=float(args.dt),
        mu_true=float(args.mu_true),
        mass=float(args.mass),
        radius=float(args.radius),
        gravity=float(args.gravity),
        ground_height=float(args.ground_height),
        vx_range=(float(args.vx_min), float(args.vx_max)),
        contact_slop=float(args.contact_slop),
        impact_velocity_min=float(args.impact_velocity_min),
    )
    batch_val = _make_rollout_batch(
        device=device,
        dtype=dtype,
        seed=int(args.seed) + 1,
        n_traj=int(args.val_n_traj),
        steps=int(args.rollout_steps),
        dt=float(args.dt),
        mu_true=float(args.mu_true),
        mass=float(args.mass),
        radius=float(args.radius),
        gravity=float(args.gravity),
        ground_height=float(args.ground_height),
        vx_range=(float(args.val_vx_min), float(args.val_vx_max)),
        contact_slop=float(args.contact_slop),
        impact_velocity_min=float(args.impact_velocity_min),
    )

    engine_baseline = _build_engine(
        device=device,
        dtype=dtype,
        mass=float(args.mass),
        radius=float(args.radius),
        gravity=float(args.gravity),
        ground_height=float(args.ground_height),
        mu=float(args.mu_init),
        contact_slop=float(args.contact_slop),
        impact_velocity_min=float(args.impact_velocity_min),
    )
    with torch.no_grad():
        baseline_loss = _rollout_loss(
            batch=batch,
            engine=engine_baseline,
            steps=int(args.rollout_steps),
            w_q=float(args.w_q),
            w_v=float(args.w_v),
        )
        baseline_loss_val = _rollout_loss(
            batch=batch_val,
            engine=engine_baseline,
            steps=int(args.rollout_steps),
            w_q=float(args.w_q),
            w_v=float(args.w_v),
        )

    model = ScalarMu(init=float(args.mu_init), mu_max=float(args.mu_max), device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    for it in range(int(args.train_iters)):
        opt.zero_grad(set_to_none=True)
        mu = model()
        engine_pred = _build_engine(
            device=device,
            dtype=dtype,
            mass=float(args.mass),
            radius=float(args.radius),
            gravity=float(args.gravity),
            ground_height=float(args.ground_height),
            mu=mu,
            contact_slop=float(args.contact_slop),
            impact_velocity_min=float(args.impact_velocity_min),
        )
        loss = _rollout_loss(
            batch=batch,
            engine=engine_pred,
            steps=int(args.rollout_steps),
            w_q=float(args.w_q),
            w_v=float(args.w_v),
        )
        loss.backward()
        opt.step()

        if it % 20 == 0 or it == int(args.train_iters) - 1:
            mu_value = float(model().detach().cpu().item())
            print(f"iter={it:04d} loss={float(loss.detach()):.6e} mu={mu_value:.6f}")

    with torch.no_grad():
        mu_fit = float(model().detach().cpu().item())
        engine_fit = _build_engine(
            device=device,
            dtype=dtype,
            mass=float(args.mass),
            radius=float(args.radius),
            gravity=float(args.gravity),
            ground_height=float(args.ground_height),
            mu=float(mu_fit),
            contact_slop=float(args.contact_slop),
            impact_velocity_min=float(args.impact_velocity_min),
        )
        fit_loss = _rollout_loss(
            batch=batch,
            engine=engine_fit,
            steps=int(args.rollout_steps),
            w_q=float(args.w_q),
            w_v=float(args.w_v),
        )
        fit_loss_val = _rollout_loss(
            batch=batch_val,
            engine=engine_fit,
            steps=int(args.rollout_steps),
            w_q=float(args.w_q),
            w_v=float(args.w_v),
        )

    improvement = float((baseline_loss / (fit_loss + 1e-12)).detach().cpu().item())
    improvement_val = float((baseline_loss_val / (fit_loss_val + 1e-12)).detach().cpu().item())
    print(
        "done:",
        f"mu_true={float(args.mu_true):.6f}",
        f"mu_init={float(args.mu_init):.6f}",
        f"mu_fit={mu_fit:.6f}",
        f"baseline_loss={float(baseline_loss.detach()):.6e}",
        f"fit_loss={float(fit_loss.detach()):.6e}",
        f"improvement_x={improvement:.2f}",
        f"baseline_loss_val={float(baseline_loss_val.detach()):.6e}",
        f"fit_loss_val={float(fit_loss_val.detach()):.6e}",
        f"improvement_val_x={improvement_val:.2f}",
        sep=" ",
    )


if __name__ == "__main__":
    main()
