from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import torch

from scone.engine import Engine
from scone.learned.mu import ScalarMu
from scone.layers.constraints import NoOpConstraintLayer
from scone.layers.dissipation import LinearDampingLayer
from scone.layers.events import DiskContactPGSEventLayer
from scone.layers.symplectic import SymplecticEulerSeparable
from scone.state import State
from scone.systems.rigid2d import Disk2D
from scone.utils.determinism import set_determinism


@dataclass(frozen=True)
class StackRollout:
    q: torch.Tensor  # (T+1, 3, 3)
    v: torch.Tensor  # (T+1, 3, 3)
    dt: float


def _stack_initial_state(*, device: torch.device, dtype: torch.dtype, omega_top: float, vx_top: float) -> State:
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
    v0[2, 0] = float(vx_top)
    v0[2, 2] = float(omega_top)
    return State(q=q0, v=v0, t=0.0)


def _build_engine(
    *,
    device: torch.device,
    dtype: torch.dtype,
    mu: float | torch.Tensor,
    mass: float,
    radius: float,
    gravity: float,
    ground_height: float,
    contact_slop: float,
    impact_velocity_min: float,
    pgs_iters: int,
    pgs_relaxation: float,
    baumgarte_beta: float,
    residual_tol: float,
    warm_start: bool,
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
            contact_slop=contact_slop,
            impact_velocity_min=impact_velocity_min,
            pgs_iters=pgs_iters,
            pgs_relaxation=pgs_relaxation,
            baumgarte_beta=baumgarte_beta,
            residual_tol=residual_tol,
            warm_start=warm_start,
            sleep=None,
        ),
    )


def _make_dataset(
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    n: int,
    steps: int,
    dt: float,
    mu_true: float,
    omega_range: tuple[float, float],
    vx_range: tuple[float, float],
    engine_params: dict[str, float | int | bool],
) -> list[StackRollout]:
    set_determinism(seed=seed, deterministic=True)
    omegas = torch.rand((n,), device=device, dtype=dtype) * (omega_range[1] - omega_range[0]) + omega_range[0]
    signs = torch.where(torch.rand((n,), device=device, dtype=dtype) > 0.5, 1.0, -1.0)
    omegas = (omegas * signs).tolist()

    vxs = torch.rand((n,), device=device, dtype=dtype) * (vx_range[1] - vx_range[0]) + vx_range[0]
    vxs = vxs.tolist()

    engine_true = _build_engine(
        device=device,
        dtype=dtype,
        mu=float(mu_true),
        **engine_params,
    )

    dataset: list[StackRollout] = []
    for omega_top, vx_top in zip(omegas, vxs, strict=True):
        state0 = _stack_initial_state(device=device, dtype=dtype, omega_top=float(omega_top), vx_top=float(vx_top))
        state = state0
        context: dict[str, object] = {"failsafe": {}}
        qs: list[torch.Tensor] = [state0.q.detach()]
        vs: list[torch.Tensor] = [state0.v.detach()]
        with torch.no_grad():
            for _ in range(steps):
                state, _ = engine_true.step(state=state, dt=dt, context=context)
                qs.append(state.q.detach())
                vs.append(state.v.detach())
        dataset.append(StackRollout(q=torch.stack(qs), v=torch.stack(vs), dt=dt))
    return dataset


def _dataset_loss(
    *,
    dataset: list[StackRollout],
    build_engine_fn: Callable[[float | torch.Tensor], Engine],
    mu: float | torch.Tensor,
    steps: int,
    w_q: float,
    w_v: float,
    w_pen: float,
    w_comp: float,
    w_res: float,
) -> torch.Tensor:
    total = None
    for rollout in dataset:
        engine = build_engine_fn(mu)
        zero = torch.tensor(0.0, device=rollout.q.device, dtype=rollout.q.dtype)
        context: dict[str, object] = {"failsafe": {}}
        loss = torch.tensor(0.0, device=rollout.q.device, dtype=rollout.q.dtype)
        for step_idx in range(steps):
            q_in = rollout.q[step_idx].detach().clone()
            v_in = rollout.v[step_idx].detach().clone()
            state_in = State(q=q_in, v=v_in, t=float(step_idx) * float(rollout.dt))
            state_out, diag = engine.step(state=state_in, dt=rollout.dt, context=context)
            q_ref = rollout.q[step_idx + 1]
            v_ref = rollout.v[step_idx + 1]

            # Focus on top disk (index 2): x/theta and vx/omega are most sensitive to friction.
            q_pred = state_out.q[2, [0, 2]]
            v_pred = state_out.v[2, [0, 2]]
            q_tgt = q_ref[2, [0, 2]]
            v_tgt = v_ref[2, [0, 2]]

            loss = loss + w_q * torch.mean((q_pred - q_tgt) ** 2) + w_v * torch.mean((v_pred - v_tgt) ** 2)

            contacts = diag.get("contacts", {})
            if isinstance(contacts, dict):
                penetration = contacts.get("penetration_max", zero)
                complementarity = contacts.get("complementarity_residual_max", zero)
            else:
                penetration = zero
                complementarity = zero

            solver = diag.get("solver", {})
            residual = solver.get("residual_max", zero) if isinstance(solver, dict) else zero

            if not isinstance(penetration, torch.Tensor):
                penetration = zero
            if not isinstance(complementarity, torch.Tensor):
                complementarity = zero
            if not isinstance(residual, torch.Tensor):
                residual = zero

            loss = loss + w_pen * penetration + w_comp * complementarity + w_res * residual

        term = loss / float(max(1, steps))
        total = term if total is None else (total + term)
    assert total is not None
    return total / float(max(1, len(dataset)))


def _dataset_loss_open_loop(
    *,
    dataset: list[StackRollout],
    build_engine_fn: Callable[[float | torch.Tensor], Engine],
    mu: float | torch.Tensor,
    steps: int,
    w_q: float,
    w_v: float,
    w_pen: float,
    w_comp: float,
    w_res: float,
) -> torch.Tensor:
    total = None
    for rollout in dataset:
        engine = build_engine_fn(mu)
        zero = torch.tensor(0.0, device=rollout.q.device, dtype=rollout.q.dtype)
        q0 = rollout.q[0].detach().clone()
        v0 = rollout.v[0].detach().clone()
        state = State(q=q0, v=v0, t=0.0)
        context: dict[str, object] = {"failsafe": {}}
        loss = torch.tensor(0.0, device=rollout.q.device, dtype=rollout.q.dtype)
        for step_idx in range(steps):
            state, diag = engine.step(state=state, dt=rollout.dt, context=context)
            q_ref = rollout.q[step_idx + 1]
            v_ref = rollout.v[step_idx + 1]

            q_pred = state.q[2, [0, 2]]
            v_pred = state.v[2, [0, 2]]
            q_tgt = q_ref[2, [0, 2]]
            v_tgt = v_ref[2, [0, 2]]

            loss = loss + w_q * torch.mean((q_pred - q_tgt) ** 2) + w_v * torch.mean((v_pred - v_tgt) ** 2)

            contacts = diag.get("contacts", {})
            if isinstance(contacts, dict):
                penetration = contacts.get("penetration_max", zero)
                complementarity = contacts.get("complementarity_residual_max", zero)
            else:
                penetration = zero
                complementarity = zero

            solver = diag.get("solver", {})
            residual = solver.get("residual_max", zero) if isinstance(solver, dict) else zero

            if not isinstance(penetration, torch.Tensor):
                penetration = zero
            if not isinstance(complementarity, torch.Tensor):
                complementarity = zero
            if not isinstance(residual, torch.Tensor):
                residual = zero

            loss = loss + w_pen * penetration + w_comp * complementarity + w_res * residual

        term = loss / float(max(1, steps))
        total = term if total is None else (total + term)
    assert total is not None
    return total / float(max(1, len(dataset)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])

    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--n-train", type=int, default=8)
    parser.add_argument("--n-val", type=int, default=8)

    parser.add_argument("--mu-true", type=float, default=0.6)
    parser.add_argument("--mu-init", type=float, default=0.1)
    parser.add_argument("--mu-max", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--train-iters", type=int, default=120)

    parser.add_argument("--omega-min", type=float, default=3.0)
    parser.add_argument("--omega-max", type=float, default=8.0)
    parser.add_argument("--val-omega-min", type=float, default=1.0)
    parser.add_argument("--val-omega-max", type=float, default=4.0)

    parser.add_argument("--vx-min", type=float, default=0.0)
    parser.add_argument("--vx-max", type=float, default=0.0)
    parser.add_argument("--val-vx-min", type=float, default=0.0)
    parser.add_argument("--val-vx-max", type=float, default=0.0)

    parser.add_argument("--w-q", type=float, default=0.1)
    parser.add_argument("--w-v", type=float, default=1.0)
    parser.add_argument("--loss-mode", type=str, default="open_loop", choices=["open_loop", "teacher_forcing"])

    parser.add_argument("--w-pen", type=float, default=0.0, help="Penalty weight for contacts.penetration_max")
    parser.add_argument(
        "--w-comp",
        type=float,
        default=0.0,
        help="Penalty weight for contacts.complementarity_residual_max",
    )
    parser.add_argument("--w-res", type=float, default=0.0, help="Penalty weight for solver.residual_max")

    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--ground-height", type=float, default=0.0)
    parser.add_argument("--contact-slop", type=float, default=1e-3)
    parser.add_argument("--impact-velocity-min", type=float, default=0.2)

    parser.add_argument("--pgs-iters", type=int, default=30)
    parser.add_argument("--pgs-relaxation", type=float, default=1.0)
    parser.add_argument("--baumgarte-beta", type=float, default=0.2)
    parser.add_argument("--residual-tol", type=float, default=1e-6)
    parser.add_argument("--warm-start", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    engine_params: dict[str, float | int | bool] = {
        "mass": float(args.mass),
        "radius": float(args.radius),
        "gravity": float(args.gravity),
        "ground_height": float(args.ground_height),
        "contact_slop": float(args.contact_slop),
        "impact_velocity_min": float(args.impact_velocity_min),
        "pgs_iters": int(args.pgs_iters),
        "pgs_relaxation": float(args.pgs_relaxation),
        "baumgarte_beta": float(args.baumgarte_beta),
        "residual_tol": float(args.residual_tol),
        "warm_start": bool(args.warm_start),
    }

    dataset_train = _make_dataset(
        device=device,
        dtype=dtype,
        seed=int(args.seed),
        n=int(args.n_train),
        steps=int(args.rollout_steps),
        dt=float(args.dt),
        mu_true=float(args.mu_true),
        omega_range=(float(args.omega_min), float(args.omega_max)),
        vx_range=(float(args.vx_min), float(args.vx_max)),
        engine_params=engine_params,
    )
    dataset_val = _make_dataset(
        device=device,
        dtype=dtype,
        seed=int(args.seed) + 1,
        n=int(args.n_val),
        steps=int(args.rollout_steps),
        dt=float(args.dt),
        mu_true=float(args.mu_true),
        omega_range=(float(args.val_omega_min), float(args.val_omega_max)),
        vx_range=(float(args.val_vx_min), float(args.val_vx_max)),
        engine_params=engine_params,
    )

    def build_engine(mu: float | torch.Tensor) -> Engine:
        return _build_engine(device=device, dtype=dtype, mu=mu, **engine_params)

    loss_fn: Callable[..., torch.Tensor]
    if str(args.loss_mode) == "open_loop":
        loss_fn = _dataset_loss_open_loop
    else:
        loss_fn = _dataset_loss

    with torch.no_grad():
        baseline_loss = loss_fn(
            dataset=dataset_train,
            build_engine_fn=build_engine,
            mu=float(args.mu_init),
            steps=int(args.rollout_steps),
            w_q=float(args.w_q),
            w_v=float(args.w_v),
            w_pen=float(args.w_pen),
            w_comp=float(args.w_comp),
            w_res=float(args.w_res),
        )
        baseline_loss_val = loss_fn(
            dataset=dataset_val,
            build_engine_fn=build_engine,
            mu=float(args.mu_init),
            steps=int(args.rollout_steps),
            w_q=float(args.w_q),
            w_v=float(args.w_v),
            w_pen=float(args.w_pen),
            w_comp=float(args.w_comp),
            w_res=float(args.w_res),
        )

    model = ScalarMu(init=float(args.mu_init), mu_max=float(args.mu_max), device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    for it in range(int(args.train_iters)):
        opt.zero_grad(set_to_none=True)
        mu = model()
        loss = loss_fn(
            dataset=dataset_train,
            build_engine_fn=build_engine,
            mu=mu,
            steps=int(args.rollout_steps),
            w_q=float(args.w_q),
            w_v=float(args.w_v),
            w_pen=float(args.w_pen),
            w_comp=float(args.w_comp),
            w_res=float(args.w_res),
        )
        loss.backward()
        opt.step()

        if it % 20 == 0 or it == int(args.train_iters) - 1:
            mu_value = float(model().detach().cpu().item())
            print(f"iter={it:04d} loss={float(loss.detach()):.6e} mu={mu_value:.6f}")

    with torch.no_grad():
        mu_fit = float(model().detach().cpu().item())
        fit_loss = loss_fn(
            dataset=dataset_train,
            build_engine_fn=build_engine,
            mu=mu_fit,
            steps=int(args.rollout_steps),
            w_q=float(args.w_q),
            w_v=float(args.w_v),
            w_pen=float(args.w_pen),
            w_comp=float(args.w_comp),
            w_res=float(args.w_res),
        )
        fit_loss_val = loss_fn(
            dataset=dataset_val,
            build_engine_fn=build_engine,
            mu=mu_fit,
            steps=int(args.rollout_steps),
            w_q=float(args.w_q),
            w_v=float(args.w_v),
            w_pen=float(args.w_pen),
            w_comp=float(args.w_comp),
            w_res=float(args.w_res),
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
