from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import torch

from scone.engine import Engine
from scone.learned.mu import ContactMuMLP, ContactMuMaterialMLP
from scone.layers.constraints import NoOpConstraintLayer
from scone.layers.dissipation import LinearDampingLayer
from scone.layers.events import DiskContactPGSEventLayer
from scone.layers.symplectic import SymplecticEulerSeparable
from scone.state import State
from scone.systems.rigid2d import Disk2D
from scone.utils.determinism import set_determinism


ContactMuFn = Callable[[dict[str, torch.Tensor]], torch.Tensor]


@dataclass(frozen=True)
class StackRollout:
    q: torch.Tensor  # (T+1, 3, 3)
    v: torch.Tensor  # (T+1, 3, 3)
    dt: float
    body_material_ids: tuple[int, ...]
    ground_material_id: int


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    return tuple(int(part) for part in parts)


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    return tuple(float(part) for part in parts)


def _parse_pattern_csv(text: str) -> tuple[tuple[int, int, int], ...]:
    raw = str(text).strip()
    if not raw:
        return tuple()
    patterns: list[tuple[int, int, int]] = []
    for block in raw.split(";"):
        block = block.strip()
        if not block:
            continue
        parts = [part.strip() for part in block.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(f"Each material pattern must contain exactly 3 ids, got: {block!r}")
        patterns.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return tuple(patterns)


def _normalize_material_patterns(
    patterns: tuple[tuple[int, ...], ...] | tuple[tuple[int, int, int], ...] | None,
) -> tuple[tuple[int, int, int], ...]:
    if not patterns:
        return tuple()
    normalized: list[tuple[int, int, int]] = []
    for pattern in patterns:
        if len(pattern) != 3:
            raise ValueError(f"Each material pattern must have exactly 3 ids, got length={len(pattern)}")
        normalized.append((int(pattern[0]), int(pattern[1]), int(pattern[2])))
    return tuple(normalized)


def _material_bias_tensor(
    *,
    material_bias: tuple[float, ...] | None,
    max_material_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if max_material_id < 0:
        raise ValueError("max_material_id must be non-negative")
    size = int(max_material_id) + 1
    values = [0.0 for _ in range(size)]
    if material_bias is not None:
        for idx, val in enumerate(material_bias):
            if idx >= size:
                break
            values[idx] = float(val)
    return torch.tensor(values, device=device, dtype=dtype)


def _material_index(value: torch.Tensor, *, max_material_id: int) -> int:
    rounded = torch.round(value.detach()).to(dtype=torch.long)
    idx = int(rounded.reshape(-1)[0].cpu().item())
    return max(0, min(int(max_material_id), idx))


def _stack_initial_state(
    *,
    device: torch.device,
    dtype: torch.dtype,
    omega_top: float,
    vx_top: float,
    vx_bottom0: float,
    vx_bottom1: float,
) -> State:
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
    v0[0, 0] = float(vx_bottom0)
    v0[1, 0] = float(vx_bottom1)
    v0[2, 0] = float(vx_top)
    v0[2, 2] = float(omega_top)
    return State(q=q0, v=v0, t=0.0)


def make_true_mu_rule(
    *,
    device: torch.device,
    dtype: torch.dtype,
    mu_max: float,
    base_ground: float,
    base_pair: float,
    speed_gain: float,
    speed_scale: float,
    vn_gain: float,
    pen_gain: float,
    mu_min: float = 0.02,
    material_bias: tuple[float, ...] | None = None,
    max_material_id: int = 4,
) -> ContactMuFn:
    mu_max_t = torch.tensor(float(mu_max), device=device, dtype=dtype)
    mu_min_t = torch.tensor(float(mu_min), device=device, dtype=dtype)
    base_ground_t = torch.tensor(float(base_ground), device=device, dtype=dtype)
    base_pair_t = torch.tensor(float(base_pair), device=device, dtype=dtype)
    speed_gain_t = torch.tensor(float(speed_gain), device=device, dtype=dtype)
    speed_scale_t = torch.tensor(float(speed_scale), device=device, dtype=dtype)
    vn_gain_t = torch.tensor(float(vn_gain), device=device, dtype=dtype)
    pen_gain_t = torch.tensor(float(pen_gain), device=device, dtype=dtype)
    material_bias_t = _material_bias_tensor(
        material_bias=material_bias,
        max_material_id=int(max_material_id),
        device=device,
        dtype=dtype,
    )

    def _rule(features: dict[str, torch.Tensor]) -> torch.Tensor:
        phi = features["phi"].reshape(())
        vn = features["vn"].reshape(())
        vt = features["vt"].reshape(())
        is_pair = torch.clamp(features["is_pair"].reshape(()), min=0.0, max=1.0)
        default_zero = torch.tensor(0.0, device=device, dtype=dtype)
        material_i = features.get("material_i", default_zero).reshape(())
        material_j = features.get("material_j", default_zero).reshape(())

        base = base_ground_t * (1.0 - is_pair) + base_pair_t * is_pair
        speed = torch.sqrt(vt * vt + 0.25 * vn * vn)
        speed_term = speed_gain_t * torch.tanh(speed / speed_scale_t)
        normal_term = vn_gain_t * torch.tanh(torch.abs(vn) / speed_scale_t)
        penetration = torch.clamp(-phi, min=0.0)
        pen_term = pen_gain_t * torch.tanh(10.0 * penetration)

        material_i_idx = _material_index(material_i, max_material_id=int(max_material_id))
        material_j_idx = _material_index(material_j, max_material_id=int(max_material_id))
        material_term = 0.5 * (material_bias_t[material_i_idx] + material_bias_t[material_j_idx])

        mu = base + speed_term + normal_term + pen_term + material_term
        return torch.clamp(mu, min=mu_min_t, max=mu_max_t)

    return _rule


def _build_engine(
    *,
    device: torch.device,
    dtype: torch.dtype,
    mu_default: float | torch.Tensor,
    mu_fn: ContactMuFn | None,
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
    body_material_ids: tuple[int, ...] | None = None,
    ground_material_id: int = 0,
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
            friction_mu=mu_default,
            friction_mu_pair=mu_default,
            friction_mu_fn=mu_fn,
            body_material_ids=body_material_ids,
            ground_material_id=int(ground_material_id),
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
    mu_true_fn: ContactMuFn,
    omega_range: tuple[float, float],
    vx_range: tuple[float, float],
    vx_bottom_range: tuple[float, float],
    engine_params: dict[str, float | int | bool],
    body_material_pool: tuple[int, ...] | None = None,
    body_material_patterns: tuple[tuple[int, int, int], ...] | None = None,
    ground_material_id: int = 0,
) -> list[StackRollout]:
    set_determinism(seed=seed, deterministic=True)
    omegas = torch.rand((n,), device=device, dtype=dtype) * (omega_range[1] - omega_range[0]) + omega_range[0]
    signs = torch.where(torch.rand((n,), device=device, dtype=dtype) > 0.5, 1.0, -1.0)
    omegas = (omegas * signs).tolist()

    vxs = torch.rand((n,), device=device, dtype=dtype) * (vx_range[1] - vx_range[0]) + vx_range[0]
    vxs = vxs.tolist()

    vx_bottom0 = (
        torch.rand((n,), device=device, dtype=dtype) * (vx_bottom_range[1] - vx_bottom_range[0]) + vx_bottom_range[0]
    )
    vx_bottom1 = (
        torch.rand((n,), device=device, dtype=dtype) * (vx_bottom_range[1] - vx_bottom_range[0]) + vx_bottom_range[0]
    )
    signs0 = torch.where(torch.rand((n,), device=device, dtype=dtype) > 0.5, 1.0, -1.0)
    signs1 = torch.where(torch.rand((n,), device=device, dtype=dtype) > 0.5, 1.0, -1.0)
    vx_bottom0 = (vx_bottom0 * signs0).tolist()
    vx_bottom1 = (vx_bottom1 * signs1).tolist()

    patterns = _normalize_material_patterns(body_material_patterns)
    if patterns:
        patterns_tensor = torch.tensor(patterns, device=device, dtype=torch.long)
        pattern_indices = torch.randint(0, int(patterns_tensor.shape[0]), (n,), device=device)
        sample_materials = patterns_tensor[pattern_indices].cpu().tolist()
    else:
        material_pool = tuple(int(x) for x in body_material_pool) if body_material_pool else (0,)
        if len(material_pool) == 0:
            material_pool = (0,)
        pool_tensor = torch.tensor(material_pool, device=device, dtype=torch.long)
        sample_idx = torch.randint(0, len(material_pool), (n, 3), device=device)
        sample_materials = pool_tensor[sample_idx].cpu().tolist()

    dataset: list[StackRollout] = []
    for idx, (omega_top, vx_top, vx_b0, vx_b1) in enumerate(zip(omegas, vxs, vx_bottom0, vx_bottom1, strict=True)):
        body_material_ids = tuple(int(x) for x in sample_materials[idx])
        engine_true = _build_engine(
            device=device,
            dtype=dtype,
            mu_default=0.3,
            mu_fn=mu_true_fn,
            body_material_ids=body_material_ids,
            ground_material_id=int(ground_material_id),
            **engine_params,
        )

        state0 = _stack_initial_state(
            device=device,
            dtype=dtype,
            omega_top=float(omega_top),
            vx_top=float(vx_top),
            vx_bottom0=float(vx_b0),
            vx_bottom1=float(vx_b1),
        )
        state = state0
        context: dict[str, object] = {"failsafe": {}}
        qs: list[torch.Tensor] = [state0.q.detach()]
        vs: list[torch.Tensor] = [state0.v.detach()]
        with torch.no_grad():
            for _ in range(steps):
                state, _ = engine_true.step(state=state, dt=dt, context=context)
                qs.append(state.q.detach())
                vs.append(state.v.detach())
        dataset.append(
            StackRollout(
                q=torch.stack(qs),
                v=torch.stack(vs),
                dt=dt,
                body_material_ids=body_material_ids,
                ground_material_id=int(ground_material_id),
            )
        )
    return dataset


def _dataset_loss_open_loop(
    *,
    dataset: list[StackRollout],
    build_engine_fn: Callable[[float | torch.Tensor, ContactMuFn | None, tuple[int, ...], int], Engine],
    mu_default: float | torch.Tensor,
    mu_fn: ContactMuFn | None,
    steps: int,
    w_q: float,
    w_v: float,
    w_pen: float,
    w_comp: float,
    w_res: float,
) -> torch.Tensor:
    total = None
    for rollout in dataset:
        engine = build_engine_fn(mu_default, mu_fn, rollout.body_material_ids, rollout.ground_material_id)
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

            q_pred = state.q[:, [0, 2]]
            v_pred = state.v[:, [0, 2]]
            q_tgt = q_ref[:, [0, 2]]
            v_tgt = v_ref[:, [0, 2]]
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


def probe_mu_mae(
    *,
    device: torch.device,
    dtype: torch.dtype,
    mu_pred_fn: ContactMuFn,
    mu_true_fn: ContactMuFn,
    material_ids: tuple[int, ...] | None = None,
    ground_material_id: int = 0,
) -> torch.Tensor:
    phis = [-0.02, -0.005, 0.0]
    vns = [-2.0, -0.5, 0.0, 0.5]
    vts = [-3.0, -1.0, 0.0, 1.0, 3.0]
    kinds = [0.0, 1.0]
    mats = tuple(int(x) for x in material_ids) if material_ids else (0,)
    if len(mats) == 0:
        mats = (0,)

    errs: list[torch.Tensor] = []
    for is_pair in kinds:
        if is_pair < 0.5:
            material_pairs = [(material_i, int(ground_material_id)) for material_i in mats]
        else:
            material_pairs = [(material_i, material_j) for material_i in mats for material_j in mats]

        for phi in phis:
            for vn in vns:
                for vt in vts:
                    for material_i, material_j in material_pairs:
                        feat = {
                            "phi": torch.tensor(phi, device=device, dtype=dtype),
                            "vn": torch.tensor(vn, device=device, dtype=dtype),
                            "vt": torch.tensor(vt, device=device, dtype=dtype),
                            "is_pair": torch.tensor(is_pair, device=device, dtype=dtype),
                            "material_i": torch.tensor(float(material_i), device=device, dtype=dtype),
                            "material_j": torch.tensor(float(material_j), device=device, dtype=dtype),
                        }
                        mu_pred = mu_pred_fn(feat)
                        mu_true = mu_true_fn(feat)
                        errs.append(torch.abs(mu_pred - mu_true))
    return torch.stack(errs).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])

    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--n-train", type=int, default=8)
    parser.add_argument("--n-val", type=int, default=8)

    parser.add_argument("--mu-init", type=float, default=0.2)
    parser.add_argument("--mu-max", type=float, default=1.0)
    parser.add_argument("--mlp-hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--train-iters", type=int, default=100)

    parser.add_argument("--true-base-ground", type=float, default=0.55)
    parser.add_argument("--true-base-pair", type=float, default=0.22)
    parser.add_argument("--true-speed-gain", type=float, default=0.18)
    parser.add_argument("--true-speed-scale", type=float, default=2.5)
    parser.add_argument("--true-vn-gain", type=float, default=0.08)
    parser.add_argument("--true-pen-gain", type=float, default=0.06)
    parser.add_argument("--use-material-features", action="store_true", default=False)
    parser.add_argument("--max-material-id", type=int, default=4)
    parser.add_argument("--ground-material-id", type=int, default=0)
    parser.add_argument("--body-material-ids", type=str, default="1,2,3")
    parser.add_argument("--body-material-patterns", type=str, default="")
    parser.add_argument("--val-body-material-patterns", type=str, default="")
    parser.add_argument("--true-material-bias", type=str, default="")

    parser.add_argument("--omega-min", type=float, default=3.0)
    parser.add_argument("--omega-max", type=float, default=8.0)
    parser.add_argument("--val-omega-min", type=float, default=1.0)
    parser.add_argument("--val-omega-max", type=float, default=4.0)

    parser.add_argument("--vx-min", type=float, default=0.0)
    parser.add_argument("--vx-max", type=float, default=0.2)
    parser.add_argument("--val-vx-min", type=float, default=0.0)
    parser.add_argument("--val-vx-max", type=float, default=0.2)

    parser.add_argument("--vx-bottom-min", type=float, default=2.0)
    parser.add_argument("--vx-bottom-max", type=float, default=3.0)
    parser.add_argument("--val-vx-bottom-min", type=float, default=2.0)
    parser.add_argument("--val-vx-bottom-max", type=float, default=3.0)

    parser.add_argument("--w-q", type=float, default=0.1)
    parser.add_argument("--w-v", type=float, default=1.0)
    parser.add_argument("--w-pen", type=float, default=0.0)
    parser.add_argument("--w-comp", type=float, default=0.0)
    parser.add_argument("--w-res", type=float, default=0.0)

    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--ground-height", type=float, default=0.0)
    parser.add_argument("--contact-slop", type=float, default=1e-3)
    parser.add_argument("--impact-velocity-min", type=float, default=0.2)
    parser.add_argument("--pgs-iters", type=int, default=20)
    parser.add_argument("--pgs-relaxation", type=float, default=1.0)
    parser.add_argument("--baumgarte-beta", type=float, default=0.2)
    parser.add_argument("--residual-tol", type=float, default=1e-6)
    parser.add_argument("--warm-start", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    set_determinism(seed=int(args.seed), deterministic=True)

    max_material_id = int(args.max_material_id)
    if max_material_id < 0:
        raise ValueError("--max-material-id must be non-negative")

    use_material_features = bool(args.use_material_features)
    body_material_pool = _parse_csv_ints(args.body_material_ids)
    body_material_patterns_train = _parse_pattern_csv(args.body_material_patterns)
    body_material_patterns_val = _parse_pattern_csv(args.val_body_material_patterns)
    if not use_material_features:
        body_material_pool = (0,)
        body_material_patterns_train = tuple()
        body_material_patterns_val = tuple()
    if len(body_material_pool) == 0:
        body_material_pool = (0,)
    body_material_patterns_train = _normalize_material_patterns(body_material_patterns_train)
    body_material_patterns_val = _normalize_material_patterns(body_material_patterns_val)
    if not body_material_patterns_val:
        body_material_patterns_val = body_material_patterns_train

    ground_material_id = int(args.ground_material_id) if use_material_features else 0
    for material_id in (*body_material_pool, ground_material_id):
        if material_id < 0 or material_id > max_material_id:
            raise ValueError(
                f"material id {material_id} out of range [0, {max_material_id}] "
                f"(body_pool={body_material_pool}, ground={ground_material_id})"
            )
    for pattern in (*body_material_patterns_train, *body_material_patterns_val):
        for material_id in pattern:
            if material_id < 0 or material_id > max_material_id:
                raise ValueError(f"material id {material_id} in pattern out of range [0, {max_material_id}]")

    material_bias = _parse_csv_floats(args.true_material_bias)
    if not use_material_features:
        material_bias = tuple()

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

    mu_true_fn = make_true_mu_rule(
        device=device,
        dtype=dtype,
        mu_max=float(args.mu_max),
        base_ground=float(args.true_base_ground),
        base_pair=float(args.true_base_pair),
        speed_gain=float(args.true_speed_gain),
        speed_scale=float(args.true_speed_scale),
        vn_gain=float(args.true_vn_gain),
        pen_gain=float(args.true_pen_gain),
        material_bias=material_bias if len(material_bias) > 0 else None,
        max_material_id=max_material_id,
    )

    dataset_train = _make_dataset(
        device=device,
        dtype=dtype,
        seed=int(args.seed),
        n=int(args.n_train),
        steps=int(args.rollout_steps),
        dt=float(args.dt),
        mu_true_fn=mu_true_fn,
        omega_range=(float(args.omega_min), float(args.omega_max)),
        vx_range=(float(args.vx_min), float(args.vx_max)),
        vx_bottom_range=(float(args.vx_bottom_min), float(args.vx_bottom_max)),
        engine_params=engine_params,
        body_material_pool=body_material_pool,
        body_material_patterns=body_material_patterns_train,
        ground_material_id=ground_material_id,
    )
    dataset_val = _make_dataset(
        device=device,
        dtype=dtype,
        seed=int(args.seed) + 1,
        n=int(args.n_val),
        steps=int(args.rollout_steps),
        dt=float(args.dt),
        mu_true_fn=mu_true_fn,
        omega_range=(float(args.val_omega_min), float(args.val_omega_max)),
        vx_range=(float(args.val_vx_min), float(args.val_vx_max)),
        vx_bottom_range=(float(args.val_vx_bottom_min), float(args.val_vx_bottom_max)),
        engine_params=engine_params,
        body_material_pool=body_material_pool,
        body_material_patterns=body_material_patterns_val,
        ground_material_id=ground_material_id,
    )

    def build_engine(
        mu_default: float | torch.Tensor,
        mu_fn: ContactMuFn | None,
        rollout_body_material_ids: tuple[int, ...],
        rollout_ground_material_id: int,
    ) -> Engine:
        return _build_engine(
            device=device,
            dtype=dtype,
            mu_default=mu_default,
            mu_fn=mu_fn,
            body_material_ids=rollout_body_material_ids,
            ground_material_id=rollout_ground_material_id,
            **engine_params,
        )

    with torch.no_grad():
        baseline_loss = _dataset_loss_open_loop(
            dataset=dataset_train,
            build_engine_fn=build_engine,
            mu_default=float(args.mu_init),
            mu_fn=None,
            steps=int(args.rollout_steps),
            w_q=float(args.w_q),
            w_v=float(args.w_v),
            w_pen=float(args.w_pen),
            w_comp=float(args.w_comp),
            w_res=float(args.w_res),
        )
        baseline_loss_val = _dataset_loss_open_loop(
            dataset=dataset_val,
            build_engine_fn=build_engine,
            mu_default=float(args.mu_init),
            mu_fn=None,
            steps=int(args.rollout_steps),
            w_q=float(args.w_q),
            w_v=float(args.w_v),
            w_pen=float(args.w_pen),
            w_comp=float(args.w_comp),
            w_res=float(args.w_res),
        )

    if use_material_features:
        model = ContactMuMaterialMLP(
            init_mu=float(args.mu_init),
            mu_max=float(args.mu_max),
            hidden_dim=int(args.mlp_hidden),
            max_material_id=max_material_id,
            device=device,
            dtype=dtype,
        )
    else:
        model = ContactMuMLP(
            init_mu=float(args.mu_init),
            mu_max=float(args.mu_max),
            hidden_dim=int(args.mlp_hidden),
            device=device,
            dtype=dtype,
        )
    model_mu_fn: ContactMuFn = model.from_contact_features
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    for _ in range(int(args.train_iters)):
        opt.zero_grad(set_to_none=True)
        loss = _dataset_loss_open_loop(
            dataset=dataset_train,
            build_engine_fn=build_engine,
            mu_default=float(args.mu_init),
            mu_fn=model_mu_fn,
            steps=int(args.rollout_steps),
            w_q=float(args.w_q),
            w_v=float(args.w_v),
            w_pen=float(args.w_pen),
            w_comp=float(args.w_comp),
            w_res=float(args.w_res),
        )
        loss.backward()
        opt.step()

    with torch.no_grad():
        fit_loss = _dataset_loss_open_loop(
            dataset=dataset_train,
            build_engine_fn=build_engine,
            mu_default=float(args.mu_init),
            mu_fn=model_mu_fn,
            steps=int(args.rollout_steps),
            w_q=float(args.w_q),
            w_v=float(args.w_v),
            w_pen=float(args.w_pen),
            w_comp=float(args.w_comp),
            w_res=float(args.w_res),
        )
        fit_loss_val = _dataset_loss_open_loop(
            dataset=dataset_val,
            build_engine_fn=build_engine,
            mu_default=float(args.mu_init),
            mu_fn=model_mu_fn,
            steps=int(args.rollout_steps),
            w_q=float(args.w_q),
            w_v=float(args.w_v),
            w_pen=float(args.w_pen),
            w_comp=float(args.w_comp),
            w_res=float(args.w_res),
        )
        probe_mae = probe_mu_mae(
            device=device,
            dtype=dtype,
            mu_pred_fn=model_mu_fn,
            mu_true_fn=mu_true_fn,
            material_ids=tuple(sorted(set(int(x) for x in body_material_pool))),
            ground_material_id=ground_material_id,
        )

    eps = 1e-12
    b_train = float(baseline_loss.detach().cpu().item())
    f_train = float(fit_loss.detach().cpu().item())
    b_val = float(baseline_loss_val.detach().cpu().item())
    f_val = float(fit_loss_val.detach().cpu().item())
    ratio_train = float(f_train / (b_train + eps))
    ratio_val = float(f_val / (b_val + eps))
    probe_mae_v = float(probe_mae.detach().cpu().item())

    print(f"baseline_loss={b_train:.6e}")
    print(f"fit_loss={f_train:.6e}")
    print(f"baseline_loss_val={b_val:.6e}")
    print(f"fit_loss_val={f_val:.6e}")
    print(f"loss_ratio={ratio_train:.6f}")
    print(f"loss_ratio_val={ratio_val:.6f}")
    print(f"mu_probe_mae={probe_mae_v:.6f}")
    print(f"use_material_features={int(use_material_features)}")
    print(f"body_material_pool={','.join(str(int(x)) for x in body_material_pool)}")
    print(
        "body_material_patterns_train="
        + ";".join(",".join(str(int(x)) for x in pattern) for pattern in body_material_patterns_train)
    )
    print(
        "body_material_patterns_val="
        + ";".join(",".join(str(int(x)) for x in pattern) for pattern in body_material_patterns_val)
    )
    print(f"ground_material_id={int(ground_material_id)}")


if __name__ == "__main__":
    main()
