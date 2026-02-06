from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from scone.engine import Engine
from scone.layers.constraints import LinearConstraintProjectionLayer, NoOpConstraintLayer
from scone.layers.dissipation import LinearDampingLayer
from scone.layers.events import (
    BouncingBallEventLayer,
    DiskContactPGSEventLayer,
    DiskGroundContactEventLayer,
    NoOpEventLayer,
)
from scone.layers.symplectic import SymplecticEulerSeparable
from scone.sleep import SleepConfig, SleepManager
from scone.state import State
from scone.systems.rigid2d import Disk2D
from scone.systems.toy import BouncingBall1D, HarmonicOscillator1D
from scone.utils.determinism import set_determinism


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def _device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    raise ValueError(f"Unsupported device: {name}")


def _tensor_from_param(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, (int, float)):
        return torch.tensor([float(value)], device=device, dtype=dtype)
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Empty initial value")
        if isinstance(value[0], (int, float)):
            return torch.tensor([[float(x) for x in value]], device=device, dtype=dtype)
        if isinstance(value[0], (list, tuple)):
            return torch.tensor([[float(x) for x in row] for row in value], device=device, dtype=dtype)
    raise TypeError(f"Unsupported initial value type: {type(value)}")


def _to_float(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().reshape(-1)[0].cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def _build_constraint_layer(config: dict[str, Any]) -> NoOpConstraintLayer | LinearConstraintProjectionLayer:
    kind = str(config.get("kind", "none")).lower()
    if kind == "linear":
        return LinearConstraintProjectionLayer(
            eps=float(config.get("eps", 1e-9)),
            enforce_position=bool(config.get("enforce_position", True)),
            enforce_velocity=bool(config.get("enforce_velocity", True)),
        )
    return NoOpConstraintLayer()


def _constraint_context(config: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    data = config.get("data", {})
    if not isinstance(data, dict):
        return {}
    out: dict[str, torch.Tensor] = {}
    for key in ("A_pos", "b_pos", "A_vel", "b_vel"):
        if key not in data:
            continue
        out[key] = torch.as_tensor(data[key], device=device, dtype=dtype)
    return out


@dataclass(frozen=True)
class Rollout:
    q: torch.Tensor
    v: torch.Tensor
    series: dict[str, list[float]]
    summary: dict[str, Any]


def _build_engine_and_state(
    *,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Engine, State, dict[str, Any], float, int, str]:
    demo = str(config["demo"])
    dt = float(config["dt"])
    steps = int(config["steps"])
    params: dict[str, Any] = dict(config.get("params", {}))
    failsafe_cfg: dict[str, Any] = dict(config.get("failsafe", {}))
    sleep_cfg: dict[str, Any] = dict(config.get("sleep", {}))
    constraint_cfg: dict[str, Any] = dict(config.get("constraints", {}))

    sleep_manager = None
    if sleep_cfg:
        sleep_manager = SleepManager(
            SleepConfig(
                enabled=bool(sleep_cfg.get("enabled", True)),
                v_sleep=float(sleep_cfg.get("v_sleep", 0.1)),
                v_wake=float(sleep_cfg.get("v_wake", 0.2)),
                steps_to_sleep=int(sleep_cfg.get("steps_to_sleep", 1)),
                freeze_core=bool(sleep_cfg.get("freeze_core", True)),
            )
        )

    if demo in {"harmonic_1d", "damped_oscillator_1d"}:
        system = HarmonicOscillator1D(
            mass=float(params["mass"]),
            stiffness=float(params["stiffness"]),
            device=device,
            dtype=dtype,
        )
        symplectic = SymplecticEulerSeparable(system=system)
        damping = float(params.get("damping", 0.0))
        dissipation = LinearDampingLayer(damping=damping)
        constraints = _build_constraint_layer(constraint_cfg)
        events = NoOpEventLayer()
    elif demo == "bouncing_ball_1d":
        system = BouncingBall1D(
            mass=float(params["mass"]),
            gravity=float(params["gravity"]),
            ground_height=float(params.get("ground_height", 0.0)),
            device=device,
            dtype=dtype,
        )
        symplectic = SymplecticEulerSeparable(system=system)
        dissipation = LinearDampingLayer(damping=0.0)
        constraints = _build_constraint_layer(constraint_cfg)
        events = BouncingBallEventLayer(
            mass=float(params["mass"]),
            gravity=float(params["gravity"]),
            restitution=float(params.get("restitution", 1.0)),
            ground_height=float(params.get("ground_height", 0.0)),
            contact_slop=float(params.get("q_slop", params.get("contact_slop", 1e-3))),
            impact_velocity_min=float(params.get("impact_velocity_min", sleep_cfg.get("v_sleep", 0.1))),
            sleep=sleep_manager,
        )
    elif demo == "disk_roll_2d":
        system = Disk2D(
            mass=float(params["mass"]),
            radius=float(params["radius"]),
            gravity=float(params.get("gravity", 9.81)),
            ground_height=float(params.get("ground_height", 0.0)),
            inertia=float(params["inertia"]) if "inertia" in params else None,
            device=device,
            dtype=dtype,
        )
        symplectic = SymplecticEulerSeparable(system=system)
        dissipation = LinearDampingLayer(damping=0.0)
        constraints = _build_constraint_layer(constraint_cfg)
        inertia = float(params.get("inertia", 0.5 * float(params["mass"]) * float(params["radius"]) ** 2))
        events = DiskGroundContactEventLayer(
            mass=float(params["mass"]),
            inertia=inertia,
            radius=float(params["radius"]),
            gravity=float(params.get("gravity", 9.81)),
            friction_mu=float(params.get("friction_mu", 0.5)),
            restitution=float(params.get("restitution", 0.0)),
            ground_height=float(params.get("ground_height", 0.0)),
            contact_slop=float(params.get("contact_slop", 1e-3)),
            impact_velocity_min=float(params.get("impact_velocity_min", 0.1)),
            sleep=sleep_manager,
        )
    elif demo == "disk_stack_2d":
        system = Disk2D(
            mass=float(params["mass"]),
            radius=float(params["radius"]),
            gravity=float(params.get("gravity", 9.81)),
            ground_height=float(params.get("ground_height", 0.0)),
            inertia=float(params["inertia"]) if "inertia" in params else None,
            device=device,
            dtype=dtype,
        )
        symplectic = SymplecticEulerSeparable(system=system)
        dissipation = LinearDampingLayer(damping=0.0)
        constraints = _build_constraint_layer(constraint_cfg)
        inertia = float(params.get("inertia", 0.5 * float(params["mass"]) * float(params["radius"]) ** 2))
        friction_mu_pair = params.get("friction_mu_pair")
        events = DiskContactPGSEventLayer(
            mass=float(params["mass"]),
            inertia=inertia,
            radius=float(params["radius"]),
            gravity=float(params.get("gravity", 9.81)),
            friction_mu=float(params.get("friction_mu", 0.5)),
            friction_mu_pair=float(friction_mu_pair) if friction_mu_pair is not None else None,
            restitution=float(params.get("restitution", 0.0)),
            ground_height=float(params.get("ground_height", 0.0)),
            contact_slop=float(params.get("contact_slop", 1e-3)),
            impact_velocity_min=float(params.get("impact_velocity_min", 0.1)),
            pgs_iters=int(params.get("pgs_iters", 20)),
            pgs_relaxation=float(params.get("pgs_relaxation", 1.0)),
            baumgarte_beta=float(params.get("baumgarte_beta", 0.2)),
            residual_tol=float(params.get("residual_tol", 1e-6)),
            warm_start=bool(params.get("warm_start", True)),
            sleep=sleep_manager,
        )
    else:
        raise ValueError(f"Unknown demo: {demo}")

    engine = Engine(
        system=system,
        core=symplectic,
        dissipation=dissipation,
        constraints=constraints,
        events=events,
    )

    state0 = State(
        q=_tensor_from_param(params["q0"], device=device, dtype=dtype),
        v=_tensor_from_param(params["v0"], device=device, dtype=dtype),
        t=0.0,
    )
    context: dict[str, Any] = {"failsafe": failsafe_cfg}
    constraints_context = _constraint_context(constraint_cfg, device=device, dtype=dtype)
    if constraints_context:
        context["constraints"] = constraints_context
    return engine, state0, context, dt, steps, demo


def _run_rollout(
    *,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    max_steps: int | None,
) -> Rollout:
    engine, state, context, dt, steps, demo = _build_engine_and_state(config=config, device=device, dtype=dtype)
    steps = int(min(steps, max_steps)) if max_steps is not None else int(steps)

    _, _, e_total0 = engine.system.energy(state)
    e_total0_value = float(e_total0.detach().cpu().item())

    q_hist: list[torch.Tensor] = []
    v_hist: list[torch.Tensor] = []

    series: dict[str, list[float]] = {
        "energy.E_total": [],
        "contacts.penetration_max": [],
        "contacts.complementarity_residual_max": [],
        "solver.residual_max": [],
        "solver.iters": [],
        "sleep.sleeping_count": [],
        "failsafe.triggered": [],
        "failsafe.soft_triggered": [],
        "failsafe.alpha_blend": [],
    }

    mode_counts_total: dict[str, int] = {}
    solver_status_counts: dict[str, int] = {}
    failsafe_reason_counts: dict[str, int] = {}
    failsafe_soft_reason_counts: dict[str, int] = {}

    for _ in range(steps):
        state, diag = engine.step(state=state, dt=dt, context=context)
        q_hist.append(state.q.detach())
        v_hist.append(state.v.detach())

        e_total = _to_float(diag["energy"]["E_total"])
        series["energy.E_total"].append(e_total)

        contacts = diag.get("contacts", {})
        series["contacts.penetration_max"].append(_to_float(contacts.get("penetration_max")))
        series["contacts.complementarity_residual_max"].append(_to_float(contacts.get("complementarity_residual_max")))

        solver = diag.get("solver", {})
        series["solver.residual_max"].append(_to_float(solver.get("residual_max")))
        series["solver.iters"].append(_to_float(solver.get("iters")))
        status = solver.get("status")
        if isinstance(status, str) and status:
            solver_status_counts[status] = solver_status_counts.get(status, 0) + 1

        contacts_mode_counts = contacts.get("mode_counts")
        if isinstance(contacts_mode_counts, dict):
            for k, v in contacts_mode_counts.items():
                if isinstance(v, int):
                    mode_counts_total[str(k)] = mode_counts_total.get(str(k), 0) + v

        sleep = diag.get("sleep", {})
        series["sleep.sleeping_count"].append(float(sleep.get("sleeping_count", 0)))

        failsafe = diag.get("failsafe", {})
        triggered = bool(failsafe.get("triggered", False))
        series["failsafe.triggered"].append(1.0 if triggered else 0.0)
        if triggered:
            reason = str(failsafe.get("reason", ""))
            failsafe_reason_counts[reason] = failsafe_reason_counts.get(reason, 0) + 1

        soft_reasons = failsafe.get("soft_reasons", [])
        soft_step_triggered = False
        if isinstance(soft_reasons, list):
            for reason in soft_reasons:
                reason_name = str(reason)
                if not reason_name:
                    continue
                failsafe_soft_reason_counts[reason_name] = failsafe_soft_reason_counts.get(reason_name, 0) + 1
                soft_step_triggered = True
        elif isinstance(soft_reasons, str) and soft_reasons:
            failsafe_soft_reason_counts[soft_reasons] = failsafe_soft_reason_counts.get(soft_reasons, 0) + 1
            soft_step_triggered = True
        series["failsafe.soft_triggered"].append(1.0 if soft_step_triggered else 0.0)
        series["failsafe.alpha_blend"].append(_to_float(failsafe.get("alpha_blend"), default=1.0))

    q_seq = torch.stack(q_hist) if q_hist else torch.zeros((0,))
    v_seq = torch.stack(v_hist) if v_hist else torch.zeros((0,))

    e_series = series["energy.E_total"]
    drift_abs_max = float(max(abs(e - e_total0_value) for e in e_series)) if e_series else 0.0
    denom = max(1e-12, abs(e_total0_value))
    drift_rel_max = float(drift_abs_max / denom)
    solver_status_counts = {str(k): int(v) for k, v in solver_status_counts.items()}
    for key in ("converged", "max_iter", "diverged", "na"):
        solver_status_counts[key] = int(solver_status_counts.get(key, 0))

    status_steps = int(sum(solver_status_counts.values()))
    max_iter_count = int(solver_status_counts.get("max_iter", 0))
    converged_count = int(solver_status_counts.get("converged", 0))
    diverged_count = int(solver_status_counts.get("diverged", 0))
    na_count = int(solver_status_counts.get("na", 0))
    max_iter_ratio = float(max_iter_count / max(1, status_steps)) if status_steps > 0 else 0.0
    converged_ratio = float(converged_count / max(1, status_steps)) if status_steps > 0 else 0.0
    diverged_ratio = float(diverged_count / max(1, status_steps)) if status_steps > 0 else 0.0
    na_ratio = float(na_count / max(1, status_steps)) if status_steps > 0 else 0.0

    summary: dict[str, Any] = {
        "demo": demo,
        "dt": float(dt),
        "steps": int(steps),
        "device": str(device),
        "dtype": str(dtype),
        "energy": {
            "E0": e_total0_value,
            "E_final": float(e_series[-1]) if e_series else 0.0,
            "drift_abs_max": drift_abs_max,
            "drift_rel_max": drift_rel_max,
        },
        "contacts": {
            "penetration_max": float(max(series["contacts.penetration_max"]) if steps else 0.0),
            "complementarity_residual_max": float(
                max(series["contacts.complementarity_residual_max"]) if steps else 0.0
            ),
            "mode_counts_total": mode_counts_total,
        },
        "solver": {
            "iters_mean": float(sum(series["solver.iters"]) / max(1, steps)),
            "iters_max": float(max(series["solver.iters"]) if steps else 0.0),
            "residual_max": float(max(series["solver.residual_max"]) if steps else 0.0),
            "status_counts": solver_status_counts,
            "status_steps": status_steps,
            "max_iter_steps": max_iter_count,
            "converged_steps": converged_count,
            "diverged_steps": diverged_count,
            "na_steps": na_count,
            "max_iter_ratio": max_iter_ratio,
            "converged_ratio": converged_ratio,
            "diverged_ratio": diverged_ratio,
            "na_ratio": na_ratio,
        },
        "sleep": {
            "sleeping_count_final": int(series["sleep.sleeping_count"][-1]) if steps else 0,
            "sleeping_count_max": int(max(series["sleep.sleeping_count"]) if steps else 0),
        },
        "failsafe": {
            "triggered_any": bool(any(v > 0.0 for v in series["failsafe.triggered"])),
            "triggered_steps": int(sum(1 for v in series["failsafe.triggered"] if v > 0.0)),
            "reason_counts": failsafe_reason_counts,
            "soft_triggered_any": bool(any(v > 0.0 for v in series["failsafe.soft_triggered"])),
            "soft_triggered_steps": int(sum(1 for v in series["failsafe.soft_triggered"] if v > 0.0)),
            "soft_reason_counts": failsafe_soft_reason_counts,
            "alpha_blend_min": float(min(series["failsafe.alpha_blend"]) if steps else 1.0),
            "alpha_blend_mean": float(sum(series["failsafe.alpha_blend"]) / max(1, steps)),
            "alpha_blend_drop_max": float(max(0.0, 1.0 - (min(series["failsafe.alpha_blend"]) if steps else 1.0))),
            "alpha_blend_drop_mean": float(max(0.0, 1.0 - (sum(series["failsafe.alpha_blend"]) / max(1, steps)))),
        },
    }
    return Rollout(q=q_seq, v=v_seq, series=series, summary=summary)


def _default_out_dir() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / "bench" / f"{timestamp}-eval"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="mvp", choices=["mvp", "mvp_ood"])
    parser.add_argument("--configs", nargs="*", type=Path, default=[])
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "float64"])
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--determinism-steps", type=int, default=100)
    parser.add_argument("--determinism-tol", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.configs:
        config_paths = [Path(p) for p in args.configs]
    else:
        if str(args.suite) == "mvp":
            config_paths = sorted(Path("configs").glob("mvp_*.yaml"))
        elif str(args.suite) == "mvp_ood":
            config_paths = sorted(Path("configs").glob("mvp_*.yaml")) + sorted(Path("configs").glob("bench_ood_*.yaml"))
        else:
            raise ValueError(f"Unknown suite: {args.suite}")
    if not config_paths:
        raise RuntimeError("No configs found")

    out_dir = args.out_dir or _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"runs": [], "meta": {}}
    results["meta"] = {
        "repeat": int(max(1, args.repeat)),
        "deterministic": bool(args.deterministic),
        "determinism_steps": int(max(0, args.determinism_steps)),
        "determinism_tol": float(args.determinism_tol),
        "max_steps": int(args.max_steps) if args.max_steps is not None else None,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
    }

    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        seed = int(config.get("seed", 0))
        dtype_name = str(args.dtype or config.get("dtype", "float64"))
        dtype = _torch_dtype(dtype_name)

        requested_device = args.device or str(config.get("device", "cpu"))
        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"
        device = _device(requested_device)

        set_determinism(seed=seed, deterministic=bool(args.deterministic))
        rollout = _run_rollout(config=config, device=device, dtype=dtype, max_steps=args.max_steps)

        determinism: dict[str, Any] = {"repeat": int(max(1, args.repeat))}
        if int(args.repeat) > 1:
            base = rollout
            max_steps_det = int(min(int(rollout.summary["steps"]), int(args.determinism_steps)))
            for rep in range(1, int(args.repeat)):
                set_determinism(seed=seed, deterministic=bool(args.deterministic))
                other = _run_rollout(config=config, device=device, dtype=dtype, max_steps=max_steps_det)
                dq = float((base.q[: other.q.shape[0]] - other.q).abs().max().detach().cpu().item())
                dv = float((base.v[: other.v.shape[0]] - other.v).abs().max().detach().cpu().item())
                determinism[f"rep{rep}"] = {"max_abs_dq": dq, "max_abs_dv": dv}

            diffs = [determinism[f"rep{rep}"] for rep in range(1, int(args.repeat))]
            max_dq = max(d["max_abs_dq"] for d in diffs) if diffs else 0.0
            max_dv = max(d["max_abs_dv"] for d in diffs) if diffs else 0.0
            determinism["max_abs_dq"] = float(max_dq)
            determinism["max_abs_dv"] = float(max_dv)
            determinism["ok"] = bool(max(max_dq, max_dv) <= float(args.determinism_tol))

        run_summary = {
            "config": str(config_path),
            "seed": seed,
            "summary": rollout.summary,
            "determinism": determinism,
        }
        results["runs"].append(run_summary)

    (out_dir / "bench_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_dir / 'bench_summary.json'}")
    for run in results["runs"]:
        cfg = run["config"]
        demo = run["summary"]["demo"]
        pen = run["summary"]["contacts"]["penetration_max"]
        res = run["summary"]["solver"]["residual_max"]
        drift = run["summary"]["energy"]["drift_rel_max"]
        det_ok = run["determinism"].get("ok")
        det_s = "" if det_ok is None else f" det_ok={det_ok}"
        print(f"{cfg} demo={demo} pen={pen:.3e} res={res:.3e} drift_rel={drift:.3e}{det_s}")


if __name__ == "__main__":
    main()
