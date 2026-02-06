from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml

from scone.diagnostics import Diagnostics
from scone.engine import Engine
from scone.layers.constraints import NoOpConstraintLayer
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


def _default_out_dir(config_path: Path, demo_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-{demo_name}-{config_path.stem}"
    return Path("outputs") / run_id


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _flatten_diagnostics(diag: Diagnostics) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for group_key, group_value in diag.items():
        if isinstance(group_value, dict):
            for key, value in group_value.items():
                flat[f"{group_key}.{key}"] = _to_jsonable(value)
        else:
            flat[group_key] = _to_jsonable(group_value)
    return flat


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


def _plot_series(out_dir: Path, series: dict[str, list[float]]) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def plot_xy(x_key: str, y_keys: list[str], filename: str, title: str) -> None:
        plt.figure(figsize=(8, 4))
        x = series[x_key]
        for y_key in y_keys:
            plt.plot(x, series[y_key], label=y_key)
        plt.title(title)
        plt.xlabel(x_key)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / filename)
        plt.close()

    if "q" in series:
        plot_xy("t", ["q"], "state_q.png", "Position")
    if "v" in series:
        plot_xy("t", ["v"], "state_v.png", "Velocity")
    if "x" in series and "y" in series:
        plot_xy("t", ["x", "y"], "state_xy.png", "Position (x,y)")
    if "theta" in series:
        plot_xy("t", ["theta"], "state_theta.png", "Yaw (theta)")
    if "vx" in series and "vy" in series:
        plot_xy("t", ["vx", "vy"], "state_vxy.png", "Velocity (vx,vy)")
    if "omega" in series:
        plot_xy("t", ["omega"], "state_omega.png", "Angular velocity (omega)")
    multi_y_keys = sorted(
        [k for k in series.keys() if re.fullmatch(r"y\d+", k)],
        key=lambda key: int(key[1:]),
    )
    if multi_y_keys:
        plot_xy("t", multi_y_keys, "state_y.png", "Heights (y per body)")
    plot_xy(
        "t",
        ["energy.E_kin", "energy.E_pot", "energy.E_total"],
        "energy.png",
        "Energy",
    )
    if "contacts.penetration_max" in series and any(v != 0.0 for v in series["contacts.penetration_max"]):
        plot_xy("t", ["contacts.penetration_max"], "penetration.png", "Penetration")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--deterministic", action="store_true", default=False)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    demo = str(config["demo"])
    seed = int(config.get("seed", 0))
    dt = float(config["dt"])
    steps = int(config["steps"])
    dtype = _torch_dtype(str(config.get("dtype", "float64")))

    requested_device = args.device or str(config.get("device", "cpu"))
    device = _device(requested_device)
    set_determinism(seed=seed, deterministic=args.deterministic)

    params: dict[str, Any] = dict(config.get("params", {}))
    failsafe_cfg: dict[str, Any] = dict(config.get("failsafe", {}))
    sleep_cfg: dict[str, Any] = dict(config.get("sleep", {}))
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
        constraints = NoOpConstraintLayer()
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
        constraints = NoOpConstraintLayer()
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
        constraints = NoOpConstraintLayer()
        sleep_manager = None
        if sleep_cfg:
            sleep_manager = SleepManager(
                SleepConfig(
                    enabled=bool(sleep_cfg.get("enabled", True)),
                    v_sleep=float(sleep_cfg.get("v_sleep", 0.1)),
                    v_wake=float(sleep_cfg.get("v_wake", 0.2)),
                    steps_to_sleep=int(sleep_cfg.get("steps_to_sleep", 5)),
                    freeze_core=bool(sleep_cfg.get("freeze_core", True)),
                )
            )
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
        constraints = NoOpConstraintLayer()
        sleep_manager = None
        if sleep_cfg:
            sleep_manager = SleepManager(
                SleepConfig(
                    enabled=bool(sleep_cfg.get("enabled", True)),
                    v_sleep=float(sleep_cfg.get("v_sleep", 0.1)),
                    v_wake=float(sleep_cfg.get("v_wake", 0.2)),
                    steps_to_sleep=int(sleep_cfg.get("steps_to_sleep", 20)),
                    freeze_core=bool(sleep_cfg.get("freeze_core", True)),
                )
            )
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

    state = State(
        q=_tensor_from_param(params["q0"], device=device, dtype=dtype),
        v=_tensor_from_param(params["v0"], device=device, dtype=dtype),
        t=0.0,
    )
    n_bodies = int(state.q.shape[0]) if state.q.ndim == 2 else 1

    out_dir = args.out_dir or _default_out_dir(args.config, demo)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "config.yaml")

    rows: list[dict[str, Any]] = []
    series: dict[str, list[float]] = {"t": [], "energy.E_kin": [], "energy.E_pot": [], "energy.E_total": []}
    if demo in {"harmonic_1d", "damped_oscillator_1d", "bouncing_ball_1d"}:
        series.update({"q": [], "v": [], "contacts.penetration_max": []})
    elif demo == "disk_roll_2d":
        series.update(
            {
                "x": [],
                "y": [],
                "theta": [],
                "vx": [],
                "vy": [],
                "omega": [],
                "contacts.penetration_max": [],
            }
        )
    elif demo == "disk_stack_2d":
        series.update({"contacts.penetration_max": []})
        for body_i in range(n_bodies):
            series[f"x{body_i}"] = []
            series[f"y{body_i}"] = []

    context: dict[str, Any] = {"failsafe": failsafe_cfg}
    for step_index in range(steps):
        next_state, diag = engine.step(state=state, dt=dt, context=context)
        flat_diag = _flatten_diagnostics(diag)
        row = {
            "step": step_index,
            "t": float(next_state.t),
            "q": _to_jsonable(next_state.q),
            "v": _to_jsonable(next_state.v),
        }
        row.update(flat_diag)
        rows.append(row)

        series["t"].append(float(next_state.t))
        if "q" in series:
            series["q"].append(float(next_state.q.detach().flatten().cpu().item()))
        if "v" in series:
            series["v"].append(float(next_state.v.detach().flatten().cpu().item()))
        if demo == "disk_roll_2d":
            q0 = next_state.q.detach().cpu()[0]
            v0 = next_state.v.detach().cpu()[0]
            series["x"].append(float(q0[0].item()))
            series["y"].append(float(q0[1].item()))
            series["theta"].append(float(q0[2].item()))
            series["vx"].append(float(v0[0].item()))
            series["vy"].append(float(v0[1].item()))
            series["omega"].append(float(v0[2].item()))
        if demo == "disk_stack_2d":
            q_cpu = next_state.q.detach().cpu()
            for body_i in range(min(n_bodies, int(q_cpu.shape[0]))):
                series[f"x{body_i}"].append(float(q_cpu[body_i, 0].item()))
                series[f"y{body_i}"].append(float(q_cpu[body_i, 1].item()))
        series["energy.E_kin"].append(float(diag["energy"]["E_kin"].detach().cpu().item()))
        series["energy.E_pot"].append(float(diag["energy"]["E_pot"].detach().cpu().item()))
        series["energy.E_total"].append(float(diag["energy"]["E_total"].detach().cpu().item()))
        penetration = diag.get("contacts", {}).get("penetration_max", torch.tensor(0.0))
        if "contacts.penetration_max" in series:
            series["contacts.penetration_max"].append(float(_to_jsonable(penetration)))

        state = next_state

    _write_jsonl(out_dir / "logs" / "diagnostics.jsonl", rows)
    _plot_series(out_dir, series)

    meta = {
        "demo": demo,
        "device": str(device),
        "dtype": str(dtype),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "seed": seed,
        "dt": dt,
        "steps": steps,
        "params": params,
        "failsafe": failsafe_cfg,
        "sleep": sleep_cfg,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
