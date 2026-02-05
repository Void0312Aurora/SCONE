from __future__ import annotations

import argparse
import json
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
from scone.layers.events import BouncingBallEventLayer, NoOpEventLayer
from scone.layers.symplectic import SymplecticEulerSeparable
from scone.state import State
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


def _to_float(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    return value


def _flatten_diagnostics(diag: Diagnostics) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for group_key, group_value in diag.items():
        if isinstance(group_value, dict):
            for key, value in group_value.items():
                flat[f"{group_key}.{key}"] = _to_float(value)
        else:
            flat[group_key] = _to_float(group_value)
    return flat


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

    plot_xy("t", ["q"], "state_q.png", "Position")
    plot_xy("t", ["v"], "state_v.png", "Velocity")
    plot_xy(
        "t",
        ["energy.E_kin", "energy.E_pot", "energy.E_total"],
        "energy.png",
        "Energy",
    )
    if "contacts.penetration_max" in series:
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
        events = BouncingBallEventLayer(
            mass=float(params["mass"]),
            gravity=float(params["gravity"]),
            restitution=float(params.get("restitution", 1.0)),
            ground_height=float(params.get("ground_height", 0.0)),
            q_slop=float(params.get("q_slop", 1e-3)),
            v_sleep=float(params.get("v_sleep", 0.1)),
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
        q=torch.tensor([float(params["q0"])], device=device, dtype=dtype),
        v=torch.tensor([float(params["v0"])], device=device, dtype=dtype),
        t=0.0,
    )

    out_dir = args.out_dir or _default_out_dir(args.config, demo)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "config.yaml")

    rows: list[dict[str, Any]] = []
    series: dict[str, list[float]] = {
        "t": [],
        "q": [],
        "v": [],
        "energy.E_kin": [],
        "energy.E_pot": [],
        "energy.E_total": [],
        "contacts.penetration_max": [],
    }

    context: dict[str, Any] = {"failsafe": failsafe_cfg}
    for step_index in range(steps):
        next_state, diag = engine.step(state=state, dt=dt, context=context)
        flat_diag = _flatten_diagnostics(diag)
        row = {"step": step_index, "t": float(next_state.t), "q": _to_float(next_state.q), "v": _to_float(next_state.v)}
        row.update(flat_diag)
        rows.append(row)

        series["t"].append(float(next_state.t))
        series["q"].append(float(next_state.q.detach().cpu().item()))
        series["v"].append(float(next_state.v.detach().cpu().item()))
        series["energy.E_kin"].append(float(diag["energy"]["E_kin"].detach().cpu().item()))
        series["energy.E_pot"].append(float(diag["energy"]["E_pot"].detach().cpu().item()))
        series["energy.E_total"].append(float(diag["energy"]["E_total"].detach().cpu().item()))
        penetration = diag.get("contacts", {}).get("penetration_max", torch.tensor(0.0))
        series["contacts.penetration_max"].append(float(_to_float(penetration)))

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
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
