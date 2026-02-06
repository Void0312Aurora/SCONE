from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

from scone.learned.mu import GroundPairMu, ScalarMu
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


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _as_range(x: Any) -> tuple[float, float]:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return float(x[0]), float(x[1])
    raise TypeError("Expected a 2-element list/tuple range")


def _clamp_int(value: int, cap: int | None) -> int:
    if cap is None:
        return int(value)
    return int(min(int(value), int(cap)))


@dataclass(frozen=True)
class ScalarTaskResult:
    summary: dict[str, Any]
    metrics: dict[str, float]


@dataclass(frozen=True)
class PairTaskResult:
    summary: dict[str, Any]
    metrics: dict[str, float]


def _run_scalar_task(
    *,
    demo: str,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    deterministic: bool,
    max_train_iters: int | None,
    max_rollout_steps: int | None,
    max_n_train: int | None,
    max_n_val: int | None,
) -> ScalarTaskResult:
    task = config.get("task", {})
    if not isinstance(task, dict):
        raise TypeError("config.task must be a mapping")
    loss_cfg = config.get("loss", {})
    if not isinstance(loss_cfg, dict):
        loss_cfg = {}
    engine_cfg = config.get("engine", {})
    if not isinstance(engine_cfg, dict):
        engine_cfg = {}

    seed = int(config.get("seed", 0))
    set_determinism(seed=seed, deterministic=deterministic)

    mu_true = float(task.get("mu_true", 0.6))
    mu_init = float(task.get("mu_init", 0.1))
    mu_max = float(task.get("mu_max", 1.0))
    lr = float(task.get("lr", 0.3))

    loss_mode = str(task.get("loss_mode", "open_loop"))
    rollout_steps = _clamp_int(int(task.get("rollout_steps", 20)), max_rollout_steps)
    train_iters = _clamp_int(int(task.get("train_iters", 120)), max_train_iters)

    dt = float(task.get("dt", config.get("dt", 0.01)))
    n_train = _clamp_int(int(task.get("n_train", 8)), max_n_train)
    n_val = _clamp_int(int(task.get("n_val", 8)), max_n_val)

    omega_range = _as_range(task.get("omega_range", (3.0, 8.0)))
    vx_range = _as_range(task.get("vx_range", (0.0, 0.0)))
    val_omega_range = _as_range(task.get("val_omega_range", (1.0, 4.0)))
    val_vx_range = _as_range(task.get("val_vx_range", (0.0, 0.0)))

    w_q = float(loss_cfg.get("w_q", 0.1))
    w_v = float(loss_cfg.get("w_v", 1.0))
    w_pen = float(loss_cfg.get("w_pen", 0.0))
    w_comp = float(loss_cfg.get("w_comp", 0.0))
    w_res = float(loss_cfg.get("w_res", 0.0))

    engine_params: dict[str, float | int | bool] = {
        "mass": float(engine_cfg.get("mass", 1.0)),
        "radius": float(engine_cfg.get("radius", 0.5)),
        "gravity": float(engine_cfg.get("gravity", 9.81)),
        "ground_height": float(engine_cfg.get("ground_height", 0.0)),
        "contact_slop": float(engine_cfg.get("contact_slop", 1e-3)),
        "impact_velocity_min": float(engine_cfg.get("impact_velocity_min", 0.2)),
        "pgs_iters": int(engine_cfg.get("pgs_iters", 20)),
        "baumgarte_beta": float(engine_cfg.get("baumgarte_beta", 0.2)),
        "residual_tol": float(engine_cfg.get("residual_tol", 1e-6)),
        "warm_start": bool(engine_cfg.get("warm_start", True)),
    }

    scalar_mod = _load_module(Path("scripts/train_m2_mu_pgs_stack.py"), name="scone_scripts_train_m2_mu_pgs_stack")

    dataset_train = scalar_mod._make_dataset(
        device=device,
        dtype=dtype,
        seed=seed,
        n=n_train,
        steps=rollout_steps,
        dt=dt,
        mu_true=mu_true,
        omega_range=omega_range,
        vx_range=vx_range,
        engine_params=engine_params,
    )
    dataset_val = scalar_mod._make_dataset(
        device=device,
        dtype=dtype,
        seed=seed + 1,
        n=n_val,
        steps=rollout_steps,
        dt=dt,
        mu_true=mu_true,
        omega_range=val_omega_range,
        vx_range=val_vx_range,
        engine_params=engine_params,
    )

    def build_engine(mu: float | torch.Tensor) -> Any:
        return scalar_mod._build_engine(device=device, dtype=dtype, mu=mu, **engine_params)

    loss_fn: Callable[..., torch.Tensor]
    if loss_mode == "open_loop":
        loss_fn = scalar_mod._dataset_loss_open_loop
    elif loss_mode == "teacher_forcing":
        loss_fn = scalar_mod._dataset_loss
    else:
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

    with torch.no_grad():
        baseline_loss = loss_fn(
            dataset=dataset_train,
            build_engine_fn=build_engine,
            mu=mu_init,
            steps=rollout_steps,
            w_q=w_q,
            w_v=w_v,
            w_pen=w_pen,
            w_comp=w_comp,
            w_res=w_res,
        )
        baseline_loss_val = loss_fn(
            dataset=dataset_val,
            build_engine_fn=build_engine,
            mu=mu_init,
            steps=rollout_steps,
            w_q=w_q,
            w_v=w_v,
            w_pen=w_pen,
            w_comp=w_comp,
            w_res=w_res,
        )

    ood_cfg = task.get("ood", {})
    dataset_ood = None
    dt_ood = None
    engine_ood = None
    if isinstance(ood_cfg, dict) and bool(ood_cfg.get("enabled", False)):
        dt_ood = float(ood_cfg.get("dt", dt))
        n_ood = _clamp_int(int(ood_cfg.get("n", n_val)), max_n_val)
        steps_ood = _clamp_int(int(ood_cfg.get("steps", rollout_steps)), max_rollout_steps)
        omega_ood = _as_range(ood_cfg.get("omega_range", val_omega_range))
        vx_ood = _as_range(ood_cfg.get("vx_range", val_vx_range))
        engine_ood = dict(engine_params)
        overrides = ood_cfg.get("engine_overrides", {})
        if isinstance(overrides, dict):
            for k, v in overrides.items():
                if k in engine_ood:
                    engine_ood[k] = v

        dataset_ood = scalar_mod._make_dataset(
            device=device,
            dtype=dtype,
            seed=seed + 2,
            n=n_ood,
            steps=steps_ood,
            dt=dt_ood,
            mu_true=mu_true,
            omega_range=omega_ood,
            vx_range=vx_ood,
            engine_params=engine_ood,
        )

    model = ScalarMu(init=mu_init, mu_max=mu_max, device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(train_iters):
        opt.zero_grad(set_to_none=True)
        mu = model()
        loss = loss_fn(
            dataset=dataset_train,
            build_engine_fn=build_engine,
            mu=mu,
            steps=rollout_steps,
            w_q=w_q,
            w_v=w_v,
            w_pen=w_pen,
            w_comp=w_comp,
            w_res=w_res,
        )
        loss.backward()
        opt.step()

    with torch.no_grad():
        mu_fit = float(model().detach().cpu().item())
        fit_loss = loss_fn(
            dataset=dataset_train,
            build_engine_fn=build_engine,
            mu=mu_fit,
            steps=rollout_steps,
            w_q=w_q,
            w_v=w_v,
            w_pen=w_pen,
            w_comp=w_comp,
            w_res=w_res,
        )
        fit_loss_val = loss_fn(
            dataset=dataset_val,
            build_engine_fn=build_engine,
            mu=mu_fit,
            steps=rollout_steps,
            w_q=w_q,
            w_v=w_v,
            w_pen=w_pen,
            w_comp=w_comp,
            w_res=w_res,
        )

        baseline_ood = None
        fit_ood = None
        if dataset_ood is not None:
            assert engine_ood is not None

            def build_engine_ood(mu: float | torch.Tensor) -> Any:
                return scalar_mod._build_engine(device=device, dtype=dtype, mu=mu, **engine_ood)

            steps_ood_eval = int(dataset_ood[0].q.shape[0] - 1) if dataset_ood else rollout_steps
            baseline_ood = loss_fn(
                dataset=dataset_ood,
                build_engine_fn=build_engine_ood,
                mu=mu_init,
                steps=steps_ood_eval,
                w_q=w_q,
                w_v=w_v,
                w_pen=w_pen,
                w_comp=w_comp,
                w_res=w_res,
            )
            fit_ood = loss_fn(
                dataset=dataset_ood,
                build_engine_fn=build_engine_ood,
                mu=mu_fit,
                steps=steps_ood_eval,
                w_q=w_q,
                w_v=w_v,
                w_pen=w_pen,
                w_comp=w_comp,
                w_res=w_res,
            )

    eps = 1e-12
    baseline_loss_v = float(baseline_loss.detach().cpu().item())
    fit_loss_v = float(fit_loss.detach().cpu().item())
    baseline_loss_val_v = float(baseline_loss_val.detach().cpu().item())
    fit_loss_val_v = float(fit_loss_val.detach().cpu().item())
    loss_ratio = float(fit_loss_v / (baseline_loss_v + eps))
    loss_ratio_val = float(fit_loss_val_v / (baseline_loss_val_v + eps))

    baseline_loss_ood_v = None
    fit_loss_ood_v = None
    loss_ratio_ood = None
    if dataset_ood is not None:
        assert baseline_ood is not None and fit_ood is not None
        baseline_loss_ood_v = float(baseline_ood.detach().cpu().item())
        fit_loss_ood_v = float(fit_ood.detach().cpu().item())
        loss_ratio_ood = float(fit_loss_ood_v / (baseline_loss_ood_v + eps))

    learn: dict[str, Any] = {
        "kind": "mu_stack_scalar",
        "loss_mode": loss_mode,
        "mu_true": float(mu_true),
        "mu_fit": float(mu_fit),
        "mu_abs_error": float(abs(mu_fit - mu_true)),
        "baseline_loss": baseline_loss_v,
        "fit_loss": fit_loss_v,
        "loss_ratio": loss_ratio,
        "baseline_loss_val": baseline_loss_val_v,
        "fit_loss_val": fit_loss_val_v,
        "loss_ratio_val": loss_ratio_val,
        "ood_enabled": bool(dataset_ood is not None),
        "baseline_loss_ood": baseline_loss_ood_v,
        "fit_loss_ood": fit_loss_ood_v,
        "loss_ratio_ood": loss_ratio_ood,
        "dt_train": float(dt),
        "dt_ood": float(dt_ood) if dt_ood is not None else None,
        "train_iters": int(train_iters),
        "rollout_steps": int(rollout_steps),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "w_q": float(w_q),
        "w_v": float(w_v),
        "w_pen": float(w_pen),
        "w_comp": float(w_comp),
        "w_res": float(w_res),
    }

    summary: dict[str, Any] = {
        "demo": demo,
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype),
        "learn": learn,
    }
    metrics = {
        "mu_fit": float(mu_fit),
        "mu_abs_error": float(abs(mu_fit - mu_true)),
        "loss_ratio_val": float(loss_ratio_val),
        "loss_ratio_ood": float(loss_ratio_ood) if loss_ratio_ood is not None else 0.0,
    }
    return ScalarTaskResult(summary=summary, metrics=metrics)


def _run_pair_task(
    *,
    demo: str,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    deterministic: bool,
    max_train_iters: int | None,
    max_rollout_steps: int | None,
    max_n_train: int | None,
    max_n_val: int | None,
) -> PairTaskResult:
    task = config.get("task", {})
    if not isinstance(task, dict):
        raise TypeError("config.task must be a mapping")
    loss_cfg = config.get("loss", {})
    if not isinstance(loss_cfg, dict):
        loss_cfg = {}
    engine_cfg = config.get("engine", {})
    if not isinstance(engine_cfg, dict):
        engine_cfg = {}

    seed = int(config.get("seed", 0))
    set_determinism(seed=seed, deterministic=deterministic)

    mu_ground_true = float(task.get("mu_ground_true", 0.6))
    mu_pair_true = float(task.get("mu_pair_true", 0.2))
    mu_ground_init = float(task.get("mu_ground_init", 0.1))
    mu_pair_init = float(task.get("mu_pair_init", 0.1))
    mu_max = float(task.get("mu_max", 1.0))
    lr = float(task.get("lr", 0.3))

    rollout_steps = _clamp_int(int(task.get("rollout_steps", 10)), max_rollout_steps)
    train_iters = _clamp_int(int(task.get("train_iters", 120)), max_train_iters)
    dt = float(task.get("dt", config.get("dt", 0.01)))
    n_train = _clamp_int(int(task.get("n_train", 8)), max_n_train)
    n_val = _clamp_int(int(task.get("n_val", 8)), max_n_val)

    omega_range = _as_range(task.get("omega_range", (3.0, 8.0)))
    vx_range = _as_range(task.get("vx_range", (0.0, 1.0)))
    vx_bottom_range = _as_range(task.get("vx_bottom_range", (0.0, 1.0)))
    val_omega_range = _as_range(task.get("val_omega_range", (1.0, 4.0)))
    val_vx_range = _as_range(task.get("val_vx_range", (0.0, 1.0)))
    val_vx_bottom_range = _as_range(task.get("val_vx_bottom_range", (0.0, 1.0)))

    w_q = float(loss_cfg.get("w_q", 0.1))
    w_v = float(loss_cfg.get("w_v", 1.0))
    w_pen = float(loss_cfg.get("w_pen", 0.0))
    w_comp = float(loss_cfg.get("w_comp", 0.0))
    w_res = float(loss_cfg.get("w_res", 0.0))

    engine_params: dict[str, float | int | bool] = {
        "mass": float(engine_cfg.get("mass", 1.0)),
        "radius": float(engine_cfg.get("radius", 0.5)),
        "gravity": float(engine_cfg.get("gravity", 9.81)),
        "ground_height": float(engine_cfg.get("ground_height", 0.0)),
        "contact_slop": float(engine_cfg.get("contact_slop", 1e-3)),
        "impact_velocity_min": float(engine_cfg.get("impact_velocity_min", 0.2)),
        "pgs_iters": int(engine_cfg.get("pgs_iters", 20)),
        "baumgarte_beta": float(engine_cfg.get("baumgarte_beta", 0.2)),
        "residual_tol": float(engine_cfg.get("residual_tol", 1e-6)),
        "warm_start": bool(engine_cfg.get("warm_start", True)),
    }

    pair_mod = _load_module(Path("scripts/train_m2_mu_pgs_stack_pair.py"), name="scone_scripts_train_m2_mu_pgs_stack_pair")

    dataset_train = pair_mod._make_dataset(
        device=device,
        dtype=dtype,
        seed=seed,
        n=n_train,
        steps=rollout_steps,
        dt=dt,
        mu_ground_true=mu_ground_true,
        mu_pair_true=mu_pair_true,
        omega_range=omega_range,
        vx_range=vx_range,
        vx_bottom_range=vx_bottom_range,
        engine_params=engine_params,
    )
    dataset_val = pair_mod._make_dataset(
        device=device,
        dtype=dtype,
        seed=seed + 1,
        n=n_val,
        steps=rollout_steps,
        dt=dt,
        mu_ground_true=mu_ground_true,
        mu_pair_true=mu_pair_true,
        omega_range=val_omega_range,
        vx_range=val_vx_range,
        vx_bottom_range=val_vx_bottom_range,
        engine_params=engine_params,
    )

    def build_engine(mu_ground: float | torch.Tensor, mu_pair: float | torch.Tensor) -> Any:
        return pair_mod._build_engine(device=device, dtype=dtype, mu_ground=mu_ground, mu_pair=mu_pair, **engine_params)

    with torch.no_grad():
        baseline_loss = pair_mod._dataset_loss_open_loop(
            dataset=dataset_train,
            build_engine_fn=build_engine,
            mu_ground=mu_ground_init,
            mu_pair=mu_pair_init,
            steps=rollout_steps,
            w_q=w_q,
            w_v=w_v,
            w_pen=w_pen,
            w_comp=w_comp,
            w_res=w_res,
        )
        baseline_loss_val = pair_mod._dataset_loss_open_loop(
            dataset=dataset_val,
            build_engine_fn=build_engine,
            mu_ground=mu_ground_init,
            mu_pair=mu_pair_init,
            steps=rollout_steps,
            w_q=w_q,
            w_v=w_v,
            w_pen=w_pen,
            w_comp=w_comp,
            w_res=w_res,
        )

    ood_cfg = task.get("ood", {})
    dataset_ood = None
    dt_ood = None
    engine_ood = None
    if isinstance(ood_cfg, dict) and bool(ood_cfg.get("enabled", False)):
        dt_ood = float(ood_cfg.get("dt", dt))
        n_ood = _clamp_int(int(ood_cfg.get("n", n_val)), max_n_val)
        steps_ood = _clamp_int(int(ood_cfg.get("steps", rollout_steps)), max_rollout_steps)
        omega_ood = _as_range(ood_cfg.get("omega_range", val_omega_range))
        vx_ood = _as_range(ood_cfg.get("vx_range", val_vx_range))
        vx_bottom_ood = _as_range(ood_cfg.get("vx_bottom_range", val_vx_bottom_range))
        engine_ood = dict(engine_params)
        overrides = ood_cfg.get("engine_overrides", {})
        if isinstance(overrides, dict):
            for k, v in overrides.items():
                if k in engine_ood:
                    engine_ood[k] = v

        dataset_ood = pair_mod._make_dataset(
            device=device,
            dtype=dtype,
            seed=seed + 2,
            n=n_ood,
            steps=steps_ood,
            dt=dt_ood,
            mu_ground_true=mu_ground_true,
            mu_pair_true=mu_pair_true,
            omega_range=omega_ood,
            vx_range=vx_ood,
            vx_bottom_range=vx_bottom_ood,
            engine_params=engine_ood,
        )

    model = GroundPairMu(
        init_ground=mu_ground_init,
        init_pair=mu_pair_init,
        mu_max=mu_max,
        device=device,
        dtype=dtype,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(train_iters):
        opt.zero_grad(set_to_none=True)
        mu_ground, mu_pair = model()
        loss = pair_mod._dataset_loss_open_loop(
            dataset=dataset_train,
            build_engine_fn=build_engine,
            mu_ground=mu_ground,
            mu_pair=mu_pair,
            steps=rollout_steps,
            w_q=w_q,
            w_v=w_v,
            w_pen=w_pen,
            w_comp=w_comp,
            w_res=w_res,
        )
        loss.backward()
        opt.step()

    with torch.no_grad():
        mu_ground_fit, mu_pair_fit = model()
        mu_ground_fit_v = float(mu_ground_fit.detach().cpu().item())
        mu_pair_fit_v = float(mu_pair_fit.detach().cpu().item())
        fit_loss = pair_mod._dataset_loss_open_loop(
            dataset=dataset_train,
            build_engine_fn=build_engine,
            mu_ground=mu_ground_fit_v,
            mu_pair=mu_pair_fit_v,
            steps=rollout_steps,
            w_q=w_q,
            w_v=w_v,
            w_pen=w_pen,
            w_comp=w_comp,
            w_res=w_res,
        )
        fit_loss_val = pair_mod._dataset_loss_open_loop(
            dataset=dataset_val,
            build_engine_fn=build_engine,
            mu_ground=mu_ground_fit_v,
            mu_pair=mu_pair_fit_v,
            steps=rollout_steps,
            w_q=w_q,
            w_v=w_v,
            w_pen=w_pen,
            w_comp=w_comp,
            w_res=w_res,
        )

        baseline_ood = None
        fit_ood = None
        if dataset_ood is not None:
            assert engine_ood is not None

            def build_engine_ood(mu_ground: float | torch.Tensor, mu_pair: float | torch.Tensor) -> Any:
                return pair_mod._build_engine(
                    device=device, dtype=dtype, mu_ground=mu_ground, mu_pair=mu_pair, **engine_ood
                )

            steps_ood_eval = int(dataset_ood[0].q.shape[0] - 1) if dataset_ood else rollout_steps
            baseline_ood = pair_mod._dataset_loss_open_loop(
                dataset=dataset_ood,
                build_engine_fn=build_engine_ood,
                mu_ground=mu_ground_init,
                mu_pair=mu_pair_init,
                steps=steps_ood_eval,
                w_q=w_q,
                w_v=w_v,
                w_pen=w_pen,
                w_comp=w_comp,
                w_res=w_res,
            )
            fit_ood = pair_mod._dataset_loss_open_loop(
                dataset=dataset_ood,
                build_engine_fn=build_engine_ood,
                mu_ground=mu_ground_fit_v,
                mu_pair=mu_pair_fit_v,
                steps=steps_ood_eval,
                w_q=w_q,
                w_v=w_v,
                w_pen=w_pen,
                w_comp=w_comp,
                w_res=w_res,
            )

    eps = 1e-12
    baseline_loss_v = float(baseline_loss.detach().cpu().item())
    fit_loss_v = float(fit_loss.detach().cpu().item())
    baseline_loss_val_v = float(baseline_loss_val.detach().cpu().item())
    fit_loss_val_v = float(fit_loss_val.detach().cpu().item())
    loss_ratio_val = float(fit_loss_val_v / (baseline_loss_val_v + eps))

    baseline_loss_ood_v = None
    fit_loss_ood_v = None
    loss_ratio_ood = None
    if dataset_ood is not None:
        assert baseline_ood is not None and fit_ood is not None
        baseline_loss_ood_v = float(baseline_ood.detach().cpu().item())
        fit_loss_ood_v = float(fit_ood.detach().cpu().item())
        loss_ratio_ood = float(fit_loss_ood_v / (baseline_loss_ood_v + eps))

    learn: dict[str, Any] = {
        "kind": "mu_stack_pair",
        "mu_ground_true": float(mu_ground_true),
        "mu_pair_true": float(mu_pair_true),
        "mu_ground_fit": float(mu_ground_fit_v),
        "mu_pair_fit": float(mu_pair_fit_v),
        "mu_ground_abs_error": float(abs(mu_ground_fit_v - mu_ground_true)),
        "mu_pair_abs_error": float(abs(mu_pair_fit_v - mu_pair_true)),
        "baseline_loss": baseline_loss_v,
        "fit_loss": fit_loss_v,
        "loss_ratio": float(fit_loss_v / (baseline_loss_v + eps)),
        "baseline_loss_val": baseline_loss_val_v,
        "fit_loss_val": fit_loss_val_v,
        "loss_ratio_val": loss_ratio_val,
        "ood_enabled": bool(dataset_ood is not None),
        "baseline_loss_ood": baseline_loss_ood_v,
        "fit_loss_ood": fit_loss_ood_v,
        "loss_ratio_ood": loss_ratio_ood,
        "dt_train": float(dt),
        "dt_ood": float(dt_ood) if dt_ood is not None else None,
        "train_iters": int(train_iters),
        "rollout_steps": int(rollout_steps),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "w_q": float(w_q),
        "w_v": float(w_v),
        "w_pen": float(w_pen),
        "w_comp": float(w_comp),
        "w_res": float(w_res),
    }

    summary: dict[str, Any] = {
        "demo": demo,
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype),
        "learn": learn,
    }
    metrics = {
        "mu_ground_fit": float(mu_ground_fit_v),
        "mu_pair_fit": float(mu_pair_fit_v),
        "mu_pair_abs_error": float(abs(mu_pair_fit_v - mu_pair_true)),
        "loss_ratio_val": float(loss_ratio_val),
        "loss_ratio_ood": float(loss_ratio_ood) if loss_ratio_ood is not None else 0.0,
    }
    return PairTaskResult(summary=summary, metrics=metrics)


def _default_out_dir() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / "bench" / f"{timestamp}-learn_eval"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="mu_scalar", choices=["mu_scalar", "mu_pair", "mu_all"])
    parser.add_argument("--configs", nargs="*", type=Path, default=[])
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "float64"])
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--determinism-tol", type=float, default=0.0)
    parser.add_argument("--max-train-iters", type=int, default=None)
    parser.add_argument("--max-rollout-steps", type=int, default=None)
    parser.add_argument("--max-n-train", type=int, default=None)
    parser.add_argument("--max-n-val", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.configs:
        config_paths = [Path(p) for p in args.configs]
    else:
        if str(args.suite) == "mu_scalar":
            config_paths = sorted(Path("configs").glob("learn_mu_stack_*.yaml"))
        elif str(args.suite) == "mu_pair":
            config_paths = sorted(Path("configs").glob("learn_mu_pair_*.yaml"))
        elif str(args.suite) == "mu_all":
            config_paths = sorted(Path("configs").glob("learn_mu_stack_*.yaml")) + sorted(
                Path("configs").glob("learn_mu_pair_*.yaml")
            )
        else:
            raise ValueError(f"Unknown suite: {args.suite}")
    if not config_paths:
        raise RuntimeError("No learn configs found (expected configs/learn_mu_stack_*.yaml and/or configs/learn_mu_pair_*.yaml)")

    out_dir = args.out_dir or _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"runs": [], "meta": {}}
    results["meta"] = {
        "repeat": int(max(1, args.repeat)),
        "deterministic": bool(args.deterministic),
        "determinism_tol": float(args.determinism_tol),
        "max_train_iters": int(args.max_train_iters) if args.max_train_iters is not None else None,
        "max_rollout_steps": int(args.max_rollout_steps) if args.max_rollout_steps is not None else None,
        "max_n_train": int(args.max_n_train) if args.max_n_train is not None else None,
        "max_n_val": int(args.max_n_val) if args.max_n_val is not None else None,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
    }

    repeat = int(max(1, args.repeat))
    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise TypeError(f"Config must be a mapping: {config_path}")

        demo = str(config.get("demo", config_path.stem))
        seed = int(config.get("seed", 0))

        dtype_name = str(args.dtype or config.get("dtype", "float32"))
        dtype = _torch_dtype(dtype_name)

        requested_device = args.device or str(config.get("device", "cpu"))
        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"
        device = _device(requested_device)

        task = config.get("task", {})
        if not isinstance(task, dict):
            raise TypeError("config.task must be a mapping")
        kind = str(task.get("kind", "mu_stack_scalar"))

        base_summary: dict[str, Any] | None = None
        base_metrics: dict[str, float] | None = None
        det: dict[str, Any] = {"repeat": repeat}

        for rep in range(repeat):
            set_determinism(seed=seed, deterministic=bool(args.deterministic))
            if kind == "mu_stack_scalar":
                result = _run_scalar_task(
                    demo=demo,
                    config=config,
                    device=device,
                    dtype=dtype,
                    deterministic=bool(args.deterministic),
                    max_train_iters=args.max_train_iters,
                    max_rollout_steps=args.max_rollout_steps,
                    max_n_train=args.max_n_train,
                    max_n_val=args.max_n_val,
                )
                summary = result.summary
                metrics = result.metrics
            elif kind == "mu_stack_pair":
                result2 = _run_pair_task(
                    demo=demo,
                    config=config,
                    device=device,
                    dtype=dtype,
                    deterministic=bool(args.deterministic),
                    max_train_iters=args.max_train_iters,
                    max_rollout_steps=args.max_rollout_steps,
                    max_n_train=args.max_n_train,
                    max_n_val=args.max_n_val,
                )
                summary = result2.summary
                metrics = result2.metrics
            else:
                raise ValueError(f"Unknown task kind: {kind}")

            if base_summary is None:
                base_summary = summary
                base_metrics = metrics
            else:
                assert base_metrics is not None
                diffs: dict[str, float] = {}
                for k, v in metrics.items():
                    if k in base_metrics:
                        diffs[k] = float(abs(float(v) - float(base_metrics[k])))
                det[f"rep{rep}"] = diffs

        assert base_summary is not None and base_metrics is not None
        if repeat > 1:
            max_abs: dict[str, float] = {}
            for rep in range(1, repeat):
                rep_diffs = det.get(f"rep{rep}", {})
                if not isinstance(rep_diffs, dict):
                    continue
                for k, v in rep_diffs.items():
                    if not isinstance(v, (int, float)):
                        continue
                    max_abs[k] = float(max(float(max_abs.get(k, 0.0)), float(v)))
            for k, v in max_abs.items():
                det[f"max_abs_{k}"] = float(v)
            det["ok"] = bool(all(float(v) <= float(args.determinism_tol) for v in max_abs.values()))
        else:
            det["ok"] = True

        run_summary = {
            "config": str(config_path),
            "seed": seed,
            "summary": base_summary,
            "determinism": det,
        }
        results["runs"].append(run_summary)

    (out_dir / "bench_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_dir / 'bench_summary.json'}")
    for run in results["runs"]:
        cfg = run["config"]
        demo = run["summary"]["demo"]
        learn = run["summary"].get("learn", {})
        if isinstance(learn, dict):
            kind = str(learn.get("kind", ""))
            mu_abs_err = learn.get("mu_abs_error") or learn.get("mu_ground_abs_error")
            ratio_val = learn.get("loss_ratio_val")
            ratio_ood = learn.get("loss_ratio_ood")
            extra = "" if ratio_ood is None else f" ratio_ood={float(ratio_ood):.3f}"
            print(f"{cfg} demo={demo} kind={kind} mu_abs_err={float(mu_abs_err):.3f} ratio_val={float(ratio_val):.3f}{extra}")


if __name__ == "__main__":
    main()
