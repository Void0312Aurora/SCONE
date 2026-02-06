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

from scone.learned.mu import ContactMuMLP, ContactMuMaterialMLP, GroundPairMu, ScalarMu
from scone.state import State
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


def _as_material_patterns(x: Any) -> tuple[tuple[int, int, int], ...]:
    if x is None:
        return tuple()
    if not isinstance(x, (list, tuple)):
        raise TypeError("Expected material patterns as list/tuple")
    patterns: list[tuple[int, int, int]] = []
    for item in x:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise TypeError("Each material pattern must be a 3-element list/tuple")
        patterns.append((int(item[0]), int(item[1]), int(item[2])))
    return tuple(patterns)


def _clamp_int(value: int, cap: int | None) -> int:
    if cap is None:
        return int(value)
    return int(min(int(value), int(cap)))


def _canonical_material_pair(material_i: int, material_j: int) -> tuple[int, int]:
    a = int(material_i)
    b = int(material_j)
    return (a, b) if a <= b else (b, a)


def _material_body_set(
    *,
    body_pool: tuple[int, ...],
    body_patterns: tuple[tuple[int, int, int], ...],
) -> set[int]:
    if body_patterns:
        out: set[int] = set()
        for pattern in body_patterns:
            out.update(int(material_id) for material_id in pattern)
        return out
    return {int(material_id) for material_id in body_pool}


def _material_pair_set(
    *,
    body_pool: tuple[int, ...],
    body_patterns: tuple[tuple[int, int, int], ...],
    ground_material_id: int,
) -> set[tuple[int, int]]:
    ground = int(ground_material_id)
    out: set[tuple[int, int]] = set()

    if body_patterns:
        for pattern in body_patterns:
            a, b, c = (int(pattern[0]), int(pattern[1]), int(pattern[2]))
            out.add(_canonical_material_pair(a, ground))
            out.add(_canonical_material_pair(b, ground))
            out.add(_canonical_material_pair(c, ground))
            out.add(_canonical_material_pair(a, b))
            out.add(_canonical_material_pair(a, c))
            out.add(_canonical_material_pair(b, c))
        return out

    for material_i in body_pool:
        mi = int(material_i)
        out.add(_canonical_material_pair(mi, ground))
    for material_i in body_pool:
        mi = int(material_i)
        for material_j in body_pool:
            mj = int(material_j)
            out.add(_canonical_material_pair(mi, mj))
    return out


def _probe_mu_mae_on_material_pairs(
    *,
    device: torch.device,
    dtype: torch.dtype,
    mu_pred_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    mu_true_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    material_pairs: set[tuple[int, int]],
    ground_material_id: int,
) -> torch.Tensor | None:
    if not material_pairs:
        return None

    phis = (-0.02, -0.005, 0.0)
    vns = (-2.0, -0.5, 0.0, 0.5)
    vts = (-3.0, -1.0, 0.0, 1.0, 3.0)
    ground = int(ground_material_id)
    pair_list = sorted(_canonical_material_pair(int(a), int(b)) for (a, b) in material_pairs)
    errs: list[torch.Tensor] = []

    for material_i, material_j in pair_list:
        if material_i == ground and material_j == ground:
            continue
        if material_i == ground or material_j == ground:
            body_material = material_j if material_i == ground else material_i
            pair_specs = ((float(body_material), float(ground), 0.0),)
        else:
            if material_i == material_j:
                pair_specs = ((float(material_i), float(material_j), 1.0),)
            else:
                pair_specs = (
                    (float(material_i), float(material_j), 1.0),
                    (float(material_j), float(material_i), 1.0),
                )
        for phi in phis:
            for vn in vns:
                for vt in vts:
                    for pair_i, pair_j, is_pair in pair_specs:
                        feat = {
                            "phi": torch.tensor(phi, device=device, dtype=dtype),
                            "vn": torch.tensor(vn, device=device, dtype=dtype),
                            "vt": torch.tensor(vt, device=device, dtype=dtype),
                            "is_pair": torch.tensor(is_pair, device=device, dtype=dtype),
                            "material_i": torch.tensor(pair_i, device=device, dtype=dtype),
                            "material_j": torch.tensor(pair_j, device=device, dtype=dtype),
                        }
                        mu_pred = mu_pred_fn(feat)
                        mu_true = mu_true_fn(feat)
                        errs.append(torch.abs(mu_pred - mu_true))
    if not errs:
        return None
    return torch.stack(errs).mean()


@dataclass(frozen=True)
class ScalarTaskResult:
    summary: dict[str, Any]
    metrics: dict[str, float]


@dataclass(frozen=True)
class PairTaskResult:
    summary: dict[str, Any]
    metrics: dict[str, float]


@dataclass(frozen=True)
class FieldTaskResult:
    summary: dict[str, Any]
    metrics: dict[str, float]


def _normalize_soft_reasons(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            name = str(item)
            if name:
                out.append(name)
        return out
    if isinstance(value, str):
        return [value] if value else []
    return []


def _collect_open_loop_diag(
    *,
    dataset: list[Any],
    build_engine_fn: Callable[[], Any] | None = None,
    build_engine_for_rollout_fn: Callable[[Any], Any] | None = None,
    steps: int,
) -> dict[str, Any]:
    if build_engine_for_rollout_fn is None and build_engine_fn is None:
        raise ValueError("Either build_engine_fn or build_engine_for_rollout_fn must be provided")

    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    soft_reason_counts: dict[str, int] = {}

    hard_steps = 0
    soft_steps = 0
    penetration_max = 0.0
    comp_res_max = 0.0
    solver_residual_max = 0.0
    alpha_blend_min = 1.0
    alpha_blend_sum = 0.0
    alpha_blend_count = 0

    for rollout in dataset:
        if build_engine_for_rollout_fn is not None:
            engine = build_engine_for_rollout_fn(rollout)
        else:
            assert build_engine_fn is not None
            engine = build_engine_fn()
        q0 = rollout.q[0].detach().clone()
        v0 = rollout.v[0].detach().clone()
        state = State(q=q0, v=v0, t=0.0)
        context: dict[str, object] = {"failsafe": {}}
        steps_eval = int(min(int(steps), int(rollout.q.shape[0] - 1)))

        for _ in range(steps_eval):
            state, diag = engine.step(state=state, dt=rollout.dt, context=context)

            contacts = diag.get("contacts", {})
            if isinstance(contacts, dict):
                pen = contacts.get("penetration_max", 0.0)
                comp = contacts.get("complementarity_residual_max", 0.0)
                if isinstance(pen, torch.Tensor):
                    penetration_max = max(penetration_max, float(pen.detach().max().cpu().item()))
                elif isinstance(pen, (int, float)):
                    penetration_max = max(penetration_max, float(pen))
                if isinstance(comp, torch.Tensor):
                    comp_res_max = max(comp_res_max, float(comp.detach().max().cpu().item()))
                elif isinstance(comp, (int, float)):
                    comp_res_max = max(comp_res_max, float(comp))

            solver = diag.get("solver", {})
            if isinstance(solver, dict):
                status = solver.get("status")
                if isinstance(status, str) and status:
                    status_counts[status] = status_counts.get(status, 0) + 1
                residual = solver.get("residual_max")
                if isinstance(residual, torch.Tensor):
                    solver_residual_max = max(solver_residual_max, float(residual.detach().max().cpu().item()))
                elif isinstance(residual, (int, float)):
                    solver_residual_max = max(solver_residual_max, float(residual))

            failsafe = diag.get("failsafe", {})
            if isinstance(failsafe, dict):
                alpha_blend = failsafe.get("alpha_blend", 1.0)
                if isinstance(alpha_blend, torch.Tensor):
                    alpha_blend_value = float(alpha_blend.detach().reshape(-1)[0].cpu().item())
                elif isinstance(alpha_blend, (int, float)):
                    alpha_blend_value = float(alpha_blend)
                else:
                    alpha_blend_value = 1.0
                alpha_blend_min = min(alpha_blend_min, alpha_blend_value)
                alpha_blend_sum += alpha_blend_value
                alpha_blend_count += 1

                if bool(failsafe.get("triggered", False)):
                    hard_steps += 1
                    reason = str(failsafe.get("reason", ""))
                    if reason:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                soft_reasons = _normalize_soft_reasons(failsafe.get("soft_reasons", []))
                if soft_reasons:
                    soft_steps += 1
                    for reason in soft_reasons:
                        soft_reason_counts[reason] = soft_reason_counts.get(reason, 0) + 1

    status_counts = {str(k): int(v) for k, v in status_counts.items()}
    for key in ("converged", "max_iter", "diverged", "na"):
        status_counts[key] = int(status_counts.get(key, 0))

    status_steps = int(sum(status_counts.values()))
    max_iter_steps = int(status_counts.get("max_iter", 0))
    converged_steps = int(status_counts.get("converged", 0))
    diverged_steps = int(status_counts.get("diverged", 0))
    na_steps = int(status_counts.get("na", 0))
    max_iter_ratio = float(max_iter_steps / max(1, status_steps)) if status_steps > 0 else 0.0
    converged_ratio = float(converged_steps / max(1, status_steps)) if status_steps > 0 else 0.0
    diverged_ratio = float(diverged_steps / max(1, status_steps)) if status_steps > 0 else 0.0
    na_ratio = float(na_steps / max(1, status_steps)) if status_steps > 0 else 0.0

    return {
        "contacts": {
            "penetration_max": penetration_max,
            "complementarity_residual_max": comp_res_max,
        },
        "solver": {
            "residual_max": solver_residual_max,
            "status_counts": status_counts,
            "status_steps": status_steps,
            "max_iter_steps": max_iter_steps,
            "converged_steps": converged_steps,
            "diverged_steps": diverged_steps,
            "na_steps": na_steps,
            "max_iter_ratio": max_iter_ratio,
            "converged_ratio": converged_ratio,
            "diverged_ratio": diverged_ratio,
            "na_ratio": na_ratio,
        },
        "failsafe": {
            "triggered_any": bool(hard_steps > 0),
            "triggered_steps": int(hard_steps),
            "reason_counts": reason_counts,
            "soft_triggered_any": bool(soft_steps > 0),
            "soft_triggered_steps": int(soft_steps),
            "soft_reason_counts": soft_reason_counts,
            "alpha_blend_min": float(alpha_blend_min),
            "alpha_blend_mean": float(alpha_blend_sum / max(1, alpha_blend_count)),
            "alpha_blend_drop_max": float(max(0.0, 1.0 - alpha_blend_min)),
            "alpha_blend_drop_mean": float(max(0.0, 1.0 - (alpha_blend_sum / max(1, alpha_blend_count)))),
        },
    }


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
        "pgs_relaxation": float(engine_cfg.get("pgs_relaxation", 1.0)),
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

    def build_engine_fit() -> Any:
        return scalar_mod._build_engine(device=device, dtype=dtype, mu=mu_fit, **engine_params)

    eval_val = _collect_open_loop_diag(
        dataset=dataset_val,
        build_engine_fn=build_engine_fit,
        steps=rollout_steps,
    )
    eval_ood = None
    if dataset_ood is not None:
        assert engine_ood is not None

        def build_engine_fit_ood() -> Any:
            return scalar_mod._build_engine(device=device, dtype=dtype, mu=mu_fit, **engine_ood)

        eval_ood = _collect_open_loop_diag(
            dataset=dataset_ood,
            build_engine_fn=build_engine_fit_ood,
            steps=int(dataset_ood[0].q.shape[0] - 1),
        )

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
        "eval_val": eval_val,
        "eval_ood": eval_ood,
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
        "eval_val_solver_max_iter_ratio": float(eval_val["solver"]["max_iter_ratio"]),
        "eval_val_failsafe_soft_triggered_any": float(bool(eval_val["failsafe"]["soft_triggered_any"])),
        "eval_val_failsafe_triggered_any": float(bool(eval_val["failsafe"]["triggered_any"])),
    }
    if eval_ood is not None:
        metrics["eval_ood_solver_max_iter_ratio"] = float(eval_ood["solver"]["max_iter_ratio"])
        metrics["eval_ood_failsafe_soft_triggered_any"] = float(bool(eval_ood["failsafe"]["soft_triggered_any"]))
        metrics["eval_ood_failsafe_triggered_any"] = float(bool(eval_ood["failsafe"]["triggered_any"]))
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
        "pgs_relaxation": float(engine_cfg.get("pgs_relaxation", 1.0)),
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

    ood_runs: list[dict[str, Any]] = []
    ood_cases_cfg = task.get("ood_cases")
    if isinstance(ood_cases_cfg, list) and ood_cases_cfg:
        for idx, ood_case_cfg in enumerate(ood_cases_cfg):
            if not isinstance(ood_case_cfg, dict):
                continue
            if not bool(ood_case_cfg.get("enabled", True)):
                continue

            case_name = str(ood_case_cfg.get("name", f"ood_case_{idx}"))
            dt_case = float(ood_case_cfg.get("dt", dt))
            n_case = _clamp_int(int(ood_case_cfg.get("n", n_val)), max_n_val)
            steps_case = _clamp_int(int(ood_case_cfg.get("steps", rollout_steps)), max_rollout_steps)
            omega_case = _as_range(ood_case_cfg.get("omega_range", val_omega_range))
            vx_case = _as_range(ood_case_cfg.get("vx_range", val_vx_range))
            vx_bottom_case = _as_range(ood_case_cfg.get("vx_bottom_range", val_vx_bottom_range))

            engine_case = dict(engine_params)
            overrides = ood_case_cfg.get("engine_overrides", {})
            if isinstance(overrides, dict):
                for k, v in overrides.items():
                    if k in engine_case:
                        engine_case[k] = v

            dataset_case = pair_mod._make_dataset(
                device=device,
                dtype=dtype,
                seed=seed + 2 + idx,
                n=n_case,
                steps=steps_case,
                dt=dt_case,
                mu_ground_true=mu_ground_true,
                mu_pair_true=mu_pair_true,
                omega_range=omega_case,
                vx_range=vx_case,
                vx_bottom_range=vx_bottom_case,
                engine_params=engine_case,
            )
            ood_runs.append(
                {
                    "name": case_name,
                    "dt": float(dt_case),
                    "steps": int(steps_case),
                    "dataset": dataset_case,
                    "engine": engine_case,
                }
            )
    else:
        ood_cfg = task.get("ood", {})
        if isinstance(ood_cfg, dict) and bool(ood_cfg.get("enabled", False)):
            dt_case = float(ood_cfg.get("dt", dt))
            n_case = _clamp_int(int(ood_cfg.get("n", n_val)), max_n_val)
            steps_case = _clamp_int(int(ood_cfg.get("steps", rollout_steps)), max_rollout_steps)
            omega_case = _as_range(ood_cfg.get("omega_range", val_omega_range))
            vx_case = _as_range(ood_cfg.get("vx_range", val_vx_range))
            vx_bottom_case = _as_range(ood_cfg.get("vx_bottom_range", val_vx_bottom_range))

            engine_case = dict(engine_params)
            overrides = ood_cfg.get("engine_overrides", {})
            if isinstance(overrides, dict):
                for k, v in overrides.items():
                    if k in engine_case:
                        engine_case[k] = v

            dataset_case = pair_mod._make_dataset(
                device=device,
                dtype=dtype,
                seed=seed + 2,
                n=n_case,
                steps=steps_case,
                dt=dt_case,
                mu_ground_true=mu_ground_true,
                mu_pair_true=mu_pair_true,
                omega_range=omega_case,
                vx_range=vx_case,
                vx_bottom_range=vx_bottom_case,
                engine_params=engine_case,
            )
            ood_runs.append(
                {
                    "name": "ood",
                    "dt": float(dt_case),
                    "steps": int(steps_case),
                    "dataset": dataset_case,
                    "engine": engine_case,
                }
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

        ood_case_results: list[dict[str, Any]] = []
        for ood_case in ood_runs:
            dataset_ood = ood_case["dataset"]
            engine_ood = ood_case["engine"]
            steps_ood = int(ood_case["steps"])

            def build_engine_ood(mu_ground: float | torch.Tensor, mu_pair: float | torch.Tensor) -> Any:
                return pair_mod._build_engine(
                    device=device, dtype=dtype, mu_ground=mu_ground, mu_pair=mu_pair, **engine_ood
                )

            baseline_ood_t = pair_mod._dataset_loss_open_loop(
                dataset=dataset_ood,
                build_engine_fn=build_engine_ood,
                mu_ground=mu_ground_init,
                mu_pair=mu_pair_init,
                steps=steps_ood,
                w_q=w_q,
                w_v=w_v,
                w_pen=w_pen,
                w_comp=w_comp,
                w_res=w_res,
            )
            fit_ood_t = pair_mod._dataset_loss_open_loop(
                dataset=dataset_ood,
                build_engine_fn=build_engine_ood,
                mu_ground=mu_ground_fit_v,
                mu_pair=mu_pair_fit_v,
                steps=steps_ood,
                w_q=w_q,
                w_v=w_v,
                w_pen=w_pen,
                w_comp=w_comp,
                w_res=w_res,
            )

            def build_engine_fit_ood() -> Any:
                return pair_mod._build_engine(
                    device=device,
                    dtype=dtype,
                    mu_ground=mu_ground_fit_v,
                    mu_pair=mu_pair_fit_v,
                    **engine_ood,
                )

            eval_ood_case = _collect_open_loop_diag(
                dataset=dataset_ood,
                build_engine_fn=build_engine_fit_ood,
                steps=steps_ood,
            )

            baseline_ood_v = float(baseline_ood_t.detach().cpu().item())
            fit_ood_v = float(fit_ood_t.detach().cpu().item())
            ood_case_results.append(
                {
                    "name": str(ood_case["name"]),
                    "dt": float(ood_case["dt"]),
                    "body_material_pool": [int(x) for x in ood_case.get("body_material_pool", [])],
                    "body_material_patterns": [list(map(int, p)) for p in ood_case.get("body_material_patterns", [])],
                    "ground_material_id": int(ood_case.get("ground_material_id", 0)),
                    "baseline_loss": baseline_ood_v,
                    "fit_loss": fit_ood_v,
                    "loss_ratio": float(fit_ood_v / (baseline_ood_v + 1e-12)),
                    "eval": eval_ood_case,
                }
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
    dt_ood = None
    ood_case_count = int(len(ood_case_results))
    loss_ratio_ood_worst = None
    if ood_case_results:
        worst_case = max(ood_case_results, key=lambda x: float(x["loss_ratio"]))
        baseline_loss_ood_v = float(worst_case["baseline_loss"])
        fit_loss_ood_v = float(worst_case["fit_loss"])
        loss_ratio_ood = float(worst_case["loss_ratio"])
        loss_ratio_ood_worst = float(loss_ratio_ood)
        dt_ood = float(worst_case["dt"])

    def build_engine_fit() -> Any:
        return pair_mod._build_engine(
            device=device, dtype=dtype, mu_ground=mu_ground_fit_v, mu_pair=mu_pair_fit_v, **engine_params
        )

    eval_val = _collect_open_loop_diag(
        dataset=dataset_val,
        build_engine_fn=build_engine_fit,
        steps=rollout_steps,
    )
    eval_ood = None
    if ood_case_results:
        max_iter_ratio_worst = max(float(case["eval"]["solver"]["max_iter_ratio"]) for case in ood_case_results)
        diverged_ratio_worst = max(float(case["eval"]["solver"].get("diverged_ratio", 0.0)) for case in ood_case_results)
        na_ratio_worst = max(float(case["eval"]["solver"].get("na_ratio", 0.0)) for case in ood_case_results)
        residual_worst = max(float(case["eval"]["solver"]["residual_max"]) for case in ood_case_results)
        hard_any = any(bool(case["eval"]["failsafe"]["triggered_any"]) for case in ood_case_results)
        soft_any = any(bool(case["eval"]["failsafe"]["soft_triggered_any"]) for case in ood_case_results)
        alpha_drop_max_worst = max(float(case["eval"]["failsafe"].get("alpha_blend_drop_max", 0.0)) for case in ood_case_results)
        alpha_drop_mean_worst = max(
            float(case["eval"]["failsafe"].get("alpha_blend_drop_mean", 0.0)) for case in ood_case_results
        )
        worst_by_solver = max(ood_case_results, key=lambda x: float(x["eval"]["solver"]["max_iter_ratio"]))

        eval_ood = {
            "case_count": int(len(ood_case_results)),
            "worst_case": str(worst_by_solver["name"]),
            "solver": {
                "max_iter_ratio": float(max_iter_ratio_worst),
                "diverged_ratio": float(diverged_ratio_worst),
                "na_ratio": float(na_ratio_worst),
                "residual_max": float(residual_worst),
            },
            "failsafe": {
                "triggered_any": bool(hard_any),
                "soft_triggered_any": bool(soft_any),
                "alpha_blend_drop_max": float(alpha_drop_max_worst),
                "alpha_blend_drop_mean": float(alpha_drop_mean_worst),
            },
            "cases": [
                {
                    "name": str(case["name"]),
                    "dt": float(case["dt"]),
                    "loss_ratio": float(case["loss_ratio"]),
                    "eval": case["eval"],
                }
                for case in ood_case_results
            ],
        }

    mu_gap_true = float(mu_ground_true - mu_pair_true)
    mu_gap_fit = float(mu_ground_fit_v - mu_pair_fit_v)
    mu_gap_abs_error = float(abs(mu_gap_fit - mu_gap_true))
    if abs(mu_gap_true) < 1e-12:
        mu_gap_sign_correct = bool(abs(mu_gap_fit) <= 5e-2)
    else:
        mu_gap_sign_correct = bool((mu_gap_true * mu_gap_fit) > 0.0)

    learn: dict[str, Any] = {
        "kind": "mu_stack_pair",
        "mu_ground_true": float(mu_ground_true),
        "mu_pair_true": float(mu_pair_true),
        "mu_ground_fit": float(mu_ground_fit_v),
        "mu_pair_fit": float(mu_pair_fit_v),
        "mu_ground_abs_error": float(abs(mu_ground_fit_v - mu_ground_true)),
        "mu_pair_abs_error": float(abs(mu_pair_fit_v - mu_pair_true)),
        "mu_gap_true": float(mu_gap_true),
        "mu_gap_fit": float(mu_gap_fit),
        "mu_gap_abs_error": float(mu_gap_abs_error),
        "mu_gap_sign_correct": bool(mu_gap_sign_correct),
        "baseline_loss": baseline_loss_v,
        "fit_loss": fit_loss_v,
        "loss_ratio": float(fit_loss_v / (baseline_loss_v + eps)),
        "baseline_loss_val": baseline_loss_val_v,
        "fit_loss_val": fit_loss_val_v,
        "loss_ratio_val": loss_ratio_val,
        "ood_enabled": bool(ood_case_results),
        "ood_case_count": int(ood_case_count),
        "baseline_loss_ood": baseline_loss_ood_v,
        "fit_loss_ood": fit_loss_ood_v,
        "loss_ratio_ood": loss_ratio_ood,
        "loss_ratio_ood_worst": loss_ratio_ood_worst,
        "ood_cases": ood_case_results,
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
        "eval_val": eval_val,
        "eval_ood": eval_ood,
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
        "mu_ground_abs_error": float(abs(mu_ground_fit_v - mu_ground_true)),
        "mu_pair_abs_error": float(abs(mu_pair_fit_v - mu_pair_true)),
        "mu_gap_abs_error": float(mu_gap_abs_error),
        "loss_ratio_val": float(loss_ratio_val),
        "loss_ratio_ood": float(loss_ratio_ood) if loss_ratio_ood is not None else 0.0,
        "eval_val_solver_max_iter_ratio": float(eval_val["solver"]["max_iter_ratio"]),
        "eval_val_failsafe_soft_triggered_any": float(bool(eval_val["failsafe"]["soft_triggered_any"])),
        "eval_val_failsafe_triggered_any": float(bool(eval_val["failsafe"]["triggered_any"])),
    }
    if eval_ood is not None:
        metrics["eval_ood_solver_max_iter_ratio"] = float(eval_ood["solver"]["max_iter_ratio"])
        metrics["eval_ood_failsafe_soft_triggered_any"] = float(bool(eval_ood["failsafe"]["soft_triggered_any"]))
        metrics["eval_ood_failsafe_triggered_any"] = float(bool(eval_ood["failsafe"]["triggered_any"]))
    return PairTaskResult(summary=summary, metrics=metrics)


def _run_field_task(
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
) -> FieldTaskResult:
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

    mu_init = float(task.get("mu_init", 0.2))
    mu_max = float(task.get("mu_max", 1.0))
    mlp_hidden = int(task.get("mlp_hidden", 16))
    lr = float(task.get("lr", 0.2))

    rollout_steps = _clamp_int(int(task.get("rollout_steps", 8)), max_rollout_steps)
    train_iters = _clamp_int(int(task.get("train_iters", 100)), max_train_iters)
    dt = float(task.get("dt", config.get("dt", 0.01)))
    n_train = _clamp_int(int(task.get("n_train", 8)), max_n_train)
    n_val = _clamp_int(int(task.get("n_val", 8)), max_n_val)

    omega_range = _as_range(task.get("omega_range", (3.0, 8.0)))
    vx_range = _as_range(task.get("vx_range", (0.0, 0.2)))
    vx_bottom_range = _as_range(task.get("vx_bottom_range", (2.0, 3.0)))
    val_omega_range = _as_range(task.get("val_omega_range", (1.0, 4.0)))
    val_vx_range = _as_range(task.get("val_vx_range", (0.0, 0.2)))
    val_vx_bottom_range = _as_range(task.get("val_vx_bottom_range", (2.0, 3.0)))

    rule_cfg = task.get("true_rule", {})
    if not isinstance(rule_cfg, dict):
        rule_cfg = {}
    base_ground = float(rule_cfg.get("base_ground", 0.55))
    base_pair = float(rule_cfg.get("base_pair", 0.22))
    speed_gain = float(rule_cfg.get("speed_gain", 0.18))
    speed_scale = float(rule_cfg.get("speed_scale", 2.5))
    vn_gain = float(rule_cfg.get("vn_gain", 0.08))
    pen_gain = float(rule_cfg.get("pen_gain", 0.06))
    mu_min = float(rule_cfg.get("mu_min", 0.02))
    material_bias_raw = rule_cfg.get("material_bias", [])
    if isinstance(material_bias_raw, (list, tuple)):
        material_bias = tuple(float(x) for x in material_bias_raw)
    else:
        material_bias = tuple()

    material_cfg = task.get("material", {})
    if not isinstance(material_cfg, dict):
        material_cfg = {}
    use_material_features = bool(material_cfg.get("enabled", False))
    max_material_id = int(material_cfg.get("max_material_id", 4))
    ground_material_id = int(material_cfg.get("ground_material_id", 0 if use_material_features else 0))
    body_pool_raw = material_cfg.get("body_material_pool", [1, 2, 3] if use_material_features else [0])
    if isinstance(body_pool_raw, (list, tuple)):
        body_material_pool = tuple(int(x) for x in body_pool_raw)
    else:
        body_material_pool = (0,)
    train_pool_raw = material_cfg.get("train_body_material_pool", body_material_pool)
    if isinstance(train_pool_raw, (list, tuple)):
        train_body_material_pool = tuple(int(x) for x in train_pool_raw)
    else:
        train_body_material_pool = body_material_pool
    val_pool_raw = material_cfg.get("val_body_material_pool", train_body_material_pool)
    if isinstance(val_pool_raw, (list, tuple)):
        val_body_material_pool = tuple(int(x) for x in val_pool_raw)
    else:
        val_body_material_pool = train_body_material_pool

    patterns_fallback = material_cfg.get("body_material_patterns", None)
    train_patterns_raw = material_cfg.get("train_body_material_patterns", patterns_fallback)
    val_patterns_raw = material_cfg.get("val_body_material_patterns", train_patterns_raw)
    train_body_material_patterns = _as_material_patterns(train_patterns_raw)
    val_body_material_patterns = _as_material_patterns(val_patterns_raw)
    if not use_material_features:
        body_material_pool = (0,)
        train_body_material_pool = (0,)
        val_body_material_pool = (0,)
        train_body_material_patterns = tuple()
        val_body_material_patterns = tuple()
        ground_material_id = 0
    if len(body_material_pool) == 0:
        body_material_pool = (0,)
    if len(train_body_material_pool) == 0:
        train_body_material_pool = body_material_pool
    if len(val_body_material_pool) == 0:
        val_body_material_pool = train_body_material_pool

    def _validate_material_id(material_id: int, *, where: str) -> None:
        if material_id < 0 or material_id > max_material_id:
            raise ValueError(f"material id {material_id} out of range [0, {max_material_id}] in {where}")

    if max_material_id < 0:
        raise ValueError("task.material.max_material_id must be non-negative")
    for material_id in body_material_pool:
        _validate_material_id(int(material_id), where="material.body_material_pool")
    for material_id in train_body_material_pool:
        _validate_material_id(int(material_id), where="material.train_body_material_pool")
    for material_id in val_body_material_pool:
        _validate_material_id(int(material_id), where="material.val_body_material_pool")
    _validate_material_id(int(ground_material_id), where="material.ground_material_id")
    for pattern in train_body_material_patterns:
        for material_id in pattern:
            _validate_material_id(int(material_id), where="material.train_body_material_patterns")
    for pattern in val_body_material_patterns:
        for material_id in pattern:
            _validate_material_id(int(material_id), where="material.val_body_material_patterns")

    train_body_material_set = _material_body_set(
        body_pool=train_body_material_pool,
        body_patterns=train_body_material_patterns,
    )
    val_body_material_set = _material_body_set(
        body_pool=val_body_material_pool,
        body_patterns=val_body_material_patterns,
    )
    all_body_material_set = _material_body_set(
        body_pool=body_material_pool,
        body_patterns=tuple(),
    )
    all_body_material_set.update(train_body_material_set)
    all_body_material_set.update(val_body_material_set)
    if not all_body_material_set:
        all_body_material_set = {0}

    train_material_pair_set = _material_pair_set(
        body_pool=train_body_material_pool,
        body_patterns=train_body_material_patterns,
        ground_material_id=ground_material_id,
    )

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
        "pgs_relaxation": float(engine_cfg.get("pgs_relaxation", 1.0)),
        "baumgarte_beta": float(engine_cfg.get("baumgarte_beta", 0.2)),
        "residual_tol": float(engine_cfg.get("residual_tol", 1e-6)),
        "warm_start": bool(engine_cfg.get("warm_start", True)),
    }

    field_mod = _load_module(
        Path("scripts/train_m2_mu_pgs_stack_field.py"), name="scone_scripts_train_m2_mu_pgs_stack_field"
    )

    mu_true_fn = field_mod.make_true_mu_rule(
        device=device,
        dtype=dtype,
        mu_max=mu_max,
        base_ground=base_ground,
        base_pair=base_pair,
        speed_gain=speed_gain,
        speed_scale=speed_scale,
        vn_gain=vn_gain,
        pen_gain=pen_gain,
        mu_min=mu_min,
        material_bias=material_bias if (use_material_features and len(material_bias) > 0) else None,
        max_material_id=max_material_id,
    )

    dataset_train = field_mod._make_dataset(
        device=device,
        dtype=dtype,
        seed=seed,
        n=n_train,
        steps=rollout_steps,
        dt=dt,
        mu_true_fn=mu_true_fn,
        omega_range=omega_range,
        vx_range=vx_range,
        vx_bottom_range=vx_bottom_range,
        engine_params=engine_params,
        body_material_pool=train_body_material_pool,
        body_material_patterns=train_body_material_patterns,
        ground_material_id=ground_material_id,
    )
    dataset_val = field_mod._make_dataset(
        device=device,
        dtype=dtype,
        seed=seed + 1,
        n=n_val,
        steps=rollout_steps,
        dt=dt,
        mu_true_fn=mu_true_fn,
        omega_range=val_omega_range,
        vx_range=val_vx_range,
        vx_bottom_range=val_vx_bottom_range,
        engine_params=engine_params,
        body_material_pool=val_body_material_pool,
        body_material_patterns=val_body_material_patterns,
        ground_material_id=ground_material_id,
    )

    def build_engine(
        mu_default: float | torch.Tensor,
        mu_fn: Any | None,
        rollout_body_material_ids: tuple[int, ...],
        rollout_ground_material_id: int,
    ) -> Any:
        return field_mod._build_engine(
            device=device,
            dtype=dtype,
            mu_default=mu_default,
            mu_fn=mu_fn,
            body_material_ids=rollout_body_material_ids,
            ground_material_id=rollout_ground_material_id,
            **engine_params,
        )

    with torch.no_grad():
        baseline_loss = field_mod._dataset_loss_open_loop(
            dataset=dataset_train,
            build_engine_fn=build_engine,
            mu_default=mu_init,
            mu_fn=None,
            steps=rollout_steps,
            w_q=w_q,
            w_v=w_v,
            w_pen=w_pen,
            w_comp=w_comp,
            w_res=w_res,
        )
        baseline_loss_val = field_mod._dataset_loss_open_loop(
            dataset=dataset_val,
            build_engine_fn=build_engine,
            mu_default=mu_init,
            mu_fn=None,
            steps=rollout_steps,
            w_q=w_q,
            w_v=w_v,
            w_pen=w_pen,
            w_comp=w_comp,
            w_res=w_res,
        )

    ood_runs: list[dict[str, Any]] = []
    ood_cases_cfg = task.get("ood_cases")
    if isinstance(ood_cases_cfg, list) and ood_cases_cfg:
        for idx, ood_case_cfg in enumerate(ood_cases_cfg):
            if not isinstance(ood_case_cfg, dict):
                continue
            if not bool(ood_case_cfg.get("enabled", True)):
                continue

            case_name = str(ood_case_cfg.get("name", f"ood_case_{idx}"))
            dt_case = float(ood_case_cfg.get("dt", dt))
            n_case = _clamp_int(int(ood_case_cfg.get("n", n_val)), max_n_val)
            steps_case = _clamp_int(int(ood_case_cfg.get("steps", rollout_steps)), max_rollout_steps)
            omega_case = _as_range(ood_case_cfg.get("omega_range", val_omega_range))
            vx_case = _as_range(ood_case_cfg.get("vx_range", val_vx_range))
            vx_bottom_case = _as_range(ood_case_cfg.get("vx_bottom_range", val_vx_bottom_range))

            engine_case = dict(engine_params)
            overrides = ood_case_cfg.get("engine_overrides", {})
            if isinstance(overrides, dict):
                for k, v in overrides.items():
                    if k in engine_case:
                        engine_case[k] = v

            case_material_overrides = ood_case_cfg.get("material_overrides", {})
            if not isinstance(case_material_overrides, dict):
                case_material_overrides = {}
            case_ground_material_id = int(case_material_overrides.get("ground_material_id", ground_material_id))
            case_pool_raw = case_material_overrides.get("body_material_pool", val_body_material_pool)
            if isinstance(case_pool_raw, (list, tuple)):
                case_body_material_pool = tuple(int(x) for x in case_pool_raw)
            else:
                case_body_material_pool = val_body_material_pool
            case_patterns_raw = case_material_overrides.get("body_material_patterns", val_body_material_patterns)
            case_body_material_patterns = _as_material_patterns(case_patterns_raw)
            if not use_material_features:
                case_ground_material_id = 0
                case_body_material_pool = (0,)
                case_body_material_patterns = tuple()
            if len(case_body_material_pool) == 0:
                case_body_material_pool = val_body_material_pool
            _validate_material_id(int(case_ground_material_id), where=f"task.ood_cases[{idx}].material_overrides.ground_material_id")
            for material_id in case_body_material_pool:
                _validate_material_id(
                    int(material_id), where=f"task.ood_cases[{idx}].material_overrides.body_material_pool"
                )
            for pattern in case_body_material_patterns:
                for material_id in pattern:
                    _validate_material_id(
                        int(material_id),
                        where=f"task.ood_cases[{idx}].material_overrides.body_material_patterns",
                    )

            case_body_material_set = _material_body_set(
                body_pool=case_body_material_pool,
                body_patterns=case_body_material_patterns,
            )
            if not case_body_material_set:
                case_body_material_set = {0}
            case_material_pair_set = _material_pair_set(
                body_pool=case_body_material_pool,
                body_patterns=case_body_material_patterns,
                ground_material_id=case_ground_material_id,
            )
            case_unseen_material_ids = sorted(int(x) for x in (case_body_material_set - train_body_material_set))
            case_holdout_pairs = sorted(case_material_pair_set - train_material_pair_set)

            dataset_case = field_mod._make_dataset(
                device=device,
                dtype=dtype,
                seed=seed + 2 + idx,
                n=n_case,
                steps=steps_case,
                dt=dt_case,
                mu_true_fn=mu_true_fn,
                omega_range=omega_case,
                vx_range=vx_case,
                vx_bottom_range=vx_bottom_case,
                engine_params=engine_case,
                body_material_pool=case_body_material_pool,
                body_material_patterns=case_body_material_patterns,
                ground_material_id=case_ground_material_id,
            )
            ood_runs.append(
                {
                    "name": case_name,
                    "dt": float(dt_case),
                    "steps": int(steps_case),
                    "dataset": dataset_case,
                    "engine": engine_case,
                    "body_material_pool": [int(x) for x in case_body_material_pool],
                    "body_material_patterns": [list(map(int, p)) for p in case_body_material_patterns],
                    "ground_material_id": int(case_ground_material_id),
                    "body_material_set": [int(x) for x in sorted(case_body_material_set)],
                    "material_pair_set": [[int(a), int(b)] for (a, b) in sorted(case_material_pair_set)],
                    "unseen_material_ids": [int(x) for x in case_unseen_material_ids],
                    "holdout_material_pairs": [[int(a), int(b)] for (a, b) in case_holdout_pairs],
                }
            )
    else:
        ood_cfg = task.get("ood", {})
        if isinstance(ood_cfg, dict) and bool(ood_cfg.get("enabled", False)):
            dt_case = float(ood_cfg.get("dt", dt))
            n_case = _clamp_int(int(ood_cfg.get("n", n_val)), max_n_val)
            steps_case = _clamp_int(int(ood_cfg.get("steps", rollout_steps)), max_rollout_steps)
            omega_case = _as_range(ood_cfg.get("omega_range", val_omega_range))
            vx_case = _as_range(ood_cfg.get("vx_range", val_vx_range))
            vx_bottom_case = _as_range(ood_cfg.get("vx_bottom_range", val_vx_bottom_range))

            engine_case = dict(engine_params)
            overrides = ood_cfg.get("engine_overrides", {})
            if isinstance(overrides, dict):
                for k, v in overrides.items():
                    if k in engine_case:
                        engine_case[k] = v

            case_material_overrides = ood_cfg.get("material_overrides", {})
            if not isinstance(case_material_overrides, dict):
                case_material_overrides = {}
            case_ground_material_id = int(case_material_overrides.get("ground_material_id", ground_material_id))
            case_pool_raw = case_material_overrides.get("body_material_pool", val_body_material_pool)
            if isinstance(case_pool_raw, (list, tuple)):
                case_body_material_pool = tuple(int(x) for x in case_pool_raw)
            else:
                case_body_material_pool = val_body_material_pool
            case_patterns_raw = case_material_overrides.get("body_material_patterns", val_body_material_patterns)
            case_body_material_patterns = _as_material_patterns(case_patterns_raw)
            if not use_material_features:
                case_ground_material_id = 0
                case_body_material_pool = (0,)
                case_body_material_patterns = tuple()
            if len(case_body_material_pool) == 0:
                case_body_material_pool = val_body_material_pool
            _validate_material_id(int(case_ground_material_id), where="task.ood.material_overrides.ground_material_id")
            for material_id in case_body_material_pool:
                _validate_material_id(int(material_id), where="task.ood.material_overrides.body_material_pool")
            for pattern in case_body_material_patterns:
                for material_id in pattern:
                    _validate_material_id(int(material_id), where="task.ood.material_overrides.body_material_patterns")

            case_body_material_set = _material_body_set(
                body_pool=case_body_material_pool,
                body_patterns=case_body_material_patterns,
            )
            if not case_body_material_set:
                case_body_material_set = {0}
            case_material_pair_set = _material_pair_set(
                body_pool=case_body_material_pool,
                body_patterns=case_body_material_patterns,
                ground_material_id=case_ground_material_id,
            )
            case_unseen_material_ids = sorted(int(x) for x in (case_body_material_set - train_body_material_set))
            case_holdout_pairs = sorted(case_material_pair_set - train_material_pair_set)

            dataset_case = field_mod._make_dataset(
                device=device,
                dtype=dtype,
                seed=seed + 2,
                n=n_case,
                steps=steps_case,
                dt=dt_case,
                mu_true_fn=mu_true_fn,
                omega_range=omega_case,
                vx_range=vx_case,
                vx_bottom_range=vx_bottom_case,
                engine_params=engine_case,
                body_material_pool=case_body_material_pool,
                body_material_patterns=case_body_material_patterns,
                ground_material_id=case_ground_material_id,
            )
            ood_runs.append(
                {
                    "name": "ood",
                    "dt": float(dt_case),
                    "steps": int(steps_case),
                    "dataset": dataset_case,
                    "engine": engine_case,
                    "body_material_pool": [int(x) for x in case_body_material_pool],
                    "body_material_patterns": [list(map(int, p)) for p in case_body_material_patterns],
                    "ground_material_id": int(case_ground_material_id),
                    "body_material_set": [int(x) for x in sorted(case_body_material_set)],
                    "material_pair_set": [[int(a), int(b)] for (a, b) in sorted(case_material_pair_set)],
                    "unseen_material_ids": [int(x) for x in case_unseen_material_ids],
                    "holdout_material_pairs": [[int(a), int(b)] for (a, b) in case_holdout_pairs],
                }
            )

    if use_material_features:
        model = ContactMuMaterialMLP(
            init_mu=mu_init,
            mu_max=mu_max,
            hidden_dim=mlp_hidden,
            max_material_id=max_material_id,
            device=device,
            dtype=dtype,
        )
    else:
        model = ContactMuMLP(init_mu=mu_init, mu_max=mu_max, hidden_dim=mlp_hidden, device=device, dtype=dtype)
    model_mu_fn = model.from_contact_features
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(train_iters):
        opt.zero_grad(set_to_none=True)
        loss = field_mod._dataset_loss_open_loop(
            dataset=dataset_train,
            build_engine_fn=build_engine,
            mu_default=mu_init,
            mu_fn=model_mu_fn,
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
        fit_loss = field_mod._dataset_loss_open_loop(
            dataset=dataset_train,
            build_engine_fn=build_engine,
            mu_default=mu_init,
            mu_fn=model_mu_fn,
            steps=rollout_steps,
            w_q=w_q,
            w_v=w_v,
            w_pen=w_pen,
            w_comp=w_comp,
            w_res=w_res,
        )
        fit_loss_val = field_mod._dataset_loss_open_loop(
            dataset=dataset_val,
            build_engine_fn=build_engine,
            mu_default=mu_init,
            mu_fn=model_mu_fn,
            steps=rollout_steps,
            w_q=w_q,
            w_v=w_v,
            w_pen=w_pen,
            w_comp=w_comp,
            w_res=w_res,
        )
        probe_material_set: set[int] = set(int(x) for x in all_body_material_set)
        train_probe_material_set: set[int] = set(int(x) for x in train_body_material_set)
        if not train_probe_material_set:
            train_probe_material_set = {0}

        holdout_material_set: set[int] = set()
        holdout_material_pair_set: set[tuple[int, int]] = set()
        for ood_case in ood_runs:
            probe_material_set.update(int(x) for x in ood_case.get("body_material_set", []))
            holdout_material_set.update(int(x) for x in ood_case.get("unseen_material_ids", []))
            for pair_values in ood_case.get("holdout_material_pairs", []):
                if not isinstance(pair_values, (list, tuple)) or len(pair_values) != 2:
                    continue
                holdout_material_pair_set.add(
                    _canonical_material_pair(int(pair_values[0]), int(pair_values[1]))
                )
        if not probe_material_set:
            probe_material_set = {0}
        mu_probe_mae_t = field_mod.probe_mu_mae(
            device=device,
            dtype=dtype,
            mu_pred_fn=model_mu_fn,
            mu_true_fn=mu_true_fn,
            material_ids=tuple(sorted(probe_material_set)),
            ground_material_id=ground_material_id,
        )
        mu_probe_mae_seen_t = field_mod.probe_mu_mae(
            device=device,
            dtype=dtype,
            mu_pred_fn=model_mu_fn,
            mu_true_fn=mu_true_fn,
            material_ids=tuple(sorted(train_probe_material_set)),
            ground_material_id=ground_material_id,
        )
        mu_probe_mae_holdout_material_t = (
            field_mod.probe_mu_mae(
                device=device,
                dtype=dtype,
                mu_pred_fn=model_mu_fn,
                mu_true_fn=mu_true_fn,
                material_ids=tuple(sorted(holdout_material_set)),
                ground_material_id=ground_material_id,
            )
            if holdout_material_set
            else None
        )
        mu_probe_pair_mae_seen_t = _probe_mu_mae_on_material_pairs(
            device=device,
            dtype=dtype,
            mu_pred_fn=model_mu_fn,
            mu_true_fn=mu_true_fn,
            material_pairs=train_material_pair_set,
            ground_material_id=ground_material_id,
        )
        mu_probe_pair_mae_holdout_t = _probe_mu_mae_on_material_pairs(
            device=device,
            dtype=dtype,
            mu_pred_fn=model_mu_fn,
            mu_true_fn=mu_true_fn,
            material_pairs=holdout_material_pair_set,
            ground_material_id=ground_material_id,
        )

        ood_case_results: list[dict[str, Any]] = []
        for ood_case in ood_runs:
            dataset_ood = ood_case["dataset"]
            engine_ood = ood_case["engine"]
            steps_ood = int(ood_case["steps"])

            def build_engine_ood(
                mu_default: float | torch.Tensor,
                mu_fn: Any | None,
                rollout_body_material_ids: tuple[int, ...],
                rollout_ground_material_id: int,
            ) -> Any:
                return field_mod._build_engine(
                    device=device,
                    dtype=dtype,
                    mu_default=mu_default,
                    mu_fn=mu_fn,
                    body_material_ids=rollout_body_material_ids,
                    ground_material_id=rollout_ground_material_id,
                    **engine_ood,
                )

            baseline_ood_t = field_mod._dataset_loss_open_loop(
                dataset=dataset_ood,
                build_engine_fn=build_engine_ood,
                mu_default=mu_init,
                mu_fn=None,
                steps=steps_ood,
                w_q=w_q,
                w_v=w_v,
                w_pen=w_pen,
                w_comp=w_comp,
                w_res=w_res,
            )
            fit_ood_t = field_mod._dataset_loss_open_loop(
                dataset=dataset_ood,
                build_engine_fn=build_engine_ood,
                mu_default=mu_init,
                mu_fn=model_mu_fn,
                steps=steps_ood,
                w_q=w_q,
                w_v=w_v,
                w_pen=w_pen,
                w_comp=w_comp,
                w_res=w_res,
            )

            def build_engine_fit_ood(rollout: Any) -> Any:
                rollout_materials = tuple(int(x) for x in getattr(rollout, "body_material_ids", (0, 0, 0)))
                rollout_ground_material = int(getattr(rollout, "ground_material_id", 0))
                return field_mod._build_engine(
                    device=device,
                    dtype=dtype,
                    mu_default=mu_init,
                    mu_fn=model_mu_fn,
                    body_material_ids=rollout_materials,
                    ground_material_id=rollout_ground_material,
                    **engine_ood,
                )

            eval_ood_case = _collect_open_loop_diag(
                dataset=dataset_ood,
                build_engine_for_rollout_fn=build_engine_fit_ood,
                steps=steps_ood,
            )

            baseline_ood_v = float(baseline_ood_t.detach().cpu().item())
            fit_ood_v = float(fit_ood_t.detach().cpu().item())
            ood_case_results.append(
                {
                    "name": str(ood_case["name"]),
                    "dt": float(ood_case["dt"]),
                    "body_material_set": [int(x) for x in ood_case.get("body_material_set", [])],
                    "material_pair_set": [list(map(int, p)) for p in ood_case.get("material_pair_set", [])],
                    "unseen_material_ids": [int(x) for x in ood_case.get("unseen_material_ids", [])],
                    "holdout_material_pairs": [list(map(int, p)) for p in ood_case.get("holdout_material_pairs", [])],
                    "unseen_material_count": int(len(ood_case.get("unseen_material_ids", []))),
                    "holdout_pair_count": int(len(ood_case.get("holdout_material_pairs", []))),
                    "baseline_loss": baseline_ood_v,
                    "fit_loss": fit_ood_v,
                    "loss_ratio": float(fit_ood_v / (baseline_ood_v + 1e-12)),
                    "eval": eval_ood_case,
                }
            )

    eps = 1e-12
    baseline_loss_v = float(baseline_loss.detach().cpu().item())
    fit_loss_v = float(fit_loss.detach().cpu().item())
    baseline_loss_val_v = float(baseline_loss_val.detach().cpu().item())
    fit_loss_val_v = float(fit_loss_val.detach().cpu().item())
    loss_ratio = float(fit_loss_v / (baseline_loss_v + eps))
    loss_ratio_val = float(fit_loss_val_v / (baseline_loss_val_v + eps))
    mu_probe_mae = float(mu_probe_mae_t.detach().cpu().item())
    mu_probe_mae_seen = float(mu_probe_mae_seen_t.detach().cpu().item())
    mu_probe_mae_holdout_material = (
        float(mu_probe_mae_holdout_material_t.detach().cpu().item()) if mu_probe_mae_holdout_material_t is not None else None
    )
    mu_probe_pair_mae_seen = (
        float(mu_probe_pair_mae_seen_t.detach().cpu().item()) if mu_probe_pair_mae_seen_t is not None else None
    )
    mu_probe_pair_mae_holdout = (
        float(mu_probe_pair_mae_holdout_t.detach().cpu().item()) if mu_probe_pair_mae_holdout_t is not None else None
    )
    mu_probe_pair_holdout_ratio = None
    mu_probe_pair_holdout_gap = None
    if mu_probe_pair_mae_seen is not None and mu_probe_pair_mae_holdout is not None:
        mu_probe_pair_holdout_ratio = float(mu_probe_pair_mae_holdout / (mu_probe_pair_mae_seen + eps))
        mu_probe_pair_holdout_gap = float(mu_probe_pair_mae_holdout - mu_probe_pair_mae_seen)

    baseline_loss_ood_v = None
    fit_loss_ood_v = None
    loss_ratio_ood = None
    dt_ood = None
    ood_case_count = int(len(ood_case_results))
    loss_ratio_ood_worst = None
    if ood_case_results:
        worst_case = max(ood_case_results, key=lambda x: float(x["loss_ratio"]))
        baseline_loss_ood_v = float(worst_case["baseline_loss"])
        fit_loss_ood_v = float(worst_case["fit_loss"])
        loss_ratio_ood = float(worst_case["loss_ratio"])
        loss_ratio_ood_worst = float(loss_ratio_ood)
        dt_ood = float(worst_case["dt"])

    def build_engine_fit(rollout: Any) -> Any:
        rollout_materials = tuple(int(x) for x in getattr(rollout, "body_material_ids", (0, 0, 0)))
        rollout_ground_material = int(getattr(rollout, "ground_material_id", 0))
        return field_mod._build_engine(
            device=device,
            dtype=dtype,
            mu_default=mu_init,
            mu_fn=model_mu_fn,
            body_material_ids=rollout_materials,
            ground_material_id=rollout_ground_material,
            **engine_params,
        )

    eval_val = _collect_open_loop_diag(
        dataset=dataset_val,
        build_engine_for_rollout_fn=build_engine_fit,
        steps=rollout_steps,
    )
    eval_ood = None
    if ood_case_results:
        max_iter_ratio_worst = max(float(case["eval"]["solver"]["max_iter_ratio"]) for case in ood_case_results)
        diverged_ratio_worst = max(float(case["eval"]["solver"].get("diverged_ratio", 0.0)) for case in ood_case_results)
        na_ratio_worst = max(float(case["eval"]["solver"].get("na_ratio", 0.0)) for case in ood_case_results)
        residual_worst = max(float(case["eval"]["solver"]["residual_max"]) for case in ood_case_results)
        hard_any = any(bool(case["eval"]["failsafe"]["triggered_any"]) for case in ood_case_results)
        soft_any = any(bool(case["eval"]["failsafe"]["soft_triggered_any"]) for case in ood_case_results)
        alpha_drop_max_worst = max(float(case["eval"]["failsafe"].get("alpha_blend_drop_max", 0.0)) for case in ood_case_results)
        alpha_drop_mean_worst = max(
            float(case["eval"]["failsafe"].get("alpha_blend_drop_mean", 0.0)) for case in ood_case_results
        )
        worst_by_solver = max(ood_case_results, key=lambda x: float(x["eval"]["solver"]["max_iter_ratio"]))

        eval_ood = {
            "case_count": int(len(ood_case_results)),
            "worst_case": str(worst_by_solver["name"]),
            "solver": {
                "max_iter_ratio": float(max_iter_ratio_worst),
                "diverged_ratio": float(diverged_ratio_worst),
                "na_ratio": float(na_ratio_worst),
                "residual_max": float(residual_worst),
            },
            "failsafe": {
                "triggered_any": bool(hard_any),
                "soft_triggered_any": bool(soft_any),
                "alpha_blend_drop_max": float(alpha_drop_max_worst),
                "alpha_blend_drop_mean": float(alpha_drop_mean_worst),
            },
            "cases": [
                {
                    "name": str(case["name"]),
                    "dt": float(case["dt"]),
                    "loss_ratio": float(case["loss_ratio"]),
                    "eval": case["eval"],
                }
                for case in ood_case_results
            ],
        }

    ood_case_metrics: dict[str, dict[str, Any]] = {}
    for case in ood_case_results:
        name = str(case["name"])
        case_eval = case["eval"] if isinstance(case.get("eval"), dict) else {}
        case_solver = case_eval.get("solver", {}) if isinstance(case_eval, dict) else {}
        case_failsafe = case_eval.get("failsafe", {}) if isinstance(case_eval, dict) else {}
        ood_case_metrics[name] = {
            "loss_ratio": float(case.get("loss_ratio", 0.0)),
            "solver_max_iter_ratio": float(case_solver.get("max_iter_ratio", 0.0)),
            "solver_residual_max": float(case_solver.get("residual_max", 0.0)),
            "failsafe_triggered_any": bool(case_failsafe.get("triggered_any", False)),
            "failsafe_soft_triggered_any": bool(case_failsafe.get("soft_triggered_any", False)),
            "unseen_material_count": int(case.get("unseen_material_count", 0)),
            "holdout_pair_count": int(case.get("holdout_pair_count", 0)),
        }

    learn: dict[str, Any] = {
        "kind": "mu_stack_field",
        "mu_init": float(mu_init),
        "mu_max": float(mu_max),
        "mlp_hidden": int(mlp_hidden),
        "use_material_features": bool(use_material_features),
        "max_material_id": int(max_material_id),
        "ground_material_id": int(ground_material_id),
        "body_material_pool": [int(x) for x in body_material_pool],
        "train_body_material_pool": [int(x) for x in train_body_material_pool],
        "val_body_material_pool": [int(x) for x in val_body_material_pool],
        "train_body_material_patterns": [list(map(int, p)) for p in train_body_material_patterns],
        "val_body_material_patterns": [list(map(int, p)) for p in val_body_material_patterns],
        "train_body_material_set": [int(x) for x in sorted(train_body_material_set)],
        "val_body_material_set": [int(x) for x in sorted(val_body_material_set)],
        "all_body_material_set": [int(x) for x in sorted(probe_material_set)],
        "holdout_material_set": [int(x) for x in sorted(holdout_material_set)],
        "train_material_pair_count": int(len(train_material_pair_set)),
        "holdout_material_pair_count": int(len(holdout_material_pair_set)),
        "holdout_material_pairs": [[int(a), int(b)] for (a, b) in sorted(holdout_material_pair_set)],
        "mu_probe_mae": float(mu_probe_mae),
        "mu_probe_mae_seen": float(mu_probe_mae_seen),
        "mu_probe_mae_holdout_material": mu_probe_mae_holdout_material,
        "mu_probe_pair_mae_seen": mu_probe_pair_mae_seen,
        "mu_probe_pair_mae_holdout": mu_probe_pair_mae_holdout,
        "mu_probe_pair_holdout_ratio": mu_probe_pair_holdout_ratio,
        "mu_probe_pair_holdout_gap": mu_probe_pair_holdout_gap,
        "baseline_loss": baseline_loss_v,
        "fit_loss": fit_loss_v,
        "loss_ratio": loss_ratio,
        "baseline_loss_val": baseline_loss_val_v,
        "fit_loss_val": fit_loss_val_v,
        "loss_ratio_val": loss_ratio_val,
        "ood_enabled": bool(ood_case_results),
        "ood_case_count": int(ood_case_count),
        "baseline_loss_ood": baseline_loss_ood_v,
        "fit_loss_ood": fit_loss_ood_v,
        "loss_ratio_ood": loss_ratio_ood,
        "loss_ratio_ood_worst": loss_ratio_ood_worst,
        "ood_cases": ood_case_results,
        "ood_case_metrics": ood_case_metrics,
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
        "eval_val": eval_val,
        "eval_ood": eval_ood,
    }

    summary: dict[str, Any] = {
        "demo": demo,
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype),
        "learn": learn,
    }
    metrics = {
        "mu_probe_mae": float(mu_probe_mae),
        "mu_probe_mae_seen": float(mu_probe_mae_seen),
        "loss_ratio_val": float(loss_ratio_val),
        "loss_ratio_ood": float(loss_ratio_ood) if loss_ratio_ood is not None else 0.0,
        "eval_val_solver_max_iter_ratio": float(eval_val["solver"]["max_iter_ratio"]),
        "eval_val_failsafe_soft_triggered_any": float(bool(eval_val["failsafe"]["soft_triggered_any"])),
        "eval_val_failsafe_triggered_any": float(bool(eval_val["failsafe"]["triggered_any"])),
    }
    if mu_probe_mae_holdout_material is not None:
        metrics["mu_probe_mae_holdout_material"] = float(mu_probe_mae_holdout_material)
    if mu_probe_pair_mae_seen is not None:
        metrics["mu_probe_pair_mae_seen"] = float(mu_probe_pair_mae_seen)
    if mu_probe_pair_mae_holdout is not None:
        metrics["mu_probe_pair_mae_holdout"] = float(mu_probe_pair_mae_holdout)
    if mu_probe_pair_holdout_ratio is not None:
        metrics["mu_probe_pair_holdout_ratio"] = float(mu_probe_pair_holdout_ratio)
    if eval_ood is not None:
        metrics["eval_ood_solver_max_iter_ratio"] = float(eval_ood["solver"]["max_iter_ratio"])
        metrics["eval_ood_failsafe_soft_triggered_any"] = float(bool(eval_ood["failsafe"]["soft_triggered_any"]))
        metrics["eval_ood_failsafe_triggered_any"] = float(bool(eval_ood["failsafe"]["triggered_any"]))
    return FieldTaskResult(summary=summary, metrics=metrics)


def _default_out_dir() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / "bench" / f"{timestamp}-learn_eval"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        type=str,
        default="mu_scalar",
        choices=["mu_scalar", "mu_pair", "mu_field", "mu_field_gate", "mu_all"],
    )
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
        elif str(args.suite) == "mu_field":
            config_paths = sorted(Path("configs").glob("learn_mu_field_*.yaml"))
        elif str(args.suite) == "mu_field_gate":
            config_paths = [Path("configs/learn_mu_field_stack_material_holdout.yaml")]
        elif str(args.suite) == "mu_all":
            config_paths = sorted(Path("configs").glob("learn_mu_stack_*.yaml")) + sorted(
                Path("configs").glob("learn_mu_pair_*.yaml")
            ) + sorted(Path("configs").glob("learn_mu_field_*.yaml"))
        else:
            raise ValueError(f"Unknown suite: {args.suite}")
    if not config_paths:
        raise RuntimeError(
            "No learn configs found (expected configs/learn_mu_stack_*.yaml, "
            "configs/learn_mu_pair_*.yaml, and/or configs/learn_mu_field_*.yaml)"
        )

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
            elif kind == "mu_stack_field":
                result3 = _run_field_task(
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
                summary = result3.summary
                metrics = result3.metrics
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
            mu_abs_err = learn.get("mu_abs_error")
            if mu_abs_err is None:
                mu_abs_err = learn.get("mu_ground_abs_error")
            if mu_abs_err is None:
                mu_abs_err = learn.get("mu_probe_mae")
            ratio_val = learn.get("loss_ratio_val")
            ratio_ood = learn.get("loss_ratio_ood")
            extra = "" if ratio_ood is None else f" ratio_ood={float(ratio_ood):.3f}"
            print(f"{cfg} demo={demo} kind={kind} mu_abs_err={float(mu_abs_err):.3f} ratio_val={float(ratio_val):.3f}{extra}")


if __name__ == "__main__":
    main()
