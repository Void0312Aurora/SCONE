from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import torch
import yaml

from scone.learned.mu import ContactMuMLP, ContactMuMaterialMLP
from scone.state import State
from scone.utils.determinism import set_determinism


ContactMuFn = Callable[[dict[str, torch.Tensor]], torch.Tensor]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    w_pen: float
    w_comp: float
    w_res: float


@dataclass(frozen=True)
class TaskBundle:
    demo: str
    seed: int
    device: torch.device
    dtype: torch.dtype
    dt: float
    train_iters: int
    rollout_steps: int
    n_train: int
    n_val: int
    mu_init: float
    mu_max: float
    mlp_hidden: int
    lr: float
    w_q: float
    w_v: float
    use_material_features: bool
    max_material_id: int
    ground_material_id: int
    probe_material_ids: tuple[int, ...]
    calibration_material_ids: tuple[int, ...]
    engine_params: dict[str, float | int | bool]
    dataset_train: list[Any]
    dataset_val: list[Any]
    field_mod: Any
    mu_true_fn: ContactMuFn
    build_engine: Callable[[float | torch.Tensor, ContactMuFn | None, tuple[int, ...], int], Any]


def _timestamped_out_dir() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / "bench" / f"{timestamp}-p3-mu-interpret"


def _to_float(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().reshape(-1)[0].cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def _as_range(value: Any, fallback: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return float(fallback[0]), float(fallback[1])


def _as_material_patterns(value: Any) -> tuple[tuple[int, int, int], ...]:
    if value is None:
        return tuple()
    if not isinstance(value, (list, tuple)):
        raise TypeError("Material patterns must be a list/tuple")
    out: list[tuple[int, int, int]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise TypeError("Each material pattern must have exactly 3 ids")
        out.append((int(row[0]), int(row[1]), int(row[2])))
    return tuple(out)


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    return tuple(float(part) for part in parts)


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    return tuple(int(part) for part in parts)


def _material_id_set(
    *,
    material_pool: tuple[int, ...],
    patterns: tuple[tuple[int, int, int], ...],
) -> tuple[int, ...]:
    out: set[int] = {int(x) for x in material_pool}
    for pattern in patterns:
        out.update(int(x) for x in pattern)
    if not out:
        out = {0}
    return tuple(sorted(out))


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pearson_corr(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    x_center = xt - xt.mean()
    y_center = yt - yt.mean()
    denom = torch.sqrt((x_center * x_center).sum() * (y_center * y_center).sum())
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float(((x_center * y_center).sum() / denom).item())


def _rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda kv: kv[1])
    ranks = [0.0 for _ in values]
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def _spearman_rank_corr(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    return _pearson_corr(_rankdata(x), _rankdata(y))


def _linear_fit(x: list[float], y: list[float]) -> dict[str, float]:
    if len(x) != len(y) or len(x) == 0:
        return {"slope": 0.0, "intercept": 0.0, "r2": 0.0}
    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    x_mean = xt.mean()
    y_mean = yt.mean()
    denom = ((xt - x_mean) ** 2).sum()
    if float(denom.item()) <= 1e-12:
        intercept = float(y_mean.item())
        return {"slope": 0.0, "intercept": intercept, "r2": 0.0}
    slope = ((xt - x_mean) * (yt - y_mean)).sum() / denom
    intercept = y_mean - slope * x_mean
    y_hat = slope * xt + intercept
    ss_res = ((yt - y_hat) ** 2).sum()
    ss_tot = ((yt - y_mean) ** 2).sum()
    if float(ss_tot.item()) <= 1e-12:
        r2 = 1.0 if float(ss_res.item()) <= 1e-12 else 0.0
    else:
        r2 = float((1.0 - (ss_res / ss_tot)).item())
    return {"slope": float(slope.item()), "intercept": float(intercept.item()), "r2": r2}


def _collect_open_loop_physics(
    *,
    dataset: list[Any],
    build_engine_fn: Callable[[float | torch.Tensor, ContactMuFn | None, tuple[int, ...], int], Any],
    mu_default: float | torch.Tensor,
    mu_fn: ContactMuFn | None,
    steps: int,
) -> dict[str, float]:
    pen_max = 0.0
    residual_max = 0.0
    drift_rel_max = 0.0
    status_counts: dict[str, int] = {}
    failsafe_steps = 0

    for rollout in dataset:
        body_material_ids = tuple(int(x) for x in getattr(rollout, "body_material_ids", (0, 0, 0)))
        ground_material_id = int(getattr(rollout, "ground_material_id", 0))
        engine = build_engine_fn(mu_default, mu_fn, body_material_ids, ground_material_id)
        state = State(q=rollout.q[0].detach().clone(), v=rollout.v[0].detach().clone(), t=0.0)
        context: dict[str, object] = {"failsafe": {}}

        e0 = _to_float(engine.system.energy(state)[2], default=0.0)
        denom = max(1e-12, abs(e0))
        steps_eval = int(min(int(steps), int(rollout.q.shape[0] - 1)))
        for _ in range(steps_eval):
            state, diag = engine.step(state=state, dt=float(rollout.dt), context=context)
            contacts = diag.get("contacts", {})
            if isinstance(contacts, dict):
                pen_max = max(pen_max, _to_float(contacts.get("penetration_max"), default=0.0))
            solver = diag.get("solver", {})
            if isinstance(solver, dict):
                residual_max = max(residual_max, _to_float(solver.get("residual_max"), default=0.0))
                status = solver.get("status")
                if isinstance(status, str) and status:
                    status_counts[status] = status_counts.get(status, 0) + 1
            failsafe = diag.get("failsafe", {})
            if isinstance(failsafe, dict) and bool(failsafe.get("triggered", False)):
                failsafe_steps += 1
            e = _to_float(engine.system.energy(state)[2], default=e0)
            drift_rel_max = max(drift_rel_max, abs(e - e0) / denom)

    status_steps = int(sum(status_counts.values()))
    max_iter_steps = int(status_counts.get("max_iter", 0))
    diverged_steps = int(status_counts.get("diverged", 0))
    return {
        "penetration_max": float(pen_max),
        "solver_residual_max": float(residual_max),
        "energy_drift_rel_max": float(drift_rel_max),
        "solver_max_iter_ratio": float(max_iter_steps / max(1, status_steps)) if status_steps > 0 else 0.0,
        "solver_diverged_ratio": float(diverged_steps / max(1, status_steps)) if status_steps > 0 else 0.0,
        "failsafe_triggered_any": bool(failsafe_steps > 0),
    }


def _logged_mu_fn(mu_fn: ContactMuFn) -> tuple[ContactMuFn, list[float]]:
    values: list[float] = []

    def _wrapped(features: dict[str, torch.Tensor]) -> torch.Tensor:
        mu = mu_fn(features)
        values.append(_to_float(mu, default=0.0))
        return mu

    return _wrapped, values


def _single_disk_state(*, radius: float, vx0: float, device: torch.device, dtype: torch.dtype) -> State:
    q0 = torch.tensor([[0.0, float(radius), 0.0]], device=device, dtype=dtype)
    v0 = torch.tensor([[float(vx0), 0.0, 0.0]], device=device, dtype=dtype)
    return State(q=q0, v=v0, t=0.0)


def _simulate_sliding_distance(
    *,
    engine: Any,
    state0: State,
    dt: float,
    steps: int,
) -> tuple[float, bool]:
    state = state0
    context: dict[str, object] = {"failsafe": {}}
    x0 = _to_float(state.q[0, 0], default=0.0)
    failsafe_any = False
    for _ in range(int(steps)):
        state, diag = engine.step(state=state, dt=float(dt), context=context)
        failsafe = diag.get("failsafe", {})
        if isinstance(failsafe, dict):
            failsafe_any = failsafe_any or bool(failsafe.get("triggered", False))
    x_final = _to_float(state.q[0, 0], default=x0)
    return float(abs(x_final - x0)), bool(failsafe_any)


def _run_calibration_sweep(
    *,
    bundle: TaskBundle,
    mu_pred_fn: ContactMuFn,
    material_ids: tuple[int, ...],
    vx_values: tuple[float, ...],
    steps: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    radius = float(bundle.engine_params["radius"])
    for material_id in material_ids:
        for vx0 in vx_values:
            mu_true_logged, mu_true_values = _logged_mu_fn(bundle.mu_true_fn)
            mu_pred_logged, mu_pred_values = _logged_mu_fn(mu_pred_fn)
            engine_true = bundle.build_engine(bundle.mu_init, mu_true_logged, (int(material_id),), bundle.ground_material_id)
            engine_pred = bundle.build_engine(bundle.mu_init, mu_pred_logged, (int(material_id),), bundle.ground_material_id)

            state0 = _single_disk_state(radius=radius, vx0=float(vx0), device=bundle.device, dtype=bundle.dtype)
            with torch.no_grad():
                dist_true, fail_true = _simulate_sliding_distance(
                    engine=engine_true,
                    state0=state0,
                    dt=bundle.dt,
                    steps=int(steps),
                )
                dist_pred, fail_pred = _simulate_sliding_distance(
                    engine=engine_pred,
                    state0=state0,
                    dt=bundle.dt,
                    steps=int(steps),
                )

            mu_true_avg = float(sum(mu_true_values) / max(1, len(mu_true_values)))
            mu_pred_avg = float(sum(mu_pred_values) / max(1, len(mu_pred_values)))
            rows.append(
                {
                    "material_id": int(material_id),
                    "vx0": float(vx0),
                    "mu_true_avg": float(mu_true_avg),
                    "mu_pred_avg": float(mu_pred_avg),
                    "mu_abs_error": float(abs(mu_pred_avg - mu_true_avg)),
                    "slide_distance_true": float(dist_true),
                    "slide_distance_pred": float(dist_pred),
                    "slide_distance_abs_error": float(abs(dist_pred - dist_true)),
                    "failsafe_true": bool(fail_true),
                    "failsafe_pred": bool(fail_pred),
                }
            )

    true_mu = [float(row["mu_true_avg"]) for row in rows]
    pred_mu = [float(row["mu_pred_avg"]) for row in rows]
    true_dist = [float(row["slide_distance_true"]) for row in rows]
    pred_dist = [float(row["slide_distance_pred"]) for row in rows]
    dist_abs_err = [float(row["slide_distance_abs_error"]) for row in rows]
    mu_abs_err = [float(row["mu_abs_error"]) for row in rows]
    mape_terms = [float(err / max(1e-12, abs(tgt))) for err, tgt in zip(dist_abs_err, true_dist, strict=True)]

    fit = _linear_fit(true_dist, pred_dist)
    summary = {
        "case_count": int(len(rows)),
        "mu_mae": float(sum(mu_abs_err) / max(1, len(mu_abs_err))),
        "slide_distance_mae": float(sum(dist_abs_err) / max(1, len(dist_abs_err))),
        "slide_distance_mape": float(sum(mape_terms) / max(1, len(mape_terms))),
        "mu_pearson": float(_pearson_corr(true_mu, pred_mu)),
        "mu_spearman": float(_spearman_rank_corr(true_mu, pred_mu)),
        "slide_distance_pearson": float(_pearson_corr(true_dist, pred_dist)),
        "slide_distance_spearman": float(_spearman_rank_corr(true_dist, pred_dist)),
        "slide_distance_fit_slope": float(fit["slope"]),
        "slide_distance_fit_intercept": float(fit["intercept"]),
        "slide_distance_fit_r2": float(fit["r2"]),
        "failsafe_triggered_any_true": bool(any(bool(row["failsafe_true"]) for row in rows)),
        "failsafe_triggered_any_pred": bool(any(bool(row["failsafe_pred"]) for row in rows)),
    }
    return {"summary": summary, "rows": rows}


def _build_task_bundle(
    *,
    config_path: Path,
    device_name: str,
    dtype_name: str,
    deterministic: bool,
    train_iters_override: int | None,
    n_train_override: int | None,
    n_val_override: int | None,
) -> TaskBundle:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a mapping: {config_path}")

    task = config.get("task", {})
    if not isinstance(task, dict):
        raise TypeError("config.task must be a mapping")
    if str(task.get("kind", "mu_stack_field")) != "mu_stack_field":
        raise ValueError("P3-2 script currently supports task.kind=mu_stack_field only")
    loss_cfg = config.get("loss", {})
    if not isinstance(loss_cfg, dict):
        loss_cfg = {}
    engine_cfg = config.get("engine", {})
    if not isinstance(engine_cfg, dict):
        engine_cfg = {}

    seed = int(config.get("seed", 0))
    set_determinism(seed=seed, deterministic=deterministic)

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(device_name)
    dtype = torch.float32 if dtype_name == "float32" else torch.float64

    demo = str(config.get("demo", "mu_field_explain"))
    dt = float(task.get("dt", config.get("dt", 0.01)))
    rollout_steps = int(task.get("rollout_steps", 5))
    train_iters = int(task.get("train_iters", 140))
    if train_iters_override is not None:
        train_iters = int(min(train_iters, int(train_iters_override)))
    n_train = int(task.get("n_train", 16))
    n_val = int(task.get("n_val", 10))
    if n_train_override is not None:
        n_train = int(min(n_train, int(n_train_override)))
    if n_val_override is not None:
        n_val = int(min(n_val, int(n_val_override)))

    mu_init = float(task.get("mu_init", 0.2))
    mu_max = float(task.get("mu_max", 1.0))
    lr = float(task.get("lr", 0.1))
    mlp_hidden = int(task.get("mlp_hidden", 24))

    omega_range = _as_range(task.get("omega_range"), (3.0, 8.0))
    vx_range = _as_range(task.get("vx_range"), (0.0, 0.2))
    vx_bottom_range = _as_range(task.get("vx_bottom_range"), (2.0, 3.0))
    val_omega_range = _as_range(task.get("val_omega_range"), (1.0, 4.0))
    val_vx_range = _as_range(task.get("val_vx_range"), (0.0, 0.2))
    val_vx_bottom_range = _as_range(task.get("val_vx_bottom_range"), (2.0, 3.0))

    material_cfg = task.get("material", {})
    if not isinstance(material_cfg, dict):
        material_cfg = {}
    use_material_features = bool(material_cfg.get("enabled", False))
    max_material_id = int(material_cfg.get("max_material_id", 4))
    if max_material_id < 0:
        raise ValueError("task.material.max_material_id must be non-negative")
    ground_material_id = int(material_cfg.get("ground_material_id", 0 if use_material_features else 0))
    body_pool_raw = material_cfg.get("body_material_pool", [1, 2, 3] if use_material_features else [0])
    train_pool_raw = material_cfg.get("train_body_material_pool", body_pool_raw)
    val_pool_raw = material_cfg.get("val_body_material_pool", train_pool_raw)
    body_material_pool = tuple(int(x) for x in body_pool_raw) if isinstance(body_pool_raw, (list, tuple)) else (0,)
    train_body_material_pool = (
        tuple(int(x) for x in train_pool_raw) if isinstance(train_pool_raw, (list, tuple)) else body_material_pool
    )
    val_body_material_pool = (
        tuple(int(x) for x in val_pool_raw) if isinstance(val_pool_raw, (list, tuple)) else train_body_material_pool
    )
    train_body_material_patterns = _as_material_patterns(material_cfg.get("train_body_material_patterns"))
    val_body_material_patterns = _as_material_patterns(
        material_cfg.get("val_body_material_patterns", train_body_material_patterns)
    )
    if not use_material_features:
        body_material_pool = (0,)
        train_body_material_pool = (0,)
        val_body_material_pool = (0,)
        train_body_material_patterns = tuple()
        val_body_material_patterns = tuple()
        ground_material_id = 0

    for material_id in (
        *body_material_pool,
        *train_body_material_pool,
        *val_body_material_pool,
        ground_material_id,
    ):
        if material_id < 0 or material_id > max_material_id:
            raise ValueError(f"material id {material_id} out of range [0, {max_material_id}]")
    for pattern in (*train_body_material_patterns, *val_body_material_patterns):
        for material_id in pattern:
            if material_id < 0 or material_id > max_material_id:
                raise ValueError(f"material id {material_id} in pattern out of range [0, {max_material_id}]")

    probe_material_ids = _material_id_set(material_pool=body_material_pool, patterns=train_body_material_patterns)
    probe_material_ids = tuple(sorted(set(probe_material_ids).union(set(_material_id_set(
        material_pool=body_material_pool,
        patterns=val_body_material_patterns,
    )))))
    if not probe_material_ids:
        probe_material_ids = (0,)

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
    material_bias = (
        tuple(float(x) for x in material_bias_raw) if isinstance(material_bias_raw, (list, tuple)) else tuple()
    )

    w_q = float(loss_cfg.get("w_q", 0.1))
    w_v = float(loss_cfg.get("w_v", 1.0))
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
        Path("scripts/train_m2_mu_pgs_stack_field.py"),
        name="scone_scripts_train_m2_mu_pgs_stack_field_p3",
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
        mu_fn: ContactMuFn | None,
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

    return TaskBundle(
        demo=demo,
        seed=seed,
        device=device,
        dtype=dtype,
        dt=float(dt),
        train_iters=int(train_iters),
        rollout_steps=int(rollout_steps),
        n_train=int(n_train),
        n_val=int(n_val),
        mu_init=float(mu_init),
        mu_max=float(mu_max),
        mlp_hidden=int(mlp_hidden),
        lr=float(lr),
        w_q=float(w_q),
        w_v=float(w_v),
        use_material_features=bool(use_material_features),
        max_material_id=int(max_material_id),
        ground_material_id=int(ground_material_id),
        probe_material_ids=tuple(int(x) for x in probe_material_ids),
        calibration_material_ids=tuple(int(x) for x in sorted(set(body_material_pool))),
        engine_params=engine_params,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        field_mod=field_mod,
        mu_true_fn=mu_true_fn,
        build_engine=build_engine,
    )


def _build_model(bundle: TaskBundle) -> Any:
    if bundle.use_material_features:
        return ContactMuMaterialMLP(
            init_mu=bundle.mu_init,
            mu_max=bundle.mu_max,
            hidden_dim=bundle.mlp_hidden,
            max_material_id=bundle.max_material_id,
            device=bundle.device,
            dtype=bundle.dtype,
        )
    return ContactMuMLP(
        init_mu=bundle.mu_init,
        mu_max=bundle.mu_max,
        hidden_dim=bundle.mlp_hidden,
        device=bundle.device,
        dtype=bundle.dtype,
    )


def _run_variant(
    *,
    bundle: TaskBundle,
    variant: VariantSpec,
    eval_every: int,
    calib_steps: int,
    calib_vx: tuple[float, ...],
    calib_material_ids: tuple[int, ...],
) -> dict[str, Any]:
    model = _build_model(bundle)
    mu_pred_fn = model.from_contact_features
    opt = torch.optim.Adam(model.parameters(), lr=float(bundle.lr))
    field_mod = bundle.field_mod

    with torch.no_grad():
        baseline_loss_val_t = field_mod._dataset_loss_open_loop(
            dataset=bundle.dataset_val,
            build_engine_fn=bundle.build_engine,
            mu_default=bundle.mu_init,
            mu_fn=None,
            steps=bundle.rollout_steps,
            w_q=bundle.w_q,
            w_v=bundle.w_v,
            w_pen=variant.w_pen,
            w_comp=variant.w_comp,
            w_res=variant.w_res,
        )
    baseline_loss_val = _to_float(baseline_loss_val_t, default=1.0)

    history: list[dict[str, Any]] = []
    for step_idx in range(bundle.train_iters + 1):
        should_eval = (step_idx == 0) or (step_idx == bundle.train_iters) or (step_idx % max(1, eval_every) == 0)
        if should_eval:
            with torch.no_grad():
                fit_loss_train_t = field_mod._dataset_loss_open_loop(
                    dataset=bundle.dataset_train,
                    build_engine_fn=bundle.build_engine,
                    mu_default=bundle.mu_init,
                    mu_fn=mu_pred_fn,
                    steps=bundle.rollout_steps,
                    w_q=bundle.w_q,
                    w_v=bundle.w_v,
                    w_pen=variant.w_pen,
                    w_comp=variant.w_comp,
                    w_res=variant.w_res,
                )
                fit_loss_val_t = field_mod._dataset_loss_open_loop(
                    dataset=bundle.dataset_val,
                    build_engine_fn=bundle.build_engine,
                    mu_default=bundle.mu_init,
                    mu_fn=mu_pred_fn,
                    steps=bundle.rollout_steps,
                    w_q=bundle.w_q,
                    w_v=bundle.w_v,
                    w_pen=variant.w_pen,
                    w_comp=variant.w_comp,
                    w_res=variant.w_res,
                )
                mu_probe_mae_t = field_mod.probe_mu_mae(
                    device=bundle.device,
                    dtype=bundle.dtype,
                    mu_pred_fn=mu_pred_fn,
                    mu_true_fn=bundle.mu_true_fn,
                    material_ids=bundle.probe_material_ids,
                    ground_material_id=bundle.ground_material_id,
                )
                physics = _collect_open_loop_physics(
                    dataset=bundle.dataset_val,
                    build_engine_fn=bundle.build_engine,
                    mu_default=bundle.mu_init,
                    mu_fn=mu_pred_fn,
                    steps=bundle.rollout_steps,
                )

            fit_loss_train = _to_float(fit_loss_train_t, default=0.0)
            fit_loss_val = _to_float(fit_loss_val_t, default=0.0)
            history_point = {
                "iter": int(step_idx),
                "train_loss": float(fit_loss_train),
                "val_loss": float(fit_loss_val),
                "loss_ratio_val": float(fit_loss_val / max(1e-12, baseline_loss_val)),
                "mu_probe_mae": float(_to_float(mu_probe_mae_t, default=0.0)),
                "val_penetration_max": float(physics["penetration_max"]),
                "val_solver_residual_max": float(physics["solver_residual_max"]),
                "val_energy_drift_rel_max": float(physics["energy_drift_rel_max"]),
                "val_solver_max_iter_ratio": float(physics["solver_max_iter_ratio"]),
                "val_solver_diverged_ratio": float(physics["solver_diverged_ratio"]),
                "val_failsafe_triggered_any": bool(physics["failsafe_triggered_any"]),
            }
            history.append(history_point)
            print(
                "[{name}] iter={it:4d} loss_ratio_val={ratio:.6f} mu_probe_mae={mae:.6f} "
                "pen={pen:.3e} res={res:.3e}".format(
                    name=variant.name,
                    it=int(step_idx),
                    ratio=float(history_point["loss_ratio_val"]),
                    mae=float(history_point["mu_probe_mae"]),
                    pen=float(history_point["val_penetration_max"]),
                    res=float(history_point["val_solver_residual_max"]),
                )
            )

        if step_idx >= bundle.train_iters:
            continue
        opt.zero_grad(set_to_none=True)
        loss_t = field_mod._dataset_loss_open_loop(
            dataset=bundle.dataset_train,
            build_engine_fn=bundle.build_engine,
            mu_default=bundle.mu_init,
            mu_fn=mu_pred_fn,
            steps=bundle.rollout_steps,
            w_q=bundle.w_q,
            w_v=bundle.w_v,
            w_pen=variant.w_pen,
            w_comp=variant.w_comp,
            w_res=variant.w_res,
        )
        loss_t.backward()
        opt.step()

    calibration = _run_calibration_sweep(
        bundle=bundle,
        mu_pred_fn=mu_pred_fn,
        material_ids=calib_material_ids,
        vx_values=calib_vx,
        steps=calib_steps,
    )
    return {
        "variant": variant.name,
        "description": variant.description,
        "weights": {"w_pen": variant.w_pen, "w_comp": variant.w_comp, "w_res": variant.w_res},
        "baseline_loss_val": float(baseline_loss_val),
        "history": history,
        "final": history[-1] if history else {},
        "calibration": calibration,
    }


def _plot_history(
    *,
    results: dict[str, dict[str, Any]],
    key: str,
    title: str,
    ylabel: str,
    out_path: Path,
    log_scale: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for variant_name, result in results.items():
        history = result.get("history", [])
        if not isinstance(history, list) or len(history) == 0:
            continue
        xs = [int(row.get("iter", 0)) for row in history]
        ys = [float(row.get(key, 0.0)) for row in history]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, label=variant_name)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("train iter")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_calibration_scatter(
    *,
    results: dict[str, dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    all_x: list[float] = []
    all_y: list[float] = []
    colors = {"base": "#1f77b4", "struct_reg": "#d62728"}
    markers = {"base": "o", "struct_reg": "s"}
    for variant_name, result in results.items():
        calibration = result.get("calibration", {})
        rows = calibration.get("rows", []) if isinstance(calibration, dict) else []
        if not isinstance(rows, list):
            continue
        xs = [float(row.get(x_key, 0.0)) for row in rows]
        ys = [float(row.get(y_key, 0.0)) for row in rows]
        if not xs:
            continue
        all_x.extend(xs)
        all_y.extend(ys)
        ax.scatter(
            xs,
            ys,
            alpha=0.72,
            s=26,
            color=colors.get(variant_name, None),
            marker=markers.get(variant_name, "o"),
            label=variant_name,
        )
    if all_x and all_y:
        lo = min(min(all_x), min(all_y))
        hi = max(max(all_x), max(all_y))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1, color="black", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_markdown_summary(*, out_path: Path, results: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# P3-2 Mu Interpretability Report\n\n",
        "| variant | loss_ratio_val | mu_probe_mae | val_pen_max | val_residual_max | val_drift_rel_max | calib.mu_mae | calib.slide_mae | calib.slide_r2 |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for name, result in results.items():
        final = result.get("final", {})
        calibration = result.get("calibration", {})
        calib_summary = calibration.get("summary", {}) if isinstance(calibration, dict) else {}
        lines.append(
            "| {name} | {loss_ratio_val:.6f} | {mu_probe_mae:.6f} | {pen:.6e} | {res:.6e} | {drift:.6e} | {mu_mae:.6f} | {slide_mae:.6f} | {slide_r2:.6f} |\n".format(
                name=name,
                loss_ratio_val=float(final.get("loss_ratio_val", 0.0)),
                mu_probe_mae=float(final.get("mu_probe_mae", 0.0)),
                pen=float(final.get("val_penetration_max", 0.0)),
                res=float(final.get("val_solver_residual_max", 0.0)),
                drift=float(final.get("val_energy_drift_rel_max", 0.0)),
                mu_mae=float(calib_summary.get("mu_mae", 0.0)),
                slide_mae=float(calib_summary.get("slide_distance_mae", 0.0)),
                slide_r2=float(calib_summary.get("slide_distance_fit_r2", 0.0)),
            )
        )
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/learn_mu_field_stack_material_holdout.yaml"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--train-iters", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--struct-reg-weight", type=float, default=1.0)
    parser.add_argument("--calib-steps", type=int, default=40)
    parser.add_argument("--calib-vx", type=str, default="0.8,1.2,1.6,2.0")
    parser.add_argument("--calib-material-ids", type=str, default="")
    args = parser.parse_args()

    out_dir = args.out_dir or _timestamped_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    bundle = _build_task_bundle(
        config_path=args.config,
        device_name=str(args.device),
        dtype_name=str(args.dtype),
        deterministic=bool(args.deterministic),
        train_iters_override=args.train_iters,
        n_train_override=args.n_train,
        n_val_override=args.n_val,
    )

    calib_vx = _parse_csv_floats(args.calib_vx)
    if len(calib_vx) == 0:
        calib_vx = (0.8, 1.2, 1.6, 2.0)
    calib_material_ids = _parse_csv_ints(args.calib_material_ids)
    if len(calib_material_ids) == 0:
        calib_material_ids = bundle.calibration_material_ids
    if len(calib_material_ids) == 0:
        calib_material_ids = (0,)

    variants = (
        VariantSpec(
            name="base",
            description="No structure regularization",
            w_pen=0.0,
            w_comp=0.0,
            w_res=0.0,
        ),
        VariantSpec(
            name="struct_reg",
            description="Structure regularization enabled",
            w_pen=float(args.struct_reg_weight),
            w_comp=float(args.struct_reg_weight),
            w_res=float(args.struct_reg_weight),
        ),
    )

    results: dict[str, dict[str, Any]] = {}
    for variant in variants:
        set_determinism(seed=bundle.seed, deterministic=bool(args.deterministic))
        results[variant.name] = _run_variant(
            bundle=bundle,
            variant=variant,
            eval_every=max(1, int(args.eval_every)),
            calib_steps=int(max(1, args.calib_steps)),
            calib_vx=tuple(float(x) for x in calib_vx),
            calib_material_ids=tuple(int(x) for x in calib_material_ids),
        )

    comparison: dict[str, Any] = {}
    if "base" in results and "struct_reg" in results:
        base_final = results["base"].get("final", {})
        reg_final = results["struct_reg"].get("final", {})
        comparison = {
            "delta_loss_ratio_val": float(reg_final.get("loss_ratio_val", 0.0) - base_final.get("loss_ratio_val", 0.0)),
            "delta_mu_probe_mae": float(reg_final.get("mu_probe_mae", 0.0) - base_final.get("mu_probe_mae", 0.0)),
            "delta_val_penetration_max": float(
                reg_final.get("val_penetration_max", 0.0) - base_final.get("val_penetration_max", 0.0)
            ),
            "delta_val_solver_residual_max": float(
                reg_final.get("val_solver_residual_max", 0.0) - base_final.get("val_solver_residual_max", 0.0)
            ),
        }

    payload = {
        "meta": {
            "config": str(args.config),
            "demo": bundle.demo,
            "seed": int(bundle.seed),
            "device": str(bundle.device),
            "dtype": str(bundle.dtype),
            "train_iters": int(bundle.train_iters),
            "eval_every": int(max(1, args.eval_every)),
            "rollout_steps": int(bundle.rollout_steps),
            "n_train": int(bundle.n_train),
            "n_val": int(bundle.n_val),
            "calib_steps": int(max(1, args.calib_steps)),
            "calib_vx": [float(x) for x in calib_vx],
            "calib_material_ids": [int(x) for x in calib_material_ids],
            "struct_reg_weight": float(args.struct_reg_weight),
        },
        "variants": results,
        "comparison": comparison,
    }

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown_summary(out_path=out_dir / "summary.md", results=results)

    _plot_history(
        results=results,
        key="mu_probe_mae",
        title="P3-2: Mu Probe MAE Convergence",
        ylabel="mu_probe_mae",
        out_path=plots_dir / "mu_probe_mae_curve.png",
    )
    _plot_history(
        results=results,
        key="loss_ratio_val",
        title="P3-2: Validation Loss Ratio",
        ylabel="loss_ratio_val",
        out_path=plots_dir / "loss_ratio_val_curve.png",
        log_scale=True,
    )
    _plot_history(
        results=results,
        key="val_penetration_max",
        title="P3-2: Validation Penetration Max",
        ylabel="penetration_max",
        out_path=plots_dir / "val_penetration_curve.png",
        log_scale=True,
    )
    _plot_history(
        results=results,
        key="val_solver_residual_max",
        title="P3-2: Validation Solver Residual Max",
        ylabel="solver_residual_max",
        out_path=plots_dir / "val_solver_residual_curve.png",
        log_scale=True,
    )
    _plot_history(
        results=results,
        key="val_energy_drift_rel_max",
        title="P3-2: Validation Energy Drift (relative max)",
        ylabel="energy_drift_rel_max",
        out_path=plots_dir / "val_energy_drift_curve.png",
        log_scale=True,
    )
    _plot_calibration_scatter(
        results=results,
        x_key="slide_distance_true",
        y_key="slide_distance_pred",
        title="P3-2 Calibration: Sliding Distance",
        xlabel="true sliding distance",
        ylabel="pred sliding distance",
        out_path=plots_dir / "calibration_sliding_distance_scatter.png",
    )
    _plot_calibration_scatter(
        results=results,
        x_key="mu_true_avg",
        y_key="mu_pred_avg",
        title="P3-2 Calibration: Mu Average",
        xlabel="true avg mu",
        ylabel="pred avg mu",
        out_path=plots_dir / "calibration_mu_scatter.png",
    )

    print(f"Wrote: {summary_json}")
    print(f"Wrote: {out_dir / 'summary.md'}")
    print(f"Wrote: {plots_dir}")
    for variant_name, result in results.items():
        final = result.get("final", {})
        calib = result.get("calibration", {})
        calib_summary = calib.get("summary", {}) if isinstance(calib, dict) else {}
        print(
            "variant={name} loss_ratio_val={ratio:.6f} mu_probe_mae={mae:.6f} "
            "val_pen={pen:.3e} val_res={res:.3e} calib_slide_mae={slide_mae:.6f}".format(
                name=variant_name,
                ratio=float(final.get("loss_ratio_val", 0.0)),
                mae=float(final.get("mu_probe_mae", 0.0)),
                pen=float(final.get("val_penetration_max", 0.0)),
                res=float(final.get("val_solver_residual_max", 0.0)),
                slide_mae=float(calib_summary.get("slide_distance_mae", 0.0)),
            )
        )


if __name__ == "__main__":
    main()
