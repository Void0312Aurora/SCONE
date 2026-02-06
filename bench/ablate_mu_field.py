from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str
    overrides: dict[str, Any]


DEFAULT_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant(name="base", description="Base config", overrides={}),
    AblationVariant(
        name="struct_reg",
        description="Enable structure regularizers (w_pen/w_comp/w_res=1)",
        overrides={"loss": {"w_pen": 1.0, "w_comp": 1.0, "w_res": 1.0}},
    ),
    AblationVariant(
        name="no_warm_start",
        description="Disable solver warm-start",
        overrides={"engine": {"warm_start": False}},
    ),
    AblationVariant(
        name="low_pgs_iters",
        description="Reduce PGS iterations (8)",
        overrides={"engine": {"pgs_iters": 8}},
    ),
    AblationVariant(
        name="small_mlp",
        description="Reduce model capacity (mlp_hidden=8)",
        overrides={"task": {"mlp_hidden": 8}},
    ),
)


def _timestamped_out_dir() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / "bench" / f"{timestamp}-ablate-mu-field"


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict):
            existing = dst.get(key)
            if isinstance(existing, dict):
                _deep_update(existing, value)
            else:
                dst[key] = copy.deepcopy(value)
        else:
            dst[key] = copy.deepcopy(value)


def _load_base_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return config


def _write_variant_configs(
    *,
    base_config: dict[str, Any],
    variants: tuple[AblationVariant, ...],
    out_dir: Path,
) -> dict[str, Path]:
    configs_dir = out_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    variant_paths: dict[str, Path] = {}

    base_demo = str(base_config.get("demo", "mu_field_base"))
    for variant in variants:
        config_i = copy.deepcopy(base_config)
        _deep_update(config_i, variant.overrides)
        config_i["demo"] = f"{base_demo}__ablate_{variant.name}"

        path_i = configs_dir / f"{variant.name}.yaml"
        path_i.write_text(yaml.safe_dump(config_i, sort_keys=False, allow_unicode=True), encoding="utf-8")
        variant_paths[variant.name] = path_i
    return variant_paths


def _run_learn_eval(
    *,
    variant_config_paths: list[Path],
    run_out_dir: Path,
    device: str,
    dtype: str,
    deterministic: bool,
    repeat: int,
    max_train_iters: int | None,
    max_rollout_steps: int | None,
    max_n_train: int | None,
    max_n_val: int | None,
) -> Path:
    command: list[str] = [
        sys.executable,
        "bench/learn_eval.py",
        "--configs",
        *[str(path) for path in variant_config_paths],
        "--device",
        str(device),
        "--dtype",
        str(dtype),
        "--repeat",
        str(int(repeat)),
        "--out-dir",
        str(run_out_dir),
    ]
    if deterministic:
        command.append("--deterministic")
    if max_train_iters is not None:
        command.extend(["--max-train-iters", str(int(max_train_iters))])
    if max_rollout_steps is not None:
        command.extend(["--max-rollout-steps", str(int(max_rollout_steps))])
    if max_n_train is not None:
        command.extend(["--max-n-train", str(int(max_n_train))])
    if max_n_val is not None:
        command.extend(["--max-n-val", str(int(max_n_val))])

    subprocess.run(command, check=True)
    summary_path = run_out_dir / "bench_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing learn_eval output: {summary_path}")
    return summary_path


def _extract_metrics(
    *,
    summary_path: Path,
    variant_paths: dict[str, Path],
    variants: tuple[AblationVariant, ...],
) -> list[dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise TypeError(f"Invalid bench summary format: {summary_path}")

    config_path_to_variant: dict[str, str] = {}
    for name, path in variant_paths.items():
        config_path_to_variant[str(path)] = name
        config_path_to_variant[str(path.resolve())] = name

    rows_by_name: dict[str, dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        config_path = str(run.get("config", ""))
        variant_name = config_path_to_variant.get(config_path)
        if variant_name is None:
            continue
        summary = run.get("summary", {})
        if not isinstance(summary, dict):
            continue
        learn = summary.get("learn", {})
        if not isinstance(learn, dict):
            continue
        eval_val = learn.get("eval_val", {})
        eval_ood = learn.get("eval_ood", {})
        eval_val_solver = eval_val.get("solver", {}) if isinstance(eval_val, dict) else {}
        eval_ood_solver = eval_ood.get("solver", {}) if isinstance(eval_ood, dict) else {}
        eval_val_failsafe = eval_val.get("failsafe", {}) if isinstance(eval_val, dict) else {}
        eval_ood_failsafe = eval_ood.get("failsafe", {}) if isinstance(eval_ood, dict) else {}

        rows_by_name[variant_name] = {
            "variant": variant_name,
            "demo": str(summary.get("demo", "")),
            "loss_ratio_val": float(learn.get("loss_ratio_val", 0.0)),
            "loss_ratio_ood_worst": float(learn.get("loss_ratio_ood_worst", learn.get("loss_ratio_ood", 0.0) or 0.0)),
            "mu_probe_mae": float(learn.get("mu_probe_mae", 0.0)),
            "eval_val_solver_max_iter_ratio": float(eval_val_solver.get("max_iter_ratio", 0.0)),
            "eval_val_solver_residual_max": float(eval_val_solver.get("residual_max", 0.0)),
            "eval_ood_solver_max_iter_ratio": float(eval_ood_solver.get("max_iter_ratio", 0.0)),
            "eval_ood_solver_residual_max": float(eval_ood_solver.get("residual_max", 0.0)),
            "eval_val_failsafe_triggered_any": bool(eval_val_failsafe.get("triggered_any", False)),
            "eval_ood_failsafe_triggered_any": bool(eval_ood_failsafe.get("triggered_any", False)),
            "ood_case_count": int(learn.get("ood_case_count", 0)),
            "use_material_features": bool(learn.get("use_material_features", False)),
        }

    ordered_rows: list[dict[str, Any]] = []
    for variant in variants:
        row = rows_by_name.get(variant.name)
        if row is None:
            raise RuntimeError(f"Missing run result for variant: {variant.name}")
        row["description"] = variant.description
        ordered_rows.append(row)
    return ordered_rows


def _write_markdown_report(*, rows: list[dict[str, Any]], out_path: Path) -> None:
    header = (
        "| variant | description | loss_ratio_val | loss_ratio_ood_worst | mu_probe_mae | "
        "eval_val.max_iter_ratio | eval_ood.max_iter_ratio | val_residual_max | ood_residual_max | "
        "val_failsafe | ood_failsafe |\n"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {variant} | {description} | {loss_ratio_val:.6f} | {loss_ratio_ood_worst:.6f} | "
            "{mu_probe_mae:.6f} | {eval_val_solver_max_iter_ratio:.6f} | {eval_ood_solver_max_iter_ratio:.6f} | "
            "{eval_val_solver_residual_max:.6e} | {eval_ood_solver_residual_max:.6e} | {val_fail} | {ood_fail} |\n".format(
                variant=row["variant"],
                description=row["description"],
                loss_ratio_val=row["loss_ratio_val"],
                loss_ratio_ood_worst=row["loss_ratio_ood_worst"],
                mu_probe_mae=row["mu_probe_mae"],
                eval_val_solver_max_iter_ratio=row["eval_val_solver_max_iter_ratio"],
                eval_ood_solver_max_iter_ratio=row["eval_ood_solver_max_iter_ratio"],
                eval_val_solver_residual_max=row["eval_val_solver_residual_max"],
                eval_ood_solver_residual_max=row["eval_ood_solver_residual_max"],
                val_fail=str(bool(row["eval_val_failsafe_triggered_any"])).lower(),
                ood_fail=str(bool(row["eval_ood_failsafe_triggered_any"])).lower(),
            )
        )
    out_path.write_text("".join(lines), encoding="utf-8")


def _plot_bar_metric(*, rows: list[dict[str, Any]], key: str, title: str, out_path: Path, log_scale: bool = False) -> None:
    names = [str(row["variant"]) for row in rows]
    values = [float(row.get(key, 0.0)) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(names, values, color="#3C78D8")
    ax.set_title(title)
    ax.set_ylabel(key)
    ax.grid(axis="y", alpha=0.25)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.3e}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/learn_mu_field_stack_material_holdout.yaml"),
        help="Base config used to generate ablation variants",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-train-iters", type=int, default=80)
    parser.add_argument("--max-rollout-steps", type=int, default=None)
    parser.add_argument("--max-n-train", type=int, default=None)
    parser.add_argument("--max-n-val", type=int, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or _timestamped_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_out_dir = out_dir / "runs"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    base_config = _load_base_config(args.base_config)
    variant_paths = _write_variant_configs(base_config=base_config, variants=DEFAULT_VARIANTS, out_dir=out_dir)

    summary_path = _run_learn_eval(
        variant_config_paths=[variant_paths[v.name] for v in DEFAULT_VARIANTS],
        run_out_dir=runs_out_dir,
        device=args.device,
        dtype=args.dtype,
        deterministic=bool(args.deterministic),
        repeat=int(max(1, args.repeat)),
        max_train_iters=args.max_train_iters,
        max_rollout_steps=args.max_rollout_steps,
        max_n_train=args.max_n_train,
        max_n_val=args.max_n_val,
    )

    rows = _extract_metrics(summary_path=summary_path, variant_paths=variant_paths, variants=DEFAULT_VARIANTS)

    matrix_path = out_dir / "ablation_matrix.json"
    matrix_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown_report(rows=rows, out_path=out_dir / "ablation_matrix.md")

    _plot_bar_metric(
        rows=rows,
        key="loss_ratio_val",
        title="Ablation: loss_ratio_val",
        out_path=plots_dir / "loss_ratio_val.png",
        log_scale=True,
    )
    _plot_bar_metric(
        rows=rows,
        key="loss_ratio_ood_worst",
        title="Ablation: loss_ratio_ood_worst",
        out_path=plots_dir / "loss_ratio_ood_worst.png",
        log_scale=True,
    )
    _plot_bar_metric(
        rows=rows,
        key="mu_probe_mae",
        title="Ablation: mu_probe_mae",
        out_path=plots_dir / "mu_probe_mae.png",
        log_scale=False,
    )
    _plot_bar_metric(
        rows=rows,
        key="eval_ood_solver_max_iter_ratio",
        title="Ablation: eval_ood.solver.max_iter_ratio",
        out_path=plots_dir / "eval_ood_solver_max_iter_ratio.png",
        log_scale=False,
    )

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {matrix_path}")
    print(f"Wrote: {out_dir / 'ablation_matrix.md'}")
    print(f"Wrote: {plots_dir}")
    for row in rows:
        print(
            "variant={variant} val={val:.6f} ood={ood:.6f} mu_probe_mae={mae:.6f} "
            "ood_max_iter_ratio={max_iter:.6f}".format(
                variant=row["variant"],
                val=float(row["loss_ratio_val"]),
                ood=float(row["loss_ratio_ood_worst"]),
                mae=float(row["mu_probe_mae"]),
                max_iter=float(row["eval_ood_solver_max_iter_ratio"]),
            )
        )


if __name__ == "__main__":
    main()
