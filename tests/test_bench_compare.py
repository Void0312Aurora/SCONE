import json
import importlib.util
import subprocess
import sys
from pathlib import Path

def _load_compare_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "bench" / "compare.py"
    spec = importlib.util.spec_from_file_location("_bench_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_compare = _load_compare_module()
evaluate_run = _compare.evaluate_run


def test_bench_compare_evaluate_run_thresholds() -> None:
    suite_thresholds = {
        "disk_stack_2d": {
            "contacts.penetration_max": 1e-2,
            "solver.residual_max": 1e-3,
            "failsafe.triggered_any": False,
            "determinism.max_abs_dq": 0.0,
        }
    }
    run = {
        "config": "configs/mvp_disk_stack_2d.yaml",
        "summary": {
            "demo": "disk_stack_2d",
            "contacts": {"penetration_max": 5e-3},
            "solver": {"residual_max": 5e-4},
            "failsafe": {"triggered_any": False},
        },
        "determinism": {"max_abs_dq": 0.0},
    }
    results = evaluate_run(run=run, suite_thresholds=suite_thresholds)
    assert results
    assert all(r.ok for r in results)


def test_bench_compare_evaluate_run_fails_on_missing_key() -> None:
    suite_thresholds = {"harmonic_1d": {"energy.drift_rel_max": 0.1}}
    run = {"config": "configs/mvp_harmonic.yaml", "summary": {"demo": "harmonic_1d"}, "determinism": {}}
    results = evaluate_run(run=run, suite_thresholds=suite_thresholds)
    assert len(results) == 1
    assert results[0].ok is False
    assert results[0].value is None


def test_bench_compare_cli_exit_code(tmp_path: Path) -> None:
    # Minimal candidate bench summary that passes the suite.
    candidate = {
        "runs": [
            {
                "config": "configs/mvp_disk_stack_2d.yaml",
                "seed": 0,
                "summary": {
                    "demo": "disk_stack_2d",
                    "contacts": {"penetration_max": 1e-3, "complementarity_residual_max": 0.0},
                    "solver": {"residual_max": 1e-6, "iters_mean": 10.0},
                    "energy": {"drift_rel_max": 0.0},
                    "failsafe": {"triggered_any": False},
                },
                "determinism": {"max_abs_dq": 0.0, "max_abs_dv": 0.0},
            }
        ]
    }
    suite = {
        "thresholds": {
            "disk_stack_2d": {
                "contacts.penetration_max": 2e-3,
                "solver.residual_max": 1e-3,
                "failsafe.triggered_any": False,
            }
        }
    }
    candidate_path = tmp_path / "candidate.json"
    suite_path = tmp_path / "suite.yaml"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    suite_path.write_text(json.dumps(suite), encoding="utf-8")

    ok = subprocess.run(
        [sys.executable, "bench/compare.py", "--candidate", str(candidate_path), "--suite", str(suite_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr

    fail_suite = {
        "thresholds": {
            "disk_stack_2d": {
                "contacts.penetration_max": 1e-6,
            }
        }
    }
    suite_path.write_text(json.dumps(fail_suite), encoding="utf-8")
    bad = subprocess.run(
        [sys.executable, "bench/compare.py", "--candidate", str(candidate_path), "--suite", str(suite_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0
