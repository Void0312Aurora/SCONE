from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    key: str
    threshold: Any
    value: Any


def _get_in(mapping: dict[str, Any], path: str) -> Any:
    cur: Any = mapping
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float))


def _check_one(*, data: dict[str, Any], key: str, threshold: Any) -> CheckResult:
    value = _get_in(data, key)
    if value is None:
        return CheckResult(ok=False, key=key, threshold=threshold, value=None)

    if isinstance(threshold, bool):
        return CheckResult(ok=bool(value) is bool(threshold), key=key, threshold=threshold, value=value)
    if _is_number(threshold):
        if not _is_number(value):
            return CheckResult(ok=False, key=key, threshold=threshold, value=value)
        return CheckResult(ok=float(value) <= float(threshold), key=key, threshold=threshold, value=value)

    raise TypeError(f"Unsupported threshold type for {key}: {type(threshold)}")


def evaluate_run(*, run: dict[str, Any], suite_thresholds: dict[str, Any]) -> list[CheckResult]:
    summary = run.get("summary")
    if not isinstance(summary, dict):
        raise TypeError("run.summary missing or invalid")

    demo = summary.get("demo")
    if not isinstance(demo, str):
        raise TypeError("run.summary.demo missing or invalid")

    thresholds = suite_thresholds.get(demo, {})
    if not isinstance(thresholds, dict):
        thresholds = {}

    # We evaluate against a merged dict: summary + determinism.
    merged: dict[str, Any] = {"summary": summary, "determinism": run.get("determinism", {})}

    results: list[CheckResult] = []
    for key, threshold in thresholds.items():
        if not isinstance(key, str):
            continue
        # Threshold keys are written as "energy.drift_rel_max" etc and refer to run.summary.* unless prefixed.
        path = key
        if not (path.startswith("summary.") or path.startswith("determinism.")):
            path = "summary." + path
        results.append(_check_one(data=merged, key=path, threshold=threshold))
    return results


def _index_runs(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise TypeError("bench summary missing 'runs' list")
    indexed: dict[str, dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        config = run.get("config")
        if not isinstance(config, str):
            continue
        indexed[config] = run
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True, help="Path to candidate bench_summary.json")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional path to baseline bench_summary.json")
    parser.add_argument("--suite", type=Path, default=Path("bench/suites/mvp.yaml"))
    parser.add_argument("--strict", action="store_true", default=False, help="Fail on missing configs")
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if not isinstance(candidate, dict):
        raise TypeError("Candidate summary is not a JSON object")
    candidate_runs = _index_runs(candidate)

    baseline_runs: dict[str, dict[str, Any]] = {}
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        if not isinstance(baseline, dict):
            raise TypeError("Baseline summary is not a JSON object")
        baseline_runs = _index_runs(baseline)

    suite_doc = yaml.safe_load(args.suite.read_text(encoding="utf-8"))
    if not isinstance(suite_doc, dict):
        raise TypeError("Suite file must be a mapping")
    suite_thresholds = suite_doc.get("thresholds", {})
    if not isinstance(suite_thresholds, dict):
        raise TypeError("Suite file missing 'thresholds' mapping")

    any_fail = False
    all_configs = sorted(set(candidate_runs.keys()) | set(baseline_runs.keys()))
    for config in all_configs:
        cand = candidate_runs.get(config)
        base = baseline_runs.get(config)

        if cand is None:
            msg = f"MISSING candidate: {config}"
            print(msg)
            any_fail = any_fail or bool(args.strict)
            continue

        demo = _get_in(cand, "summary.demo")
        checks = evaluate_run(run=cand, suite_thresholds=suite_thresholds)
        ok = all(c.ok for c in checks)
        any_fail = any_fail or (not ok)

        prefix = "PASS" if ok else "FAIL"
        print(f"{prefix} {config} demo={demo}")
        for c in checks:
            status = "ok" if c.ok else "bad"
            print(f"  - {status} {c.key} value={c.value} thr={c.threshold}")

        if base is not None:
            # Show a few key deltas when available.
            for key in [
                "contacts.penetration_max",
                "contacts.complementarity_residual_max",
                "solver.residual_max",
                "solver.iters_mean",
                "energy.drift_rel_max",
            ]:
                a = _get_in(cand.get("summary", {}), key)
                b = _get_in(base.get("summary", {}), key)
                if _is_number(a) and _is_number(b):
                    print(f"    Δ {key} = {float(a) - float(b):+.3e} (cand={a}, base={b})")

    raise SystemExit(1 if any_fail else 0)


if __name__ == "__main__":
    main()

