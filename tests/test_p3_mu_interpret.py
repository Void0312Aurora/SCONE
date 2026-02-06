from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    path = Path("bench/p3_mu_interpret.py")
    module_name = "scone_bench_p3_mu_interpret_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_spearman_rank_corr_extremes() -> None:
    module = _load_module()
    assert abs(module._spearman_rank_corr([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-12
    assert abs(module._spearman_rank_corr([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) + 1.0) < 1e-12


def test_linear_fit_recovers_affine_line() -> None:
    module = _load_module()
    x = [0.0, 1.0, 2.0, 3.0]
    y = [1.0, 3.0, 5.0, 7.0]
    fit = module._linear_fit(x, y)
    assert abs(float(fit["slope"]) - 2.0) < 1e-12
    assert abs(float(fit["intercept"]) - 1.0) < 1e-12
    assert abs(float(fit["r2"]) - 1.0) < 1e-12
