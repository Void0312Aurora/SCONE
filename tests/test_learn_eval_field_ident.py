import importlib.util
import sys
from pathlib import Path

import torch


def _load_learn_eval_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "bench" / "learn_eval.py"
    spec = importlib.util.spec_from_file_location("_bench_learn_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_learn_eval = _load_learn_eval_module()


def test_material_pair_set_and_holdout_pairs() -> None:
    train_pairs = _learn_eval._material_pair_set(
        body_pool=(1, 2, 3),
        body_patterns=((1, 2, 2), (2, 3, 2)),
        ground_material_id=0,
    )
    case_pairs = _learn_eval._material_pair_set(
        body_pool=(1, 2, 3),
        body_patterns=((1, 3, 3),),
        ground_material_id=0,
    )
    holdout_pairs = case_pairs - train_pairs

    assert (0, 1) in train_pairs
    assert (1, 2) in train_pairs
    assert (1, 3) in holdout_pairs
    assert (3, 3) in holdout_pairs


def test_probe_mu_mae_on_material_pairs_zero_and_nonzero() -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    material_pairs = {(0, 1), (1, 3), (2, 3)}

    def mu_true(features: dict[str, torch.Tensor]) -> torch.Tensor:
        is_pair = features["is_pair"]
        material_i = features["material_i"]
        material_j = features["material_j"]
        base = 0.2 + 0.1 * is_pair
        return base + 0.01 * material_i + 0.02 * material_j

    mu_zero = _learn_eval._probe_mu_mae_on_material_pairs(
        device=device,
        dtype=dtype,
        mu_pred_fn=mu_true,
        mu_true_fn=mu_true,
        material_pairs=material_pairs,
        ground_material_id=0,
    )
    assert mu_zero is not None
    assert float(mu_zero.detach().cpu().item()) < 1e-8

    def mu_shifted(features: dict[str, torch.Tensor]) -> torch.Tensor:
        return mu_true(features) + 0.05

    mu_shift = _learn_eval._probe_mu_mae_on_material_pairs(
        device=device,
        dtype=dtype,
        mu_pred_fn=mu_shifted,
        mu_true_fn=mu_true,
        material_pairs=material_pairs,
        ground_material_id=0,
    )
    assert mu_shift is not None
    assert float(mu_shift.detach().cpu().item()) > 0.04
