# SCONE

结构化可学习“物理引擎 step”原型实现（MVP），与 `docs/spec.md` 对齐。

## Quickstart（venv）

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

GPU（可选）：确保安装了 CUDA 版本 PyTorch，并在运行时使用 `--device cuda`。

检查 CUDA 是否可用：

```bash
python -c "import torch; print(torch.__version__); print('cuda', torch.version.cuda); print('cuda avail', torch.cuda.is_available())"
```

## Run MVP demos

```bash
python scripts/run_mvp.py --config configs/mvp_harmonic.yaml --device cpu
python scripts/run_mvp.py --config configs/mvp_damped.yaml --device cpu
python scripts/run_mvp.py --config configs/mvp_bounce.yaml --device cuda
python scripts/run_mvp.py --config configs/mvp_disk_roll_2d.yaml --device cuda
python scripts/run_mvp.py --config configs/mvp_disk_stack_2d.yaml --device cuda
```

输出会写入 `outputs/<run_id>/`（含 `config.yaml`、`logs/diagnostics.jsonl`、`plots/*.png`）。

## Bench / Eval

对 `configs/mvp_*.yaml` 跑一遍并输出汇总指标（穿透、互补残差、能量漂移、求解器残差/迭代、sleep 统计、确定性差异等）：

```bash
python bench/eval.py --suite mvp --device cpu --deterministic --repeat 2
```

输出写入 `outputs/bench/<timestamp>-eval/bench_summary.json`。

扩展：包含轻量 OOD case（dt / mass / radius）：

```bash
python bench/eval.py --suite mvp_ood --device cpu --deterministic --repeat 2
```

将一份 candidate 结果按 suite 阈值做回归门禁（并可选对比 baseline）：

```bash
python bench/compare.py --candidate outputs/bench/<...>/bench_summary.json --suite bench/suites/mvp.yaml
python bench/compare.py --baseline outputs/bench/<...>/bench_summary.json --candidate outputs/bench/<...>/bench_summary.json --suite bench/suites/mvp.yaml

# OOD suite gate
python bench/compare.py --candidate outputs/bench/<...>/bench_summary.json --suite bench/suites/mvp_ood.yaml
```

学习切片的回归门禁（训练 + ID/OOD 评测，输出同样是 `bench_summary.json`）：

```bash
python bench/learn_eval.py --suite mu_scalar --device cpu --dtype float32 --deterministic --repeat 1
python bench/compare.py --candidate outputs/bench/<...>/bench_summary.json --suite bench/suites/learn_mu.yaml
```

两参数摩擦（ground/pair）：

```bash
python bench/learn_eval.py --suite mu_pair --device cpu --dtype float32 --deterministic --repeat 1
python bench/compare.py --candidate outputs/bench/<...>/bench_summary.json --suite bench/suites/learn_mu_pair.yaml

# 可辨识性增强 stress（多场景 OOD + mu gap gate）
python bench/learn_eval.py --configs configs/learn_mu_pair_stack_ident_stress.yaml --device cpu --dtype float32 --deterministic --repeat 1
```

状态依赖摩擦 `mu(x)`：

```bash
python bench/learn_eval.py --suite mu_field --device cpu --dtype float32 --deterministic --repeat 1
python bench/compare.py --candidate outputs/bench/<...>/bench_summary.json --suite bench/suites/learn_mu_field.yaml

# 主线 gate（material holdout + 强 OOD）
python bench/learn_eval.py --suite mu_field_gate --device cpu --dtype float32 --deterministic --repeat 1
python bench/compare.py --candidate outputs/bench/<...>/bench_summary.json --suite bench/suites/learn_mu_field_gate.yaml

# 多场景 OOD stress（dt/mass/radius）
python bench/learn_eval.py --configs configs/learn_mu_field_stack_stress.yaml --device cpu --dtype float32 --deterministic --repeat 1

# 材料条件化 stress（material-aware μ(x) + OOD）
python bench/learn_eval.py --configs configs/learn_mu_field_stack_material_stress.yaml --device cpu --dtype float32 --deterministic --repeat 1

# 可辨识性 holdout（单材质留出 + 组合留出）
python bench/learn_eval.py --configs configs/learn_mu_field_stack_material_holdout.yaml --device cpu --dtype float32 --deterministic --repeat 1

# holdout gate：额外检查 seen/holdout 可辨识性与分 case OOD 指标
# (mu_probe_mae_seen / mu_probe_mae_holdout_material / mu_probe_pair_* /
#  ood_case_metrics.<case>.loss_ratio / alpha_blend_drop_max)
python bench/compare.py --candidate outputs/bench/<...>/bench_summary.json --suite bench/suites/learn_mu_field.yaml

# M3 消融矩阵（自动生成 configs + bench_summary + markdown + plots）
python bench/ablate_mu_field.py --base-config configs/learn_mu_field_stack_material_holdout.yaml --device cpu --dtype float32 --deterministic

# P3-2 可解释性：mu 收敛曲线 + 物理误差曲线 + 滑移距离标定散点
python bench/p3_mu_interpret.py --config configs/learn_mu_field_stack_material_holdout.yaml --device cpu --dtype float32 --deterministic
```

## Learnable slices

PGS 多接触开环 BPTT（标量 `mu`）：

```bash
python scripts/train_m2_mu_pgs_stack.py --device cpu --dtype float32 --loss-mode open_loop
```

结构一致性正则（默认系数为 0，可用于消融）：

```bash
python scripts/train_m2_mu_pgs_stack.py --device cpu --dtype float32 --loss-mode open_loop --w-pen 1.0 --w-comp 1.0 --w-res 1.0
```

扩展：区分 ground vs body-body 的两参数摩擦（`mu_ground`, `mu_pair`）：

```bash
python scripts/train_m2_mu_pgs_stack_pair.py --device cpu --dtype float32
```

同样支持结构正则项（`--w-pen/--w-comp/--w-res`）。

扩展：状态依赖摩擦 `mu(x)`（接触特征驱动）：

```bash
python scripts/train_m2_mu_pgs_stack_field.py --device cpu --dtype float32

# 材料条件化 μ(x)
python scripts/train_m2_mu_pgs_stack_field.py --device cpu --dtype float32 \
  --use-material-features \
  --max-material-id 4 \
  --body-material-ids 1,2,3 \
  --ground-material-id 0 \
  --true-material-bias 0.0,-0.04,0.0,0.04,0.08
```

## Failsafe / Ledger

`Engine.step()` 现在会统一输出：
- `failsafe.alpha_blend`（软降级混合系数，默认 1.0）
- `ledger.B_balance_error_abs / C_power_error_from_energy_abs / D_energy_mismatch_abs`

可选软降级配置（写在 config 的 `failsafe`）：

```yaml
failsafe:
  solver_residual_soft: 1.0e-3
  alpha_blend_soft: 0.5          # 触发 soft reason 时与基线状态混合
  alpha_blend_baseline: pre_event # pre_event | pre_constraints | prev
  ledger_balance_soft: 1.0e-4
  event_energy_soft: 1.0e-4
```

约束层最小实现：`LinearConstraintProjectionLayer`（`src/scone/layers/constraints.py`），
支持通过 `context["constraints"]` 注入 `A_pos/b_pos/A_vel/b_vel` 做仿射投影并输出
`constraints.residual_pos/residual_vel/power_error`。
