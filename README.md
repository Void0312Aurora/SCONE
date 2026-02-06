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
