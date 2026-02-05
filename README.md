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
