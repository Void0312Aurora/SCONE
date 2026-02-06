# SCONE 进度记录（2·6 / temp-01）

> 目的：把截至当前的工程实现与可验收成果“落盘”，便于后续按 `docs/spec.md` 的 (2) 路线继续推进（接触堆叠 + 摩擦 + 可复现 + 可学习切片）。

## 0. 总览（当前已经能交付什么）

1. **引擎 MVP（可跑 + 可诊断 + 可回归）**
   - `Engine.step()` 具备分层能量账本与 fail-safe 机制（回退到上一步 state）。
   - 2D 圆盘系统支持**多接触 PGS time-stepping**、warm-start、mode 统计、sleep islands（冻结 core）。
2. **Bench / Regression gate**
   - `bench/eval.py`：对 `configs/*.yaml` 跑 rollout，汇总 penetration / residual / drift / determinism 指标。
   - `bench/compare.py` + `bench/suites/*.yaml`：按 demo 阈值做回归门禁（CI/本地改动都能用）。
   - 轻量 OOD suite：dt / mass / radius stress-case。
3. **最小可学习切片（μ）+ 可消融结构正则**
   - 标量 μ（stack 场景）与 ground/pair 双 μ（stack 场景）训练脚本可运行。
   - 支持 open-loop 多步 BPTT（PGS 计算图可回传梯度）+ 结构一致性正则项（penetration / complementarity / solver residual）。
4. **学习 bench gate（训练 + ID/OOD 一键回归）**
   - `bench/learn_eval.py`：跑小规模训练并输出 `bench_summary.json`，用同一个 `bench/compare.py` 进行门禁。

---

## 1. Bench / Eval（物理基线回归门禁）

### 1.1 入口与输出

- `bench/eval.py`：支持 `--suite mvp|mvp_ood`，输出 `outputs/bench/<timestamp>-eval/bench_summary.json`。
- `bench/compare.py`：读取 `bench_summary.json`，按 `bench/suites/*.yaml` 的阈值检查，失败返回非 0。

### 1.2 Suites 与配置

- **ID suite**
  - configs：`configs/mvp_*.yaml`
  - gate：`bench/suites/mvp.yaml`
- **ID + 轻量 OOD suite**
  - configs：`configs/mvp_*.yaml` + `configs/bench_ood_*.yaml`
  - gate：`bench/suites/mvp_ood.yaml`
  - OOD configs（当前）
    - `configs/bench_ood_disk_stack_2d_dt.yaml`（dt=0.02）
    - `configs/bench_ood_disk_stack_2d_mass.yaml`（mass=2.0）
    - `configs/bench_ood_disk_stack_2d_radius.yaml`（radius=0.4）
    - `configs/bench_ood_disk_roll_2d_mass.yaml`（mass=2.0）

> 说明：OOD dt=0.02 会把 `disk_stack_2d` 的 `penetration_max` 从 ~2e-3 推到 ~5.3e-3，因此 `mvp_ood` gate 在 `disk_stack_2d` 的穿透阈值上比 `mvp` 略放宽（0.006 vs 0.005），其余保持一致。

### 1.3 推荐命令

```bash
# ID
python bench/eval.py --suite mvp --device cpu --dtype float64 --deterministic --repeat 2
python bench/compare.py --candidate outputs/bench/<...>/bench_summary.json --suite bench/suites/mvp.yaml

# ID + OOD
python bench/eval.py --suite mvp_ood --device cpu --dtype float64 --deterministic --repeat 2
python bench/compare.py --candidate outputs/bench/<...>/bench_summary.json --suite bench/suites/mvp_ood.yaml
```

---

## 2. 最小可学习切片（μ）与结构正则（可消融）

### 2.1 学习参数

- `src/scone/learned/mu.py`
  - `ScalarMu`：单参数 μ（用 sigmoid * mu_max 保证可行域）
  - `GroundPairMu`：两参数 μ（ground vs body-body）

### 2.2 训练脚本（面向研究迭代）

- `scripts/train_m2_scalar_mu.py`
  - 单步监督信号：DiskGroundContactEventLayer 上拟合一个 penetrating state 的下一步速度（用于验证“μ 可学”）。
- `scripts/train_m2_mu_rollout.py`
  - `disk_roll_2d` 场景的 rollout 监督（多步误差对 μ 反传）。
- `scripts/train_m2_mu_pgs_stack.py`
  - `disk_stack_2d` + PGS 多接触：标量 μ 学习。
  - `--loss-mode open_loop|teacher_forcing`
  - 结构正则（默认 0，可用于 ablation）：
    - `--w-pen`：`contacts.penetration_max`
    - `--w-comp`：`contacts.complementarity_residual_max`
    - `--w-res`：`solver.residual_max`
- `scripts/train_m2_mu_pgs_stack_pair.py`
  - `disk_stack_2d` + PGS 多接触：`mu_ground` / `mu_pair` 双参数学习。
  - 同样支持 `--w-pen/--w-comp/--w-res`。

> 备注：目前的结构正则属于“结构化诊断驱动的辅助项”，不是硬约束；其核心价值是：后续换 PGS/加更多接触时，无需先重构日志与评测，就能系统做 ablation。

---

## 3. 学习 Bench / Gate（训练 + ID/OOD 的回归门禁）

### 3.1 入口与 suites

- `bench/learn_eval.py`
  - `--suite mu_scalar|mu_pair|mu_all`
  - 输出 `outputs/bench/<timestamp>-learn_eval/bench_summary.json`
  - 与 `bench/eval.py` 共用 `bench/compare.py` 门禁逻辑（阈值写在 suite yaml）

Suites：
- `bench/suites/learn_mu.yaml`：标量 μ（stack）
- `bench/suites/learn_mu_pair.yaml`：pair μ（stack，重点 gate `mu_pair_abs_error`）

### 3.2 当前学习配置（configs）

- 标量 μ（ID + OOD）
  - `configs/learn_mu_stack_id.yaml`
  - `configs/learn_mu_stack_id_reg.yaml`（开启 `w_pen/w_comp/w_res`）
- 双 μ（ID + OOD）
  - `configs/learn_mu_pair_stack_id.yaml`

### 3.3 推荐命令

```bash
# scalar μ
python bench/learn_eval.py --suite mu_scalar --device cpu --dtype float32 --deterministic --repeat 1
python bench/compare.py --candidate outputs/bench/<...>/bench_summary.json --suite bench/suites/learn_mu.yaml

# pair μ
python bench/learn_eval.py --suite mu_pair --device cpu --dtype float32 --deterministic --repeat 1
python bench/compare.py --candidate outputs/bench/<...>/bench_summary.json --suite bench/suites/learn_mu_pair.yaml
```

### 3.4 观测到的结果（本地记录，2026-02-06）

- 标量 μ（ID + dt=0.02 OOD）
  - `mu_abs_err ≈ 0.039`
  - `loss_ratio_val ≈ 0.011`
  - `loss_ratio_ood ≈ 0.011`
- 标量 μ + 结构正则（同设置）
  - `loss_ratio_val ≈ 0.019`、`loss_ratio_ood ≈ 0.016`（更“保守”，但仍显著优于 baseline）
- pair μ（重点看 `mu_pair`）
  - `mu_pair_abs_error ≈ 0.008`
  - `loss_ratio_val ≈ 0.031`、`loss_ratio_ood ≈ 0.006`
  - **注意**：`mu_ground` 在该设置下可能不稳定/偏离真值（可辨识性不足：粘着/支撑主导时 ground μ 很难被轨迹充分激励）。

---

## 4. 测试与可复现性

- `pytest -q`：当前为 `10 passed`（含 bench compare 的门禁测试）。
- 物理 bench：
  - `mvp`：`bench/suites/mvp.yaml` 通过
  - `mvp_ood`：`bench/suites/mvp_ood.yaml` 通过
- 学习 bench：
  - `learn_mu`：`bench/suites/learn_mu.yaml` 通过
  - `learn_mu_pair`：`bench/suites/learn_mu_pair.yaml` 通过

---

## 5. 下一步建议（按对后期收益排序）

1. **从“常数 μ”升级到 “μ(x)”（接触状态依赖）**
   - 用小 MLP 预测 μ（输入可从每个 contact 的局部量抽取：`phi, vn, vt, body ids/material id` 等）。
   - 先保持 PGS 求解器不变：只替换 `mu` 的闭合项，并继续沿用当前的结构正则与 learn bench gate。
2. **扩展 learn bench 的 OOD 维度**
   - 把 `mass / radius / dt` 作为可组合 stress-case，形成更完整的 “ID + light OOD regression”。
3. **可辨识性增强（pair μ）**
   - 设计激励（例如底盘/顶盘初速度分布或加入可控水平驱动）让 ground/pair 都进入滑动区间，从而可稳定拟合两参数。

