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
   - 新增状态依赖摩擦 `μ(x)`（接触特征驱动）与材料条件化 `μ(x)`（material-aware）训练脚本与 bench gate。
   - 支持 open-loop 多步 BPTT（PGS 计算图可回传梯度）+ 结构一致性正则项（penetration / complementarity / solver residual）。
4. **学习 bench gate（训练 + ID/OOD 一键回归）**
   - `bench/learn_eval.py`：跑小规模训练并输出 `bench_summary.json`，用同一个 `bench/compare.py` 进行门禁。
   - `mu_field` 已补充可辨识性 gate：`mu_probe_mae_seen`、`mu_probe_mae_holdout_material`、`mu_probe_pair_*` 与 `ood_case_metrics.<case>.loss_ratio`。
5. **M3 消融矩阵自动化（新增）**
   - `bench/ablate_mu_field.py`：基于一个 base config 自动生成 ablation configs、批量运行 `learn_eval`、产出 `ablation_matrix.{json,md}` 和 `plots/*.png`。

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
  - `ContactMuMLP`：接触特征 `phi/vn/vt/is_pair` 到 `μ(x)` 的有界映射（sigmoid * mu_max）
  - `ContactMuMaterialMLP`：在 `ContactMuMLP` 基础上加入 `material_i/material_j` 条件特征

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
- `scripts/train_m2_mu_pgs_stack_field.py`
  - `disk_stack_2d` + PGS 多接触：状态依赖摩擦 `μ(x)` 学习（保持 PGS 不变，仅替换每接触 `mu` 闭合项）。
  - 支持材料条件化特征（`material_i/material_j`）、可配置材料池与真值 `material_bias`。
  - 数据由可控真值规则生成，评测提供 `mu_probe_mae`。

> 备注：目前的结构正则属于“结构化诊断驱动的辅助项”，不是硬约束；其核心价值是：后续换 PGS/加更多接触时，无需先重构日志与评测，就能系统做 ablation。

---

## 3. 学习 Bench / Gate（训练 + ID/OOD 的回归门禁）

### 3.1 入口与 suites

- `bench/learn_eval.py`
  - `--suite mu_scalar|mu_pair|mu_field|mu_all`
  - `mu_field` 已支持 `task.ood_cases`（多场景 OOD，自动汇总 worst-case 指标）
  - 输出 `outputs/bench/<timestamp>-learn_eval/bench_summary.json`
  - 与 `bench/eval.py` 共用 `bench/compare.py` 门禁逻辑（阈值写在 suite yaml）

Suites：
- `bench/suites/learn_mu.yaml`：标量 μ（stack）
- `bench/suites/learn_mu_pair.yaml`：pair μ（stack，含 `mu_gap_abs_error/mu_gap_sign_correct` 可辨识性 gate）
- `bench/suites/learn_mu_field.yaml`：`μ(x)`（重点 gate：`mu_probe_mae` + seen/holdout probe + case-level OOD + solver/failsafe）

### 3.2 当前学习配置（configs）

- 标量 μ（ID + OOD）
  - `configs/learn_mu_stack_id.yaml`
  - `configs/learn_mu_stack_id_reg.yaml`（开启 `w_pen/w_comp/w_res`）
- 双 μ（ID + OOD）
  - `configs/learn_mu_pair_stack_id.yaml`
  - `configs/learn_mu_pair_stack_ident_stress.yaml`（`dt/mass` 组合 stress + 可辨识性实验）
- `μ(x)`（ID + OOD）
  - `configs/learn_mu_field_stack_id.yaml`
  - `configs/learn_mu_field_stack_stress.yaml`（`dt/mass/radius` + `dt+mass+radius` 组合 OOD）
  - `configs/learn_mu_field_stack_material_stress.yaml`（material-aware + `dt/mass` + `dt+mass+radius` 组合 OOD）
  - `configs/learn_mu_field_stack_material_holdout.yaml`（单材质留出 + 材质组合留出）

### 3.3 推荐命令

```bash
# scalar μ
python bench/learn_eval.py --suite mu_scalar --device cpu --dtype float32 --deterministic --repeat 1
python bench/compare.py --candidate outputs/bench/<...>/bench_summary.json --suite bench/suites/learn_mu.yaml

# pair μ
python bench/learn_eval.py --suite mu_pair --device cpu --dtype float32 --deterministic --repeat 1
python bench/compare.py --candidate outputs/bench/<...>/bench_summary.json --suite bench/suites/learn_mu_pair.yaml

# pair μ identifiability stress
python bench/learn_eval.py --configs configs/learn_mu_pair_stack_ident_stress.yaml --device cpu --dtype float32 --deterministic --repeat 1

# field μ(x)
python bench/learn_eval.py --suite mu_field --device cpu --dtype float32 --deterministic --repeat 1
python bench/compare.py --candidate outputs/bench/<...>/bench_summary.json --suite bench/suites/learn_mu_field.yaml

# field μ(x) material-aware stress
python bench/learn_eval.py --configs configs/learn_mu_field_stack_material_stress.yaml --device cpu --dtype float32 --deterministic --repeat 1

# field μ(x) material holdout（single/combo）
python bench/learn_eval.py --configs configs/learn_mu_field_stack_material_holdout.yaml --device cpu --dtype float32 --deterministic --repeat 1

# M3 消融矩阵自动化
python bench/ablate_mu_field.py --base-config configs/learn_mu_field_stack_material_holdout.yaml --device cpu --dtype float32 --deterministic
```

### 3.4 观测到的结果（本地记录，2026-02-06）

- 标量 μ（ID + dt=0.02 OOD）
  - `mu_abs_err ≈ 0.039`
  - `loss_ratio_val ≈ 0.011`
  - `loss_ratio_ood ≈ 0.011`
- 标量 μ + 结构正则（同设置）
  - `loss_ratio_val ≈ 0.019`、`loss_ratio_ood ≈ 0.016`（更“保守”，但仍显著优于 baseline）
- pair μ（重点看 `mu_pair`）
  - `mu_ground_abs_error ≈ 0.016`
  - `mu_pair_abs_error ≈ 0.037`
  - `mu_gap_abs_error ≈ 0.021`（gap 符号正确）
  - `loss_ratio_val ≈ 6.0e-4`、`loss_ratio_ood ≈ 0.008`
- pair μ（identifiability stress + 组合 OOD）
  - `mu_ground_abs_error ≈ 0.014`
  - `mu_pair_abs_error ≈ 0.017`
  - `mu_gap_abs_error ≈ 0.0029`（gap 符号正确）
  - `loss_ratio_val ≈ 4.7e-4`
  - `loss_ratio_ood_worst ≈ 0.0018`（`dt_2x/mass_2x/dt_2x_mass_2x` 三场景最坏）
- `μ(x)`（field）
  - `mu_probe_mae ≈ 0.099`
  - `loss_ratio_val ≈ 0.001`
  - `loss_ratio_ood ≈ 0.006`
- `μ(x)`（field + 多场景 OOD stress）
  - `mu_probe_mae ≈ 0.092`
  - `loss_ratio_val ≈ 2.2e-4`
  - `loss_ratio_ood_worst ≈ 0.009`（`dt_2x/mass_2x/radius_08x` 三场景中取最坏）
- `μ(x)`（field + material-aware stress）
  - `mu_probe_mae ≈ 0.106`
  - `loss_ratio_val ≈ 0.122`
  - `loss_ratio_ood_worst ≈ 0.0064`（`dt_2x/mass_2x` 两场景最坏）
  - `use_material_features = true`，`body_material_pool = [1,2,3]`
- `μ(x)`（field + material holdout）
  - train/val 仅使用预定义材质模式（不含 holdout 组合），OOD 包含：
    - `unseen_material4_only`（单材质留出）
    - `holdout_pair_13` / `holdout_pair_13_dt2x`（组合留出）
  - `mu_probe_mae ≈ 0.069`
  - `loss_ratio_val ≈ 0.0105`
  - `loss_ratio_ood_worst ≈ 0.0198`
- `μ(x)`（field + material holdout，新增可辨识性 gate 指标）
  - `mu_probe_mae_seen ≈ 0.097`
  - `mu_probe_mae_holdout_material ≈ 0.054`
  - `mu_probe_pair_mae_holdout ≈ 0.064`
  - `mu_probe_pair_holdout_ratio ≈ 0.66`
  - 分 case OOD：`unseen_material4_only≈0.046`、`holdout_pair_13≈0.0096`、`holdout_pair_13_dt2x≈0.0060`
- M3 消融（轻量设置，`max_train_iters=50, n_train=8, n_val=6`）
  - `base`: `loss_ratio_val≈0.0323`, `loss_ratio_ood_worst≈0.0623`, `mu_probe_mae≈0.0766`
  - `struct_reg`: `0.0464 / 0.0837 / 0.0766`（当前配置下偏保守，误差略升）
  - `no_warm_start`: 与 `base` 基本一致（该切片上 warm-start 非主导）
  - `low_pgs_iters`: `loss_ratio`接近，但 `eval_ood.solver.max_iter_ratio` 从 `0.0667` 升至 `0.3667`
  - `small_mlp`: `mu_probe_mae` 变差到 `≈0.106`（容量不足先影响参数场可辨识性）

---

## 4. 测试与可复现性

- `pytest -q`：当前为 `27 passed`（含 warm-start/sleep-wake/OOD residual 与 P3-2 解释性统计测试）。
- 物理 bench：
  - `mvp`：`bench/suites/mvp.yaml` 通过
  - `mvp_ood`：`bench/suites/mvp_ood.yaml` 通过
- 学习 bench：
  - `learn_mu`：`bench/suites/learn_mu.yaml` 通过
  - `learn_mu_pair`：`bench/suites/learn_mu_pair.yaml` 通过
  - `learn_mu_field`：`bench/suites/learn_mu_field.yaml` 通过

---

## 6. 与目标距离（2·6 阶段量化）

对照 `docs/spec.md` 的 (2) 路线（接触堆叠 + 摩擦 + 可复现 + 可学习切片）：

1. **已完成（可门禁）**
   - 多接触 PGS + warm-start + sleep islands + solver/failsafe 诊断
   - P3-1 门禁安全带：solver 状态分布指标（`max_iter/diverged/na`）写入 summary，并纳入 suite gate（以 `diverged_ratio` 为硬约束）
   - P3-1 回归用例：`warm-start on/off` 差异约束、`sleep+wake` 无穿透尖峰、`OOD(dt/mass/radius)` residual 稳定性
   - P3-2 可解释性切片：`bench/p3_mu_interpret.py` 产出 `mu_probe_mae/loss_ratio/penetration/residual/drift` 收敛曲线，并基于可控单盘滑移距离做 `mu_avg` 与 distance 标定对照（`summary.{json,md}` + `plots/*.png`）
   - 标量 μ / pair μ / `μ(x)` / material-aware `μ(x)` 四条最小可学习切片
   - ID + OOD bench 与 suite 门禁（含 pair 与 `μ(x)` 的多场景 OOD stress）
   - pair μ 可辨识性 gate（`mu_gap_abs_error/mu_gap_sign_correct`）与增强实验
   - material-aware `μ(x)` gate（`use_material_features=true` + seen/holdout probe + case-level OOD + solver/failsafe）
   - `μ(x)` 组合 OOD（`dt+mass+radius`）stress 配置与 gate 验证
2. **进行中（下一里程碑）**
   - 将 `μ(x)` 的可辨识性从材料 ID 进一步扩展到更强条件化特征（例如法向压力分段）
3. **未开始（后续阶段）**
   - island 级求解/并行与大规模接触性能
   - 约束层与接触层更紧耦合（关节 + 接触统一评测）

---

## 5. 下一步建议（按对后期收益排序）

1. **可辨识性增强（从参数到函数）**
   - 在现有 material-aware `μ(x)` 基础上，增加“单材质留出/组合留出”的 OOD 可辨识性测试切片。
2. **将 `μ(x)` 推进到更强条件化实验**
   - 在 `material_i/material_j` 之外加入接触类型/法向压力分段特征，并保持现有 gate 接口不变。
3. **扩展 OOD 到联合极端场景**
   - 在 `dt+mass+radius` 的基础上增加更极端组合（例如更大 `dt` 或更低迭代预算）并观察 fail-safe/solver 分布。
