我按你给的 `3ce5458214fd89255a2fca6179db8167258a9ddb`（相对 `dd4a5f9...` ahead 1 commit）把关键改动过了一遍：这次更新**很明确地把方向 (2)“接触堆叠 + 摩擦”落到了可跑的工程骨架**上，而且把“resting/sleep + 可复现”这条主线也补齐了。下面是审阅结论 + 下一步我建议你怎么走。

---

## 这次提交做对了什么（结构性收益）

### 1) 你把“接触语义契约”从 demo 提升成了引擎级对象

* 新增 `src/scone/contacts.py`：`ContactMode`（separated/impact/resting/active/sliding/sticking）+ `Contact` 数据结构与 `mode_name`，这会让**日志、评测、sleep islands、PGS**都能围绕统一字段运转。
  这一步非常关键：没有稳定的 contact schema，后面堆叠/摩擦/约束都会反复返工。

### 2) sleep/resting 被做成可复用策略，并且“冻结 core”打通了

* `src/scone/sleep.py` 的 `SleepManager` 不是简单阈值，而是：

  * 通过 contact graph 做 islands（并查集）
  * 需要“被静态体支撑”才允许入睡（supported_by_static）
  * impact 或速度超过 `v_wake` 可唤醒
* `src/scone/layers/symplectic.py` 里 `_apply_sleep_freeze`：sleeping 时把 **q 回滚到 q_prev、v 置零**，避免“数值爬行”。
  这件事在“堆叠稳定 + 长时 deterministic”里几乎是必需品。

### 3) 你已经有了两个“引擎味”场景：滚动摩擦、堆叠接触（PGS）

* `Disk2D`（`src/scone/systems/rigid2d.py`）给了一个最小刚体 state：`(x,y,theta)` / `(vx,vy,omega)`，并且能量可算。
* `DiskGroundContactEventLayer`：单体-地面 + 库仑摩擦（stick/slide）闭式更新。
* `DiskContactPGSEventLayer`：地面 + 多圆盘 + 摩擦的 PGS time-stepping 框架，已经能支撑你要的 **contact stacking** 叙事。
* `scripts/run_mvp.py` 增加了 `disk_roll_2d` 与 `disk_stack_2d` 的可跑入口、画图与 jsonl 日志，能形成闭环。

### 4) `docs/spec.md` 同步更新到了 v0.2，并写死了“当前选择 (2)”

这对你后面写论文/写 README 也很重要：读者会看到你不是“又一个 HNN”，而是 **engine-step semantics + diagnostics contract + fail-safe/determinism** 的路线。

---

## 我看到的逻辑风险 / 需要尽快加固的点（按优先级）

### P0：PGS 的“物理一致性”与“可控退化”还缺一层护栏

`DiskContactPGSEventLayer` 现在属于“能跑、结构对”，但作为引擎基线，还差这些工程护栏：

1. **收敛/失败判据**
   你目前固定迭代次数 `pgs_iters`，但没有输出类似：

* `solver.iters`（实际迭代）
* `solver.residual_max`（互补残差/速度残差）
* `solver.status`（converged / max_iter / diverged）
  否则后续引入学习或做 ablation 时，你很难解释“为什么这次爆了 / 为什么这次睡死了”。

2. **warm-start（强烈建议尽早做）**
   堆叠一旦多起来，PGS 不 warm-start 会导致：

* 需要更多 iters
* 抖动更明显
* 更依赖 sleep 去“盖住”误差
  做法很简单：把上一帧每个 contact 的 `(lambda_n, lambda_t)` 缓存在 `context`，按稳定 `contact_id` 对齐回填。

3. **“位置修正”和“速度求解”的账本要更明确地区分**
   你在 PGS 里做了 position correction（最后 `pos_next` 的修正），但 event ledger 目前用 `d_ke + d_pe` 统一记账。
   建议你把：

* **冲量导致的能量跳变**（纯速度层）
* **位置投影导致的能量改动**（几何修正）
  拆成两个字段（你 spec 里其实已经写了 `W_position_correction` 的建议），这样你后面讲“结构一致性 / 能量预算”会更站得住。

---

### P1：`DiskGroundContactEventLayer` 的 mode 判定有点“语义滑坡”

这里有一段：

```python
elif bool(jn[i].item() == 0.0) and bool(contact_active[i].item()):
    mode = ACTIVE
else:
    is_sticking = ...
    mode = STICKING/SLIDING
```

当 `contact_active` 但 `jn == 0` 时，你标为 `ACTIVE`，但这在日志语义上有点怪：

* 对地面接触而言，“active”通常意味着**法向反力/冲量正在支撑**（jn>0）。
* `jn==0` 更像“近接触但未接触/将分离”（separated-ish），或至少应该叫“touching but no impulse”。

建议你把 mode 语义做一个更严格的定义（尤其为了 sleep）：

* `RESTING`: 接触存在且被支撑（lambda_n>0 或稳定支撑判据成立）
* `ACTIVE`: 正在解算接触（接触集合里参与求解的都可叫 active）
* `SEPARATED`: `phi > slop`
* `IMPACT`: 满足冲击判据
  否则 sleep islands 的“supported_by_static”可能把一些其实没支撑的也算进去。

---

### P2：接触几何/雅可比的抽象迟早要来，否则约束与可学习闭合项会卡住

你现在的实现是“为圆盘定制的解析更新”，这是 MVP 很好的路线；但下一步如果你要：

* 加入**关节约束**

* 加入**更多形状**

* 或把接触/摩擦做成可学习闭合项
  那你会自然需要一个统一的：

* `J(q)`（接触雅可比：法向/切向）

* `M^{-1}` 的局部算子

* `b`（bias：Baumgarte/恢复系数/稳定项）

* `solve(λ)` 的接口

不一定现在就重构，但建议你至少在代码层预留一个“ContactConstraint”中间层（哪怕只给 disk 用），让 PGS 不直接写死在 event layer 里。

---

## 下一步行动方向（我建议你继续押注 (2)，并把“堆叠 + 摩擦 + 可复现”做成你最硬的卖点）

下面是一个我认为**最能产出论文/引擎价值**的 3 段式计划（每段都能交付可验收结果）。

### A. 先把 PGS 基线做成“可发布的引擎模块”（1–2 个迭代）

**目标：**堆叠稳定、参数可控、日志可解释、失败可回退

1. 在 `DiskContactPGSEventLayer` 增加 `solver.*` diagnostics：残差、iters、status
2. warm-start：`context["pgs_cache"]` 按 `contact_id` 存取 `(lambda_n, lambda_t)`
3. 更严格的 mode 定义 + 在 diagnostics 里输出 mode 统计（impact/resting/sliding…计数）
4. fail-safe：当残差超阈值或出现明显穿透增长时，降级策略：

   * 增大 `baumgarte_beta` / 增大 iters（软）
   * 或直接切到“无摩擦/无恢复”的保守版本（硬）

**验收用例：**

* 盒子/圆盘堆叠高度从 2 → 10，仍能进入 resting
* 复跑 deterministic 一致
* penetration、complementarity residual 有可读曲线

---

### B. 做一个“摩擦最小闭环”的可学习切片（论文味开始出现）

你 spec 里的 M2 说“先学 D 层参数”，我同意，而且你现在的结构非常适合从这两种开始：

1. **学 `mu`（摩擦系数）**：
   输入可以是接触局部状态（`vn, vt, phi, material id`），输出 `mu_hat`，再投影到 `[0, mu_max]`。
   损失可以用：

* 轨迹拟合（终态/多步）
* 约束：摩擦锥违反率（`|lambda_t| <= mu * lambda_n`）

2. **学“切向速度衰减/微滑”闭合**（比直接学 mu 更稳）：
   让网络输出一个 `alpha in [0,1]`，把 `vt_target = (1-alpha) * vt`，这更像“可控耗散”，训练更好收敛。

**关键：**你已经有 ledger 与 fail-safe 叙事了，所以这块很容易做出“结构化学习”对比。

---

### C. 扩展到“可写进引擎 README 的 feature”：堆叠 + 摩擦 + sleep islands

等 A/B 跑顺后，你就可以把项目的核心卖点写成：

* time-stepping + PGS（deterministic）
* sleep islands（可选 freeze core）
* ledger + diagnostics 合同
* 可学习闭合项（但带 fail-safe）

这会比单纯 “contact neural net” 更像“可用的 engine”。

---

## 我会怎么定位你现在的项目阶段

你现在已经从“弹跳球修 Zeno”跨过了一个门槛：开始出现**真实引擎会有的模块**（PGS、sleep islands、determinism、可读日志）。
所以我建议下一步不要急着扩形状/扩约束，而是先把 **PGS 基线 + diagnostics + fail-safe** 做到“像一个小引擎”，然后再引入最小学习切片去证明“结构化可学习”确实有增益。

如果你愿意，我也可以基于你现在 `disk_stack_2d` 的配置与日志字段，帮你把“验收指标表”（penetration、residual、iters、sleeping_count、mode 分布、能量账本）整理成一份固定模板，后面你每次改 solver 直接对比曲线就行。
