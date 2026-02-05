我把你现在 `Void0312Aurora/SCONE` 里 MVP 路径（`scripts/run_mvp.py` → `Engine` → A/B/C/D 四层）过了一遍，结合你汇报的测试现象，代码整体“按规范落地”做得很扎实：接口清晰、日志与 plots 可复现、GPU 确定性也确实被 `torch.use_deterministic_algorithms(True)` 这套兜住了（`--deterministic`）。你这版实现已经很好地对齐了你 spec 里“引擎级 step + Diagnostics + 确定性”的目标（见你最初规范草案 ）。

下面是我认为**最关键的逻辑问题/改进点**，按优先级排序（基本都对应你现在 bouncing_ball 的 Zeno/抖动现象，以及你下一阶段要扩展到 time-stepping+PGS 的基础工作）。

---

## P0：BouncingBall 事件层的“位置修正”导致账本语义不完整（并放大 Zeno）

在 `BouncingBallEventLayer.resolve()` 里你做了两件事：

1. 穿透时把 `q` 直接夹到 `ground`：`next_q = ground`
2. 同时对速度做恢复系数反弹：`next_v = -e*v`

但你 `event.dE_event` 只统计了**动能变化** `d_ke`，没有把 **“把球从 q<ground 抬到 ground”**造成的**势能跳变**记入事件账本（而 `Engine` 的 `dE_D` 会包含这部分，因为它用 `system.energy()`算了总能量差）。

这会导致两类问题：

* **账本对不上**：`event.dE_event` ≠ `energy.dE_D`（尤其穿透不为 0 时）；你后面要做“事件账本一致性约束/可训练闭合项”时会很痛。
* **Zeno 更容易出现**：你每一步都在“强行抬起”+“反弹”，离散误差会把系统卡在地面附近抖动。

**建议（最小修复）**
把事件层的 `dE_event` 改成“事件前后总能量差”，或者至少加上势能跳变项（按你 `BouncingBall1D.energy` 的定义，势能是 `m*g*(q-ground)`）：

* `d_pe = m*g*(next_q - q)`
* `dE_event = d_ke + d_pe`

并把 `W_impulse` 的语义也固定：如果你把“位置夹紧”视为**非物理的数值校正**，那就应该单独记录 `W_position_correction`（否则会混到冲量做功里）。

---

## P0：现在的 Zeno/抖动不是 bug，是“缺一个静止接触模型”

你自己也判断对了：当前是“简单恢复系数反弹”，在离散检测下必然会出现微反弹链（你观察到 ~1023 次）。原因很简单：

* `dt` 固定；
* `q` 一旦略微穿透就会被夹回地面；
* `v` 只要仍是负且穿透，就会再次触发 `do_bounce`；
* 当能量很小后，系统会进入“反复触发事件”的极限循环。

**最低成本的引擎式修法：加入 “resting contact / sleep” 阈值**
给事件层加两个阈值（写进 config）：

* `v_sleep`：当 `abs(v) < v_sleep` 且 `q <= ground + q_slop`，直接令 `v=0` 并保持 `q=ground`，标记 `contact_mode="resting"`。
* `q_slop`：接触容差（避免数值噪声导致反复进出穿透状态）

这会立刻把 Zeno 链砍掉，也更符合你 spec 里“工程可控 + 可回退”的精神。

---

## P1：`Engine` 里的 `P_diss` 计算方式会“把错误藏起来”

你现在是：

```python
p_diss = torch.clamp(-d_e_b / dt, min=0.0)
```

这意味着：

* 如果 B 层（耗散层）由于数值/实现问题 **导致能量增加**（`dE_B>0`），你会把 `P_diss` 直接夹到 0，看起来“一切正常”，但其实 ledger 已经违背了“耗散层不应注入能量”的语义。

**建议**

* 同时记录 `power.P_diss_signed = -dE_B/dt` 和 `power.P_diss = clamp(...)`（一个用于诊断/约束，一个用于“物理上解释”）。
* 或者更干净：让 `DissipationInputLayer` 直接输出它“理论上”的 `P_diss_model`（比如线性阻尼可以直接算），然后把 `dE_B` 只当作误差。

---

## P1：事件层的互补残差目前是常数 0，后续扩展会卡住

`BouncingBallEventLayer` 里：

```python
"complementarity_residual": 0.0
```

这在 MVP 可以接受，但如果你下一步要上 time-stepping/PGS 或者想做论文级指标，这个字段应该至少能反映：

* 非穿透：`phi(q+) >= -slop`
* 法向冲量：`lambda_n >= 0`
* 互补：`lambda_n * phi(q+) ≈ 0`（或速度级条件）

哪怕先给一个简化版本，也能让“接触质量曲线”真正有意义。

---

## P2：Fail-safe 现在只兜底了 non-finite，建议按 spec 把 hard/soft fail 分级做起来

你现在 `failsafe.triggered` 只有 `_all_finite_state` 失败才会 True。

结合你 spec 的目标（可回退、可 debug），下一步最好把以下 hard fail 变成真实逻辑（即使先不回退，也要记录）：

* `penetration_max > penetration_hard`
* `abs(v) > v_hard`
* `energy.E_total` 非物理爆炸（比如超过初值 * 某倍数，且无输入功率）

软 fail 则用于触发“限幅/睡眠/混合”。这一步做好，你后面引入学习闭合项会非常省心。

---

## 你现在的结果如何解读（你测的这些值都“合理”）

* harmonic 能量 ±1% 漂移：符合 `SymplecticEulerSeparable` 的特性（有界不严格守恒）。
* damped 能量严格单调下降：你 B 层用的是精确指数衰减 `exp(-c dt)`，因此非常干净。
* bounce 首次触地时刻接近理论值：说明 A 层（重力势）+ 事件触发逻辑总体没错；Zeno 是缺“静止接触模式”，不是你整体设计错。

---

## 我建议你下一步的最小改动（能最快把 bounce 变“引擎级”）

1. **事件账本补齐**：`dE_event` 改为总能量跳变（至少加势能项）。
2. **加入 resting contact 阈值**：`v_sleep` + `q_slop`，把 Zeno 杀掉。
3. **Diagnostics 更诚实**：记录 `P_diss_signed`、真实互补残差（哪怕简化）。

做完这三步，你的 bouncing_ball 会从“可跑 demo”升级到“可以当论文图的 demo”，同时也为你说的 time-stepping+PGS 铺好字段与评测协议。

如果你希望我进一步“审阅到具体行级改动建议”，你把你准备改的分支名/commit SHA 发我（或直接给我 PR 链接），我可以按文件逐段给 review（包括建议的阈值默认值与日志字段命名）。
