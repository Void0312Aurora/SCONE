我看了你给的 commit `83b3a05a05701901a89374400c0082ccabbefad6`（相对 `3ce5458214fd...` ahead 1 个 commit）。结论：**这次改动把“能跑的 demo”升级成了“可量化评测 + 可回归门禁 + 可学习切片”的工程形态**，方向选择很对；同时也引入了几类下一步必须盯紧的数值/工程风险点（主要在 PGS 的收敛性、warm-start 缓存一致性、sleep 与接触求解的交互）。

> 小透明说明：我这边的 `file_search(slurm_github)` 代码检索返回 0（像是连接器侧没搜到），所以我用 GitHub 直链打开了你这次提交的源码文件来审阅（等你那边确认一下 repo 选择/权限后，我也可以再用 file_search 给你带行号引用式的 code review）。

---

## 1) 你这次结果“好在哪里”（工程验收视角）

从 `temp-01.md` 的“落盘”目标来看，你已经把 (2) 路线（接触堆叠+摩擦+可复现+可学习切片）真正闭环了：

* **Bench / regression gate 成型**：`bench/eval.py` 产出统一 `bench_summary.json`，`bench/compare.py` + suite YAML 做阈值门禁；并且加了 OOD suite（dt/mass/radius stress）。这一步非常关键：它把“物理引擎研究”变成“工程可迭代的产品形态”。
* **可学习切片明确且最小**：把学习目标聚焦在 μ（标量 / ground+pair 双 μ），并且明确用 open-loop 多步 BPTT（“PGS 计算图可回传梯度”）+ 结构一致性正则（penetration / complementarity / residual）。这比泛泛的“学一个 contact solver”可控太多。
* **PGS 求解器工程化**：你在 `DiskContactPGSEventLayer` 上补齐了

  * `residual_tol` 与收敛状态（`status/iters/residual_max`），
  * `warm_start` + `pgs_cache`（并且缓存值 `detach`），
  * `mode_counts` 统计，
  * 以及对 sleep 上下文的适配（读取 `sleeping_mask`）。

这四件套一上，后面做“接触堆叠 + 摩擦学习”才有抓手。

---

## 2) 代码层面我认为最关键的改动点（以及意义）

我重点读了这个文件（head commit）：

* `src/scone/layers/events.py`（你改动最大的地方，新增 warm-start、residual、μ 可学习等）
* `src/scone/engine.py`（failsafe + soft/hard gate 扩展）

### A. μ 从常数变成 “float | Tensor”，并支持 ground/pair 分离

在 `DiskGroundContactEventLayer` / `DiskContactPGSEventLayer` 里把 `friction_mu` 升级为 `float | torch.Tensor`，并引入 `friction_mu_pair`（pair contact 可以不同 μ）。这一步是把“引擎摩擦”自然过渡到“可学习摩擦”的核心桥梁。

### B. warm-start 缓存的实现方式是对的（但要补一致性测试）

你做了三件正确的事：

1. 缓存值在写回时 `detach`（避免把跨 step 的求解历史塞进计算图，图爆炸）。
2. warm-start 时会把 λ 投影回摩擦锥（按当前 μ 和 λn clamp λt）。
3. stale contact id 会清理（避免 cache 漂移污染）。

**风险点**：contact id 的稳定性（尤其 disk-disk 的 pair 在排序/重排/并行时是否稳定），以及 sleep 参与时对 cache 的跳过逻辑是否会导致 “醒来第一帧”冲击。

### C. 你引入了 residual 检查，这是后续“学习门禁”的灵魂

你现在每轮迭代后会扫一遍 contacts 计算一个 `residual_max`，并且满足阈值就提前停止（status=converged）。
这给了你一个非常干净的 **fail-soft / fail-hard** 接口：比如 “残差太大就降 dt / 提升迭代数 / 或触发回退”。

### D. `_index_add_*` 这类写法是在向“GPU 确定性”靠拢

你用 `torch.index_add` 去避免一些 in-place scatter 在不同设备上的非确定性，这个动机非常好，且和你前面强调 `--deterministic` 完全一致。

---

## 3) 我认为现在最该警惕的逻辑/数值风险（按优先级）

### (P0) PGS 的收敛性与“门禁阈值”可能会互相打架

你现在有了 residual，但还需要在 bench suite 里明确：

* 在 **哪些场景** residual 要求严格（例如 stack 静止、resting contact）
* 在 **哪些场景** residual 可以放松（高速碰撞、强约束冲击阶段）

否则会出现：为了过 gate 不得不把 solver_iters 拉很高 → 速度慢；或阈值太松 → 学习阶段梯度信号变脏。

### (P0) sleep 与 warm-start/cache 的交互需要一个最小可复现测试

你已经读取 `sleeping_mask` 并在 warm-start 时跳过 sleeping bodies 的 cache 应用，但仍建议加一个回归用例：

* A：堆叠静止进入 sleep
* B：外力轻推唤醒上层一个 disk
* 期望：不会出现 “第一帧爆冲/穿透突增/λ 缓存错误注入”。

### (P1) residual 的计算目前是 per-contact 扫描，会成为大规模接触的瓶颈

短期 OK（你现在主要是 MVP/bench）。但你下一条路线如果是“接触堆叠 + 摩擦”（引擎向），很快会触及性能问题。
建议：把 residual 计算也逐步 tensorize（至少把 contacts_work 打包成张量结构）。

### (P1) contact 生成顺序/ID 稳定性 = warm-start 的生命线

你现在 disk-disk 的 ID 形如 `disk{i}-disk{j}`，只要 i/j 遍历顺序稳定就行。后续如果引入并行 broadphase / island 分块，很容易把顺序打乱，需要提前把 “id 生成规则”固化成**与遍历无关**的规范。

---

## 4) 下一步行动方向（我建议你就沿 (2) 路线打穿，但按这个节奏）

你问“请评估这一结果并规划下一步”，我的建议是：

### 下一步 1（最优先，1–2 个迭代内）：把“门禁”真正变成安全带

* 给 `bench/suites/*.yaml` 增加 **solver** 指标阈值（iters/residual/status 分布），并把 failsafe 的 soft_reasons 也写进 summary。
* 增加 2–3 个最小回归测试：

  1. warm-start on/off：同样的 stack 场景 residual 与 penetration 的差值应落在阈值内
  2. sleep + wake：唤醒后 penetration_max 不应出现尖峰
  3. OOD（mass/radius/dt）下 residual 不应系统性爆炸

### 下一步 2：把 μ-learning 的“可解释性”做出来（论文价值点）

你已经能训练 μ 了，接下来最好补齐：

* 用一组可控的合成数据（比如不同 μ 的滑移距离/停止距离）去验证 learned μ 的标定正确性；
* 输出“μ 的收敛曲线 + 物理误差曲线（penetration/residual/drift）”的对照；
* 做一个 ablation：没有结构正则 vs 有结构正则，看看 residual 是否更稳、OOD 是否更好。

这会直接形成你方法论文的核心图。

### 下一步 3：进入“接触堆叠 + 摩擦”的引擎级特性（但别一下子贪多）

按你现在架构，最自然的扩展顺序是：

1. **静摩擦/动摩擦切换的更稳健判据**（你已经有 mode + residual）
2. **island 级求解**（先做拓扑分解 + 分块 PGS，再谈并行）
3. 约束（关节）可以后置：因为它会把求解结构复杂度再上一个台阶。

---

## 5) 我会怎么“评价你现在的阶段”

如果按“能写进论文/能做成引擎”的标准，你已经越过了一个很关键的门槛：

> 过去：跑通 demo（可展示）
> 现在：可评测、可回归、可学习（可迭代、可发表、可工程化）

接下来真正决定你路线成败的，不是再加更多 feature，而是：**把“门禁指标”打磨成能长期保护迭代质量的体系**（尤其是接触/摩擦这种极容易回归的模块）。

---

如果你愿意，我可以按你现有的 `bench_summary.json` 结构（你在 `temp-01.md` 里提到的那套字段）帮你设计一套“最小但足够强”的 suite 阈值（ID / OOD 分开），让它既不苛刻到拖慢开发，也不宽松到形同虚设。
