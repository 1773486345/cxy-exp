# PatternAD 下一轮严格实验计划

> 状态：严格协议与执行工具已实现；核心 P0 单测、Weather dry-run、最小 GPU 训练 smoke、同 seed 精确重跑及 tiny Gaussian sigma 边界诊断已通过。正式多 seed 前先执行一次真实 Weather 四格 one-seed P0；P1 synthetic suite 与 hierarchical bootstrap 见对应脚本。

本文档只承担实验协议与预注册职责：冻结因子、切分、seed、运行顺序、推进门槛和结果判读。研究动机、方法演进与论文叙事仍以 `../tsad_direction_A_pattern_aware_scoring.md` 为主，工程接口以 `README.md` 和 `scripts/patternad/README.md` 为准，三处不重复维护大段实现说明。P2 开始前应将审定版本固化为带版本号的 protocol，记录文件 hash；后续想法进入下一版，不回写已经看过结果的冻结协议。

## 1. 本轮要回答的问题

本轮只回答三个可以被实验否证的问题：

1. 在相同的无泄漏条件重构协议下，局部动态 context 是否带来独立增益？
2. 相比 raw squared residual，条件尺度/条件分布是否更准确地表达 residual 的异常含义？
3. complementary masking 的增益来自无目标泄漏的条件评分，还是仅来自额外计算量或随机扰动？

本轮不再同时改变 context、训练 mask、推理 mask 和 scorer。当前 `PatternAD` 与 `PatternAD_raw` 的比较同时改变了 dynamic visible context 和 conditional scoring，不能用来归因 context 的效果。

## 2. 执行前必须修复或验证的协议问题

以下项目是正式多 seed 实验的硬约束。当前实现已逐项修复或纳入可执行配置：

1. **Seed 必须真实生效。** strategy seed 在模型构造前传给 `fix_random_seed(seed)`，并写入结果与 attempt 元数据。
2. **阈值不得读取测试分数或测试标签。** 严格 label 路径先验证 official-train 标签全部为 0，再只用其尾部的 calibration scores；发现非零或非有限 train label 时直接失败。
3. **不得在测试集上枚举 anomaly ratio 后取最好结果。** 主实验预声明一个 ratio；leaderboard 分开保留不同 ratio，严格 summarizer 不做 test-oracle max。
4. **Complementary mask 必须完整覆盖。** 每个 `(time, variable)` 在 K 个推理 pass 中恰好被遮蔽并计分一次。
5. **Context 必须是 target-blind。** 改变被 mask 的目标值不会改变该位置的 context。
6. **Context rolling statistics 必须排除 mask token。** 统计量只由 visible values 和显式有效计数构成。
7. **训练 mask 流必须可配对。** 训练 mask 与 DataLoader shuffle 使用独立 seeded generator；严格矩阵固定 `dropout=0`，避免 context 路径额外消耗随机数后破坏配对。
8. **窗口边界不得共享原始点。** train/validation 由不重叠的连续片段各自生成窗口；model-fit 输入与 calibration 之间另留 `seq_len - 1` 个点的 gap。

建议加入自动测试：

```text
tests/test_patternad_core.py
tests/test_anomaly_protocol.py
```

## 3. 因子定义

### 3.1 Context 因子 C

```text
C0: constant context control
    context 输入为可学习的 dataset-level 常量并经过相同 context_proj/FiLM 路径。
    保留同样的主干、张量尺寸和名义参数量。

C1: visible context
    使用 mask-aware local mean/std、trend、high-frequency activity 和 mask state。
    所有统计只由 visible values 计算。
```

若 C1 获胜，再补一个 `C_shuffle` 诊断：在 batch 内打乱 context，保持 context 的边际分布和网络容量但破坏样本对应关系。它不是首轮主矩阵的一部分。

### 3.2 Distribution 因子 D

第一轮只比较最干净的两种形式，不把 Gaussian、Student-t、quantile 和 flow 一次性混在一起。

```text
D0: deterministic mean
    训练目标：masked MSE + 固定权重的 full MSE
    分数：masked squared residual 的变量均值

D1: heteroscedastic Gaussian
    输出：mu(x_visible, c), log_sigma(x_visible, c)
    训练目标：masked Gaussian NLL + 固定权重的 full mean-MSE
    分数：0.5 * ((x-mu)^2 / sigma^2 + 2*log(sigma))
```

实现约束：

- `sigma` 只能由 masked input 和 visible context 预测，不能接收真实 target 或 realized residual。
- 对 `log_sigma` 使用数值边界和 variance floor；记录落到上下界的比例。
- D0 与 D1 使用相同的 `mu` backbone、训练窗口、mask schedule、epoch budget 和 early-stop patience。
- Gaussian 是机制识别版本。只有 D1 明确有效后，再把 Student-t 作为二阶段 robustness 优化；否则无法判断收益来自 conditional scale 还是 heavy tail。

### 3.3 Mask 因子 M

```text
M0: unmasked self-reconstruction
    单 pass，输入包含当前目标，所有变量以 self-reconstruction residual/NLL 计分。
    仅作为诊断基线，不具备 conditional-density 的无目标泄漏解释。

M1: complementary conditional masking
    对 (t,d) 网格构造 K 个互补 mask，例如 (t+d) mod K = k。
    K 个 pass 的 mask 互斥且并集为全部位置，每个位置恰好计分一次。
```

建议首轮固定 `K=3`，并报告推理吞吐量。训练阶段在所有 C/D/M cell 中统一保留：

```text
train_mask_ratio = 0.25
train_variable_mask_ratio = 0.15
reconstruction_full_loss_weight = 0.10
```

训练 mask 类型不是主矩阵的 M 因子。它在模型胜出后单独比较 `point-only` 与 `point+whole-variable`，避免把训练任务和推理协议再次混杂。

## 4. 最小公平消融：分两阶段共 6 个唯一配置

### 4.1 主消融：2 x 2 context/distribution，固定 M1

| ID | Context | Distribution | Mask | 直接回答 |
|---|---|---|---|---|
| A00 | C0 | D0 | M1 | 无 context 的 masked raw-residual 基线 |
| A10 | C1 | D0 | M1 | context 是否改善条件均值重构 |
| A01 | C0 | D1 | M1 | 条件尺度模型是否不依赖动态 context 也有效 |
| A11 | C1 | D1 | M1 | 完整假设：context-conditioned residual distribution |

主效应和交互作用：

```text
context main effect      = mean(A10-A00, A11-A01)
distribution main effect = mean(A01-A00, A11-A10)
interaction              = (A11-A01) - (A10-A00)
```

解释规则：

- `A11 > A01` 才能支持 context 对条件分布有增量价值。
- 若 `A01 ~= A11 > A00`，证据只支持 uncertainty/distribution modeling，不支持 pattern-aware context 的核心主张。
- 若 `A10 ~= A11 > A00`，主要收益来自条件均值重构，而非 residual semantics/calibration。
- 若只有 A11 获胜且 interaction 为正，最契合当前动机。

### 4.2 次级 mask 实验：只新增两个 cell

复用 A00 和 A11，只新增：

| ID | Context | Distribution | Mask | 配对对象 |
|---|---|---|---|---|
| B00 | C0 | D0 | M0 | 与 A00 比较 M1 对普通残差基线的作用 |
| B11 | C1 | D1 | M0 | 与 A11 比较 M1 对完整模型的作用 |

这样可用 6 个唯一配置获得三因素的最小证据，不需要第一轮直接运行完整 2 x 2 x 2 的 8 个 cell。仅当 `A00-B00` 与 `A11-B11` 方向相反时，再补 `C1/D0/M0` 和 `C0/D1/M0`，完成全 8 格以定位 mask interaction。

M0 即使指标更高，也不能直接作为“conditional residual distribution”方法，因为它看到了待评分目标。M0 的作用是量化 target visibility 带来的表观收益。

所有 A/B cell 的 early stopping 均使用相同的 complementary masked validation loss，M 只改变最终评分路径。不过 B11 的 Gaussian scale 只在 hidden-target 条件下训练，M0 又在 target-visible 条件下使用该 scale，因此 B11 仍只是 protocol-shift 诊断，不能作为干净的概率 mask 因果效应；mask 主结论应优先依据 A00/B00，并把 B11 放在附录。

## 5. 数据切分和阈值协议

### 5.1 Official train 内部切分

当前可执行协议对每个实体按时间顺序切分 official train，并要求其标签全部为 0。先留出尾部 20% calibration，并在 model-fit 输入与 calibration 之间删除 `seq_len - 1` 个点；PatternAD 再把 model-fit 输入按 80/20 切为 optimization train 和 validation。因此忽略 gap 与取整时，实际约为：

```text
optimization train  64%
validation          16%
calibration  20%
```

optimization train 与 validation 是不重叠的原始片段，各自在片段内部生成窗口；model-fit 与 calibration 之间额外保留 gap。Scaler 只在 optimization train 拟合；validation 用于 early stopping；calibration 只用于固定阈值和正常校准诊断。当前代码不实现 label-aware 窗口过滤；若 official train 含已知异常，严格协议会拒绝运行，必须先在数据版本层重定义/清理 official train 并记录 provenance。

当前代码只暴露 outer `calibration_fraction`，PatternAD 内部 validation 固定为 model-fit 输入的 20%，因此默认约为 `64/16/20`。若要精确使用 `70/15/15`，需先新增显式 validation-fraction 配置并重做切分测试；不能只改 outer fraction 后把结果写成 `70/15/15`。

### 5.2 固定阈值

主阈值使用 calibration normal score 的有限样本 conformal quantile：

```text
alpha = 0.01
k = ceil((n_cal + 1) * (1 - alpha))
threshold = kth order statistic of calibration scores (1-indexed)
```

若 `k > n_cal`，该 calibration set 不支持目标 alpha，应令阈值为 `+inf` 或预先改用可支持的 alpha，不能静默截断 k。`alpha=0.005` 和 `0.02` 只做敏感性分析。小数据集应报告可达到的最小经验 alpha，不得借测试标签换阈值。时间依赖会削弱普通 conformal 的严格交换性保证，因此论文中应称为 calibration-only empirical/conformal quantile；block conformal 属于后续增强项。

严禁：

- 拼接 train/test score 后取 percentile；
- 根据测试异常比例设置 threshold；
- 在 42 个 anomaly ratio 中选测试 F1 最大者作为主结果；
- 根据 test AUC 选择 context、窗口或分布族。

## 6. 数据集分组

### 6.1 Smoke 和机制组

```text
Synthetic contextual suite  机制必要性、校准和泄漏测试
Weather                     快速端到端 smoke，且作为 raw residual 强势压力集
MetroPT3                    长序列运行时间、内存和吞吐量检查，不用于证明核心动机
```

### 6.2 Motivation development 组

用于选择 C/D 机制和少量超参数：

```text
HAI21_part1
Daphnet
Energy
SMD_machine-3-2
SMD_machine-2-2
SMD_machine-3-9
```

SMD 三个实体先在内部 macro 后作为一个 dataset family 计入总 macro，不能让三个 SMD 实体获得三倍于 HAI/Daphnet/Energy 的权重。

### 6.3 Robustness development 组

只在 A00/A11 和必要的失败诊断上运行，不用于挑模型：

```text
Weather, Genesis, GECCO, SKAB, MSDS, MetroPT3
SMD_machine-1-1, SMD_machine-2-8
```

这些数据可回答“新方法会不会破坏幅值已经足够的场景”，但不能作为筛选契合方向的数据集后再单独报告。

### 6.4 从现在起锁定的 confirmation 组

```text
HAI21_part2, HAI21_part3
SMD 其余 23 台机器
```

在第一次运行前保存：dataset 文件 hash、代码 commit、配置 hash、锁定时间和唯一候选配置。HAI 和 SMD 各自先做 entity macro，再做 family macro。它们仍不是完全独立的外部验证；最终论文最好再加入未参与方向迭代的 SWaT/WADI/BATADAL、Exathlon 或明确版本的 TEP。

原始七数据集和已经查看过结果的 HAI21_part1/MetroPT3 均应视为 development evidence，不能再称为 untouched test。

## 7. Seed 和配对运行

```text
开发：2021, 2022, 2023
锁定确认：2021, 2022, 2023, 2024, 2025
合成机制：20 个 generator seeds，每个 generator seed 固定 paired model seeds
```

相同 dataset/seed 下必须固定：数据切分、Scaler fit 段、训练窗口顺序生成规则、训练 mask schedule、complementary mask partitions。不同 cell 仅改变被声明的因子。

结果长表至少包含：

```text
run_id, git_commit, config_hash, dataset, entity, seed,
context_mode, distribution, score_mask_mode, train_mask_mode,
fit/val/cal/test lengths, threshold_alpha, threshold,
metric_name, metric_value, runtime, peak_memory, status
```

不要把 seed 当作独立数据集样本。统计时先计算同一 entity 的 paired seed difference，再做分层聚合。

## 8. 主要指标和统计汇总

### 8.1 模型选择指标

唯一首要选择指标：

```text
test AUC-PR，按 entity -> dataset family -> dataset macro 的顺序聚合
```

关键次要指标：

```text
VUS-PR
AUC-ROC
VUS-ROC
```

阈值指标只使用 calibration-only 固定阈值：

```text
point F1 / precision / recall
Affiliation F1（或明确实现的 event F1）
point-adjusted F1，仅作补充
false alarms per 10,000 normal points
```

机制和校准指标：

```text
normal FPR@alpha by regime/context bin
max-regime FPR gap
matched-pair ordering accuracy
normal score 与 local scale/trend 的 Spearman correlation
Gaussian NLL、标准化 residual coverage、sigma clamp rate
```

补充报告参数量、训练时间、单点推理时间和 K-pass 吞吐量。类别极不平衡时可附 normalized AP `(AP-prevalence)/(1-prevalence)`，但不能替换原始 AP。

### 8.2 不确定性

所有比较使用相同 dataset/entity/seed 的 paired difference。报告均值、标准差和 95% hierarchical bootstrap CI，建议 10,000 次。P2 在四个以上 development families 上按 family -> entity -> seed 重采样；P4 只有预声明的 HAI21/SMD 两个 family，不把它们当作 family 超总体，而将二者固定为 strata、只在各 family 内重采样 entity/seed。主表不按序列长度加权。

## 9. 合成机制实验

合成数据不是为了制造一个容易赢的 benchmark，而是验证方法是否真的实现了论文命题。训练、validation、calibration 全部只含 normal regimes；异常只注入 test。

### 9.1 基础生成过程

使用 switching VAR/state-space process：

```text
z_t in {low-variance, high-variance}, Markov dwell time 200-1000
x_t = A_z x_(t-1) + L_z epsilon_t
D = 5（当前可执行机制 fixture）；机制成立后再冻结 D in {8, 32} 的扩展配置
variance ratio sigma_high/sigma_low in {2, 4}
```

固定稳定的 `A_z` 和已知 cross-variable covariance，保存所有生成参数和注入区间。

### 9.2 四类机制场景

1. **Same deviation, different context。** 高波动 regime 中生成正常脉冲；在低波动 regime 注入完全相同幅度和形状的脉冲并标异常。构造一一匹配的 hard-negative/positive pairs。
2. **Slow drift vs abrupt shift。** 正常 reference 先缓慢 ramp 到 Delta，保持恰好 `abrupt_length` 的 plateau 后再缓慢回落；异常样本以相同 Delta 突然 step。matched ordering 与 raw-control tie 只比较等长、等注入幅值的 gradual plateau 和 abrupt event，不把整段 ramp 的累计能量伪装成相等。
3. **Dependency break。** 在保持单变量边际分布和幅度近似不变的情况下，置换一个通道片段或替换其 innovation，破坏 cross-variable conditional relation。
4. **Context OOD。** 生成训练未出现但内部条件关系仍可自洽的全局 regime 组合，用于检验 conditional reconstructor 是否把 collective/context anomaly 完全解释掉。

变化因素：异常持续时间 `{1, 8, 32}`、受影响变量比例 `{1/D, 0.25, 1.0}`、SNR 和 regime dwell time。每个条件使用相同注入位置比较所有 cell。

### 9.3 机制成功指标

```text
matched ordering = P(score(low-regime anomaly) > score(matched high-regime normal))
regime FPR gap   = max_z |FPR_z - alpha|
dependency AP   = dependency-break anomalies 的 AUC-PR
context-OOD AP  = collective/context anomaly 的 AUC-PR
raw-control tie = matched pair 的注入 squared-deviation margin 应近似 0
```

机制推进门槛见下一节。`context_ood` 是 conditional-only score 的预声明负对照，预计需要 Priority 2 的独立 rare-context score；不能拿它的失败否定条件残差机制，也不能把它隐藏。若 A11 只在普通幅值尖峰上提升，而 matched ordering、raw-control tie 和 regime FPR gap 不改善，不能声称 residual semantics 得到验证。

## 10. 阶段化执行顺序与推进/停止判据

### P0：协议单元测试和单 seed smoke

运行 synthetic tiny + Weather 单 seed。必须全部满足：

- seed 变化能改变初始化/训练结果，同 seed 重跑在允许误差内一致；
- complementary masks 对每个位置覆盖恰好一次；
- masked target perturbation 不改变其 `mu/sigma`；
- 修改测试标签或附加测试 score 不改变 threshold；
- 输出 score 与 label 等长且没有尾部复制造成的系统偏差；
- 无 NaN/Inf，`sigma` 落边界比例低于 1%。

任一项失败：停止正式实验并修协议。

### P1：合成机制，4 个主 cell

开发阶段只在 generator seeds 3101-3110 上运行 A00/A10/A01/A11。预先分别定义 `A11-A00`（完整方法对基线）、`A11-A01`（Gaussian 下 dynamic context 增量）和 `A11-A10`（dynamic context 下 distribution 增量），不得看完结果后选择 observed-best comparator。候选、组合规则和所有门槛冻结后，才在 untouched seeds 3111-3120 上运行一次确认。开发阶段至少满足：

- `A11-A01` 的 matched-ordering paired mean 提升至少 0.05，且 95% CI 下界大于 0；
- regime FPR gap 相对 A00 降低至少 25%；
- dependency-break AP 不低于 A00，普通大幅尖峰 AP 的退化不超过 0.02。

若 matched ordering 和 FPR calibration 均不改善：停止扩展真实数据实验，先修改 conditional distribution/context。若只有 A01 改善：将研究主线改为 conditional uncertainty，而不是继续宣称 pattern context 有效。

### P2：Motivation development，4 个主 cell，3 seeds

用 macro AUC-PR 选择一个候选。若预注册候选为 A11，必须分别检查 `A11-A00`、`A11-A10`、`A11-A01` 三个预声明 comparison；不能定义或选择 observed-best comparator。完整的 context-conditioned distribution 主张要求相应 pair 均满足下列推进条件；若只有某一 pair 成立，则按第 13 节收窄主张：

- family-macro AUC-PR 提升至少 0.01；
- HAI21、SMD、Daphnet、Energy 四个 family 中至少三个方向为正；
- 任一 motivation family 的 AUC-PR 退化不超过 0.02；
- family-macro VUS-PR 不为负；
- 提升不是单一 seed 或单一实体驱动。

未达到：不运行 HAI/SMD full。根据四格效应决定是放弃 context、放弃 distribution，还是修正模型。

### P3：次级 mask 和 robustness

运行 B00/B11，复用 A00/A11；仅在 interaction 冲突时补齐 8 格。同时在 robustness development 组运行候选与 A00。

推进条件：

- M1 在 dependency-break synthetic 上相对 M0 的 AP 至少提高 0.05；
- A11-M1 的 motivation macro AUC-PR 不低于 B11-M0；
- robustness family-macro AUC-PR 退化不超过 0.02，且没有单数据集退化超过 0.05；
- K-pass 的计算代价被完整报告。

若 M0 更高但 target-invariance 测试表明其使用了目标信息，只能把它作为 conventional autoencoder baseline，不能替换提案中的 M1。

### P4：冻结并运行 confirmation，5 seeds

只允许一个候选配置和预先冻结的 A00 基线进入 HAI21_part2/3 与 SMD held-out。禁止看结果后改窗口、alpha、分布族或数据集权重。

确认成功需满足：

- entity-paired family-macro AUC-PR delta 的 95% hierarchical bootstrap CI 下界大于 0；
- family-macro VUS-PR delta 非负；
- HAI21 与 SMD 两个 family 均无超过 0.01 的负向差异；
- fixed-threshold false-alarm rate 没有显著偏离预设 alpha，或偏离得到明确报告。

若 CI 包含 0 且绝对提升小于 0.005：停止“普遍优于 raw residual”的主张，不得继续在这些 locked entities 上调参。可以报告机制成立但真实数据效果有限，或换一套从未查看的外部数据重新预注册。

## 11. 建议新增的脚本、配置和指南

### P0 必需

```text
config/unfixed_detect_label_multi_config.json
    已实现：fit/cal split、gap、alpha、固定指标和 seed 行为。

config/patternad/factorial_ablation.json
    A00/A10/A01/A11/B00/B11 的唯一配置源，禁止复制多份 shell 后漂移。

scripts/patternad/run_factorial_ablation.py
    按 manifest 生成 run_id、逐 dataset/seed 运行、支持 resume、失败不覆盖。

scripts/patternad/summarize_factorial.py
    已实现：读取冻结任务清单、拒绝缺失/混杂结果、计算 paired delta 和 family macro；不做 test-oracle max。

scripts/patternad/bootstrap_factorial.py
    已实现：deterministic hierarchical/fixed-strata bootstrap CI、fail-closed provenance 检查和逐项推进诊断。

run_factorial_ablation.py 生成的 run_plan.json
    已实现：写出 clean git commit、关键源码/config/data hash、候选 ID、完整预期网格和锁定时间。

README.md + EXPERIMENT_PLAN_DRAFT.md
    已实现：对外可复现的切分、阈值、指标、聚合和 forbidden operations。
```

### P1 机制实验

```text
scripts/patternad/generate_contextual_synthetic.py
scripts/patternad/evaluate_contextual_mechanisms.py
config/patternad/synthetic_suite.json
    已实现：20 个冻结 generator seeds、四种机制、外部分数契约和 PatternAD variant 端到端模式。
```

生成数据默认写入独立 ignored 目录，只提交 generator、参数 manifest 和小型 deterministic fixture，不提交大量生成 CSV。

### P2/P4 运行辅助

```text
config/patternad/dataset_groups.json
run_factorial_ablation.py --group confirmation --allow-locked
summarize_factorial.py 内置 completeness fail-fast
result/patternad_strict/<run_id>/<group>/run_plan.json
result/patternad_strict/<run_id>/summary/*.csv
```

现有每数据集 `PatternAD.sh` 可保留用于历史复现，但严格 factorial 不应继续依靠手工复制的 shell 作为配置真源。

## 12. 与当前动机最契合的创新方向优先级

### Priority 1：条件尾概率与 regime-invariant calibration

把研究命题从“context 帮助重构”提升为可检验原则：

```text
对正常数据，校准后的 tail probability 应在每个运行状态下都近似 uniform，
即 anomaly score 不应系统性依赖正常 regime。
```

可实现为 heteroscedastic Gaussian/Student-t、conditional quantile/CDF 或 conditional normalizing flow，并加入跨 context-bin 的 coverage/FPR invariance regularization。这最直接回答“同 residual 在不同背景下含义不同”。

### Priority 2：conditional inconsistency 与 context OOD 双分数

单纯 `p(x_masked | x_visible, context)` 会漏掉所有变量共同进入异常状态但彼此仍可重构的情况。将异常拆为：

```text
S_cond = -log p(x_masked | x_visible, context)
S_ctx  = -log p(context) 或 context 的 normal-state tail probability
```

两者先各自用 normal calibration 转为 p-value，再用预先固定的组合规则聚合。它既修复 collective anomaly failure，也形成比“再加一个 scorer”更清晰的新动机：条件不一致和条件本身罕见是两种正交异常证据。

### Priority 3：全变量 masked pseudo-likelihood 与可归因性

用 complementary masks 估计每个变量的 conditional surprise，再得到：

```text
time score, variable score, conditional dependency attribution
```

创新重点不是随机 mask，而是完整覆盖、无泄漏、可校准的 multivariate pseudo-likelihood，以及在不增加 K 倍成本时的蒸馏/并行近似。

### Priority 4：学习连续运行状态，而不是继续堆手工 rolling features

由 visible-only state encoder 学习 context，并用以下约束保证它服务于 residual semantics：

- normal calibrated score 对 context 不敏感；
- context 能预测正常 residual scale/quantiles；
- masked target perturbation 不改变 context；
- 慢状态变化与短时异常通过不同时间尺度编码。

这比增加更多 scale/trend/frequency 手工组件更有扩展性，但应在 Priority 1 的分布与校准闭环成立后再做。

### Priority 5：时间依赖下的 online/block conformal calibration

针对正常分布缓慢漂移，用 block conformal、weighted conformal 或 EVT tail 更新阈值，并显式控制每个运行状态的 false-alarm budget。它适合部署和在线 TSAD，但工程与理论成本更高，不应先于离线条件分布机制验证。

### Priority 6：counterfactual transition scoring

把“允许的慢 regime transition”和“相同幅度的突然跳变”写成两个反事实：在保持 context trajectory 与替换 transition mechanism 下分别计算 likelihood。这一方向新颖但风险最高，适合在 synthetic slow-drift/abrupt-shift 结果显示现有局部 context 无法区分二者后推进。

## 13. 论文主张边界

根据实验结果选择主张，不应预先固定结论：

```text
A11 独立胜出且 interaction > 0:
    支持 pattern-conditioned residual distribution / calibration。

A01 与 A11 相当且都胜出:
    支持 uncertainty-aware residual scoring，不支持 context 是关键贡献。

A10 与 A11 相当且都胜出:
    支持 context-conditioned reconstruction，不支持 distributional semantics 是关键。

M1 是唯一主要增益:
    支持 leakage-free masked pseudo-likelihood，应调整论文动机。

Synthetic 通过而 locked real data 不通过:
    只能声称机制在控制条件下成立，不能声称一般 benchmark 提升。
```

最有潜力且与方向 A 最闭环的表述不是“FiLM 加了局部统计”，而是：

```text
An anomaly score should represent conditional tail probability and remain calibrated
across normal operating regimes, while separately detecting rare contexts themselves.
```
