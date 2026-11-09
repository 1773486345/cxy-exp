# 八组修改决定与 PSM 训练范围越界回放

2026-11-10；实际 HEAD `5f36b94b56a4c8f5cc3c2cd55d1ee478efd7afaa`，开始时工作树干净，无适用 AGENTS.md。使用 catch_env Python 3.8.20 / NumPy 1.24.4，进程级单线程。以历史 [dataset_diagnosis.md](../../MNRF-Comparison-Diagnosis-20261102T125000Z/dataset_diagnosis.md) 为索引，补入后续已完成的对照；保留历史文件。本轮只新增 PSM 保存产物的统计，不加载权重、前向、拟合、训练或重校准，不重跑 baseline。

**决定：PSM 值得把“超出训练支持范围”单列为提示，但不能把它当作误报屏蔽器。八组中最优先的下一项受控验证，是在真正独立数据上确认 Creditcard 的固定 R 平方残差读出；当前最缺的是独立数据，不是另一轮头部搜索。**

## 比较范围

- **论文全轴参照**：沿用已确认的 16 方法论文成绩及历史 [paper_comparison.csv](../../MNRF-Benchmark-Main/benchmark-main-20261103T200322Z-851688/paper_comparison.csv)。[main_results.csv](../../MNRF-Benchmark-Main/benchmark-main-20261103T200322Z-851688/main_results.csv) 中 MNRF 为七组全轴完成、Creditcard 全轴留空。MSL/CICIDS 的 masked cold-start、旧 CalIt2 modulo-10 均须注明；本轮不把有效点结果放入全轴空格，也不更改该表。
- **共同点诊断**：下述模型优劣主要来自相同坐标上的 Full、CATCH-common、sklearn IF、sklearn PCA-residual 及冻结 R 读出。它们不是论文原生实现的替代值。各自保存的 q=.995 阈值用于报警，实际 FPR 不相同。测试标签扫描的 FP 预算仅用于事后排序诊断。
- **修正版数据/任务**：MSL 序列边界 × 评分对象四格、CalIt2 RawCount 各自单列，不能拼进历史八组平均分。没有将辅助 R 残差/NLL 或条件评分称为完整 Flow。

## PSM：越界标记覆盖什么

固定源 `full_gmm-full-20261024T115119Z-3976714`。共同轴为全局 `[132672,220322)` 的 **87,650 点**，异常 24,378、正常 63,272；官方测试轴为 `[132481,220322)` 的 87,841 点。十等分先在官方测试轴上划分，再映射共同点；没有对紧凑分数重新分段。

从 [psm_calibration_coverage.csv](psm_calibration_coverage.csv) 的 `all_rows / all` 行读取未舍入最大值。官方训练侧 `[0,132481)` 包括内部 train、validation、calibration 和两个 purge，五段无重叠且覆盖完整；并核对保存的标准化输入得到相同最大值：

```text
feature_1 > 2.0010499954223633
或 feature_3 > 2.231389284133911
```

这是原训练统计标准化后的输入范围，不是原始物理量上限或异常定义。严格使用 `>`，未按测试值修改上限；两个字段本身来自此前测试诊断，所以本次仍属探索性。标记共 **43,331 点**（49.44%），其中正常 **25,591**（占全部正常 40.45%）、异常 **17,740**（占全部异常 72.77%）。仅 feature_1 越界 21,931，仅 feature_3 越界 18,360，同时越界 3,040。

| 保存读出 | 原 TP / FP / FN / TN | 越界覆盖原 FP | 越界覆盖原 TP | 越界覆盖原 FN | 保存 Recall / FPR |
|---|---|---|---|---|---|
| Full GMM | 14,694 / 17,305 / 9,684 / 45,967 | 17,189 / 17,305（99.33%） | 14,478 / 14,694（98.53%） | 3,262 / 9,684（33.68%） | 60.28% / 27.35% |
| R 平方残差 | 8,630 / 14,826 / 15,748 / 48,446 | 14,450 / 14,826（97.46%） | 7,102 / 8,630（82.29%） | 10,638 / 15,748（67.55%） | 35.40% / 23.43% |

保存阈值分别为 `15.702126131057883`、`157.7610897827152`，预测始终为原 `score > threshold`。以下每格为“越界覆盖数 / 原数量”，**没有移除任何越界点或改变分母**。

| 官方十等分 | Full FP | Full TP | Full FN | R FP | R TP | R FN |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3 / 14 | 66 / 136 | 129 / 1,034 | 2 / 25 | 38 / 456 | 157 / 714 |
| 2 | 65 / 89 | 182 / 195 | 471 / 478 | 1 / 25 | 15 / 22 | 638 / 651 |
| 3 | 6 / 6 | 6,164 / 6,164 | 2,197 / 2,197 | 0 / 0 | 77 / 77 | 8,284 / 8,284 |
| 4 | 6,964 / 6,989 | 571 / 584 | 0 / 6 | 4,291 / 4,308 | 64 / 71 | 507 / 519 |
| 5 | 0 / 12 | 0 / 6 | 0 / 126 | 0 / 38 | 0 / 33 | 0 / 99 |
| 6 | 3,765 / 3,765 | 813 / 813 | 465 / 465 | 3,777 / 3,777 | 231 / 231 | 1,047 / 1,047 |
| 7 | 5,500 / 5,500 | 3,284 / 3,284 | 0 / 0 | 5,493 / 5,493 | 3,281 / 3,281 | 3 / 3 |
| 8 | 886 / 903 | 3,398 / 3,438 | 0 / 142 | 886 / 1,011 | 3,396 / 3,496 | 2 / 84 |
| 9 | 0 / 6 | 0 / 19 | 0 / 672 | 0 / 34 | 0 / 52 | 0 / 639 |
| 10 | 0 / 21 | 0 / 55 | 0 / 4,564 | 0 / 115 | 0 / 911 | 0 / 3,708 |

事件沿完整标签、原 segment ID 和原全局起止定义，不跨间隔连接。测试共 **72 个原事件，71 个有评分**；其中 **28 个事件的全部有效点越界、2 个部分有效点越界、41 个无有效点越界、1 个完全无评分**。这里“全部”只指有效点，不声称覆盖事件中未评分的点。30 个至少部分越界的事件中，Full 原命中 24 个，R 原命中 22 个；41 个没有越界的有评分事件中，Full 命中 24 个，R 命中 27 个。事件明细保存在 [psm_training_range_replay.csv](psm_training_range_replay.csv)；逐段事件数按事件原起点归属，整段事件不截断或重复计数。

**解释与行动边界。** 越界能给 17,189 个 Full 误报和 14,450 个 R 误报一个直接可见的共同输入背景，但同样涉及大量真正异常与原漏检；它不是“这些报警无用”的证明。第 3 段几乎全是越界异常，第 4/6/7 段又有大量越界正常，说明同一范围提示不能区分二者。不能将其作为扣分、删点或重校准的依据。下一版可预先固定并单列此提示，保留独立异常分数与报警，在新的独立时期验证提示的工况解释价值。

已有 [误报逐点归因](psm_fp_attribution.md) 发现两读出共同误报 14,464 点，其中约 99.37% 由相同的 feature_1/3 主导；整个 192 步窗口都无标注异常时仍有 12,471 个共同误报。两变量的相应 MAD 未触底。此次证明这些输入值甚至超出**整个官方训练侧**最大值，收紧了“只是原校准遗漏了训练中同幅状态”的解释。模型对输入外推、表示响应和评分响应各自的因果份额仍不能仅凭关联拆分。R 校准期逐变量残差未保存；此前未找到覆盖这些强状态的合法额外预留区段，不能将训练/验证/purge 或已看过的测试重包装为校准确认集。

同时保留低误报排序结论：1% FP 上限下，Full 为 337 TP / 631 FP，CATCH-common 为 966 / 631，R 为 5,332 / 631；R 中 4,689 TP 来自第 7/8 段。排除这两段后的探索性重算，R 为 680 / 469，仍高于 Full 108 / 416，却低于 IF 813 / 524、CATCH-common 724 / 524。R 并没有同时解决跨区段排序和固定阈值报警。

## 八组修改决策表

“采用”指本阶段数据/任务和报告规则；“候选”不是按八个测试集分别选最优头组成新方法。原历史 Full 与所有已完成对照均保留。机器可读表见 [eight_dataset_decisions.csv](eight_dataset_decisions.csv)。

| 数据集 | 相对哪个对照好/差；直接证据 | 现在采用的设置 | 候选改动、尚缺证据与改变决定的条件 |
|---|---|---|---|
| **PSM** | 87,650 点 Full AP .4373，R .5294；Full 低误报尾部弱于 CATCH-common，R 的收益集中于第 7/8 段。越界覆盖 Full/R 原 FP 的 99.33%/97.46%，也覆盖 72.77% 的真实异常点。[P1–P3] | 保留全部字段、原分数、阈值和原分母，单列低 FPR 排序与保存阈值报警；越界只作探索性范围提示。 | 支持优先检查训练支持范围外的工况与外推，不直接删除 feature_1/3。独立越界提示值得验证，自动抑警/改变校准暂缓；须有合法独立时段及状态证据，不能把当前越界异常压下去。 |
| **MSL** | 同一 68,572 点：旧拼接 all55/col_0 AP .1715/.3053，新分段 all55/col_0 .1322/.3073；评分目标改动的增益明显大于 col_0 下分段重训的 +.0020。新 col_0 AUC .7790、AP .3073，高于 CATCH-common .6683/.2008、IF .6074/.1536、PCA-residual .5093/.1541。[M] | 采用真实 27 条训练/测试序列边界、55 输入、col_0 最终评分，保留命令条件输入。历史拼接 Full 单列；CATCH-common 仍拼接训练，各自保存阈值不是等 FPR。 | 边界是正确任务语义，即使 AP 增益小也保留。尚未验证的模型改动是把目标/条件职责纳入训练损失；不是再重复目标列后处理或分段训练。 |
| **GECCO** | 固定表示 full→diag、max 不变：AUC .4425→.8472，AP .2233→.2459；1% FPR 召回 27.40%→27.67%，0.1% 则 25.48%→24.25%。CATCH-common AP .4672、1% 召回 60.00%。calibration 286/6,812 端点异常，窗口含异常 1,598；上尾 35/35 均异常。[G] | 保留原始训练/测试划分和污染，排序与报警分别报告；不按标签清洗。diag 留作协方差敏感性对照，不升级默认头。 | 排序变化直接支持密度建模影响；校准污染只支持报警阈值受污染分布影响，不能解释 AUC 差距。是否采用更简单头须同时通过独立 AP/低 FPR 验证；报警端需预留代表性数据，不能靠测试清洗凑效果。 |
| **Genesis** | Full AP .0468 高于 CATCH-common .0144/IF .0055，但远低于 PCA-residual .5155；Full mean .1005 仍低。1% FPR 召回 Full 22%、PCA 94%，保存阈值 PCA FPR 7.05% 又高于 Full 3.03%。Full FP 的 80.8%、TP 的 82.6% 由 Motor_Pos2reached 主导，FN 的 70.4% 由 IsAcceleration 主导。仅 50 异常点/3 段。[G] | 保留原输入及离散状态字段，报告三个事件与 132 点小校准集，不仅凭二值/高归因改成条件。 | 值得查正常状态切换，但目前只有归因关联。须看到独立正常工况切换与高分稳定对应、来源确认该字段职责，并确认未删掉真实故障信息，才考虑状态条件化或分型读出；不能仅凭 PCA 更好就删除字段。 |
| **NYC** | Full AP .0441；E 残差 .0215，低于 persistence .0224、目标均值 .0388；仅客流联合评分 AUC .5511，条件评分 .5171，均低于全变量 Full .6913。唯一有评分事件：E 客流平方残差中位数前8点 .007720、内部83点 .001747、末8点 .000485；事件前8个正常点 .011118 更高。[N] | 客流为目标、时刻/星期为条件；既有全部对照单列，不能把低预测误差当检测收益。 | 固定慢速/季节客流参照值得验证，以检验持续异常被快速跟随的假设；不先认定机制。单事件只支持该段描述，且正常前邻域反证简单“起点残差一定大”。需不读未来/异常真值的参照及新增独立事件。 |
| **CalIt2** | 修正版 RawCount Full AUC/AP .8535/.1703，优于旧 modulo-10 .7130/.0540；同版 CATCH .8816/.2815、IF .8902/.2538 仍更好，PCA-residual .8224/.1466 较低。旧全维 PCA AP .0801、旧数据残差 PCA .0858、RawCount 残差 PCA .1466 分列。固定幅度分数 AP .2323；≤183 FP 下，IF 检出而 Full 漏掉的27点中，幅度找回26、PCA找回17、共同16。[C] | 后续修正版采用 RawCount，官方时间、标签及划分不变；旧 modulo-10 只保留历史表。 | 固定两路计数幅度已完成离线对照，支持把直接幅度读出列为下一版候选，优先于把这批差距归于计数线性关系偏离；PCA另找回1点仍保留。历史测试探索性扫描不是新报警阈值；只有24个原校准窗口，未验证候选独立报警能力，不改原模型或分数。 |
| **Creditcard** | 142,213 点，R 残差 AP .3561、同解码器 NLL .001756、Full .0510。≤222 FP：R 130 TP、Full 26、IF 48；≤1,480 FP：R 185、Full 54，Full 已完整扫描。R 主预算在第2–9段均有 TP；不是单个首段的收益。[R] | 优先保留全部29字段、原 median/MAD/max 的 R 平方残差为固定候选，不换变量。保存阈值 R 184 TP/1,167 FP，排名与报警分开确认。Full 全轴仍因 3/5,800 特征超差留空；有效点证据独立保留。 | 独立确认首要缺新增合格数据，继而缺该轴冻结分数；旧测试删段不算独立。逐点 μ/σ 已导出且粗分箱已修订，不再列为未执行；它支持单变量 NLL 排序损失，仍不能量化“固定残差下 sigma 的因果贡献”，更不能直接归因 Full GMM。[R2] |
| **CICIDS** | 有效84,925点 Full AP .000917，低于 CATCH-common .001360、IF .001295、PCA-residual .001066；保存报警四者 TP 均0。新增前缀191点全报警，含5异常/186正常；全轴 AP .004596 不能解释为原有效段变好。[I] | 保留逐行数据结果和 current-only/masked 前缀标注，分别报告有效点与全轴；不从缺失 session 信息创造边界。 | 已有逐行 IF/PCA 能检验同一批行的静态检测力，不能证明 E/P 的时序收益。优先取得可信 flow/session 与原行映射，再判断序列建模是否成立；元数据缺失前不把跨行模型修改列为已支持方案。 |

### 每项决定的直接证据

以下路径均相对 APD-CATCH；链接提供相同产物，未重新生成旧对照。

- **P1**：[本轮越界回放](psm_training_range_replay.csv)、[训练侧覆盖](psm_calibration_coverage.csv)、[误报逐点归因](psm_fp_attribution.md)。原始输入是固定 Full 的 `cache/normalized_values.npy`，评分轴/事件来自同源 `analysis_inputs.npz`，Full 分数为同源 `results/aligned_scores.npz`，R 为 `result/analysis/MNRF-FrozenDecoder/PSM/full_gmm-full-20261024T115119Z-3976714/R_squared_residual_test_scores.npz`。
- **P2 / R**：[readout results.csv](results.csv)，含完整全局预算扫描、保存阈值与官方十等分。Creditcard 主预算 R 各段 TP 为 `60,5,3,8,6,12,11,19,6,0`，Full 为 `26,0,0,0,0,0,0,0,0,0`；第1段以外 R 仍增加70个 TP，不是独立确认。预算只在全局选一次，不逐段挑阈值。
- **P3**：[PSM 低 FPR 表](../../MNRF-PSM-LowFPR/low-fpr-20261103T121447Z/psm_low_fpr_operating_points.csv)。
- **M**：[四格原表](../../MNRF-ViewProbe-MSL-Boundary/full_gmm-sequence-bounded-20261102T105427Z-2415811/four_cell_comparison.csv)、[同点四方法](../../MNRF-ViewProbe-MSL-Boundary/same-point-baselines-20261103T121447Z/msl_same_point_comparison.csv)。新增模型 Run ID 为 `full_gmm-sequence-bounded-20261102T105427Z-2415811`。不拿历史 73,538 点 AP .1590 或目标后处理 AP .2792 直接填入 68,572 点四格。
- **G**：[Full/diag/max/mean 与低 FPR](../../MNRF-ViewProbe-FullDiagnostic/full-head-diagnostic-20261030T062555Z-3226097/loss_decomposition.csv)、[上尾上下文](../../MNRF-ViewProbe-FullDiagnostic/full-head-diagnostic-20261030T062555Z-3226097/calibration_tail_context.csv)、[变量 FP/FN/TP 归因](../../MNRF-ViewProbe-FullDiagnostic/full-head-diagnostic-20261030T062555Z-3226097/variable_error_attribution.csv)。GECCO 6,812 为有效校准端点数，286/6,812=4.20%；不把合法训练侧含异常直接判成标签/转换错误。Genesis 归因限 `configuration=Full-full-max`，不可混用其他配置同名字段。
- **N**：[NYC k=8 事件邻域](../../MNRF-FrozenDecoder-Analysis-20261102-final/nyc_event_neighborhood.csv)、[简单参照](../../MNRF-FrozenDecoder-Analysis-20261102-final/reference_metrics.csv)、[字段职责与条件评分](../../MNRF-ViewProbe-ConditionalScore/conditional-score-20261030T-run2/conditional_scores.csv)。前8点全局 `[14701,14709)`，内部 `[14709,14792)`，末8点 `[14792,14800)`；冻结时间索引不冒充新恢复的真实时刻。客流为原 `channnel1`，原字段拼写保留。
- **C**：[RawCount 四法配对](../../MNRF-ViewProbe-DataRepair/calit2-rawcount-20261031/paired_comparison.json)；直接计数参照绑定见 `result/analysis/MNRF-ViewProbe-DataRepair/calit2-rawcount-20261031/baselines-v2/IsolationForest/evidence/baseline.json` 和同级 `PCA/evidence/baseline.json`，两者 complete 且绑定 RawCount Full `rawcount-full-20261031T20261031T092320Z-1093827`。IF 使用标准化后的原计数，RawCount PCA 保留1个成分/1维残差。三条 PCA 记录必须分开：配对文件中的旧 modulo-10 **全维** PCA AP **.0801**；[旧数据残差规则重算](../../MNRF-ViewProbe-FullDiagnostic/full-head-diagnostic-20261030T062555Z-3226097/pca_residual_results.csv) AP **.0858**；RawCount 残差 PCA AP **.1466**。同残差规则的数据参照是 .0858→.1466，.0801→.1466 同时改变规则和数据，不能全归于数据修复；原始文件均保留。2026-11-10 补充的[固定计数幅度对照](../../MNRF-ViewProbe-DataRepair/calit2-rawcount-20261031/count-amplitude-20261110/README.md)与[同点表](../../MNRF-ViewProbe-DataRepair/calit2-rawcount-20261031/count-amplitude-20261110/same_point_results.csv)使用2,329个共同点/70异常，未重新运行四法。幅度公式为 `max(abs(z9), abs(z7))`，直接读取原1,710训练行绑定的 IF/PCA 均值与尺度；目标27点中幅度/PCA共同找回16、仅幅度10、仅PCA1，支持幅度读出候选，不证明完整表示无用。
- **R2**：[Creditcard μ/σ 修订证据](../../MNRF-FrozenDecoder/Creditcard/R-loss-20261102-revision/README.md)。V14 平方残差/NLL AP .7725/.1419，V17 .7079/.0581，恶化已在单变量层面发生；正尺度 affine 变换不改变该变量排序。粗残差箱不是完全匹配，撤回的63.4/76.2倍说法不沿用。[独立确认缺口](independent_confirmation.md) 未发现登记的独立数据；全轴复现阻塞见上述主表 Creditcard 行。
- **I**：[有效点四方法](../../MNRF-Comparison-Adapter-20261102/common_comparison.csv)、[含前缀全轴主表](../../MNRF-Benchmark-Main/benchmark-main-20261103T200322Z-851688/main_results.csv)。原有效点 TP/FP=0/1,411；加前缀后为5/1,597，FPR 1.8780%。新增前缀的标签计数由上述全轴结果与旧有效标签计数对应，并未重算输入。

历史八组源及变量顺序沿用 [source_bindings.csv](../../MNRF-FrozenDecoder-Analysis-20261102-final/source_bindings.csv)。本轮不按日期或成绩替换源；MSL 新模型和 CalIt2 RawCount 都明确另列。

## 下一项最高优先级的受控验证

**优先：Creditcard 固定 R 平方残差的独立确认，决定是否将它推进为下一版异常读出。** 这是八组里已有同解码器公式对照、完整同预算扫描和多交易区段增益共同支持的候选；PSM 的收益更受时段限制，GECCO 的 diag 尚未补上低误报召回，Genesis/NYC 只有少数事件，其他组首先需要任务或数据定义到位。因此先确认一个读出是否能外推，比继续挑头/调阈值更能改变设计决定。

1. **前置输入**：新增、同字段且未参与训练、标准化、校准、人工候选选择的独立交易数据与标签、原始坐标及来源记录。当前没有合格产物；历史测试重分段/删段仍是探索性。数据已具备后才另行授权生成缺少的冻结分数，本轮不执行。
2. **固定项**：Creditcard 源不变；29字段全部保留，R 平方残差、保存 validation median/MAD、max 不变。对照同解码器 R NLL、同源 Full GMM，以及可绑定同一新轴的现有 IF/CATCH-common/PCA-residual；不借用论文汇总值替代新轴分数。不训练、不调尺度、不按新标签筛变量。
3. **评价预注册**：AP；主 FP 上限用新共同轴正常数 `N0*222//141992`，严格并列整组，报告 TP/Recall/实际FP/FPR。扫描工作点仍明确为评估标签事后排序诊断。报警另用各自已保存训练侧阈值，报告全局和预先固定十段，不能拿扫描阈值替代。缺分数的方法明确缺项，不按结果筛选参照。
4. **能改变的决定**：若 R 对 Full/NLL 的排序优势在独立总体和多区段重现，支持优先发展残差读出而非继续直接使用辅助异方差 NLL；若不重现，降为旧测试局部案例。只有保存阈值下的 FPR/Recall 也能满足事先约定的应用要求，才支持报警可用；排序通过不能自动批准阈值。此实验不证明 Full GMM 根因，也不证明所有四分支互补。

PSM 范围提示是次一级、可解释的候选记录项，不在这轮自动改报警规则。MSL 的真实边界和 col_0、CalIt2 RawCount 是已有语义/数据证据支持的设置决定，不需要等待上述独立确认才采纳；具体新训练目标仍未验证。

## 本轮实际计算与保护

实际调用已有 `load_task / deciles / segment_counts_from_axis` 读取固定来源和逐点保存分数；没有新建通用分析框架。19 个消费文件经读取前后摘要核对无变化，包含 PSM 源/分数/标签/输入与使用的 baseline 评分证据。保存输入实际最大值与覆盖 CSV 未舍入值逐项相等。两读出的原混淆矩阵与现有 results.csv 一致，10段点计数严格加总回全局；原事件72/有评分71仍在表中。

定向合成断言覆盖标记仍保留 TP/FP/FN/TN 分母及事件的 none/partial/all/no-valid 分类。另独立复核新 CSV 的所有分段加总、事件计数、同坐标两读出的同一越界标记数和来源链接。仅新增本表述、八组决定 CSV 与 PSM 回放 CSV；历史报告、分数、阈值、模型和数据未修改。
