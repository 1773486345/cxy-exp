# R 残差独立确认：锁定方案与输入缺口

核对日期：2026-11-10。实际及参考 HEAD 均为 `8611b10dbeedb07624b273fa3c197c7cff49d3f8`；开始时工作树干净。未发现适用 AGENTS.md。本次仅核对 metadata、登记、既有评分轴和脚本；没有加载权重、前向、拟合、重新校准或重跑 baseline。本文与原报告共用当前目录，不新建批次或审查包。

## 结论与优先级

**独立确认未完成，当前首要缺口是独立评估数据，而不是只有某个方法缺分数。** 在本次检查的登记及产物内，没有找到 PSM/Creditcard 的未用于训练、输入/分数标准化、校准和此前候选选择的新增评估区间。已有测试标签和六种方法的测试分数均存在，但不能充当独立输入。仓库外是否另有未登记的新数据未知。

继续优先确认 Creditcard 的 R 平方残差，PSM 用来检验跨时段外推；不批准直接替换正式评分或部署报警。已有结果仍为探索性：Creditcard 222 FP 预算下 R 为 130 TP / 222 FP，Full GMM 为 26 / 201，最佳 baseline（IF）为 48 / 215。PSM 全部共同点在 1% 上限下 R 为 5,332 / 631，Full 为 337 / 631，但 4,689 个 R TP 来自第 7、8 段。去掉两段后 R 为 680 / 469，低于 IF 的 813 / 524 和 CATCH-common 的 724 / 524，仍高于 Full 的 108 / 416。后一次筛除分析没有产生新留出集。

报警另外判断：保存阈值下 PSM R 残差 FPR=23.4322%、Recall=35.401%，Full 为 27.3502% / 60.276%；R 仍有高误报，不能只以整体 FPR 下降宣布成功。Creditcard R 保存报警为 184 TP / 1,167 FP、FPR=0.8219%，同样没有独立确认。数值直接来自当前 results.csv，本轮未重新计算原指标。

## 已检查的数据与来源

所有坐标为原完整 CSV 的 **0-based 全局行号**，以下范围均为左闭右开。时间值是冻结输入中的纳秒索引，不能据此声称真实日历时间或实体连续性已确认。

| 数据集 | 固定 Full Run ID | 内部 train / validation / calibration | 官方测试轴 | 六方法已保存共同评分轴 | 共同标签计数 | 独立资格 |
|---|---|---|---|---|---|---|
| Creditcard | full_gmm-full-20261024T115131Z-3977864 | [0,113616) / [113807,128009) / [128200,142403) | [142403,284807)，142,404 点；冻结时间 [142404,284807] | [142594,284807)，142,213 点 | 异常 221，正常 141,992 | 已用于候选分析，不合格 |
| PSM | full_gmm-full-20261024T115119Z-3976714 | [0,105679) / [105870,119079) / [119270,132481) | [132481,220322)，87,841 点；冻结时间 [132482,220322] | [132672,220322)，87,650 点 | 异常 24,378，正常 63,272 | 已用于候选分析，不合格 |

内部范围来自原 source_identity，非本轮新切分。官方测试轴各有 191 点不在现有 R 分数包；不从其他头的补前缀结果推造 R 分数，也不把这些旧测试前缀自动认定为独立数据。训练、验证、校准及 purge 间隔也不重新包装为确认集。

实际检查路径（相对于 APD-CATCH）：

- `dataset/anomaly_detect/DETECT_META.csv`：1,855 条登记，PSM 与 Creditcard 各有一条，官方 train/test 长度分别为 132481/87841、142403/142404；同时查看该数据根下两组 CSV 文件登记，未读取全量 CSV 或重新散列大型缓存。
- `dataset/external_validation/EXTERNAL_DETECT_META.csv` 及 `dataset/external_validation/data/` 文件清单：20 条登记，覆盖 BATADAL、HAI、Metro、Occupancy、OPPORTUNITY、SWAN_SF、MetroPT3，没有 PSM/Creditcard 的独立区间。它们不能因名称为 external_validation 就替代本轮相同任务的独立数据。
- `result/analysis/MNRF-ViewProbe-Ablation/current_sources/PSM.json`、`Creditcard.json`：固定源与以下 Run ID 相符。
- `result/score/MNRF-ViewProbe/{dataset}/{固定RunID}/probe_manifest.json` 和 `analysis_inputs.npz`：来源、变量顺序、冻结 train/val/cal 范围、官方测试坐标/时间/标签。
- `result/analysis/MNRF-FrozenDecoder/{dataset}/{固定RunID}/diagnostic_manifest.json`、`R_squared_residual_test_scores.npz`、`R_nll_test_scores.npz`：登记绑定与上述源一致，保存的 R 分数仅覆盖上表历史测试轴；同目录 E 分数也只有同一测试数据，不是独立确认产物。
- `result/score/MNRF-ViewProbe/{dataset}/` 的 `results/aligned_scores.npz`，以及 `result/score/MNRF-ViewProbe-Common/` 中两组的 `results/aligned_scores.npz`：定向列举实际保存坐标，相关 Full、CATCH-common、IsolationForest、PCA 均落在上表历史区间，没有新增独立评分轴。
- `result/analysis/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/{dataset}/common_contract/common_contract.json`：记录相同原始数据、固定 Full、train/val/cal 和测试坐标。旧 `common-baseline-20261029T123512Z-2049683/PSM/` 只有既有契约，不构成新数据。

来源身份摘要（读取冻结记录，未重新散列大缓存）：PSM `source_identity_sha256=8711277a408942ac1ee53b0a6e1bdc92a82d16dfbed328b82c3e6a2f86a6ba1e`；Creditcard `4cb154b889ac7ad98df849b3c05dd4a9ae084b2f771d1ad69135bfcdeeef2f90`。对应 feature_fingerprint 分别为 `13e2ad46057f0d38cfc6031c5276fb375533ce5ab724817257bc7610bc0c3063`、`9e30484ba13240f715d4353794781140a82aa01a160432292e3b8bbffd6f15a9`。

## 冻结候选与对照：看新评估标签之前锁定

1. 候选只用当前冻结 R 解码器的逐变量平方残差。PSM 保留原 25 个字段、Creditcard 保留原 29 个字段及变量顺序；取各自已保存的 validation center/scale，`z=(residual**2-center)/scale`，跨全部变量 max。不改字段、尺度、聚合、公式，不使用本次测试分段结果选变量。固定上述 Full/解码器身份；新数据的输入标准化也使用原训练侧统计。
2. 对照为同一解码器辅助 NLL、同源原 Full GMM，以及能绑定相同新数据的 CATCH-common、sklearn IsolationForest、sklearn PCA-residual。各方法使用其自己的原评分和保存阈值。common 实现不冒充论文原生实现；不按新结果删除落后方法。
3. persistence、训练目标均值、latent 训练均值替换在现有诊断中只有汇总指标，没有可对齐的逐点分数包。它们不是本轮同预算参照已完成的项目；如果未来要加入，须事先登记并另行授权冻结评分，不能用汇总 AP 推出逐点排序。
4. 选择/规则锁定在本文件和当前源身份上。新数据的来源、排除范围、时间/实体边界、可用信息、共同有效点及缺失点原因必须在查看分数表现前登记。新数据不要求与旧数据全文件摘要相同，但要绑定自身原文件摘要、标签、坐标和完全一致的输入字段语义；模型来源身份不能换。

## 独立数据的资格与最小输入

| 项目 | Creditcard | PSM | 最小补齐动作 |
|---|---|---|---|
| 独立数据及来源 | 未找到已登记的独立交易区间；现有官方测试已用于选择残差候选 | 未找到已登记的独立时段；旧测试分块/删段已看过 | 先提供或登记从未用于训练、统计估计、校准和候选选择的同字段新数据；保留来源、时间/行号、实体证据和隔离说明。不要求重训 |
| 标签与评价轴 | 旧数据有标签；不存在可核验的新数据标签/轴 | 同左 | 对合格新数据提供固定标签、全局/原始行号、时间、边界和有效掩码；若只有新数据缺标签，单列“缺标签”，不能评 AP/召回 |
| 各方法冻结分数 | 没有独立区间的 R residual/NLL、Full 或 baseline 分数 | 同左 | 数据先合格；再另行授权固定权重/统计下生成缺失方法的分数及源绑定，无拟合、无新校准。已有新数据/标签而只有某方法缺分数时，应明确写其方法名与缺失路径，不能称“缺独立数据” |
| 简单参照逐点分数 | 仅历史汇总 | 仅历史汇总 | 非当前独立确认的先决阻塞；若纳入下一次比较，预先登记后保存逐点输出 |

新数据必须证明未参与之前的人工模型/评分选择；仅与 train/val/cal 行号不重叠还不够。需要记录取得时间、此前访问/分析情况、数据生产来源和参与选择的排除范围。Creditcard 缺实体/session 信息仍是时序解释限制，不能创造边界；PSM 同序列历史上下文的合法性需要来源证明。若不能恢复真实时间，按原行序描述跨区段，不称跨日历时期泛化。

## 可直接执行的离线评价规则（有合格分数后）

先读取新分数包和来源记录，逐项比较模型 Run ID、变量顺序、数据版本、全局坐标、时间、标签及有效掩码，拒绝按长度截齐或插值。先在每个预先登记方法间形成共同有效轴，再对所有方法用同一轴和标签；报告官方/指定新轴长度、共同点数、正常/异常数及各方法覆盖缺口。不能为残差获得好成绩而变更评价轴。任何方法缺包则明确缺项，保留其他合格比较，并注明参与方法及共同点变化。

排序诊断固定规则：

- AP 用 `sklearn.metrics.average_precision_score`，全部共同有效点，不依阈值；无正例时留空并说明，不能报告独立性能已确认。
- Creditcard 主 FPR 比例固定为 **222/141992 = 0.1563468364%**。新共同轴正常点数为 `N0`，整数 FP 上限用精确整数运算 `N0*222//141992`，不继续硬用 222。
- PSM 主上限固定 1%，FP 上限 `N0//100`。不得用旧全轴 63,272 的分母代替新数据实际共同正常点数。
- 复用 `analyze_mnrf_r_residual_readout.py::budget_point`：枚举连续分数可达到的严格 `score > threshold` 工作点，精确并列分数整组处理；预算内 TP 最大，TP 相同则 FP 最少，不插值。输出 AP、TP、FP、Recall、实际 FPR、Precision 和相对各预先固定对照的差值。它使用评估标签，必须标为**事后排序诊断**，不是从独立数据学习部署阈值。

报警验收独立输出：使用新评估数据之前已保存且绑定评分公式的训练侧阈值，不用上述扫描阈值替换；严格 `score > saved_threshold`，报告 TP/FP/Recall/FPR（及各分母），不重算 center、scale、校准分位数。IF/PCA/CATCH-common 直接校准分位数，MNRF 三种读出先按各自 validation 参数标准化/max，再用各自校准阈值。q=.995 相同不代表实际 FPR 相同。排序改善与报警能否使用分别得出结论；本方案不擅自设定新的业务报警合格线。

PSM 跨时段检验预先固定为：对**登记的新完整官方/指定评价轴**按 `floor(j*N/10), j=0..10` 分为十段，再映射共同有效坐标；不能对紧凑分数重切。公布全部十段的实际起止坐标、时间、异常/正常点数以及保存阈值下 TP/FP/Recall/FPR，不只展示有益段。排序预算只在新全局共同轴选一次工作点，再投影到十段；不逐段选阈值、不删除第 7/8 段重称确认实验。缺真实时间时称行序区段。各段样本不足/无事件应单列，不能以局部最好区段宣布成功。

## 本轮实际修改与保护

生成器按方法区分阈值来源。现有 `results.csv` 444 行中只改 `threshold_source` 93 行：IF 31、PCA-residual 31、CATCH-common 31；数值字段及其字符串精度、行序和其他说明字段均不变。去掉 threshold_source 后，按原列顺序序列化全部行值的 SHA256 修订前后同为 `950ee7bf11d80e9a23bf34b5cd34c41f1b23c6c1117e3f1f7354648c778f74d7`。原始评分、保存阈值、源文件和模型无写入；未重新运行历史测试集分析来冒充独立确认。

使用 catch_env Python 3.8.20、进程级单线程预算执行 `python -m pytest -q ts_benchmark/baselines/mnrf_view_probe/tests/test_r_residual_readout_analysis.py`：13 passed，7.84 秒。新增六方法参数化用例调用真实行生成函数，核对各自阈值来源、保存阈值与 TP/FP 不变及事后预算说明独立。两个修改 Python 文件的 py_compile、git diff --check 通过；另逐行比较历史 CSV，并核对 current_sources、Full manifest、诊断收据的固定源路径及身份一致。本轮没有运行全测试套件或真实实验。
