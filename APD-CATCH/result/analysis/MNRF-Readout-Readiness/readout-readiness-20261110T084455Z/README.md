# 状态幅度读出与八组决策检查

**两组正式 Flow 的原工作目录已找到，原生验收及三路同源离线对照均完成（2/2）。** 1% FPR预算内，CalIt2 RawCount 的 Flow→Flow加幅度为9TP/15FP→9TP/12FP；PSM为177TP/575FP→182TP/577FP。新增幅度未证明普遍补检出：按各自q=.995报警，CalIt2净增13TP但增加113FP，PSM不增TP、增加2FP。没有把历史Full GMM或冻结R读出当成正式Flow。

本次分析起始HEAD：`98657f7025c6e2a8776414dfd316ef6b8d7f1857`，分支`main`，起始工作树干净；仓库根为 `/media/h3c/users/wangyueyang1/cxy`，工作目录保持 `APD-CATCH`。没有适用AGENTS.md。原准备轮次基于`4c264c9`，以下六组历史检查保留其原记录。本次只运行CPU离线核验与统计：没有训练、恢复、模型前向、拟合或baseline；两个新增读出的阈值仅由同次运行保存的验证/独立校准资料复现，没有调整原Flow阈值。

catch_env：Python 3.8.20、PyTorch 2.4.1+cu121、NumPy 1.24.4、pandas 1.5.3、scikit-learn 1.3.2。统计进程各用单线程；独立任务并行检查，共用评分入口只实现一份。

## 八组状态与本次决定

| 数据集 | 状态、已经执行的检查 | 可采用的设置与下一项缺口 |
|---|---|---|
| CalIt2 RawCount | 正式Flow三读出验收完成；隔离根、字段 `[9,7]`、2520/2520官方划分、L192、2329同点 | 24个校准窗口限制保留；同预算没有增加TP，独立校准报警增加误报。历史183FP预算单列 |
| PSM | 正式Flow三读出验收完成；L192、87650同点、13020校准窗口 | 加幅度在1%预算净增5TP/2FP；保存阈值下高误报未改善。历史GMM/R独立报告，越界仅分层 |
| MSL | 只读链路检查完成；55 列保留，但真实序列 ID 未传入原生 launcher，聚合前 col_0 选择也未接通 | 不能把原生 Flow 入口称为已采用边界修正版。最小修补点见下文；已完成的 col_0 成绩是 Full GMM |
| GECCO | 原 full/diag GMM 保存结果与同源端点、指纹、坐标、标签、预测、上尾标签复核完成 | diag 可保留协方差敏感性对照；整体 AUC 提升不等于低 FPR 检出明显改善。校准污染针对报警解释，不能解释 AUC |
| Genesis | 固定 ±8 行的状态切换邻域与稳定区统计已完成，三个异常段均核对 | 有支持工况读出的局部证据，但不同状态变量方向相反，不能统一把二值字段移出评分 |
| NYC | 训练侧日历历史清点和已有事件残差核对完成 | 训练侧每个事件日历键有 30–31 个正常例，足以准备冻结日历参照；尚未执行新参照评分，单事件不证明普遍跟随机制 |
| Creditcard | 数据资格盘点完成，缺独立数据及其分数 | 保持 R 残差候选，不宣称独立确认完成；全轴 3/5800 特征超差是另一个未解决问题 |
| CICIDS | 历史 GMM、逐行 IF/PCA 同 84925 点结果复核完成；session/entity 映射缺失 | 逐行参照可判断现有字段的行级排序；不能证明行邻接是物理时序。191 点掩码前缀另列 |

机器可读状态：[dataset_status.csv](dataset_status.csv)。[formal_flow_results.csv](formal_flow_results.csv)含两组正式Flow的六个读出位置、各自报警/预算点（12行）；[formal_flow_psm_segments.csv](formal_flow_psm_segments.csv)含PSM十等分60行。越界分层及完整比较行保存在对应离线输出目录的`results.csv`中；[formal_flow_sources.json](formal_flow_sources.json)保存实际来源、绑定摘要与校准参数。其余六组沿用已完成的检查，本次没有重算这些模型。

## 共用入口及公式

实现：[compare_mnrf_frozen_readouts.py](../../../../scripts/compare_mnrf_frozen_readouts.py)。输入一个完成的原生 Flow run 和独立、原先不存在的输出目录。调用原生 `validate_mnrf_lite_v2_run` 一次，核对任务、seed=2021、q=.995、布局、变量顺序、官方切分、训练输入/时间身份、窗口/分段端点、全轴标签/掩码、全局/局部索引、保存阈值及严格预测。没有 fit 或模型 forward 调用。

旧原生 benchmark 没有绑定测试观测字节，仅有标签/时间相同不能证明测试数据相同。因此入口还要求同次运行的 `readout_data_contract.json`，绑定全输入、时间、规范化二元标签、变量顺序、Run ID 和数据版本；缺失时拒绝，不能事后猜测认证。下述未来启动入口在训练前捕获这些内容，成功后独占写入新 run。已有旧 run 如果没有同等证据，要先从其启动时保存的记录恢复绑定，不能只给当前 CSV 补一个“已验证”标记。

- `flow`：原生逐变量 Flow NLL 经过原验证 median/MAD 后跨变量 max；照用原保存阈值。若保存测试 NLL，重新聚合核对，不从 GMM 或 R 分数估计 Flow。
- `state_amplitude`：S 是同端点的原训练输入标准化观测，分数 `max_v(abs(S[t,v]))`。校准 S 和源原始端点观测逐项核对；测试 S 使用同源冻结输入统计，不用 GMM 特征标准化/噪声。
- `flow_plus_amplitude`：复用 `readout.py`，两路分量分别以同次验证集的 median/MAD（同 floor）缩放后取 max。优先核对/使用部署包保存的幅度参数；旧包用同次 val/cal NLL 和原始 S 离线恢复。两种新读出只从原独立 calibration 计算自己的 q=.995 阈值。

缺 validation Flow NLL 时，仍能验收已有 Flow 和 S 幅度；联合读出在无保存校准参数时标为缺文件，整体 partial、CLI 非零。坏身份、非有限值、错轴或原阈值无法复现不降级绕过。旧源不写入，输出以独占新目录及本次私有暂存发布；重复输出拒绝覆盖。

输出 `results.csv`、`scores.npz`、`manifest.json`：保存三路真实分数/预测/阈值、同一坐标与原掩码、校准窗口数和源小文件摘要。排序诊断每路只在全轴选一次 `floor(N_normal/100)` 预算点，并列分数组整体移动，严格 `>`，TP 最大后 FP 最少；保存 q=.995 报警独立列出。PSM 再按官方测试轴十等分投影，并报告相对 Flow 的新增/丢失 TP、FP。feature_1/3 超官方训练最大值只作附加分层，不重新选预算或阈值。

## CalIt2 RawCount 与 PSM：已完成正式Flow同源对照

[formal_flow_inventory.json](formal_flow_inventory.json)是准备阶段的历史盘点，不代表当前缺产物。本次按[native_flow_benchmark_results.csv](native_flow_benchmark_results.csv)的两个Run ID定位原目录，未按AP或时间换源。两次实际运行均保留在`result/.mnrf-work/score/MNRF-Lite-V2/<TASK>/<RUN_ID>`，原生发布目录只含`.tar.gz`和`test_report*.csv`，不是完整可验收工作目录。

| 正式来源Run ID | 数据与窗口 | 原目录及离线输出 |
|---|---|---|
| `run-20261110T105831Z-3223502` | CalIt2 RawCount，`[9,7]`，L192，seed2021，batch/encoding128 | [原工作目录](../../../.mnrf-work/score/MNRF-Lite-V2/CalIt2/run-20261110T105831Z-3223502)；[比较输出](../../../score/MNRF-Lite-V2/CalIt2/run-20261110T105831Z-3223502-readout-comparison) |
| `run-20261110T105920Z-3229230` | PSM原benchmark数据，25字段，L192，seed2021，batch/encoding128 | [原工作目录](../../../.mnrf-work/score/MNRF-Lite-V2/PSM/run-20261110T105920Z-3229230)；[比较输出](../../../score/MNRF-Lite-V2/PSM/run-20261110T105920Z-3229230-readout-comparison) |

实际使用现有`compare_mnrf_frozen_readouts.py`，每个源调用一次原生validator，随后核对同次输入合同、变量/布局、时间/坐标/标签/掩码、窗口轴、保存NLL、验证median/MAD、原Flow阈值与预测、新读出校准参数。两个CLI均返回0、manifest均complete；比较进程将`torch.nn.Module.__call__`与`MNRFLiteV2.detect_fit`设为失败哨兵，未触发。源被消费文件的SHA256在导出前后相同。没有事后补造输入合同，没有写回源。准备阶段的[flow_preflight.json](flow_preflight.json)仍保留为历史记录。

| 数据 | 内部 train / validation / calibration 范围（左闭右开） | 窗口数 | 官方测试 / 有效测试 |
|---|---|---|---|
| RawCount | `[0,1710)` / `[1901,2114)` / `[2305,2520)` | 1519 / 22 / 24 | 2520 / 2329 |
| PSM | `[0,105679)` / `[105870,119079)` / `[119270,132481)` | 105488 / 13018 / 13020 | 87841 / 87650 |

### 1% FPR预算：事后排序诊断

每路在全局共同轴上仅选一次最高TP、再最少FP工作点，并列分数组整体处理，严格`score > threshold`。CalIt2为70异常/2259正常，FP上限22；PSM为24378异常/63272正常，FP上限632。不是沿用历史GMM的631或183预算，也不是部署阈值。

| 数据 | 读出 | AP | TP / FP | Recall / 实际FPR | 相对Flow 新增/丢失TP | 新增/减少FP |
|---|---|---:|---|---|---|---|
| RawCount | Flow | .240618 | 9 / 15 | 12.857% / .664% | 0 / 0 | 0 / 0 |
| RawCount | 幅度 | .232275 | 9 / 12 | 12.857% / .531% | 6 / 6 | 10 / 13 |
| RawCount | Flow＋幅度 | .237730 | 9 / 12 | 12.857% / .531% | 6 / 6 | 10 / 13 |
| PSM | Flow | .451237 | 177 / 575 | .726% / .909% | 0 / 0 | 0 / 0 |
| PSM | 幅度 | .420669 | 388 / 631 | 1.592% / .997% | 348 / 137 | 629 / 573 |
| PSM | Flow＋幅度 | .451709 | 182 / 577 | .747% / .912% | 5 / 0 | 2 / 0 |

RawCount联合读出替换了检出对象而没有净增TP，少3FP；PSM联合读出净增5TP/2FP，只有很小的前列收益，且幅度单路丢失137个Flow检出。以上测试标签已用于预算扫描，只是探索性排序诊断，不能称为独立确认或逐组选择最佳正式读出。

### 各自q=.995：真实保存/复现阈值的报警

Flow照用原保存阈值；另外两路用同次运行原验证集统计与独立校准集复现其自身阈值。没有在测试集上重新校准。

| 数据 | 读出 | 阈值 | 校准窗口 | TP / FP | Recall / 实际FPR | 相对Flow 新增/丢失TP | 新增/减少FP |
|---|---|---:|---:|---|---|---|---|
| RawCount | Flow | 6.100638411 | 24 | 35 / 158 | 50.000% / 6.994% | 0 / 0 | 0 / 0 |
| RawCount | 幅度 | 2.136992382 | 24 | 48 / 271 | 68.571% / 11.996% | 23 / 10 | 220 / 107 |
| RawCount | Flow＋幅度 | 1526.882734985 | 24 | 48 / 271 | 68.571% / 11.996% | 23 / 10 | 220 / 107 |
| PSM | Flow | 11.801858068 | 13020 | 11722 / 13937 | 48.084% / 22.027% | 0 / 0 | 0 / 0 |
| PSM | 幅度 | 10.544049320 | 13020 | 77 / 10 | .316% / .0158% | 1 / 11646 | 2 / 13929 |
| PSM | Flow＋幅度 | 5.096372330 | 13020 | 11722 / 13939 | 48.084% / 22.030% | 0 / 0 | 2 / 0 |

RawCount仅24个校准窗口。其幅度分量的验证尺度触及.001下限（Flow尺度1.159456），联合数值阈值大不代表更保守或概率更低；PSM两路验证尺度为1.832686/.503429。两组都不能由AP或阈值数值大小推断实际误报。PSM幅度单路误报很少却几乎不召回；联合仍继承Flow的高误报。

PSM第4/6/7/8段，保存阈值下Flow的FP分别为7002/1694/4125/930，联合完全相同；幅度为3/0/0/0。1%预算下Flow四段TP/FP分别为38/517、26/0、1/9、22/7；联合只有第7段变为2/9。全轴增加的5TP分布为第1/2/7/10段各1/2/1/1点，新增2FP在第1/2段各1点。完整十段分母、逐点集合变化见[逐段表](formal_flow_psm_segments.csv)，越界分层见原离线比较目录的结果CSV；每个分层没有重新选阈值。

### 磁盘与后续保留策略

[formal_flow_storage.csv](formal_flow_storage.csv)列出两次运行原工作目录、原生发布与本次离线比较的全部文件及字节数；[formal_flow_sources.json](formal_flow_sources.json)记录`du`实际分配/表观占用。原工作目录分别占2,842,624字节（2.71MiB）和349,609,984字节（333.41MiB），31文件/组。原生两文件发布分别45,056/692,224字节，比较输出分别69,632/1,646,592字节。

PSM超过1MiB的原工作文件：train_features 263,720,128字节、calibration_features 32,550,128、validation_features 32,545,128、normalized_train 13,248,228、calibration_nll 1,302,128、validation_nll 1,301,928。CalIt2没有超过1MiB的单文件。本次PSM比较scores.npz为1,577,373字节。均仅留服务器，不加入Git。

**本次未删除或压缩任何源文件，释放空间0。** 当前原生validator依赖训练缓存与checkpoint；不能删后仍宣称原生可重验。后续两个正式`scripts/.../MNRFLiteV2.sh`默认采用`benchmark`，验证成功后仅发布原生`.tar.gz`与`test_report*.csv`并清理该次暂存工作目录，失败保留。需要离线比较/继续保留完整依赖时，启动前显式设`MNRF_ARTIFACT_POLICY=readout`；该模式不做伪装的“轻量可复现包”。现有两次源继续完整保留，未触发清理，也未启动新实验。模型、缓存、逐点包由根`.gitignore`排除，只提交代码、小型统计表和来源记录。

**独立的历史证据**：RawCount 历史 183 FP 预算下 Full GMM 23TP/180FP，固定幅度 43TP/182FP，IF 40TP/154FP，PCA-residual 30TP/180FP。IF 检出而 Full 漏掉的 27 点中，幅度找回 26、PCA 找回 17。幅度 AP .2323 对 Full .1703，但该历史幅度没有保存报警阈值，不能声称其独立校准报警已经验收，更不能把其尺度/阈值搬给未来 Flow。来源：[历史计数幅度表](../../MNRF-ViewProbe-DataRepair/calit2-rawcount-20261031/count-amplitude-20261110/same_point_results.csv)。

PSM 历史 Full GMM AP .4373、R 残差 .5294，是两种不同头；1% 预算 Full 337TP/631FP、R 5332TP/631FP，其中 R 的 4689 TP 来自第7/8段。保存阈值下 Full/R 的 FP 为 17305/14826，分别 27.35%/23.43% FPR。超训练范围覆盖它们原 FP 的 99.33%/97.46%，同时涉及 72.77% 真异常，不能作抑警器。上述不是 Flow 加幅度的收益。逐段第4、6–8段与越界证据继续引用[既有八组决定](../../MNRF-R-Residual-Readout/readout-20261110/eight_dataset_decisions.md)，没有重新生成另一套历史评分。

## MSL：原生正式链路两项缺口

已核验映射为 27 个训练数组与 27 个测试数组，55 是输入字段数。顺序为 `C-1,C-2,D-14,D-15,D-16,F-4,F-5,F-7,F-8,M-1,M-2,M-3,M-4,M-5,M-6,M-7,P-10,P-11,P-14,P-15,S-2,T-12,T-13,T-4,T-5,T-8,T-9`；映射和原始摘要来自 [msl-raw-mapping-20261031.json](../../MNRF-ViewProbe-DataRepair/msl-raw-mapping-20261031.json)。

1. **边界不通过**：`MSL_script/MNRFLiteV2.sh:5` → `run_mnrf_lite_v2_task.sh:61`（L128）→ `data/data_source.py:173 _load_series` 直接 `read_data` → `mnrf_valid_detect.py:33 split_data` 只按官方边界切分并去标签。没有调用 `scripts/mnrf_msl_sequence_boundaries.py:74 annotate_frame`。模型 `MNRFLiteV2.py:97–102` 支持 `entity_id`，但没有得到真实通道列。按真实长度静态枚举当前 L128 窗口，train/val/cal/test 分别有 2413/381/457/3302 个潜在跨通道窗口；这是当前入口的结构检查，不是新运行测量。最小修补是 MSL 专用加载分支在官方切分前调用已验证 annotate_frame，并让同一实体列贯穿全部 split；不改55输入。
2. **col_0 评分不通过**：`MNRFLiteV2.py:637 score_with_metadata`、`:692` 以及 `training.py:490 calibrate` 都通过 `flow_scores` 对全部变量 max。仅在测试后拿 col_0 而仍用 all55 校准阈值是错配。最小修补是在两个位置共享显式评分对象，先选 col_0 再做聚合及相应独立校准，其他54字段继续参与输入；本轮只定位，没有修改模型。

已完成的历史 Full GMM 四格在 68572 共同点 AP 为：旧拼接 all55/col0 = .171463/.305313，新边界 all55/col0 = .132237/.307332。说明目标选择的收益远大于这次 col0 下边界重训额外 +.002020；仍应遵守真实边界。来源：[four_cell_comparison.csv](../../MNRF-ViewProbe-MSL-Boundary/full_gmm-sequence-bounded-20261102T105427Z-2415811/four_cell_comparison.csv)。不能把它标成正式 Flow 已接通。

## GECCO：diag 的收益在哪里

同一历史源 `full_gmm-full-20261020T204951Z-1642355`、同一 69166 点（730异常、68436正常），指纹、拟合端点及校准端点绑定和保存坐标已重新核对。只从保存分数复算，未加载 GMM 或重新校准。详细表：[gecco_saved_comparison.csv](gecco_saved_comparison.csv)。

| 历史 GMM | AP / AUC | ≤1% FPR TP / FP / Recall / 实际FPR | 原阈值 TP / FP / Recall / 实际FPR |
|---|---|---|---|
| full-covariance max | .223303 / .442452 | 200 / 429 / 27.40% / .627% | 186 / 70 / 25.48% / .102% |
| diagonal max | .245850 / .847167 | 202 / 387 / 27.67% / .565% | 202 / 464 / 27.67% / .678% |

预算是 floor(68436/100)=684，因并列分数整组移动，最优点未用满预算。diag 的 AUC 大幅改善、AP 小幅改善，但此预算只多 2 TP；不支持“已经解决前列排序”。原 calibration 6812 个端点中 286 个异常（4.20%），1598 个窗口含异常；原 q=.995 上尾 35/35 端点及窗口都有异常。full 的阈值 247.4845、diag 为 40.2843，不能跨头比较数值大小；污染影响阈值的代表性，但不构成转换错误，也不能用它解释 AUC。保留污染，不按标签删点。

## Genesis：固定状态切换邻域

计算前固定：完整输入中恰好两种取值的全部13字段；正常相邻两行的状态发生变化，在同一原连续正常段内取切换行左右各8行，遇异常/段边界停止。不是先选高误报字段再定义邻域；二值不等于外生条件。保存预测和阈值不变，测试正常点 12375、FP375。全字段邻域、稳定段和全192步窗口均正常的复核详见 [genesis_state_transitions.csv](genesis_state_transitions.csv)。

- 任一二值字段切换附近：正常5169、FP338、FPR6.539%；稳定段：正常7206、FP37、FPR.513%。整个输入窗均正常时仍为 6.632% 对 .487%，不是仅由历史标注异常连接解释。
- `Motor_Pos2reached` 邻域：487正常、198FP、40.657%；稳定：11888正常、177FP、1.489%。这是强关联。
- 反证：`Motor_Pos3reached`、`Motor_Pos4reached` 切换附近 FPR 低于其稳定段，不能把“所有状态切换必然误报”当事实。字段共同变化和工况混杂未排除。
- 三个异常段 `[15538,15540)`、`[15541,15567)`、`[15916,15938)` 分别命中 1/2、5/26、17/22 点，事件3/3命中、总23TP/50异常点；不能把命中率当成点召回。

足以把工况条件化列为后续候选；尚不足以删除状态字段或改训练目标。还需字段职责证据和独立正常切换区间。事件仅三个。没有实施条件化。

## NYC：慢速客流参照有无历史可用

复用最终冻结 E/R 报告的固定 k=8 事件邻域，没有重新解码。唯一有评分事件为全局 `[14701,14800)`，真实时间 2015-05-03 06:30 至 05-05 07:30，共99点。E 客流平方残差中位数：事件前8正常 .011118、前8异常 .007720、内部83点 .001747、末8点 .000485；事件前正常误差本来更高。R 的相应走势不同，不能把 E 的描述泛化成所有分支。来源：[nyc_event_neighborhood.csv](../../MNRF-FrozenDecoder-Analysis-20261102-final/nyc_event_neighborhood.csv)。

新增只读历史清点：[nyc_training_history.json](nyc_training_history.json)。真实半小时时间映射共17520行，内部训练 `[0,10177)`（2014-07-01至2015-01-29 00:00），全部10177端点标签正常；事件99个日历键在训练侧各有30–31个正常例（83键30例、16键31例），历史336行也正常的有29–30例。因此可以定义**仅用内部训练全部行、按星期×半小时时刻固定中位数、测试期不更新**的慢速参照，无需在事件内跟随当前异常。不是从未来或测试标签估计参照；本轮没有计算新分数。

现有最近336行历史对全部4225评分端点可用，其中3791个端点的过去336行全正常；该事件起点之前336行全正常。该事实只证明可构造参照，不能证明慢速参照会提升 AP。日历变化、季节漂移、特殊节假日作用仍未知，单事件不足以建立因果机制。

## Creditcard：独立确认资格

[creditcard_data_inventory.json](creditcard_data_inventory.json) 实际读取主 DETECT_META、external validation 登记及对应文件清单。只有旧 Creditcard 官方 train/test（142403/142404行），外部登记没有同任务新数据；既有 R/Full/common 分数都落在已用于候选分析的142213个有效测试点。没有合格独立轴、独立标签或独立冻结对照分数，**本轮不做伪独立确认**。

下一项最小输入是同一29字段及同一匿名变换含义的新交易区间、来源与隔离证明、真实标签/坐标，再生成固定 R 残差、R辅助NLL、Full及已有可对齐baseline的分数。若只有新数据缺某方法分数，应单列方法缺项；目前先缺数据本身。新数据主预算严格 `floor(N_normal * 222 / 141992)`；所有读出固定后才看标签。原历史结果只是探索性：R 残差 AP .356110 对辅助NLL .001756，与正式 Flow 无关。

全轴发布另受 3/5800 原完整窗口特征元素超差阻塞；本轮没有前向重试，也没有改变容差、填补前缀或更新源。详细源运行条件与首个超差保留在[主批次 README](../../MNRF-Benchmark-Main/benchmark-main-20261103T200322Z-851688/README.md)。独立性问题与数值复现问题分别记录。

## CICIDS：逐行参照与前缀分开

源 `full_gmm-full-20261024T115157Z-3980319`；common 契约绑定同一输入/划分/Full来源。逐个核对 IF/PCA 的全局坐标、时间、标签、保存预测，与契约端点及 Full 掩码一致：84925有效点，74异常、84851正常。不是按长度截齐。baseline包本身是紧凑数组，不含mask；使用原契约的 `catch_test_indices.npy` 与 Full有效坐标一致来核对，不伪称它保存了mask。

| 保存读出（q=.995，严格>） | AP / AUC | TP / FP | 实际FPR |
|---|---|---|---|
| 历史 Full GMM | .000917 / .538825 | 0 / 1411 | 1.663% |
| sklearn IF，逐行 | .001295 / .674689 | 0 / 343 | .404% |
| sklearn PCA，逐行 | .001066 / .573874 | 0 / 383 | .451% |

数据：[cicids_same_point_checks.csv](cicids_same_point_checks.csv)，也单列1%预算IF 0TP/0FP、PCA 2TP/596FP。逐行模型较高AUC不足以说明有可用的前列检测。当前72字段是流量统计，没有 Flow ID、IP对、端口对、真实timestamp与原行到session映射；契约 `boundary_columns=[]`、`boundary_identity=null`，冻结时间是合成行索引，不是可靠会话时钟。`Flow Bytes/s` 之类统计字段不能替代实体ID。最小材料是原始流记录标识/时间以及到当前CSV行的确定映射；此前转换未保留，不能凭行邻接恢复。

历史 masked cold-start 另外191点中5异常、186正常，191点全部报警，即前缀TP5/FP186；旧有效段仍TP0/FP1411。这是 Full GMM 掩码前缀，不是正式 Flow 新读出。旧有效 AP .000917，全轴 AP .004596 的变化须带这一说明；不使用它证明时序建模有效。来源：[masked_cold_start_verification.json](../../MNRF-Benchmark-Main/benchmark-main-20261103T200322Z-851688/masked_cold_start_verification.json)。

## CPU 验证与研究边界

本次存储策略合成回归：`python -m pytest -q ts_benchmark/baselines/mnrf_lite_v2/tests/test_artifact_retention.py`，**8 passed，4.29秒**。实际执行两个正式wrapper（拦截通用launcher，不执行模型）验证默认benchmark、显式readout/full覆盖、L192/batch128/环境继承；临时目录验证两文件发布、默认删除本次工作副本、显式保留、重复发布拒绝及相邻运行不变。真实比较由catch_env的Python单线程执行，两组原生validator及三读出均成功，禁止模型执行哨兵未触发。未重跑包含合成拟合的历史测试或任何真实实验。

以下为原准备阶段已完成的回归记录，本次没有重新训练来重复这些检查：

指定原生回归实际运行 `CUDA_VISIBLE_DEVICES="" conda run --no-capture-output -n catch_env python -m pytest -q ts_benchmark/baselines/mnrf_lite_v2/tests/test_mnrf_lite_v2.py`，附加本进程 OMP/OpenBLAS/MKL单线程：**10 passed，7.50秒**。随后合并新增测试：**23 passed，34.92秒**；代码收尾后最终重跑 **23 passed，42.66秒，0失败、0跳过**。Python 编译和 Bash 语法检查退出均为0。详情 [execution_checks.json](execution_checks.json)。

新增测试真正从 `4c264c9^`（`5c763f0e85f865621e7c1312b58a52db7ab89d42`）git archive 隔离代码，由旧代码对子进程合成数据拟合并保存 checkpoint，再用当前代码加载同一输入，分数/阈值/预测/有效坐标逐项精确相同。原生新 checkpoint 三模式往返已测。离线入口与原生三读出合成结果精确相同，且比较阶段禁止 encoder/Flow forward 与 fit；测试错标签、时间、字段顺序、训练/测试观测、NLL、掩码、读出身份、缺完整输入绑定和重复输出拒绝。真实数据未拟合模型，未运行前向；少量合成拟合/前向只用于测试。

**总评一：**正式Flow同源对照已给出代价：RawCount在1%预算不增TP，保存阈值多13TP/113FP；PSM在1%预算仅多5TP/2FP，保存阈值不增TP、多2FP。幅度信息有局部互补，但当前固定联合规则未带来可靠的报警改进，不能用历史GMM/R的更大收益代替这一结论。不据此扩大实验或宣称四路互补性已证实。

**总评二：**RawCount数据、MSL真实边界/目标职责、明确的共同点与各自阈值报告规则已有证据可采用。diag头、工况条件化、慢速客流参照、R残差候选仍是待独立确认的设计选择。论文16方法原表继续保留于[论文并列表](../../MNRF-Benchmark-Main/benchmark-main-20261103T200322Z-851688/paper_comparison.csv)；这里的common/修正数据结果均分列，不替换论文值。四路互补性仍需四路与最强单路/少路组合的同预算对照，本轮未启动该研究。
