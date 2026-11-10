# 状态幅度读出与八组决策检查

**本轮不能确认“状态幅度给同一个正式 Flow 补回多少检出”：检查的两个结果根目录中没有正式 Flow 运行产物。** 没有把历史 Full GMM、冻结 R 解码器或计数幅度的成绩冒充 Flow。共用离线入口、真实旧版本 checkpoint 回归及两组数据加载预检已完成；其余六组检查的实际结果如下。未启动任何真实模型前向、训练、拟合、baseline 或真实数据新阈值计算。

实际 HEAD：`4c264c9d068f53b618ae4d39b0baf3fc2667d275`；仓库根为 `/media/h3c/users/wangyueyang1/cxy`，工作目录保持 `APD-CATCH`。没有适用的祖先或子目录 AGENTS.md。开始时已存在的三个未跟踪路径为本目录所属结果根、共用比较脚本及其测试；保留并在其上继续完成，没有 reset、切换、提交或推送。

catch_env：Python 3.8.20、PyTorch 2.4.1+cu121、NumPy 1.24.4、pandas 1.5.3、scikit-learn 1.3.2。统计进程各用单线程；独立任务并行检查，共用评分入口只实现一份。

## 八组状态与本次决定

| 数据集 | 状态、已经执行的检查 | 可采用的设置与下一项缺口 |
|---|---|---|
| CalIt2 RawCount | 入口预检通过，正式 Flow 缺产物；隔离根、字段 `[9,7]`、2520/2520 官方划分、L=192 全部核对 | RawCount 单列。未来显式 L=192 的 Flow 预期测试 2329、校准 24 窗口；小校准集限制须保留。不能套用默认 L=128 launcher 的轴或阈值 |
| PSM | 入口预检通过，正式 Flow 缺产物；L=192、87650 测试端点、13020 校准窗口 | 历史 GMM/R 结果各自保留；正式 Flow 完成后再运行同源三读出比较。越界只分层，不屏蔽或加分 |
| MSL | 只读链路检查完成；55 列保留，但真实序列 ID 未传入原生 launcher，聚合前 col_0 选择也未接通 | 不能把原生 Flow 入口称为已采用边界修正版。最小修补点见下文；已完成的 col_0 成绩是 Full GMM |
| GECCO | 原 full/diag GMM 保存结果与同源端点、指纹、坐标、标签、预测、上尾标签复核完成 | diag 可保留协方差敏感性对照；整体 AUC 提升不等于低 FPR 检出明显改善。校准污染针对报警解释，不能解释 AUC |
| Genesis | 固定 ±8 行的状态切换邻域与稳定区统计已完成，三个异常段均核对 | 有支持工况读出的局部证据，但不同状态变量方向相反，不能统一把二值字段移出评分 |
| NYC | 训练侧日历历史清点和已有事件残差核对完成 | 训练侧每个事件日历键有 30–31 个正常例，足以准备冻结日历参照；尚未执行新参照评分，单事件不证明普遍跟随机制 |
| Creditcard | 数据资格盘点完成，缺独立数据及其分数 | 保持 R 残差候选，不宣称独立确认完成；全轴 3/5800 特征超差是另一个未解决问题 |
| CICIDS | 历史 GMM、逐行 IF/PCA 同 84925 点结果复核完成；session/entity 映射缺失 | 逐行参照可判断现有字段的行级排序；不能证明行邻接是物理时序。191 点掩码前缀另列 |

机器可读状态：[dataset_status.csv](dataset_status.csv)。实际新正式 Flow 三读出结果 **0/2 组**；[formal_flow_results.csv](formal_flow_results.csv) 六个位置均明确缺产物，指标留空。其余六组有本轮可判定检查，不因前两组缺产物而停止。

## 共用入口及公式

实现：[compare_mnrf_frozen_readouts.py](../../../../scripts/compare_mnrf_frozen_readouts.py)。输入一个完成的原生 Flow run 和独立、原先不存在的输出目录。调用原生 `validate_mnrf_lite_v2_run` 一次，核对任务、seed=2021、q=.995、布局、变量顺序、官方切分、训练输入/时间身份、窗口/分段端点、全轴标签/掩码、全局/局部索引、保存阈值及严格预测。没有 fit 或模型 forward 调用。

旧原生 benchmark 没有绑定测试观测字节，仅有标签/时间相同不能证明测试数据相同。因此入口还要求同次运行的 `readout_data_contract.json`，绑定全输入、时间、规范化二元标签、变量顺序、Run ID 和数据版本；缺失时拒绝，不能事后猜测认证。下述未来启动入口在训练前捕获这些内容，成功后独占写入新 run。已有旧 run 如果没有同等证据，要先从其启动时保存的记录恢复绑定，不能只给当前 CSV 补一个“已验证”标记。

- `flow`：原生逐变量 Flow NLL 经过原验证 median/MAD 后跨变量 max；照用原保存阈值。若保存测试 NLL，重新聚合核对，不从 GMM 或 R 分数估计 Flow。
- `state_amplitude`：S 是同端点的原训练输入标准化观测，分数 `max_v(abs(S[t,v]))`。校准 S 和源原始端点观测逐项核对；测试 S 使用同源冻结输入统计，不用 GMM 特征标准化/噪声。
- `flow_plus_amplitude`：复用 `readout.py`，两路分量分别以同次验证集的 median/MAD（同 floor）缩放后取 max。优先核对/使用部署包保存的幅度参数；旧包用同次 val/cal NLL 和原始 S 离线恢复。两种新读出只从原独立 calibration 计算自己的 q=.995 阈值。

缺 validation Flow NLL 时，仍能验收已有 Flow 和 S 幅度；联合读出在无保存校准参数时标为缺文件，整体 partial、CLI 非零。坏身份、非有限值、错轴或原阈值无法复现不降级绕过。旧源不写入，输出以独占新目录及本次私有暂存发布；重复输出拒绝覆盖。

输出 `results.csv`、`scores.npz`、`manifest.json`：保存三路真实分数/预测/阈值、同一坐标与原掩码、校准窗口数和源小文件摘要。排序诊断每路只在全轴选一次 `floor(N_normal/100)` 预算点，并列分数组整体移动，严格 `>`，TP 最大后 FP 最少；保存 q=.995 报警独立列出。PSM 再按官方测试轴十等分投影，并报告相对 Flow 的新增/丢失 TP、FP。feature_1/3 超官方训练最大值只作附加分层，不重新选预算或阈值。

## CalIt2 RawCount 与 PSM：来源核查、已执行预检、待运行命令

[formal_flow_inventory.json](formal_flow_inventory.json) 记录实际遍历 `result/score`、`result/analysis`，包含被 Git 忽略的文件，检查 5572 个文件，未发现 `deployment.pt`、`run_metadata.json`、`best_fusion.pt`、`fusion.pt` 或 `flow.pt`。未检查这两个根之外的任意服务器位置，不将未找到扩大成“任何位置都不存在”。候选按数据来源与完整性选，未按时间/AP 选。

| 当前可找到的固定来源 | 真实类型 | 数据与窗口 | 为什么不能作为正式 Flow |
|---|---|---|---|
| PSM `full_gmm-full-20261024T115119Z-3976714` | 历史 Full GMM | 原 PSM，25 字段，L192 | 无原生 Flow 部署、run identity、Flow val/cal NLL 或 Flow 阈值 |
| CalIt2 `rawcount-full-20261031T20261031T092320Z-1093827` | 历史 RawCount Full GMM | 隔离 RawCount，`[9,7]`，L192 | 同上，GMM NLL 不可替代 Flow NLL |

已经实际执行两次 `formal_flow_launch.sh <task> --dry-run`，退出均为 0，未构造模型。加载真实 CSV，使用对应 GMM 的正确数组摘要算法核对输入身份，按正式 Lite V2 窗口函数计算轴；[flow_preflight.json](flow_preflight.json) 保存完整有效配置、实际 CSV/metadata 路径及命令参数。

| 数据 | 内部 train / validation / calibration 范围（左闭右开） | 窗口数 | 官方测试 / 有效测试 |
|---|---|---|---|
| RawCount | `[0,1710)` / `[1901,2114)` / `[2305,2520)` | 1519 / 22 / 24 | 2520 / 2329 |
| PSM | `[0,105679)` / `[105870,119079)` / `[119270,132481)` | 105488 / 13018 / 13020 | 87841 / 87650 |

以下是**将来人工执行**的两个完整命令，本轮仅执行了 `--dry-run`，没有执行 `--execute`。一个共用 launcher，任务独立、没有全局锁或后台调度。CPU 单线程、seed2021，RawCount 隔离根和 L192 显式绑定；不会退回默认 L128 或共享数据根。目标已存在时拒绝重新训练。

```bash
bash result/analysis/MNRF-Readout-Readiness/readout-readiness-20261110T084455Z/formal_flow_launch.sh CalIt2 --execute
bash result/analysis/MNRF-Readout-Readiness/readout-readiness-20261110T084455Z/formal_flow_launch.sh PSM --execute
```

前置条件是当前 catch_env、完整注册的相应数据、可接受正式训练资源；它们将新建真实 Flow，不能用旧 GMM 权重伪装。预期 run 分别为 `result/score/MNRF-Lite-V2/CalIt2-RawCount/flow-readout-4c264c9/`、`result/score/MNRF-Lite-V2/PSM/flow-readout-4c264c9/`。成功后调用共用比较，结果在各自同级的 `flow-readout-4c264c9-readout-comparison/`。保存真实模型/缓存/来源后才可三路验收；未补齐产物不报完成。未来已有合格产物时可单独只读执行，例如：

```bash
python scripts/compare_mnrf_frozen_readouts.py --task CalIt2 --data-root result/analysis/MNRF-ViewProbe-DataRepair/calit2-rawcount-20261031/data_root --run-dir result/score/MNRF-Lite-V2/CalIt2-RawCount/flow-readout-4c264c9 --output-dir result/score/MNRF-Lite-V2/CalIt2-RawCount/flow-readout-4c264c9-readout-comparison --cpu-threads 1
python scripts/compare_mnrf_frozen_readouts.py --task PSM --run-dir result/score/MNRF-Lite-V2/PSM/flow-readout-4c264c9 --output-dir result/score/MNRF-Lite-V2/PSM/flow-readout-4c264c9-readout-comparison --cpu-threads 1
```

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

指定原生回归实际运行 `CUDA_VISIBLE_DEVICES="" conda run --no-capture-output -n catch_env python -m pytest -q ts_benchmark/baselines/mnrf_lite_v2/tests/test_mnrf_lite_v2.py`，附加本进程 OMP/OpenBLAS/MKL单线程：**10 passed，7.50秒**。随后合并新增测试：**23 passed，34.92秒**；代码收尾后最终重跑 **23 passed，42.66秒，0失败、0跳过**。Python 编译和 Bash 语法检查退出均为0。详情 [execution_checks.json](execution_checks.json)。

新增测试真正从 `4c264c9^`（`5c763f0e85f865621e7c1312b58a52db7ab89d42`）git archive 隔离代码，由旧代码对子进程合成数据拟合并保存 checkpoint，再用当前代码加载同一输入，分数/阈值/预测/有效坐标逐项精确相同。原生新 checkpoint 三模式往返已测。离线入口与原生三读出合成结果精确相同，且比较阶段禁止 encoder/Flow forward 与 fit；测试错标签、时间、字段顺序、训练/测试观测、NLL、掩码、读出身份、缺完整输入绑定和重复输出拒绝。真实数据未拟合模型，未运行前向；少量合成拟合/前向只用于测试。

**总评一：**历史幅度证据值得准备同一正式 Flow 的对照，但当前无正式 Flow 产物，补检出及新增误报的真实代价尚不能回答；不得拿“历史 GMM 幅度更好”代替该结论。

**总评二：**RawCount数据、MSL真实边界/目标职责、明确的共同点与各自阈值报告规则已有证据可采用。diag头、工况条件化、慢速客流参照、R残差候选仍是待独立确认的设计选择。论文16方法原表继续保留于[论文并列表](../../MNRF-Benchmark-Main/benchmark-main-20261103T200322Z-851688/paper_comparison.csv)；这里的common/修正数据结果均分列，不替换论文值。四路互补性仍需四路与最强单路/少路组合的同预算对照，本轮未启动该研究。
