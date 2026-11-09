# PSM / Creditcard 冻结 R 残差读出验证

## 设计决定

优先把 **R 平方残差读出**纳入下一轮预先固定的独立验证，而不是将 R 辅助 NLL 视为等价替代。Creditcard 的收益较大且覆盖多个交易区段；PSM 的整体收益集中，第 7、8 段之外仍超过 Full GMM，却未超过 CATCH-common/IF，不能据此宣布普遍优于基线。读出候选和报警候选必须分开：PSM 保存校准阈值仍有大量误报，不能把预算内排序改善称作报警问题解决。

本轮没有找到与训练、校准及此前模型/评分选择隔离、且具有可对齐保存分数的评估集。检查范围为两组指定来源的 analysis_inputs、对应 FrozenDecoder 诊断目录，以及现有 common 比较契约/逐点包；目录中仅有已反复分析的官方测试集与训练侧统计。以下全部为**探索性、事后排序诊断**；重新切成十分位或去掉区段不构成新留出集。

## 范围、规则与核对

- 固定 seed=2021 的两组原 Full，所有字段、原 R validation median/MAD、跨变量 max 均保持不变。只读 saved score、residual/NLL 和元数据，不加载权重/特征缓存，不做前向/拟合/重新校准。R 残差、R 辅助 NLL 与四视图 Full GMM 是三种不同评分。
- 分别对齐 PSM 87,650、Creditcard 142,213 个全局坐标、标签及来源轴；baseline 同时核对时间、完成记录、分数文件摘要、源数据/变量身份和保存阈值。Full 的紧凑分数/掩码/全轴映射、NLL 重聚合与保存预测均核对；R 的原始逐变量残差/NLL 按保存 center/scale 重聚合复核。rtol=1e-5、atol=1e-6；原 AP/AUC、TP/FP 与诊断 metrics.csv 一致。
- 每种方法扫描自己的完整连续分数。相同分数整组进入/退出；最大化 TP，再最小化 FP。严格 `score > threshold`，不插值。FP 上限为整数：PSM floor(63272×.01)=632；排除 7/8 后 floor(52568×.01)=525；Creditcard 固定 222 / 1480。实际 FP 允许小于上限。
- 十等分先对完整官方测试轴按 `floor(j*N/10)` 固定，再映射共同有效点；不是对紧凑数组重分块。PSM 官方轴 87,841，Creditcard 142,404，各有 191 点未评分，本轮未补分。各分段 TP/FP 使用**全局**工作点，绝不逐段另选阈值。CSV 的 decile=global 为全局，1..10 为区段。
- 保存 q=.995 阈值逐方法保留，严格 > 复算。各方法不是共同数值阈值或相同实际 FPR。PSM 数据时间值不擅自解释为墙钟；Creditcard 十分位仅交易行序，不证明实体/session 连续性。

## 全局结果（AP 与阈值无关）

| dataset | scope | method | rule | AP | TP | FP | Recall | FPR |
|---|---|---|---|---|---|---|---|---|
| PSM | all_common | Full-GMM | saved_q995 | 0.437281 | 14694 | 17305 | 60.276% | 27.3502% |
| PSM | all_common | Full-GMM | posthoc_fpr_1pct | 0.437281 | 337 | 631 | 1.382% | 0.9973% |
| PSM | all_common | R-residual | saved_q995 | 0.529406 | 8630 | 14826 | 35.401% | 23.4322% |
| PSM | all_common | R-residual | posthoc_fpr_1pct | 0.529406 | 5332 | 631 | 21.872% | 0.9973% |
| PSM | all_common | R-auxiliary-NLL | saved_q995 | 0.297434 | 5145 | 18194 | 21.105% | 28.7552% |
| PSM | all_common | R-auxiliary-NLL | posthoc_fpr_1pct | 0.297434 | 1 | 27 | 0.004% | 0.0427% |
| PSM | all_common | CATCH-common | saved_q995 | 0.424089 | 614 | 365 | 2.519% | 0.5769% |
| PSM | all_common | CATCH-common | posthoc_fpr_1pct | 0.424089 | 966 | 631 | 3.963% | 0.9973% |
| PSM | all_common | IsolationForest | saved_q995 | 0.439688 | 422 | 11 | 1.731% | 0.0174% |
| PSM | all_common | IsolationForest | posthoc_fpr_1pct | 0.439688 | 926 | 627 | 3.799% | 0.9910% |
| PSM | all_common | PCA-residual | saved_q995 | 0.534842 | 4414 | 1640 | 18.106% | 2.5920% |
| PSM | all_common | PCA-residual | posthoc_fpr_1pct | 0.534842 | 612 | 627 | 2.510% | 0.9910% |
| PSM | exclude_deciles_7_8 | Full-GMM | saved_q995 | 0.358000 | 7972 | 10902 | 45.518% | 20.7389% |
| PSM | exclude_deciles_7_8 | Full-GMM | posthoc_fpr_1pct | 0.358000 | 108 | 416 | 0.617% | 0.7914% |
| PSM | exclude_deciles_7_8 | R-residual | saved_q995 | 0.388526 | 1853 | 8322 | 10.580% | 15.8309% |
| PSM | exclude_deciles_7_8 | R-residual | posthoc_fpr_1pct | 0.388526 | 680 | 469 | 3.883% | 0.8922% |
| PSM | exclude_deciles_7_8 | R-auxiliary-NLL | saved_q995 | 0.264095 | 1571 | 11894 | 8.970% | 22.6259% |
| PSM | exclude_deciles_7_8 | R-auxiliary-NLL | posthoc_fpr_1pct | 0.264095 | 2 | 359 | 0.011% | 0.6829% |
| PSM | exclude_deciles_7_8 | CATCH-common | saved_q995 | 0.401929 | 485 | 357 | 2.769% | 0.6791% |
| PSM | exclude_deciles_7_8 | CATCH-common | posthoc_fpr_1pct | 0.401929 | 724 | 524 | 4.134% | 0.9968% |
| PSM | exclude_deciles_7_8 | IsolationForest | saved_q995 | 0.401203 | 416 | 11 | 2.375% | 0.0209% |
| PSM | exclude_deciles_7_8 | IsolationForest | posthoc_fpr_1pct | 0.401203 | 813 | 524 | 4.642% | 0.9968% |
| PSM | exclude_deciles_7_8 | PCA-residual | saved_q995 | 0.491316 | 206 | 18 | 1.176% | 0.0342% |
| PSM | exclude_deciles_7_8 | PCA-residual | posthoc_fpr_1pct | 0.491316 | 410 | 523 | 2.341% | 0.9949% |
| Creditcard | all_common | Full-GMM | saved_q995 | 0.051007 | 54 | 1480 | 24.434% | 1.0423% |
| Creditcard | all_common | Full-GMM | posthoc_fp_222 | 0.051007 | 26 | 201 | 11.765% | 0.1416% |
| Creditcard | all_common | Full-GMM | posthoc_fp_1480 | 0.051007 | 54 | 1406 | 24.434% | 0.9902% |
| Creditcard | all_common | R-residual | saved_q995 | 0.356110 | 184 | 1167 | 83.258% | 0.8219% |
| Creditcard | all_common | R-residual | posthoc_fp_222 | 0.356110 | 130 | 222 | 58.824% | 0.1563% |
| Creditcard | all_common | R-residual | posthoc_fp_1480 | 0.356110 | 185 | 1242 | 83.710% | 0.8747% |
| Creditcard | all_common | R-auxiliary-NLL | saved_q995 | 0.001756 | 0 | 222 | 0.000% | 0.1563% |
| Creditcard | all_common | R-auxiliary-NLL | posthoc_fp_222 | 0.001756 | 0 | 0 | 0.000% | 0.0000% |
| Creditcard | all_common | R-auxiliary-NLL | posthoc_fp_1480 | 0.001756 | 1 | 1325 | 0.452% | 0.9332% |
| Creditcard | all_common | CATCH-common | saved_q995 | 0.091205 | 93 | 980 | 42.081% | 0.6902% |
| Creditcard | all_common | CATCH-common | posthoc_fp_222 | 0.091205 | 39 | 197 | 17.647% | 0.1387% |
| Creditcard | all_common | CATCH-common | posthoc_fp_1480 | 0.091205 | 128 | 1465 | 57.919% | 1.0317% |
| Creditcard | all_common | IsolationForest | saved_q995 | 0.096501 | 96 | 1083 | 43.439% | 0.7627% |
| Creditcard | all_common | IsolationForest | posthoc_fp_222 | 0.096501 | 48 | 215 | 21.719% | 0.1514% |
| Creditcard | all_common | IsolationForest | posthoc_fp_1480 | 0.096501 | 112 | 1474 | 50.679% | 1.0381% |
| Creditcard | all_common | PCA-residual | saved_q995 | 0.045347 | 54 | 650 | 24.434% | 0.4578% |
| Creditcard | all_common | PCA-residual | posthoc_fp_222 | 0.045347 | 41 | 219 | 18.552% | 0.1542% |
| Creditcard | all_common | PCA-residual | posthoc_fp_1480 | 0.045347 | 57 | 924 | 25.792% | 0.6507% |

## PSM：局部收益、外推边界与报警

全样本 1% FPR 预算内 R 残差检出 5332 点，其中第 7、8 段合计 4689（87.94%）；该两段包含 6,864/24,378 个有效异常点。去掉两段后并非沿用旧工作点：重新对余下 70,082 点（17,514 异常、52,568 正常）扫描 floor(52568×.01)=525 的预算。R 残差 TP/FP=680/469、Recall=3.883%、AP=0.388526；相对 Full GMM 仍有收益，但不能把原全局优势外推为各阶段都强于基线。

| method | TP | FP | R_minus_TP | AP |
|---|---|---|---|---|
| Full-GMM | 108 | 416 | 572 | 0.358000 |
| R-auxiliary-NLL | 2 | 359 | 678 | 0.264095 |
| CATCH-common | 724 | 524 | -44 | 0.401929 |
| IsolationForest | 813 | 524 | -133 | 0.401203 |
| PCA-residual | 410 | 523 | 270 | 0.491316 |

保存阈值的原正常高分时段对照（十分位 4、6、7、8，单元为 FP / 正常点 FPR）：

| decile | Full-GMM | R-residual | R-auxiliary-NLL |
|---|---|---|---|
| 4 | 6989 / 85.29% | 4308 / 52.58% | 6242 / 76.18% |
| 6 | 3765 / 50.16% | 3777 / 50.32% | 3763 / 50.13% |
| 7 | 5500 / 100.00% | 5493 / 99.87% | 5437 / 98.85% |
| 8 | 903 / 17.35% | 1011 / 19.43% | 863 / 16.58% |

R 残差保存阈值全局 FP=14826、FPR=23.432%，Full GMM 为 17305/27.350%；R 残差在原 4/6/7/8 段仍有 14589 个误报（占自身 FP 98.40%）。逐段变化不一致，降低总体误报不等于修复原正常高分工况。其保存阈值 Recall 为 35.401%，Full 为 60.276%，不能只报更低 FPR。
预算内 R 残差优于 R 辅助 NLL 支持对读出公式进行独立确认；保存阈值下的误报则仍要求另行验证训练侧校准代表性/漂移。这里没有重新校准，也不据测试预算选部署阈值。原时段依据：[PSM 正常高分诊断](../../MNRF-PSM-ScoreDiagnosis/psm-score-diagnosis-20261103T180743Z-306520/README.md)。

## Creditcard：完整扫描 Full GMM 后的收益

| budget | R_TP_FP | Full_TP_FP | extra_TP | extra_Recall_pp |
|---|---|---|---|---|
| posthoc_fp_222 | 130/222 | 26/201 | 104 | 47.059 |
| posthoc_fp_1480 | 185/1242 | 54/1406 | 131 | 59.276 |

Full GMM 1,480 预算是对所有工作点完整扫描，结果不是引用保存报警点：同样 54 TP 的扫描工作点仅需 1,406 FP，保存阈值则为 54 TP / 1,480 FP。R 残差 1,480 预算最佳为 185 TP / 1,242 FP（保存阈值为 184 TP / 1,167 FP）。

新增/丢失 TP 是逐点集合变化，非仅净差；以下按固定官方交易十等分统计：

| budget | decile | added_TP | lost_TP |
|---|---|---|---|
| 222 | 1 | 34 | 0 |
| 222 | 2 | 5 | 0 |
| 222 | 3 | 3 | 0 |
| 222 | 4 | 8 | 0 |
| 222 | 5 | 6 | 0 |
| 222 | 6 | 12 | 0 |
| 222 | 7 | 11 | 0 |
| 222 | 8 | 19 | 0 |
| 222 | 9 | 6 | 0 |
| 222 | 10 | 0 | 0 |
| 1480 | 1 | 22 | 0 |
| 1480 | 2 | 7 | 0 |
| 1480 | 3 | 6 | 0 |
| 1480 | 4 | 11 | 0 |
| 1480 | 5 | 12 | 0 |
| 1480 | 6 | 15 | 0 |
| 1480 | 7 | 18 | 0 |
| 1480 | 8 | 23 | 0 |
| 1480 | 9 | 10 | 0 |
| 1480 | 10 | 7 | 0 |

222 预算的净增 104 TP 分布于第 1–9 段，第 1/8 段分别增 34/19，合计占净增 50.96%；1,480 预算的净增 131 TP 覆盖全部十段，第 1/8 段分别增 22/23，合计占净增 34.35%。两个预算下均未丢失 Full GMM 已检出的异常点。第 1 段有 79/221 个异常点，不能把原始 TP 集中程度与每段检出能力混淆；完整分母在 CSV 中。

R 残差相对 R 辅助 NLL 的排序差异在同一冻结解码器上出现，未改变表示/变量选择；与 Full GMM 的差异同时包含读出公式和任务差异，不能宣称已证明 GMM 根因。交易区段跨度支持它不只是一个局部事件，但仍是已用于分析/选择候选的测试集，尚未证明独立数据上的稳定性。

## 下一版设计：读出与报警分开验收

1. **读出优先级：优先验证 R 平方残差，保留全字段、保存标准化和 max 规则作为固定候选。** Creditcard 的同预算收益跨多个区段；PSM 排除主要获益时段后仍优于当前 Full GMM，但低于 CATCH-common/IF，故只支持有数据/阶段依赖的候选，不支持统一替换正式模型或宣布全局胜出。不得按当前测试结果删除变量、调尺度或拼接逐数据集最佳头。
2. **报警决定：不批准把保存 q=.995 的 R 残差报警直接当作已解决方案。** PSM 仍有高 FPR 和原高分时段误报，排序改善与校准迁移必须独立判定；Creditcard 保存报警明显更有用，但也需独立留出评估。
3. **下一次最小验证：** 在未用于当前选择的留出数据/未来时间区间上，预先固定 R 残差与 Full、R NLL、可取得的简单参照，固定数据与坐标和训练侧阈值规则；分别验收低误报排序和保存校准报警。若没有独立数据，则不能把历史分块复验升级为确认性证据。本轮不启动该实验。

## 完整区段统计与简单参照缺口

各方法全部十等分的 TP、FP、异常/正常点数、Recall、FPR、AP，以及 PSM 排除 7/8 后的区段结果均在 [results.csv](results.csv)。每行 operating_point 指定全局选定的工作点，decile 指定统计区段；global_fp_budget 不意味着对各段分别使用该预算。所有分段计数之和均已核对等于对应全局计数。

persistence、训练目标均值、latent 训练均值替换只有既有汇总行，未保存可对齐的一维逐点总分和校准包；本轮不通过重算预测或当前输入补造。CSV 将其标为 aggregate_only，保留原 AP/AUC，仅不参与同预算和逐段比较。此缺口不等于参照失效或训练失败。

| dataset | method | status | ap | reason |
|---|---|---|---|---|
| PSM | persistence | aggregate_only | 0.37901239314995094 | aggregate metrics only; no saved pointwise score/calibration bundle |
| PSM | train_target_mean | aggregate_only | 0.37474388222333577 | aggregate metrics only; no saved pointwise score/calibration bundle |
| PSM | decoder_latent_train_mean | aggregate_only | 0.37304692967495656 | aggregate metrics only; no saved pointwise score/calibration bundle |
| Creditcard | persistence | aggregate_only | 0.00851153923326723 | aggregate metrics only; no saved pointwise score/calibration bundle |
| Creditcard | train_target_mean | aggregate_only | 0.022918339613947042 | aggregate metrics only; no saved pointwise score/calibration bundle |
| Creditcard | decoder_latent_train_mean | aggregate_only | 0.01839937616835035 | aggregate metrics only; no saved pointwise score/calibration bundle |

## 来源与文件保护

实际 HEAD：`9f5d15c50bc2a856acc0cfd81af518c6d390ab3e`。
全部已消费输入的小文件/逐点证据在分析前后 SHA256 相同；未重新计算大型特征缓存摘要。以下记录实际读取路径与摘要。

| path | sha256 |
|---|---|
| result/score/MNRF-ViewProbe/PSM/full_gmm-full-20261024T115119Z-3976714/analysis_inputs.npz | 8458e16efe71746cd157b9814250f3335eda211edf6632cc9a1ac9aaba8f5849 |
| result/score/MNRF-ViewProbe/PSM/full_gmm-full-20261024T115119Z-3976714/results/aligned_scores.npz | 1f63006449bbf20ac2f5e60c66e4096717499de25306d6994360228623f5e157 |
| result/analysis/MNRF-FrozenDecoder/PSM/full_gmm-full-20261024T115119Z-3976714/R_squared_residual_test_scores.npz | 10cad3f027787296b658d8d75fabae8adc45d782e7a1ee08c3813460a95a22a7 |
| result/analysis/MNRF-FrozenDecoder/PSM/full_gmm-full-20261024T115119Z-3976714/R_nll_test_scores.npz | c13a59f9ded6c196e4b37a00b9b2d61473209f5da0c9886056f73a5bf06ce4ab |
| result/score/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/PSM/CATCH-common/results/aligned_scores.npz | 967fb58af12428f9b27518797e4d652868a00742e2f704dc054cb50a6c78c634 |
| result/score/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/PSM/IsolationForest/results/aligned_scores.npz | 398c6577777011ace8ab1e903ea00be5a57e3ed46e6452797f4665ddfec743bf |
| result/score/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/PSM/PCA/results/aligned_scores.npz | bef5f103a2b4db011ad673424385f4ff831e10c475a12a43fbdf975cb2e88422 |
| result/score/MNRF-ViewProbe/Creditcard/full_gmm-full-20261024T115131Z-3977864/analysis_inputs.npz | b1d98a9083046d6da0888f4665812190cb84d7e0d9fbbfa3b100cb8a4d766acf |
| result/score/MNRF-ViewProbe/Creditcard/full_gmm-full-20261024T115131Z-3977864/results/aligned_scores.npz | a9333f546dae8bb705c98816671adf2f0646c12d74e0966e41d9ffe585879302 |
| result/analysis/MNRF-FrozenDecoder/Creditcard/full_gmm-full-20261024T115131Z-3977864/R_squared_residual_test_scores.npz | 3aaed44613272c360dfe1d09c8974a3bfff6f4a652fabc3faec6839de083681d |
| result/analysis/MNRF-FrozenDecoder/Creditcard/full_gmm-full-20261024T115131Z-3977864/R_nll_test_scores.npz | 65dd1ca499e491f5a138334022ba9826c48fef8c8099c5c1348b47ab4f47b517 |
| result/score/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/Creditcard/CATCH-common/results/aligned_scores.npz | 6a4dcd1ab89395164289e8d00476a4844c7234aae59b7f47d2fcd8579e5acfe8 |
| result/score/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/Creditcard/IsolationForest/results/aligned_scores.npz | 6f12687769cc51bae22ddde11b19da38a5024065d33d361e8aacd490c4721f2a |
| result/score/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/Creditcard/PCA/results/aligned_scores.npz | 2cc2beae826fcf31bb05dcb2ddb8bfd617517c0ca8575ea601b8fb2a1f9e6bcd |

另核对来源 manifest、诊断收据、metrics.csv、Full 校准/NLL、R 逐变量证据及 common 契约/完成记录；总计 34 个实际消费文件在本次读取前后摘要不变。来源与方法评分路径同时保存在结果 CSV。报告未改写原历史表、模型或评分产物。
