# MNRF 对既有 16 方法表格的评价适配方案

本方案只处理评价对齐和已有产物的只读复算，不改变模型、分数、阈值或历史结果。baseline 的配置和表格结果按已确认前提保留；论文/原生归档/common-protocol 三类结果分开报告。

共同协议批次 `result/analysis/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/` 已按各任务 `task_status.json` 逐项检查：八组的 CATCH-common、IsolationForest、PCA 共 24 个实际输出均存在 `results/aligned_scores.npz`；CATCH 使用 `results/common_manifest.json`，IF/PCA 使用 `evidence/baseline.json`，24 个文件均与各自 `common_contract.json` 的测试端点摘要一致。精确路径和哈希保存在 `result/analysis/MNRF-Comparison-Adapter-20261102/common_output_inventory.json`。

## 1. 正式采用的报告设置

主表沿用已经确认的 16 方法原生结果：每个方法保留其官方数据版本、官方 train/test 范围、原生评分头、窗口和阈值/后处理。MNRF 不把自己的 `q=0.995` 报警结果改名为论文表格的原生报警指标。

MNRF 应报告两行（或两个明确区块）：

1. **MNRF Full GMM / common valid-endpoint**：使用注册 Full 源、原始行序、完整标签轴和源保存的 valid mask；ROC-AUC、AP 只在真实有效端点上计算。报警使用独立 calibration 的 `q=0.995`、严格 `score > threshold`，不使用测试标签调阈值，不做 point adjustment。Affiliation 使用保存的连续有效区间和边界，名称保留 `affiliation_f_segment_macro`。
2. **MNRF native-table adaptation（待补分数）**：只有在为每个任务生成覆盖官方测试轴的 MNRF 分数后，才能与原表的逐点/报警指标逐项对应。不能把当前有效端点数组补零、插值或删除标签后冒充这一行。

因此，当前已有 MNRF 结果可直接复算的是第一行的 AUC、AP、q=.995 点级报警和同口径段指标；不能直接复算原表方法的分数，也不能从 MNRF 分数推导 CATCH、ModernTCN 等方法的原生报警值。原表数值继续作为已确认的 paper/native reference。

## 2. 官方轴、MNRF 轴和内部切分

官方版本统一指当前任务的 `dataset/anomaly_detect/<TASK>.csv` 与 `DETECT_META.csv` 记录的 benchmark 任务；baseline 表值的官方范围和标签定义按已确认的原表/服务器设置使用。下表的内部区间是 MNRF 在官方训练区间内的冻结 train/validation/calibration 切分，不是对官方 train/test 的重新定义。

| 数据集 | 官方 train/test 点数 | seq | MNRF 内部 train / validation / calibration 全局区间 | MNRF test 有效点 | 冷启动/尾部 | 有效覆盖、被排除异常标签 |
|---|---:|---:|---|---:|---|---|
| PSM | 132481 / 87841 | 192 | `[0,105679]` / `[105870,119079]` / `[119270,132481]` | 87650 | 前 191 点无端点评分；无尾部缺口 | 87650/87841=99.7826%；前缀排除 3 个异常标签 |
| MSL | 58317 / 73729 | 192 | `[0,46348]` / `[46539,52332]` / `[52523,58317]` | 73538 | 前 191 点；无尾部缺口 | 99.7409%；排除 0 个异常标签 |
| GECCO | 69260 / 69261 | 96 | `[0,55256]` / `[55351,62258]` / `[62353,69260]` | 69166 | 前 95 点；无尾部缺口 | 99.8628%；排除 0 个异常标签 |
| Genesis | 3604 / 12616 | 192 | `[0,2577]` / `[2768,3090]` / `[3281,3604]` | 12425 | 前 191 点；无尾部缺口 | 98.4860%；排除 0 个异常标签 |
| NYC | 13104 / 4416 | 192 | `[0,10177]` / `[10368,11640]` / `[11831,13104]` | 4225 | 前 191 点；无尾部缺口 | 95.6748%；排除 0 个异常标签 |
| CalIt2 | 2520 / 2520 | 192 | `[0,1710]` / `[1901,2114]` / `[2305,2520]` | 2329 | 前 191 点；无尾部缺口 | 92.4206%；排除 4 个异常标签 |
| Creditcard | 142403 / 142404 | 192 | `[0,113616]` / `[113807,128009]` / `[128200,142403]` | 142213 | 前 191 点；无尾部缺口 | 99.8659%；排除 2 个异常标签 |
| CICIDS | 85115 / 85116 | 192 | `[0,67786]` / `[67977,76450]` / `[76641,85115]` | 84925 | 前 191 点；无尾部缺口 | 99.7756%；排除 5 个异常标签 |

“无尾部缺口”仅说明当前 MNRF 缓存覆盖到官方测试轴最后一点，不代表它覆盖了冷启动前缀。所有数据集均保留完整 `analysis_inputs.npz` 标签/时间/全局坐标轴；无效点不参与 MNRF 当前评分集合，也不被当作正常点。

### PSM 的优先处理

PSM 官方测试标签有 87841 点，而 MNRF `aligned_scores.npz` 只有全轴中的 87650 个有效端点（全轴长度仍为 87841，前 191 点为 NaN/预测哨兵）。这 191 点没有合法 MNRF 分数，其中 3 点是真实异常。当前可以交付的严格结果是：

```text
labels = analysis_inputs.test_labels[valid_mask]
scores = aligned_scores[valid_mask]
coordinates = analysis_inputs.test_global_indices[valid_mask]
```

这给出 87650 点的精确坐标比较，覆盖率为 99.7826%。不得把它扩展成 87841 点的“完整 MNRF 表格分数”。若必须与官方全轴报警逐点对齐，后续需要一份预先批准的 MNRF warm-up/cold-start 规则和新输出；本轮不生成、不补零、不插值、不删掉那 3 个异常标签。

其余七组采用完全相同的原则，差异只在 seq、覆盖率和冷启动前缀中的异常数量。官方 test 范围仍是完整标签轴；MNRF 的内部 validation/calibration 只来自官方 train 区间，不能被描述成官方测试切分。

## 3. 排序、报警和后处理

### MNRF

- 排序：`sklearn.metrics.roc_auc_score` 与 `average_precision_score`，仅使用 valid endpoint 的真实标签和分数。
- 原始分数：Full GMM 逐变量 NLL，按保存的 feature statistics 和固定噪声得到；逐变量 validation median/MAD 标准化后沿变量轴取 max。
- 报警：calibration 分数的线性 `q=0.995` 分位数；严格 `score > threshold`。
- 后处理：保留有效连续区间、实体/段边界和冷启动无效 mask；不做 point adjustment，不跨缺口拼接。
- 现有表格中的 Full 成绩由这个 Full GMM probe 产生，不是完整 Flow，也不是冻结 E/R 解码器残差/NLL。E/R 结果只能作为辅助诊断。

### 既有 16 方法表格

原生 score 入口提供连续分数，原生 label 入口提供各自阈值和报警结果；二者不是同一次模型输出时，不拼接 score 的 AUC 与 label 的 Affiliation-F。具体窗口、评分覆盖、阈值和后处理以 `result/analysis/MNRF-Baseline-Investigation-20261031/baseline_settings.csv` 的 dataset×method×entry 记录为准。

原生适配器 `ts_benchmark/evaluation/strategy/anomaly_detect.py` 对短预测数组会在尾部 padding 0 后交给完整测试标签轴；这属于原生 label 评价行为，不等价于 MNRF 的 valid-mask 语义。原生 CATCH/其他方法的窗口展平、重叠、冷启动和尾部处理必须留在其 native 列中。阈值可能来自原生 percentile/anomaly-ratio 或其他方法规则；阈值差异只影响报警指标，不改变 AUC/AP。

## 4. 已有 MNRF 缓存能做什么

| 目标 | 当前证据 | 处理 |
|---|---|---|
| 八组 MNRF Full AUC/AP | `results/aligned_scores.npz` + `analysis_inputs.npz`，valid mask 和坐标齐全 | 可只读复算；结果限定为 valid-endpoint 范围 |
| 八组 MNRF q=.995 点级 P/R/F1/FPR、混淆矩阵 | 保存 threshold、prediction、scores、labels | 可只读复算，严格检查 `prediction == score > threshold` |
| 八组 MNRF segment-macro Affiliation | 完整标签轴、segment IDs、valid intervals 和已有评价封装 | 可只读复算；不得改名为论文 Aff-F |
| 原表 CATCH/ModernTCN 等 native AUC/AP | 当前目录只有论文/配置记录，没有对应逐点 score 文件 | 保留已确认表值；不能在本机独立复算 |
| common CATCH/IF/PCA 的独立坐标复算 | 2026-10-29 批次的 24 个 `aligned_scores.npz` 均存在；契约和摘要由 `prepare_mnrf_comparison_adapter.py --mode inventory` 核对 | 可在同一 common contract 上只读运行 common evaluator；仍与论文/原生结果分列 |
| PSM 官方 87841 点的 MNRF 全轴成绩 | 当前 MNRF 仅有 87650 合法分数 | 缺少前 191 点分数；不得制造全轴结果 |

## 5. 可执行的只读步骤

### A. 逐源完整性核验

用户已激活 `catch_env` 并进入 APD-CATCH 后，对每个注册 Full 源执行：

```bash
python scripts/validate_mnrf_view_probe.py \
  result/score/MNRF-ViewProbe/PSM/full_gmm-full-20261024T115119Z-3976714
```

将 `PSM` 和 run ID 替换为表中其余七组的实际路径。该命令只读，不训练、不拟合、不改源。

### B. 复算一个 MNRF valid-endpoint 结果

下面命令只读取保存的数组和保存阈值，不重新校准：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

run = Path("result/score/MNRF-ViewProbe/PSM/full_gmm-full-20261024T115119Z-3976714")
inputs = np.load(run / "analysis_inputs.npz", allow_pickle=False)
aligned = np.load(run / "results/aligned_scores.npz", allow_pickle=False)
labels = inputs["test_labels"].astype(np.int8, copy=False)
scores = aligned["aligned_scores"]
mask = np.asarray(aligned["valid_mask"], dtype=bool)
pred = np.asarray(aligned["aligned_predictions"])
threshold = float(np.load(run / "heads/calibration.npz", allow_pickle=False)["threshold"])
assert labels.shape == mask.shape == scores.shape == pred.shape
assert np.array_equal(pred[mask], (scores[mask] > threshold).astype(pred.dtype))
y, s = labels[mask], scores[mask]
print({"valid_points": int(mask.sum()), "roc_auc": float(roc_auc_score(y, s)),
       "average_precision": float(average_precision_score(y, s)),
       "threshold": threshold})
PY
```

对其它数据集只替换 `run`。脚本不会把无效前缀当正常点，也不会改写任何文件。若数组字段名在某个历史源中不同，应先按该源 `aligned_scores.npz` 的实际字段核对后再运行，不能猜字段或补齐数据。

### C. 已核实共同协议分数的共同坐标复核

下列入口使用已检查的 CATCH-common 文件和契约，只读比较，不训练、不拟合。PSM 示例可按 `common_output_inventory.json` 替换为其它任务：

```bash
python scripts/evaluate_mnrf_view_probe_common.py \
  --probe result/score/MNRF-ViewProbe/PSM/full_gmm-full-20261024T115119Z-3976714/results/aligned_scores.npz \
  --catch result/score/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/PSM/CATCH-common/results/aligned_scores.npz \
  --labels result/analysis/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/PSM/common_contract/labels.npy \
  --train-indices result/analysis/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/PSM/common_contract/catch_train_indices.npy \
  --validation-indices result/analysis/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/PSM/common_contract/catch_validation_indices.npy \
  --calibration-indices result/analysis/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/PSM/common_contract/catch_calibration_indices.npy \
  --common-manifest result/score/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/PSM/CATCH-common/results/common_manifest.json \
  --output result/analysis/MNRF-Comparison-Adapter-20261102/PSM-common-evaluation.json
```

该入口要求实际坐标、时间、标签和边界严格绑定；不会通过取交集来隐藏缺点。当前 CATCH-common 文件已存在，但不能将其改称论文原生 CATCH 结果。

## 6. 八组适配结论

- **PSM**：可直接复算 MNRF valid-endpoint 排序和 q=.995 报警；与原表全 87841 点报警仍差前 191 点，需后续明确 warm-up 规则才可声称全轴对应。
- **MSL**：MNRF 官方 train/test 范围一致，内部 calibration 在官方 train 内；可复算 valid-endpoint 结果。27 序列/55 字段的原生边界语义若用于 native 对照，仍以原表方法自己的处理为准，不能由拼接 MNRF 轴替代。
- **GECCO**：官方测试轴与 MNRF 有效端点可严格配对（排除 95 个冷启动点）；排序与报警分列报告，不能把独立 calibration 当成论文报警协议。
- **Genesis**：有效覆盖 12425/12616；只有 50 个测试异常，报警指标应同时报告样本量和阈值来源，不能从小样本报警差异推断排序能力。
- **NYC**：有效覆盖 4225/4416；保留客流/日历字段和原实体/段边界。MNRF common 报警不能替代原表方法的 native 窗口阈值。
- **CalIt2**：有效覆盖 2329/2520，前缀排除 4 个异常；小校准集的 q=.995 结果是共同协议诊断，不能伪装成原表报警结果。
- **Creditcard**：有效覆盖 142213/142404；实体连续性证据和原生方法边界仍按 baseline 记录，不能通过行号补造 session 边界。
- **CICIDS**：有效覆盖 84925/85116，前缀排除 5 个异常；行邻接不自动等于网络 session 连续性，Affiliation 只能在已绑定边界上报告。

## 7. 完成边界

本方案不产生新的 MNRF 分数，不训练、不前向、不拟合、不重新校准 baseline。下一步若需要把 MNRF 放进原表的全轴报警列，最小新增工作是：每个任务固定并记录 cold-start/tail 策略，生成覆盖官方测试轴的 MNRF score artifact，再分别按原表排序指标和原生报警指标评估。该工作不能通过修改现有 `aligned_scores.npz` 完成。
