# APD-CATCH 旧探索线状态

> 状态：**冻结（legacy exploratory line）**。本文件记录 APD-CATCH v1/v2 的范围、结论和保全位置；它们不是后续分解研究的活动主线。

## 保全规则

- 不删除、移动或覆盖本文件列出的代码、脚本、测试、结果、检查点、JSON/NPZ 诊断文件或历史文档。
- 不得将这些结果自动并入新的分解研究主表。它们使用的条件预测、可见性和阈值协议与原版 CATCH 重构协议不同。
- 仅在有专门研究问题明确指向“因果条件预测”时，才可重新启用此线；重新启用前须新建独立协议和结果目录。
- `result/` 中已有内容只读保全。本次归档不删除、不移动、不重算任何结果。

## v1.0：past-to-next-point 的频带候选

**范围。** v1.0 是对原版 CATCH 的探索性改写，而非原版 CATCH 复现。它以 `causal_catch`、`fixed`、`adaptive` 三个同参数预算候选比较：输入为过去窗口、目标为下一点；`fixed/adaptive` 使用低/高频完备分解，其中 `adaptive` 的截止频率由历史频谱产生。三个候选都以逐变量 Gaussian NLL 作为连续异常分数。

**已完成的工作。**

- 核心测试、论文数据展开、参数映射和 dry-run 已记录在 `CODE_MODIFICATION_LOG.md`；v1.0 当时记录为 8 项单元测试通过。
- 三个种子的合成机制门控已完成：`causal_catch=0.5165`、`fixed=0.5125`、`adaptive=0.5191` 的平均 AP。
- 单种子真实数据产物保存在 `result/paper_real_v1/workers/genesis/` 与 `result/paper_real_v1/workers/psm/`，每个数据集都有 `causal_catch`、`fixed`、`adaptive` 的 `seed_20261.json` 与 `seed_20261.npz`。

**失败结论。** `adaptive` 相对 `causal_catch` 的合成平均 AP 增益仅为 0.0026，且在 level 与 periodic 异常上更差。该门控不能支持把 adaptive frequency cutoff 作为真实数据贡献或后续默认方向；真实数据的早期结果也只能作为旧协议下的归档证据。

## v1.1：robust scale floor 的数值门控

**范围。** v1.1 在 v1.0 的 `causal_catch`、`fixed`、`adaptive` 框架上加入训练期固定的 robust scale floor，意图避免局部近常数窗口把正常状态变化放大为极端 Gaussian NLL。它没有改变 CATCH 主干、频带划分、标签协议或参数预算。

**Genesis 数值退化。** 文档记录的 Genesis 首轮中，训练期窗口 MAD 与全局 MAD 对多数变量均为零，scale floor 退化为 float32 epsilon；测试期首次变化仍产生接近零的 `prediction_scale`。最高正常点分数从约 `7.8e15` 恶化到 `3.8e22`，三个变体的分数和指标几乎相同。因此，表面排名变化不构成性能收益。

**为什么停止。** 训练期恒定、测试期进入新工况的变量没有由历史数据支持的本变量正常尺度。v1.1 未解决该可识别性问题，故停止 PSM/SWAT 扩展和多种子扩展，不将 Genesis 排名变化作为结论。

**现有保全结果。** `result/paper_real_v1_1_robust_scale/` 包含 `summary_runs.csv`、`summary_paper_comparison.csv`、`CalIt2/{causal_catch,fixed,adaptive}/` 与 `Genesis/{causal_catch,fixed,adaptive}/`。其中 CalIt2 保存三个变体的 `20261/20262/20263` JSON/NPZ；Genesis 保存的种子集合不完整。物理文件范围大于修改日志中“仅 Genesis seed 20261”的文字记录，均按现存归档保留，不据此扩张旧实验。

## v2.0：Causal-State-CATCH

**范围。** v2.0 的三个候选为 `causal_catch`、`state`、`state_scale`。`state` 与 `state_scale` 使用无参数的因果 EMA state，并由 CATCH 编码创新（innovation）；`state_scale` 额外使用 recent innovation scale。三个版本仍属于 past-to-next-point 条件预测与 Gaussian NLL 的探索，而不是原版 CATCH 重构。

**当前结果状态。** `CODE_MODIFICATION_LOG.md` 明确记录 v2 尚无真实数据结果；本次结果目录盘点也未发现 `result/causal_state_catch_v2*` 目录或 v2 的 JSON/NPZ 产物。原计划的 15-task screen 仅保留为旧探索复现实验，不应继续运行。

## 位置清单

### 现有 APD-CATCH 结果目录

```text
result/
├── paper_real_v1/
│   └── workers/{genesis,psm}/<dataset>/{causal_catch,fixed,adaptive}/
└── paper_real_v1_1_robust_scale/
    ├── summary_runs.csv
    ├── summary_paper_comparison.csv
    └── {CalIt2,Genesis}/{causal_catch,fixed,adaptive}/
```

### 旧线运行与汇总脚本

- `scripts/run_apd_catch_paper.py`
- `scripts/run_causal_state_catch_all.sh`
- `scripts/run_causal_state_catch_screen.sh`
- `scripts/run_causal_state_catch_variant.sh`
- `scripts/summarize_causal_state_catch.sh`
- `scripts/run_missing_original_catch.sh`（调用相邻 `../CATCH-master` 的原版命令；同样只作旧归档复现）
- `scripts/analysis/evaluate_apd_catch_mechanism.py`
- `scripts/analysis/summarize_apd_catch_results.py`
- `scripts/analysis/summarize_original_catch_results.py`
- `scripts/download_tab_datasets.sh`、`scripts/run_benchmark.py` 与 `scripts/install_third_party_tods.sh` 是该仓库现有的下载、基准运行和第三方安装支撑脚本；本阶段不修改其行为。

原版 CATCH 的逐数据集脚本仍位于 `scripts/multivariate_detection/detect_score/**/CATCH.sh` 与 `scripts/multivariate_detection/detect_label/**/CATCH.sh`；它们属于原版重构协议，不是 APD-CATCH 模型脚本。

### 旧线模型与测试

```text
ts_benchmark/baselines/apd_catch/
├── APDCATCH.py
├── models/APDCATCH_model.py
├── layers/{channel_mask.py,cross_channel_Transformer.py,RevIN.py}
└── utils/{ch_discover_loss.py,fre_rec_loss.py,tools.py}

tests/test_apd_catch_core.py
tests/test_paper_runner.py
```

与其并列、保留为原版 CATCH 基座的实现位于 `ts_benchmark/baselines/catch/`，包含 `CATCH.py`、`models/CATCH_model.py`、`layers/` 与 `utils/`。

## 重新启用门槛

重新启用仅限于一项专门的因果条件预测研究。该研究必须独立说明问题、可见性、数据切分、阈值和比较协议；不得将本清单中的历史分数改写为新分解主线的主表证据。
