# APD-CATCH 仓库审计

审计时间：2026-07-17 07:18 CST（Git 快照）；进程复查：07:20 CST。审计范围为 `cxy/APD-CATCH` 与相邻的 `cxy/CATCH-master`。本次只新增或修改 Markdown 文档；未运行训练、未删除或移动任何结果。

## 安全状态

`cxy/` 是 Git 仓库；工作区根目录 `/media/h3c/users/wangyueyang1` 本身不是 Git 仓库。以下为在 `cxy/` 执行的记录：

```text
$ git status --short
?? CATCH-master/result/label/CATCH/PSM/run-20260716T223734Z-776688-6344/CATCH.1784243251.h3c-R5500-G5.777325.csv.tar.gz
?? CATCH-master/result/label/CATCH/PSM/run-20260716T223734Z-776688-6344/test_report.1784243251.h3c-R5500-G5.777325.csv
?? CATCH-master/result/score/CATCH/SMD/

$ git branch --show-current
main

$ git log -1 --oneline
ed5bb2d 模型v2
```

没有已暂存或已修改的受跟踪文件；存在上述 3 项未跟踪状态项。`git ls-files --others --exclude-standard` 展开的未跟踪文件为：

```text
CATCH-master/result/label/CATCH/PSM/run-20260716T223734Z-776688-6344/CATCH.1784243251.h3c-R5500-G5.777325.csv.tar.gz
CATCH-master/result/label/CATCH/PSM/run-20260716T223734Z-776688-6344/test_report.1784243251.h3c-R5500-G5.777325.csv
CATCH-master/result/score/CATCH/SMD/run-20260716T225633Z-1008371-29270/command.sh
CATCH-master/result/score/CATCH/SMD/run-20260716T225633Z-1008371-29270/environment.txt
CATCH-master/result/score/CATCH/SMD/run-20260716T225633Z-1008371-29270/metadata.txt
CATCH-master/result/score/CATCH/SMD/run-20260716T225633Z-1008371-29270/official_CATCH.sh
CATCH-master/result/score/CATCH/SMD/run-20260716T225633Z-1008371-29270/official_CATCH.sh.sha256
```

修改摘要：这些都是 `CATCH-master/result/` 下的原版 CATCH 结果或运行溯源产物，没有发现受跟踪源码、脚本、配置、检查点或 Markdown 的预先修改。本次没有对这些未跟踪路径执行 reset、checkout、clean、stash、覆盖、删除或移动操作。

## 运行中的实验

按脚本名检查，以下任务均**未运行**：

- `run_causal_state_catch_screen.sh`
- `run_causal_state_catch_variant.sh`
- `run_missing_original_catch.sh`

GPU 0 同时存在活跃计算进程，不能声明机器处于无实验状态。其中与 CATCH 直接相关、且未被本次操作改变的进程为：

| PID | 当前命令含义 | 状态 |
| ---: | --- | --- |
| 59126 | 原版 `catch.CATCH` 的 MSL score 运行 | 运行中 |
| 1011549 | 原版 `catch.CATCH` 的 SMD score 运行 | 运行中 |
| 1989089 | 原版 `catch.CATCH` 的 SWAT label 运行 | 运行中 |

其余 GPU 进程是 TimeXer 或其他用户路径的任务，未匹配 APD-CATCH。未对任何 PID 执行 kill 或其他进程控制操作。

## 当前目录结构

```text
cxy/
├── CATCH-master/                     原版 CATCH 本地副本与原版运行结果
│   ├── ts_benchmark/baselines/catch/
│   ├── scripts/multivariate_detection/
│   └── result/{score,label}/
├── APD-CATCH/                        本审计对象；含原版基座与 APD 旧线
│   ├── ts_benchmark/baselines/{catch,apd_catch}/
│   ├── scripts/
│   ├── config/
│   ├── dataset/
│   ├── result/
│   ├── tests/
│   ├── README.md
│   ├── CODE_MODIFICATION_LOG.md
│   └── docs/
└── tsad_direction_A_pattern_aware_scoring.md
```

APD-CATCH 当前已有结果目录为：

```text
APD-CATCH/result/paper_real_v1/workers/genesis/
APD-CATCH/result/paper_real_v1/workers/psm/
APD-CATCH/result/paper_real_v1_1_robust_scale/CalIt2/{adaptive,fixed,causal_catch}/
APD-CATCH/result/paper_real_v1_1_robust_scale/Genesis/{adaptive,fixed,causal_catch}/
```

## 原版 CATCH 与 APD-CATCH 的文件边界

| 范围 | 位置 | 审计结论 |
| --- | --- | --- |
| 原版 CATCH 本地副本 | `../CATCH-master/` | 原版命令、配置和原版结果的独立归档；当前存在运行中的原版任务。 |
| APD-CATCH 内的原版基座 | `ts_benchmark/baselines/catch/` | 应保留为新阶段的完整窗口重构基座。 |
| APD 条件预测实现 | `ts_benchmark/baselines/apd_catch/` | v1/v2 legacy exploratory line；冻结。 |
| 原版运行入口 | `scripts/multivariate_detection/detect_{score,label}/**/CATCH.sh`、`scripts/run_benchmark.py` | 对应原版 `catch.CATCH`；只可作为原版协议复现或基座参考。 |
| APD 运行入口 | `scripts/run_apd_catch_paper.py`、`scripts/run_causal_state_catch_*.sh` | 旧探索线；冻结。 |

## 可直接复用与必须冻结的内容

可直接复用的基础设施限于原版 CATCH 重构路径：

- `ts_benchmark/baselines/catch/` 的 CATCH 适配器、模型、层和损失工具；
- `scripts/multivariate_detection/` 中的原版 CATCH 数据集命令与 `config/` 中对应的基准配置；
- `ts_benchmark/data/`、`common/`、`evaluation/`、`report/` 的数据和评价基础设施；
- `CATCH-master/result/` 的已有原版结果，仅作协议已知的参考，不能覆盖或替代新阶段的独立结果记录。

必须冻结的内容：

- `ts_benchmark/baselines/apd_catch/` 的全部实现；
- `scripts/run_apd_catch_paper.py`、`scripts/run_causal_state_catch_all.sh`、`scripts/run_causal_state_catch_screen.sh`、`scripts/run_causal_state_catch_variant.sh`、`scripts/summarize_causal_state_catch.sh`；
- `scripts/analysis/evaluate_apd_catch_mechanism.py` 与 APD 汇总逻辑；
- `result/paper_real_v1/`、`result/paper_real_v1_1_robust_scale/` 及其 JSON、NPZ、CSV；
- `tests/test_apd_catch_core.py`、`tests/test_paper_runner.py` 和 `CODE_MODIFICATION_LOG.md`，作为旧线的可审计快照。

## README、修改日志与实际内容的不一致

1. README 原先将 v2.0 表述为活动方向，并提供 15-task screen 的并行命令；`CODE_MODIFICATION_LOG.md` 的 “Design Status (2026-07-17)” 也称其为 active Direction A base。该状态与本次复位不一致。README 顶部已加入冻结说明；修改日志保留为历史记录，不改写其当时陈述。
2. `CODE_MODIFICATION_LOG.md` 的 v1.1 命令使用 `fixed/adaptive`，但当前 `scripts/run_apd_catch_paper.py` 的 `VARIANTS` 仅为 `causal_catch`、`state`、`state_scale`。因此该历史 v1/v1.1 命令不能由当前脚本直接复现，必须把它视为与其结果一起保存的历史版本记录。
3. v1.1 日志文字称“仅完成 Genesis、seed=20261”，而实际 `result/paper_real_v1_1_robust_scale/` 还保存了 CalIt2 三变体的 20261/20262/20263 产物，并保存了部分 Genesis 追加种子。目录事实与文本记录范围不一致；两者都保留，不从目录内容推导新的性能结论。
4. README 标题使用 “Causal-State-CATCH v2.0”，而模型包与结果历史仍统称 `APD-CATCH`，且结果目录没有 v2 产物。这不是需要重命名的理由，但在引用时必须同时说明版本、协议和结果目录。

## 测试覆盖与缺口

现有测试仅面向 APD 旧线，且本次没有运行测试：

- `tests/test_apd_catch_core.py` 覆盖历史/目标分离、三个 v2 变体参数量一致、正尺度与有限 Gaussian NLL、`state_scale` 推理确定性、常量历史下的参考归一化。
- `tests/test_paper_runner.py` 覆盖 23 个论文源文件展开、原版参数映射时排除 `score_lambda/anomaly_ratio`、三变体参数量一致、长表数据读取和 ASD 聚合。

未覆盖的关键部分包括：原版 `catch.CATCH` 的完整窗口重构契约；真实数据端到端训练/评分；原版与新研究协议的一致性；历史结果完整性和可复现性；结果目录不被覆盖的保护；以及任何未来分解的分量异常保持、专门性、互补性和相对总残差的比较。测试未运行是因为本阶段为文档归档，且禁止启动训练或改变模型行为。

## 新研究线的最少文件

在人工确认后，最少应先补齐研究协议和验证契约，而不是新增模型模块：

1. `docs/DECOMPOSITION_STUDY_PROTOCOL.md`：锁定原版 CATCH 重构基座、数据切分、评价和“一次只改变一个变量”的实验表。
2. `tests/test_catch_reconstruction_contract.py`：验证原版完整窗口重构路径和评分接口在对照中未被改变。
3. `docs/RESULT_PROVENANCE.md`：定义新主表与 legacy 结果的隔离、运行标识和结果登记规则。

这些文件尚未创建；任何实现或新结果目录均等待人工确认。
