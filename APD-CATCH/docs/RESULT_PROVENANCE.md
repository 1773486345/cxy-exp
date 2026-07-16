# 分解研究结果来源规则 v0

> 本规则适用于经人工确认后开展的检测导向固定分解研究。它不授权创建模型、运行训练或修改既有结果。

## 目录隔离

legacy APD-CATCH 结果与新研究结果必须使用完全独立的目录。新研究结果根目录预注册为：

```text
result/decomposition_study_v0/
```

以下路径是 legacy，永不纳入新研究主表、不得复制到新目录后改名为新结果：

```text
result/paper_real_v1*
result/causal_state_catch*
```

`CATCH-master/result/` 是原版 CATCH 的独立归档，只能作为原版协议参考，不能替代新研究的独立运行记录。

## 不可覆盖的运行布局

每次运行必须使用唯一 run ID，建议格式为：

```text
<UTC timestamp>_<dataset>_seed<seed>_<git-short-sha>_<nonce>
```

建议的单次运行布局为：

```text
result/decomposition_study_v0/
└── runs/<run_id>/
    ├── manifest.json
    ├── scores.npz
    ├── metrics.json
    └── checkpoint_reference.json
```

运行器必须在写入前检查目标目录不存在；若 run ID 已存在，立即失败，不覆盖、追加或复用既有文件。文件修改时间不能作为版本、配置或来源的替代信息。

## 必填来源记录

`manifest.json` 必须在运行开始时写入并在结束时补充结果索引，至少记录：

- Git commit 的完整 SHA；
- dirty working tree 状态：`git status --porcelain=v1` 原文及是否为空；
- 数据集、实际数据文件、协议名称与 seed；
- 完整 CATCH 配置及配置来源脚本；
- checkpoint 的路径、内容哈希、创建它的 Git commit（若可得）和是否为原版已有 checkpoint；
- 分解规则版本、移动平均边界规则和预注册窗口 `W`；
- `slow_score` 与 `fast_score` 标准化统计量的来源段、location、scale 与 `epsilon`；
- `original_score`、`slow_score`、`fast_score`、`fusion_score` 的文件位置、长度、dtype 和哈希；
- 评分窗口到原始时间轴的对齐规则，以及任何无法对齐时的停止原因。

`scores.npz` 必须保存四个连续分数和与其一一对齐的时间索引。若保存标签副本，它只能用于最终指标复算，不能作为分解、窗口、标准化或融合参数的输入。

## 标签与选择隔离

测试标签只能用于最终指标计算。特别禁止使用测试标签、测试异常比例或测试分数总体统计量来选择：

- 移动平均窗口；
- 分解规则；
- 融合权重；
- 分数标准化参数；
- checkpoint、seed、数据集或要报告的运行。

所有移动平均窗口、等权融合和标准化来源必须先由 `DECOMPOSITION_STUDY_PROTOCOL.md` 固定。若需要偏离，停止该运行并经人工审核创建新协议版本，而不是修改现有 run 的 metadata。

## 主表登记规则

新的主表只能引用 `result/decomposition_study_v0/runs/<run_id>/` 中有完整 manifest 的运行。每个主表单元格必须能回溯到 run ID、完整 Git SHA、checkpoint 哈希与四个连续分数文件。

legacy 目录、文件修改时间、手工摘录的旧 CSV、未记录 checkpoint 的结果和缺少 dirty-tree 状态的运行不得进入新主表。
