# MSD-CATCH / BHD-MSD-CATCH

本项目在原版 CATCH 的基础上，研究面向异常检测的结构分解与联合建模。当前代码和结果围绕 CATCH、MSD-CATCH 与 BHD-MSD-CATCH 三个模型组织。

## 当前模型

- **CATCH**：固定基线；已有结果只读复用，不作为当前改造对象。
- **MSD-CATCH**：当前主性能模型，使用独立分量建模。
- **BHD-MSD-CATCH**：当前轻量共享编码版本。

RSA、SDD 与旧 APD-CATCH 均为历史研究路线，不进入当前全数据集主实验。

## 正式评价

当前主实验仅使用：

- `detect_score`
- `total_score`
- AUC-PR 与 AUC-ROC

## 正式真实数据集范围与 ASD 汇总

ASD 必须展开为 12 个独立训练和评价任务：

```text
ASD_dataset_1–ASD_dataset_12
```

每个子数据集分别保留 CATCH、MSD-CATCH 与 BHD-MSD-CATCH 的原始 AUC-PR 和 AUC-ROC。当前每个 ASD 子数据集均有 MSD-CATCH 与 BHD-MSD-CATCH 的 `detect_score` 报告，已有结果只读复用。

完整结果或附录必须保留 12 个 ASD 子数据集的逐项行。论文主表将 ASD 视为 1 个真实数据集，只显示一行 `ASD`：对每个模型，`ASD` 的 AUC-PR 和 AUC-ROC 分别是 12 个子数据集对应指标的等权算术平均。不得拼接 score/label 后重新计算 pooled/micro AUC，也不得按子数据集长度、异常点数量或窗口数量加权。

因此，论文真实数据集总数统计中 ASD 计为 1；训练、评价与任务完成度统计中必须说明 ASD 包含 12 个独立序列任务。

## 运行单个数据集

例如运行 MSL 的 `detect_score`：

```bash
sh ./scripts/multivariate_detection/detect_score/MSL_script/MSDCATCH.sh
sh ./scripts/multivariate_detection/detect_score/MSL_script/BHDMSDCATCH.sh
```

## 结果位置

```text
result/score/by_dataset/<DATASET>/<MODEL>/
result/msd_catch_total_screen/
result/bhd_msd_catch_screen/
```

## 相关文档与测试

- [方向 A 研究主文档](../tsad_direction_A_pattern_aware_scoring.md)
- [APD-CATCH 旧路线状态](./docs/legacy/APD_CATCH_LEGACY_STATUS.md)
- [MSD-CATCH smoke test](./tests/test_msd_catch_smoke.py)
- [BHD-MSD-CATCH smoke test](./tests/test_bhd_msd_catch_smoke.py)
