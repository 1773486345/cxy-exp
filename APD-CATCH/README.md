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
