# MNRF Comparison Adapter

这是只读评价适配入口的实际盘点结果。`common_output_inventory.json` 已按 `task_status.json` 记录检查八组 `CATCH-common`、`IsolationForest`、`PCA` 输出：24/24 个 `results/aligned_scores.npz` 存在、有限、严格执行 `score > threshold`，其分数摘要、`common_manifest.json`/`evidence/baseline.json` 和 `common_contract.json` 均已核对。检查的共同协议根目录为：

```text
result/score/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/
result/analysis/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/
```

`mnrf_evaluation_plan.json` 固化了八个注册 Full 源、有效端点、冷启动前缀、段边界检查和需要的训练侧文件。当前所有源均只允许在已保存有效端点内使用同一冻结 segment 的历史；没有显式实体边界字段的任务不会被解释为已经证明了跨行 session 连续性。冷启动前缀没有保存特征，状态为 `blocked-no-saved-features`，没有补造分数。所有八组均无尾部未评分点。

## 入口

```bash
# 只读盘点共同协议实际输出及契约
python scripts/prepare_mnrf_comparison_adapter.py \
  --mode inventory \
  --output-dir result/analysis/MNRF-Comparison-Adapter-20261102

# 只读生成八组 MNRF 评价轴计划
python scripts/prepare_mnrf_comparison_adapter.py \
  --mode plan \
  --output-dir result/analysis/MNRF-Comparison-Adapter-20261102

# 合成坐标/段边界/覆盖与输出格式检查
python scripts/prepare_mnrf_comparison_adapter.py --mode synthetic
```

以下是后续经人工授权后才执行的冻结评分命令。它们只加载指定 Full 的已保存 GMM、标准化统计、固定噪声和 test feature cache；不拟合、不重新校准，并输出到新目录。当前本轮未执行：

```bash
python scripts/prepare_mnrf_comparison_adapter.py --mode score --task PSM --source-dir result/score/MNRF-ViewProbe/PSM/full_gmm-full-20261024T115119Z-3976714 --output-dir result/score/MNRF-Comparison-Adapter/PSM/valid-endpoint
python scripts/prepare_mnrf_comparison_adapter.py --mode score --task MSL --source-dir result/score/MNRF-ViewProbe/MSL/full_gmm-full-20261021T112147Z-2470335 --output-dir result/score/MNRF-Comparison-Adapter/MSL/valid-endpoint
python scripts/prepare_mnrf_comparison_adapter.py --mode score --task GECCO --source-dir result/score/MNRF-ViewProbe/GECCO/full_gmm-full-20261020T204951Z-1642355 --output-dir result/score/MNRF-Comparison-Adapter/GECCO/valid-endpoint
python scripts/prepare_mnrf_comparison_adapter.py --mode score --task Genesis --source-dir result/score/MNRF-ViewProbe/Genesis/full_gmm-full-20261020T205005Z-1645522 --output-dir result/score/MNRF-Comparison-Adapter/Genesis/valid-endpoint
python scripts/prepare_mnrf_comparison_adapter.py --mode score --task NYC --source-dir result/score/MNRF-ViewProbe/NYC/full_gmm-full-20261020T211341Z-1931786 --output-dir result/score/MNRF-Comparison-Adapter/NYC/valid-endpoint
python scripts/prepare_mnrf_comparison_adapter.py --mode score --task CalIt2 --source-dir result/score/MNRF-ViewProbe/CalIt2/full_gmm-full-20261021T112212Z-2474004 --output-dir result/score/MNRF-Comparison-Adapter/CalIt2/valid-endpoint
python scripts/prepare_mnrf_comparison_adapter.py --mode score --task Creditcard --source-dir result/score/MNRF-ViewProbe/Creditcard/full_gmm-full-20261024T115131Z-3977864 --output-dir result/score/MNRF-Comparison-Adapter/Creditcard/valid-endpoint
python scripts/prepare_mnrf_comparison_adapter.py --mode score --task CICIDS --source-dir result/score/MNRF-ViewProbe/CICIDS/full_gmm-full-20261024T115157Z-3980319 --output-dir result/score/MNRF-Comparison-Adapter/CICIDS/valid-endpoint
```

The `score` mode stops if the frozen-cache score differs from the source's existing score at any endpoint. It never converts the valid-endpoint bundle into an official full-test-axis result. A future full-axis run would require a separately approved warm-up policy and newly encoded prefix features.
