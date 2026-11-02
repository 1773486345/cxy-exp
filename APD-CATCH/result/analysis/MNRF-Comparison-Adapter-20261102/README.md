# MNRF Comparison Adapter

这是只读评价适配入口的实际盘点结果。`common_output_inventory.json` 已按 `task_status.json` 记录检查八组 `CATCH-common`、`IsolationForest`、`PCA` 输出：24/24 个 `results/aligned_scores.npz` 存在、有限、严格执行 `score > threshold`，其分数摘要、`common_manifest.json`/`evidence/baseline.json` 和 `common_contract.json` 均已核对。检查的共同协议根目录为：

```text
result/score/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/
result/analysis/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/
```

`mnrf_evaluation_plan.json` 固化了八个注册 Full 源、有效端点、冷启动前缀、段边界检查和需要的训练侧文件。已保存端点只在其自身冻结 segment 内使用历史；没有显式实体边界字段的任务不会被解释为已经证明了跨行 session 连续性。MSL 明确标为 `blocked-known-cross-channel`：不能借 T-9 的训练尾部给 C-1 的测试开头补历史。Creditcard/CICIDS 标为 `blocked-entity-session-unproven`，因为当前证据没有权威实体/session 边界。PSM、GECCO、Genesis、NYC、CalIt2 的计划只在全局坐标连续、时间单调且窗口不跨冻结 segment 时标为 `ready-frozen-forward`，这仍是窗口级证据，不是跨实体连续性的证明。冷启动前缀没有保存特征，计划状态只是后续人工前向的准入结果，当前没有补造分数。所有八组均无尾部未评分点。

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

# 八组四方法的只读严格 common-protocol 对照
python scripts/prepare_mnrf_comparison_adapter.py \
  --mode common-compare \
  --output-dir result/analysis/MNRF-Comparison-Adapter-20261102
```

`common_comparison.csv` 和 `common_comparison.json` 已生成，包含八组四种实现：
`MNRF-Full-GMM`、`CATCH-common`、`IsolationForest`、`PCA`。MNRF/CATCH 通过既有
严格评价器重新核对冻结源和缓存分数；IF/PCA 通过各自已验证的 evidence 与同一
`common_contract` 坐标、时间和标签只读核对。这是 common-protocol 结果，不替代
论文 16 方法原生成绩。`coverage` 是共同有效评分点占完整测试轴的比例；冷启动
前缀没有补零、插值或伪造分数。

| 数据集 | 共同有效点 | 完整测试轴 | 覆盖率 |
|---|---:|---:|---:|
| PSM | 87650 | 220322 | 0.397827 |
| MSL | 73538 | 132046 | 0.556912 |
| GECCO | 69166 | 138521 | 0.499318 |
| Genesis | 12425 | 16220 | 0.766030 |
| NYC | 4225 | 17520 | 0.241153 |
| CalIt2 | 2329 | 5040 | 0.462103 |
| Creditcard | 142213 | 284807 | 0.499331 |
| CICIDS | 84925 | 170231 | 0.498881 |

注册源与窗口配置来自实际 `task_status.json`/`probe_manifest.json`，没有按最新时间或成绩替换：

| 数据集 | source Run ID | seq_len | 前缀规则 |
|---|---|---:|---|
| PSM | `full_gmm-full-20261024T115119Z-3976714` | 192 | 同 frozen segment；191 点待补算 |
| MSL | `full_gmm-full-20261021T112147Z-2470335` | 192 | `blocked-known-cross-channel`；191 点禁止借 T-9 历史补算 |
| GECCO | `full_gmm-full-20261020T204951Z-1642355` | 96 | 同 frozen segment；95 点待补算 |
| Genesis | `full_gmm-full-20261020T205005Z-1645522` | 192 | 同 frozen segment；191 点待补算 |
| NYC | `full_gmm-full-20261020T211341Z-1931786` | 192 | 同 frozen segment；191 点待补算 |
| CalIt2 | `full_gmm-full-20261021T112212Z-2474004` | 192 | 同 frozen segment；191 点待补算 |
| Creditcard | `full_gmm-full-20261024T115131Z-3977864` | 192 | `blocked-entity-session-unproven`；191 点待边界证据 |
| CICIDS | `full_gmm-full-20261024T115157Z-3980319` | 192 | `blocked-entity-session-unproven`；191 点待边界证据 |

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

## 冷启动补算入口（本轮未执行）

`score-cold-start` 是后续人工授权后才运行的真实前缀补算入口。它加载已验证的
冻结 encoder checkpoint、`input_statistics.npz`、GMM、特征统计和固定噪声，在同一
frozen segment 内按当前/历史窗口前向生成缺失前缀特征，再复用原阈值评分。入口会
先在内存中复现全部已保存有效端点；任何端点不一致即停止。它不训练、不拟合 GMM、
不重新校准，也不改写 Full 源。当前计划中的缺失前缀为 PSM/MSL/Genesis/NYC/
CalIt2/Creditcard/CICIDS 各 191 点，GECCO 95 点。

示例（一次只处理一个数据集，输出必须是新目录）：

```bash
python scripts/prepare_mnrf_comparison_adapter.py --mode score-cold-start \
  --task PSM \
  --source-dir result/score/MNRF-ViewProbe/PSM/full_gmm-full-20261024T115119Z-3976714 \
  --output-dir result/score/MNRF-Comparison-Adapter/PSM/cold-start-<RUN_ID>
```

其余任务只替换 `--task` 与计划中对应固定源路径；MSL、Creditcard、CICIDS 当前会在准入阶段明确拒绝，而不是产生伪造分数。没有显式实体列的任务只能按已保存 segment 身份放行，不能解释为已经证明跨 session 连续；越过边界的窗口会被拒绝。冷启动输出采用原生结构：先填 `aligned_scores/aligned_predictions/aligned_actual/valid_mask` 全测试轴，再由 mask 派生紧凑 `scores/predictions/time_indices/time_values_ns/local_indices/actual`，因此不会覆盖紧凑数组前部冒充前缀。评价覆盖率分母始终是官方完整测试点数。NYC/CalIt2 的 PCA 与 PCA-residual 产物是不同只读产品，本适配器不会互相替换或重跑它们。
