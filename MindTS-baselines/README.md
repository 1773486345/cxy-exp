# Baseline 实验运行说明

本目录是 baseline 实验专用目录：

```text
/media/h3c/users/wangyueyang1/cxy/MindTS-baselines
```

当前对齐的主模型在：

```text
/media/h3c/users/wangyueyang1/cxy/PatternAD-main
```

`MindTS-baselines/dataset` 与 `PatternAD-main/dataset` 使用同一份当前异常检测数据；
baseline 只修改本目录下的脚本、配置和结果，不修改主模型脚本。

## 1. 先进入目录

每次运行 baseline 前，先执行：

```bash
cd /media/h3c/users/wangyueyang1/cxy/MindTS-baselines
export PYTHON_BIN=/media/h3c/users/wangyueyang1/cxy/MindTS-baselines/scripts/baselines/run_baseline_python.sh
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

后两个变量很重要：代码会加载 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` tokenizer，设置离线模式可以直接用本机缓存，避免启动实验时访问 HuggingFace 失败。

## 2. 运行前检查

先复制执行这段检查：

```bash
test -x /media/h3c/users/wangyueyang1/cxy/MindTS-baselines/scripts/baselines/run_baseline_python.sh
test -x /media/h3c/users/wangyueyang1/cxy/.env/envs/omni_tf1/bin/python
test -e dataset/anomaly_detect/data/Genesis.csv
test -e config/unfixed_detect_label_multi_config.json
test "$(readlink -f /media/h3c/users/wangyueyang1/cxy/TAB/result)" = "/media/h3c/users/wangyueyang1/cxy/MindTS-baselines/result"
bash -n scripts/baselines/run_all_requested_baselines.sh
bash -n scripts/baselines/run_daphnet_gecco_tslib_baselines.sh
$PYTHON_BIN -c 'import torch, torch_geometric, transformers, numpy, pandas, sklearn'
$PYTHON_BIN - <<'PY'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
print("DeepSeek tokenizer offline cache OK")
PY
```

如果其中任何一步报错，先不要开长实验。尤其要注意 `TAB/result` 必须指向 `MindTS-baselines/result`，否则 TAB 相关 baseline 会把结果写到错误目录。

## 3. 一键运行

默认模式：已有结果会跳过，最后刷新汇总。

```bash
bash scripts/baselines/run_train_calibrated_patternad_baselines.sh
```

当前 PatternAD 数据集 baseline，结果统一写到本目录的 `result/label`：

```bash
cd /media/h3c/users/wangyueyang1/cxy/MindTS-baselines
bash scripts/baselines/run_train_calibrated_patternad_baselines.sh
```

该入口严格遵守 `train_calibration` 阈值协议，默认数据集与当前主模型一致：
`Genesis, Weather, Energy, SKAB, MSDS, Daphnet, GECCO, MetroPT3, PSM, BATADAL`。
`BATADAL` 会将 `BATADAL_dataset04.csv` 和 `BATADAL_test.csv` 作为两个独立序列，
在同一模型报告中聚合。它默认跳过 `AnomalyTransformer`、`OmniAnomaly`、
`InterFusion`：前者需独占 GPU，后两者使用 CPU-only TF1 环境。只跑部分数据集：

```bash
bash scripts/baselines/run_train_calibrated_patternad_baselines.sh PSM BATADAL
```

只跑部分模型可用逗号分隔的 `MODEL_FILTER`，例如：

```bash
MODEL_FILTER=PCA,IsolationForest \
  bash scripts/baselines/run_train_calibrated_patternad_baselines.sh PSM
```

汇总文件：

```text
result/label/strict_patternad_baselines_summary.csv
result/label/strict_patternad_baselines_three_metrics.csv
```

`run_all_requested_baselines.sh` 和下方以 `ExathlonSmall/Metro` 为参数的示例是旧的
RMindTS 数据集入口，仅用于复现历史结果；不要用于当前 PatternAD 的 10 数据集。
当前全量严格重跑使用：

```bash
SKIP_EXISTING=0 bash scripts/baselines/run_train_calibrated_patternad_baselines.sh
```

只重跑指定数据集：

```bash
SKIP_EXISTING=0 bash scripts/baselines/run_train_calibrated_patternad_baselines.sh PSM BATADAL
```

长实验建议放到 `tmux`：

```bash
tmux new-session -d -s baseline_rerun \
  "cd /media/h3c/users/wangyueyang1/cxy/MindTS-baselines && export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 && SKIP_EXISTING=0 bash scripts/baselines/run_train_calibrated_patternad_baselines.sh"
```

## 4. 重要：跳过逻辑

默认 `SKIP_EXISTING=1` 时，不同脚本的跳过规则不完全一样：

- PCA、IsolationForest、LOF、OCSVM、TranAD、AnomalyTransformer、USAD、GDN、OmniAnomaly、InterFusion、MTAD-GAT：只要目标目录中已有 `test_report.*.csv` 就跳过。
- DAGMM、DADA、UniTS、Timer、LMixer、DLinear、PatchTST、iTransformer、TimesNet：只有 report 同时包含 `affiliation_f`、`VUS_ROC`、`VUS_PR` 三项指标才跳过。

所以，如果你怀疑某个已有结果是旧的、空的或不完整的，请用：

```bash
SKIP_EXISTING=0 bash scripts/baselines/对应脚本.sh 数据集.csv
```

## 5. 分组运行

不想全跑时，可以按模型组运行：

| 脚本 | 模型 |
| --- | --- |
| `run_classic_baselines.sh` | PCA、IsolationForest、LOF、OCSVM |
| `run_self_impl_deep_baselines.sh` | TranAD、AnomalyTransformer、USAD |
| `run_gdn_baselines.sh` | GDN |
| `run_omni_anomaly_baselines.sh` | OmniAnomaly |
| `run_interfusion_baselines.sh` | InterFusion |
| `run_mtad_gat_baselines.sh` | MTAD-GAT |
| `run_tab_supported_baselines.sh` | DAGMM、DADA、UniTS、Timer、LMixer |
| `run_daphnet_gecco_tslib_baselines.sh` | DLinear、PatchTST、iTransformer、TimesNet |

示例：

```bash
SKIP_EXISTING=0 bash scripts/baselines/run_classic_baselines.sh Genesis.csv Metro.csv
SKIP_EXISTING=0 bash scripts/baselines/run_interfusion_baselines.sh Weather.csv
SKIP_EXISTING=0 bash scripts/baselines/run_daphnet_gecco_tslib_baselines.sh Genesis.csv ExathlonSmall.csv
```

## 6. GPU 和环境说明

- Baseline 环境包装器：`/media/h3c/users/wangyueyang1/cxy/MindTS-baselines/scripts/baselines/run_baseline_python.sh`
- Baseline 环境：`/media/h3c/users/wangyueyang1/cxy/.env/envs/baseline_env`
- PatternAD 主模型环境：`/media/h3c/users/wangyueyang1/.env/envs/patternad_env`
- 旧 TF 环境：`/media/h3c/users/wangyueyang1/cxy/.env/envs/omni_tf1`
- TSLib 四模型默认用 `GPU=0`，可这样改：

```bash
GPU=1 SKIP_EXISTING=0 bash scripts/baselines/run_daphnet_gecco_tslib_baselines.sh Weather.csv
```

- TAB 里的 DADA、UniTS、Timer、LMixer 默认使用 `--gpus 0`。
- InterFusion runner 会禁用 GPU。
- `scripts/install_third_party_tods.sh` 不是当前 20 个 baseline 的必跑步骤，正常重跑不用执行。

## 7. 结果在哪里

单个模型结果：

```text
result/label/baselines_<DATASET>_<BASELINE>/
result/label/<DATASET>_<TSLIB_MODEL>_baseline/
result/label/<DATASET>_TimesNet_baseline_h0/
```

汇总结果：

```text
result/label/_baseline_logs/requested_baseline_summary.csv
result/label/_baseline_logs/requested_baseline_three_metrics.csv
```

三指标含义：

```text
Aff-F = affiliation_f
V-PR  = VUS_PR
V-ROC = VUS_ROC
```

手动刷新汇总：

```bash
$PYTHON_BIN scripts/baselines/summarize_requested_baselines.py
```

## 8. 跑完后检查

运行结束后执行：

```bash
$PYTHON_BIN scripts/baselines/summarize_requested_baselines.py
wc -l result/label/_baseline_logs/requested_baseline_three_metrics.csv
rg ',,' result/label/_baseline_logs/requested_baseline_three_metrics.csv || true
rg '^MTAD-GAT,Daphnet|^InterFusion,MSDS|^OCSVM,(SKAB|GECCO)|^USAD,Energy|^DAGMM,Energy' \
  result/label/_baseline_logs/requested_baseline_three_metrics.csv
ps -eo pid,ppid,stat,etime,cmd | rg 'MindTS-baselines|run_benchmark.py|scripts/baselines|omni_anomaly_runner|interfusion_runner|mtad_gat_runner' || true
```

正常情况：

- `requested_baseline_three_metrics.csv` 应该是 181 行：1 行表头 + 20 个 baseline x 9 个数据集。
- `rg ',,'` 不应输出空字段行。
- `ps` 检查只看到当前检查命令，表示没有 baseline 进程残留。

## 9. 已知 NaN 不是缺失实验

以下结果来自正式运行，是真实失败项，不要改成 0，也不要删掉：

```text
OCSVM:       SKAB, GECCO
USAD:        Energy
DAGMM:       Energy
MTAD-GAT:    Daphnet
InterFusion: MSDS
```

论文表里可以写成 `Fail†` 或 `-†`。含义是：实验已执行，但由于数值不稳定或无效预测比例超过评估阈值，无法计算有效指标。

## 10. 清理缓存和过程日志

确认没有实验正在运行后，可以清理可重建文件：

```bash
find scripts ts_benchmark -type d -name '__pycache__' -prune -exec rm -rf {} +
find result/label/_baseline_logs -maxdepth 1 -type f -name '*.log' -delete
find result/label/_baseline_logs -mindepth 1 -maxdepth 1 -type d \
  \( -name 'interfusion_runs' -o -name 'mtad_gat_runs' -o -name 'omni_runs' \) \
  -exec rm -rf {} +
```

不要删除：

```text
result/label/**/test_report.*.csv
result/label/**/*.csv.tar.gz
result/label/_baseline_logs/requested_baseline_summary.csv
result/label/_baseline_logs/requested_baseline_three_metrics.csv
```
