# 八数据集共同协议基线对照

批次：`common-baseline-20261029T-newrun`。共同协议使用各固定 Full 源绑定的数据文件、变量顺序、原始训练行、冻结 train/validation/calibration/test 端点；测试标签只用于指标。
CATCH 每组使用 score 归档中的任务参数、seed=2021，按冻结训练范围拟合、冻结 validation 早停、calibration 的 0.995 分位数阈值。原 benchmark 的 score/label 归档来自不同运行，分开列示，不拼成一行。
Isolation Forest 和 PCA 使用相同原始训练行拟合 StandardScaler 与模型，并在共同 calibration/test 端点评分。它们是逐点统计基线，不使用窗口上下文；与 CATCH/MNRF 的过去窗口上下文不同。三者都使用 score > 各自阈值。Aff-F 此处为 `affiliation_f_segment_macro`，只在有效连续评分段计算，不是论文原始协议结果。

## 共同协议结果

| 数据集 | 方法 | ROC-AUC | AP | 点 Precision | Recall | F1 | 正常点 FPR | TP/FP/FN/TN | Aff-F（有效段宏平均） | 命中/有分数/异常段 | 阈值 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PSM | MNRF Full | 0.744926 | 0.437281 | 0.459202 | 0.602757 | 0.521276 | 0.273502 | 14694/17305/9684/45967 | 0.736402 | 48/71/72 | 15.702126 |
| PSM | CATCH-common | 0.634849 | 0.424089 | 0.627171 | 0.025187 | 0.048428 | 0.005769 | 614/365/23764/62907 | 0.746764 | 40/71/72 | 2.075911 |
| PSM | IsolationForest | 0.683900 | 0.439688 | 0.974596 | 0.017311 | 0.034017 | 0.000174 | 422/11/23956/63261 | 0.152837 | 6/71/72 | 0.638228 |
| PSM | PCA | 0.792576 | 0.534842 | 0.729105 | 0.181065 | 0.290089 | 0.025920 | 4414/1640/19964/61632 | 0.423267 | 17/71/72 | 1.483011 |
| MSL | MNRF Full | 0.602445 | 0.159006 | 0.290260 | 0.077517 | 0.122358 | 0.022380 | 602/1472/7164/64300 | 0.541380 | 13/36/36 | 2201.669312 |
| MSL | CATCH-common | 0.655305 | 0.169585 | 0.274457 | 0.065027 | 0.105143 | 0.020297 | 505/1335/7261/64437 | 0.593781 | 16/36/36 | 0.169921 |
| MSL | IsolationForest | 0.604007 | 0.142277 | 0.542056 | 0.007468 | 0.014734 | 0.000745 | 58/49/7708/65723 | 0.416186 | 9/36/36 | 0.533007 |
| MSL | PCA | 0.505271 | 0.144016 | 0.176966 | 0.135334 | 0.153375 | 0.074317 | 1051/4888/6715/60884 | 0.711322 | 29/36/36 | 0.047276 |
| GECCO | MNRF Full | 0.442452 | 0.223303 | 0.726562 | 0.254795 | 0.377282 | 0.001023 | 186/70/544/68366 | 0.288205 | 3/22/22 | 247.484548 |
| GECCO | CATCH-common | 0.971360 | 0.467191 | 0.602694 | 0.245205 | 0.348588 | 0.001724 | 179/118/551/68318 | 0.418016 | 3/22/22 | 5.041964 |
| GECCO | IsolationForest | 0.956131 | 0.439830 | 0.249141 | 0.595890 | 0.351373 | 0.019157 | 435/1311/295/67125 | 0.799145 | 17/22/22 | 0.604275 |
| GECCO | PCA | 0.679565 | 0.361844 | 0.972826 | 0.245205 | 0.391685 | 0.000073 | 179/5/551/68431 | 0.220123 | 2/22/22 | 0.890972 |
| Genesis | MNRF Full | 0.899568 | 0.046805 | 0.057789 | 0.460000 | 0.102679 | 0.030303 | 23/375/27/12000 | 0.854378 | 3/3/3 | 45.299886 |
| Genesis | CATCH-common | 0.758450 | 0.014433 | 0.020548 | 0.180000 | 0.036885 | 0.034667 | 9/429/41/11946 | 0.822038 | 2/3/3 | 0.597480 |
| Genesis | IsolationForest | 0.650124 | 0.005486 | 0.002135 | 0.140000 | 0.004207 | 0.264323 | 7/3271/43/9104 | 0.610478 | 2/3/3 | 0.502828 |
| Genesis | PCA | 0.987562 | 0.515525 | 0.053145 | 0.980000 | 0.100823 | 0.070545 | 49/873/1/11502 | 0.842081 | 3/3/3 | 0.215497 |
| NYC | MNRF Full | 0.691339 | 0.044089 | 0.000000 | 0.000000 | 0.000000 | 0.001697 | 0/7/99/4119 | 0.670732 | 0/1/1 | 5.727228 |
| NYC | CATCH-common | 0.684937 | 0.049959 | 0.125000 | 0.020202 | 0.034783 | 0.003393 | 2/14/97/4112 | 0.672935 | 1/1/1 | 1.012044 |
| NYC | IsolationForest | 0.665662 | 0.037823 | 0.000000 | 0.000000 | 0.000000 | 0.001939 | 0/8/99/4118 | 0.755788 | 0/1/1 | 0.637791 |
| NYC | PCA | 0.517170 | 0.026369 | 0.000000 | 0.000000 | 0.000000 | 0.005332 | 0/22/99/4104 | 0.693801 | 0/1/1 | 0.000000 |
| CalIt2 | MNRF Full | 0.713040 | 0.053971 | 0.030612 | 0.042857 | 0.035714 | 0.042054 | 3/95/67/2164 | 0.613976 | 2/12/13 | 4.047460 |
| CalIt2 | CATCH-common | 0.835812 | 0.106986 | 0.112805 | 0.528571 | 0.185930 | 0.128818 | 37/291/33/1968 | 0.776543 | 11/12/13 | 3.412226 |
| CalIt2 | IsolationForest | 0.800190 | 0.101147 | 0.081136 | 0.571429 | 0.142096 | 0.200531 | 40/453/30/1806 | 0.772268 | 11/12/13 | 0.595362 |
| CalIt2 | PCA | 0.732786 | 0.080117 | 0.096386 | 0.342857 | 0.150470 | 0.099602 | 24/225/46/2034 | 0.737714 | 9/12/13 | 0.000000 |
| Creditcard | MNRF Full | 0.920188 | 0.051007 | 0.035202 | 0.244344 | 0.061538 | 0.010423 | 54/1480/167/140512 | 0.563857 | 39/202/204 | 21.887294 |
| Creditcard | CATCH-common | 0.959316 | 0.091205 | 0.086673 | 0.420814 | 0.143740 | 0.006902 | 93/980/128/141012 | 0.681985 | 75/202/204 | 0.483156 |
| Creditcard | IsolationForest | 0.955052 | 0.096501 | 0.081425 | 0.434389 | 0.137143 | 0.007627 | 96/1083/125/140909 | 0.677707 | 78/202/204 | 0.595648 |
| Creditcard | PCA | 0.800732 | 0.045347 | 0.076705 | 0.244344 | 0.116757 | 0.004578 | 54/650/167/141342 | 0.609035 | 53/202/204 | 0.967395 |
| CICIDS | MNRF Full | 0.538825 | 0.000917 | 0.000000 | 0.000000 | 0.000000 | 0.016629 | 0/1411/74/83440 | 0.125632 | 0/65/70 | 74.099782 |
| CICIDS | CATCH-common | 0.693761 | 0.001360 | 0.000000 | 0.000000 | 0.000000 | 0.007507 | 0/637/74/84214 | 0.249429 | 0/65/70 | 0.179449 |
| CICIDS | IsolationForest | 0.674689 | 0.001295 | 0.000000 | 0.000000 | 0.000000 | 0.004042 | 0/343/74/84508 | 0.214418 | 0/65/70 | 0.651881 |
| CICIDS | PCA | 0.573874 | 0.001066 | 0.000000 | 0.000000 | 0.000000 | 0.004514 | 0/383/74/84468 | 0.202970 | 0/65/70 | 0.520600 |

## 原始 CATCH 归档参照

这些是 `CATCH-master` 原生 `unfixed_detect_score` 与 `unfixed_detect_label` 的独立归档运行。score 行只含 ROC-AUC/AP；label 行只含 Affiliation P/R/F。它们的运行 ID、脚本和压缩 CSV 哈希见 `archive_references.csv`；当前未发现归档 checkpoint，因此本轮按 score 归档参数补跑共同协议 CATCH。原归档分数使用原任务协议阈值/评估过程，不等同于下表独立 calibration 0.995。

| 数据集 | 历史 score ROC-AUC | 历史 score AP | score run | 历史 label Aff-P | Aff-R | Aff-F | label run |
|---|---:|---:|---|---:|---:|---:|---|
| PSM | 0.647548 | 0.436646 | `run-20260716T222700Z-660928-22488` | 0.807580 | 0.917612 | 0.859087 | `run-20260716T223734Z-776688-6344` |
| MSL | 0.661964 | 0.165589 | `run-20260716T213635Z-58595-28964` | 0.599308 | 0.966247 | 0.739775 | `run-20260716T232646Z-1365820-1457` |
| GECCO | 0.969981 | 0.417423 | `run-20260716T213302Z-16971-16047` | 0.832445 | 0.998365 | 0.907887 | `run-20260716T213559Z-51278-24592` |
| Genesis | 0.973541 | 0.265986 | `run-20260716T213208Z-6088-16524` | 0.834907 | 0.965638 | 0.895526 | `run-20260716T221132Z-484571-23537` |
| NYC | 0.818435 | 0.080123 | `run-20260716T222643Z-657156-32728` | 1.000000 | 0.988646 | 0.994290 | `run-20260716T223043Z-701607-26288` |
| CalIt2 | 0.837335 | 0.113310 | `run-20260715T183704Z-2374669-5874` | 0.741919 | 0.954760 | 0.834990 | `run-20260715T184135Z-2437260-7989` |
| Creditcard | 0.957814 | 0.100479 | `run-20260716T202925Z-3426561-275` | 0.617548 | 0.955814 | 0.750319 | `run-20260717T085033Z-4154569-18693` |
| CICIDS | 0.790505 | 0.002173 | `run-20260715T120722Z-1345491-4045` | 0.667452 | 0.959336 | 0.787209 | `run-20260715T133707Z-2321802-10349` |

## 结果读取与边界

共同协议使用相同测试端点和完整标签轴；没有按标签清洗训练数据，也没有内部再次划分训练行。Validation 仅用于 CATCH 早停，calibration 仅用于各自 0.995 阈值。CalIt2/Genesis 校准端点较少，分位阈值方差较高，仍按统一规则报告，不作特例调整。

不能把历史 score 和 label 两个独立运行的结果合成一个历史 CATCH 运行。`archive_references.csv` 同时记录每组 score/label 的参数和压缩归档来源；当前归档中权重文件计数为零。共同协议 CATCH 是本次新训练，保存可重载 checkpoint、训练证据、归一化统计、分数和注册验证。

Isolation Forest 参数固定为 100 trees、max_samples=min(256,n_train)、contamination=auto、seed=2021；PCA 在训练行上以 95% 方差保留率拟合，按每行重建 MSE 排序。它们均没有内部 holdout。

逐点 AP/ROC 描述排序；点报警指标和有效段 affiliation 指标使用 calibration 阈值。原 CATCH score 归档还可能使用合并 train/test 能量及 anomaly_ratio 阈值，协议不同。数值并列/接近不能解释为论文协议复现。

## 每组配置与失败状态

- **PSM**：source `full_gmm-full-20261024T115119Z-3976714`；状态 `complete`；CATCH 配置 `{"Mlr":0.001,"affine":0,"anomaly_ratio":[0.1,0.5,1.0,2,3,5.0,10.0,15,20,25],"auxi_lambda":0.01,"auxi_loss":"MAE","auxi_mode":"fft","auxi_type":"complex","batch_size":128,"cf_dim":16,"d_ff":32,"d_model":16,"dc_lambda":0.05,"dropout":0.3,"e_layers":1,"head_dim":32,"head_dropout":0.1,"individual":0,"inference_patch_size":96,"inference_patch_stride":1,"lr":0.005,"lradj":"type1","mask":false,"module_first":true,"n_heads":4,"num_epochs":3,"patch_size":16,"patch_stride":8,"patience":3,"pct_start":0.3,"pretrained_model":null,"regular_lambda":0.5,"revin":1,"score_lambda":0.5,"seed":2021,"seq_len":192,"subtract_last":0,"temperature":0.07}`。校准端点数：13020。
- **MSL**：source `full_gmm-full-20261021T112147Z-2470335`；状态 `complete`；CATCH 配置 `{"Mlr":5e-05,"affine":0,"anomaly_ratio":[0.1,0.5,1.0,2,3,5.0,10.0,15,20,25],"auxi_lambda":0.005,"auxi_loss":"MAE","auxi_mode":"fft","auxi_type":"complex","batch_size":128,"cf_dim":64,"d_ff":256,"d_model":128,"dc_lambda":0.005,"dropout":0.2,"e_layers":3,"head_dim":64,"head_dropout":0.1,"individual":0,"inference_patch_size":32,"inference_patch_stride":1,"lr":0.0005,"lradj":"type1","mask":false,"module_first":true,"n_heads":2,"num_epochs":5,"patch_size":16,"patch_stride":8,"patience":3,"pct_start":0.3,"pretrained_model":null,"regular_lambda":0.5,"revin":1,"score_lambda":0.05,"seed":2021,"seq_len":192,"subtract_last":0,"temperature":0.07}`。校准端点数：5603。
- **GECCO**：source `full_gmm-full-20261020T204951Z-1642355`；状态 `complete`；CATCH 配置 `{"Mlr":0.0001,"affine":0,"anomaly_ratio":1,"auxi_lambda":0.005,"auxi_loss":"MAE","auxi_mode":"fft","auxi_type":"complex","batch_size":128,"cf_dim":64,"d_ff":256,"d_model":128,"dc_lambda":0.005,"dropout":0.2,"e_layers":2,"head_dim":64,"head_dropout":0.1,"individual":0,"inference_patch_size":32,"inference_patch_stride":1,"itr":1,"lr":0.0001,"lradj":"type1","mask":false,"module_first":true,"n_heads":2,"num_epochs":1,"patch_size":16,"patch_stride":8,"patience":10,"pct_start":0.3,"pretrained_model":null,"regular_lambda":0.5,"revin":1,"score_lambda":0.05,"seed":2021,"seq_len":96,"small_kernel_merged":"False","subtract_last":0,"temperature":0.1,"use_multi_scale":"False"}`。校准端点数：6812。
- **Genesis**：source `full_gmm-full-20261020T205005Z-1645522`；状态 `complete`；CATCH 配置 `{"Mlr":0.0001,"affine":0,"anomaly_ratio":[0.1,0.5,1.0,2,3,5.0,10.0,15,20,25],"auxi_lambda":0.005,"auxi_loss":"MAE","auxi_mode":"fft","auxi_type":"complex","batch_size":4,"cf_dim":64,"d_ff":128,"d_model":128,"dc_lambda":0.005,"dropout":0.2,"e_layers":3,"head_dim":64,"head_dropout":0.1,"individual":0,"inference_patch_size":32,"inference_patch_stride":1,"lr":0.0001,"lradj":"type1","mask":false,"module_first":true,"n_heads":16,"num_epochs":10,"patch_size":8,"patch_stride":8,"patience":3,"pct_start":0.3,"pretrained_model":null,"regular_lambda":0.5,"revin":1,"score_lambda":1,"seed":2021,"seq_len":192,"subtract_last":0,"temperature":0.07}`。校准端点数：132。
- **NYC**：source `full_gmm-full-20261020T211341Z-1931786`；状态 `complete`；CATCH 配置 `{"Mlr":1e-05,"affine":0,"anomaly_ratio":[0.1,0.5,1.0,2,3,5.0,10.0,15,20,25],"auxi_lambda":0.005,"auxi_loss":"MAE","auxi_mode":"fft","auxi_type":"complex","batch_size":128,"cf_dim":64,"d_ff":256,"d_model":256,"dc_lambda":0.005,"dropout":0.2,"e_layers":3,"head_dim":64,"head_dropout":0.1,"individual":0,"inference_patch_size":32,"inference_patch_stride":1,"lr":0.0001,"lradj":"type1","mask":false,"module_first":true,"n_heads":2,"num_epochs":2,"patch_size":16,"patch_stride":8,"patience":3,"pct_start":0.3,"pretrained_model":null,"regular_lambda":0.5,"revin":1,"score_lambda":0.05,"seed":2021,"seq_len":192,"subtract_last":0,"temperature":0.07}`。校准端点数：1082。
- **CalIt2**：source `full_gmm-full-20261021T112212Z-2474004`；状态 `complete`；CATCH 配置 `{"Mlr":1e-05,"affine":0,"anomaly_ratio":[0.1,0.5,1.0,2,3,5.0,10.0,15,20,25],"auxi_lambda":0.1,"auxi_loss":"MAE","auxi_mode":"fft","auxi_type":"complex","batch_size":128,"cf_dim":128,"d_ff":128,"d_model":128,"dc_lambda":0.5,"dropout":0.2,"e_layers":2,"head_dim":32,"head_dropout":0.1,"individual":0,"inference_patch_size":16,"inference_patch_stride":1,"lr":0.0001,"lradj":"type1","mask":false,"module_first":true,"n_heads":8,"num_epochs":10,"patch_size":16,"patch_stride":16,"patience":3,"pct_start":0.3,"pretrained_model":null,"regular_lambda":0.5,"revin":1,"score_lambda":0.5,"seed":2021,"seq_len":192,"subtract_last":0,"temperature":0.07}`。校准端点数：24。
- **Creditcard**：source `full_gmm-full-20261024T115131Z-3977864`；状态 `complete`；CATCH 配置 `{"Mlr":1e-05,"affine":0,"anomaly_ratio":[0.1,0.5,1.0,2,3,5.0,10.0,15,20,25],"auxi_lambda":0.005,"auxi_loss":"MAE","auxi_mode":"fft","auxi_type":"complex","batch_size":128,"cf_dim":64,"d_ff":128,"d_model":128,"dc_lambda":0.005,"dropout":0.2,"e_layers":3,"head_dim":64,"head_dropout":0.1,"individual":0,"inference_patch_size":16,"inference_patch_stride":1,"lr":0.0001,"lradj":"type1","mask":false,"module_first":true,"n_heads":16,"num_epochs":5,"patch_size":32,"patch_stride":32,"patience":3,"pct_start":0.3,"pretrained_model":null,"regular_lambda":0.5,"revin":1,"score_lambda":0.05,"seed":2021,"seq_len":192,"subtract_last":0,"temperature":0.07}`。校准端点数：14012。
- **CICIDS**：source `full_gmm-full-20261024T115157Z-3980319`；状态 `complete`；CATCH 配置 `{"Mlr":1e-05,"affine":0,"anomaly_ratio":[0.1,0.5,1.0,2,3,5.0,10.0,15,20,25],"auxi_lambda":0.005,"auxi_loss":"MAE","auxi_mode":"fft","auxi_type":"complex","batch_size":128,"cf_dim":64,"d_ff":128,"d_model":128,"dc_lambda":0.005,"dropout":0.2,"e_layers":3,"head_dim":16,"head_dropout":0.1,"individual":0,"inference_patch_size":16,"inference_patch_stride":1,"lr":0.0001,"lradj":"type1","mask":false,"module_first":true,"n_heads":16,"num_epochs":5,"patch_size":16,"patch_stride":16,"patience":3,"pct_start":0.3,"pretrained_model":null,"regular_lambda":0.5,"revin":1,"score_lambda":0.05,"seed":2021,"seq_len":192,"subtract_last":0,"temperature":0.07}`。校准端点数：8283。

## 协议解释

对比只涉及当前 Full 表示与共同切分下的 CATCH、Isolation Forest、PCA。若排名接近但报警差，首先体现独立校准尾部分位与 score 分布差异；若 IF/PCA 排名更强，提示点统计结构可能足以排序，但不等价于证明 MNRF 表示无效。该诊断不改 MNRF、不做分支消融，也不据此宣称 SOTA。
