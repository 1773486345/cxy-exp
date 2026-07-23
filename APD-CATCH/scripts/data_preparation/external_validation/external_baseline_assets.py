"""Static assets for the frozen external baseline comparison.

This module only renders the 300 independent shell commands, their registry, and
the README command list.  It never invokes a benchmark, a model, or a result
summarizer.  The configurations are fixed before any external baseline result is
read and are copied from the existing PSM ``detect_score`` templates.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from common import PROJECT_ROOT, RESULT_ROOT, TASK_ORDER


BENCHMARK_PYTHON = "python"
CONFIG_PATH = "unfixed_detect_score_multi_config.json"


BASELINE_SPECS = (
    {
        "paper_name": "ModernTCN",
        "framework_model_name": "self_impl.ModernTCN",
        "script_name": "ModernTCN",
        "result_name": "ModernTCN",
        "implementation_module": "ts_benchmark.baselines.self_impl.ModernTCN.ModernTCN",
        "implementation_class": "ModernTCN",
        "source_existing_script": "scripts/multivariate_detection/detect_score/PSM_script/ModernTCN.sh",
        "model_hyper_params": {"anomaly_ratio": 1, "batch_size": 128, "dims": [8], "dropout": 0.1, "ffn_ratio": 1, "head_dropout": 0.0, "itr": 1, "large_size": [51], "lr": 0.0005, "num_blocks": [1], "num_epochs": 2, "patch_size": 8, "patch_stride": 4, "patience": 10, "small_kernel_merged": False, "small_size": [5], "use_multi_scale": False},
        "adapter": None,
        "python": BENCHMARK_PYTHON,
        "environment_name": "catch_env",
        "dependency_note": "catch_env",
    },
    {
        "paper_name": "iTransformer",
        "framework_model_name": "time_series_library.iTransformer",
        "script_name": "iTransformer",
        "result_name": "iTransformer",
        "implementation_module": "ts_benchmark.baselines.time_series_library.models.iTransformer",
        "implementation_class": "iTransformer",
        "source_existing_script": "scripts/multivariate_detection/detect_score/PSM_script/iTransformer.sh",
        "model_hyper_params": {"batch_size": 64, "d_ff": 512, "d_model": 256, "e_layers": 1, "horizon": 0, "lr": 0.0001, "norm": True, "num_epochs": 5, "seq_len": 100},
        "adapter": "transformer_adapter",
        "python": BENCHMARK_PYTHON,
        "environment_name": "catch_env",
        "dependency_note": "catch_env; transformer_adapter",
    },
    {
        "paper_name": "DualTF",
        "framework_model_name": "self_impl.DualTF",
        "script_name": "DualTF",
        "result_name": "DualTF",
        "implementation_module": "ts_benchmark.baselines.self_impl.DualTF.DualTF",
        "implementation_class": "DualTF",
        "source_existing_script": "scripts/multivariate_detection/detect_score/PSM_script/DualTF.sh",
        "model_hyper_params": {"batch_size": 8, "fre_anormly_ratio": 10, "lr": 0.001, "num_epochs": 3, "seq_len": 50},
        "adapter": None,
        "python": BENCHMARK_PYTHON,
        "environment_name": "catch_env",
        "dependency_note": "catch_env",
    },
    {
        "paper_name": "AnomalyTransformer",
        "framework_model_name": "self_impl.AnomalyTransformer",
        "script_name": "AnomalyTransformer",
        "result_name": "AnomalyTransformer",
        "implementation_module": "ts_benchmark.baselines.self_impl.Anomaly_trans.AnomalyTransformer",
        "implementation_class": "AnomalyTransformer",
        "source_existing_script": "scripts/multivariate_detection/detect_score/PSM_script/AnomalyTransformer.sh",
        "model_hyper_params": {"batch_size": 128, "lr": 0.001, "num_epochs": 3, "win_size": 50},
        "adapter": None,
        "python": BENCHMARK_PYTHON,
        "environment_name": "catch_env",
        "dependency_note": "catch_env",
    },
    {
        "paper_name": "DCdetector",
        "framework_model_name": "self_impl.DCdetector",
        "script_name": "DCdetector",
        "result_name": "DCdetector",
        "implementation_module": "ts_benchmark.baselines.self_impl.DCdetector.DCdetector",
        "implementation_class": "DCdetector",
        "source_existing_script": "scripts/multivariate_detection/detect_score/PSM_script/DCdetector.sh",
        "model_hyper_params": {"batch_size": 128, "lr": 0.0001, "num_epochs": 3, "win_size": 110},
        "adapter": None,
        "python": BENCHMARK_PYTHON,
        "environment_name": "catch_env",
        "dependency_note": "catch_env",
    },
    {
        "paper_name": "TimesNet",
        "framework_model_name": "time_series_library.TimesNet",
        "script_name": "TimesNet",
        "result_name": "TimesNet",
        "implementation_module": "ts_benchmark.baselines.time_series_library.models.TimesNet",
        "implementation_class": "TimesNet",
        "source_existing_script": "scripts/multivariate_detection/detect_score/PSM_script/TimesNet.sh",
        "model_hyper_params": {"batch_size": 128, "d_ff": 8, "d_model": 8, "e_layers": 3, "horizon": 0, "norm": True, "num_epochs": 3, "seq_len": 100},
        "adapter": "transformer_adapter",
        "python": BENCHMARK_PYTHON,
        "environment_name": "catch_env",
        "dependency_note": "catch_env; transformer_adapter",
    },
    {
        "paper_name": "PatchTST",
        "framework_model_name": "time_series_library.PatchTST",
        "script_name": "PatchTST",
        "result_name": "PatchTST",
        "implementation_module": "ts_benchmark.baselines.time_series_library.models.PatchTST",
        "implementation_class": "PatchTST",
        "source_existing_script": "scripts/multivariate_detection/detect_score/PSM_script/PatchTST.sh",
        "model_hyper_params": {"batch_size": 128, "d_ff": 16, "d_model": 8, "e_layers": 1, "horizon": 0, "norm": True, "num_epochs": 1, "seq_len": 100},
        "adapter": "transformer_adapter",
        "python": BENCHMARK_PYTHON,
        "environment_name": "catch_env",
        "dependency_note": "catch_env; transformer_adapter",
    },
    {
        "paper_name": "DLinear",
        "framework_model_name": "time_series_library.DLinear",
        "script_name": "DLinear",
        "result_name": "DLinear",
        "implementation_module": "ts_benchmark.baselines.time_series_library.models.DLinear",
        "implementation_class": "DLinear",
        "source_existing_script": "scripts/multivariate_detection/detect_score/PSM_script/DLinear.sh",
        "model_hyper_params": {"batch_size": 128, "d_ff": 16, "d_model": 8, "e_layers": 1, "horizon": 0, "norm": True, "num_epochs": 1, "seq_len": 100},
        "adapter": "transformer_adapter",
        "python": BENCHMARK_PYTHON,
        "environment_name": "catch_env",
        "dependency_note": "catch_env; transformer_adapter",
    },
    {
        "paper_name": "NLinear",
        "framework_model_name": "time_series_library.NLinear",
        "script_name": "NLinear",
        "result_name": "NLinear",
        "implementation_module": "ts_benchmark.baselines.time_series_library.patchs.NLinear",
        "implementation_class": "NLinear",
        "source_existing_script": "scripts/multivariate_detection/detect_score/PSM_script/NLinear.sh",
        "model_hyper_params": {"batch_size": 128, "d_ff": 16, "d_model": 8, "e_layers": 1, "horizon": 0, "norm": True, "num_epochs": 1, "seq_len": 100},
        "adapter": "transformer_adapter",
        "python": BENCHMARK_PYTHON,
        "environment_name": "catch_env",
        "dependency_note": "catch_env; transformer_adapter",
    },
    {
        "paper_name": "TFAD",
        "framework_model_name": "self_impl.TFAD",
        "script_name": "TFAD",
        "result_name": "TFAD",
        "implementation_module": "ts_benchmark.baselines.self_impl.TFAD.TFAD",
        "implementation_class": "TFAD",
        "source_existing_script": "scripts/multivariate_detection/detect_score/PSM_script/TFAD.sh",
        "model_hyper_params": {},
        "adapter": None,
        "python": BENCHMARK_PYTHON,
        "environment_name": "catch_env",
        "dependency_note": "catch_env; implementation default window_length=192",
    },
    {
        "paper_name": "AutoEncoder",
        "framework_model_name": "merlion.AutoEncoder",
        "script_name": "AutoEncoder",
        "result_name": "AutoEncoder",
        "implementation_module": "ts_benchmark.baselines.merlion.merlion_models",
        "implementation_class": "MerlionModelAdapter(AutoEncoder)",
        "source_existing_script": "scripts/multivariate_detection/detect_score/PSM_script/AutoEncoder.sh",
        "model_hyper_params": {"hidden_size": 1, "layer_sizes": [10, 3], "lr": 0.00001, "num_epochs": 1},
        "adapter": None,
        "python": BENCHMARK_PYTHON,
        "environment_name": "catch_env",
        "dependency_note": "catch_env",
    },
    {
        "paper_name": "OCSVM",
        "framework_model_name": "tods.ocsvmski",
        "script_name": "ocsvmski",
        "result_name": "OCSVM",
        "implementation_module": "ts_benchmark.baselines.tods.tods_models",
        "implementation_class": "TodsModelAdapter(OCSVMSKI)",
        "source_existing_script": "scripts/multivariate_detection/detect_score/PSM_script/ocsvmski.sh",
        "model_hyper_params": {},
        "adapter": None,
        "python": BENCHMARK_PYTHON,
        "environment_name": "tods_legacy",
        "dependency_note": "tods_legacy",
    },
    {
        "paper_name": "IsolationForest",
        "framework_model_name": "merlion.IsolationForest",
        "script_name": "IsolationForest",
        "result_name": "IsolationForest",
        "implementation_module": "ts_benchmark.baselines.merlion.merlion_models",
        "implementation_class": "MerlionModelAdapter(IsolationForest)",
        "source_existing_script": "scripts/multivariate_detection/detect_score/PSM_script/IsolationForest.sh",
        "model_hyper_params": {},
        "adapter": None,
        "python": BENCHMARK_PYTHON,
        "environment_name": "catch_env",
        "dependency_note": "catch_env",
    },
    {
        "paper_name": "PCA",
        "framework_model_name": "tods.pcaodetectorski",
        "script_name": "pcaodetectorski",
        "result_name": "PCA",
        "implementation_module": "ts_benchmark.baselines.tods.tods_models",
        "implementation_class": "TodsModelAdapter(PCAODetectorSKI)",
        "source_existing_script": "scripts/multivariate_detection/detect_score/PSM_script/pcaodetectorski.sh",
        "model_hyper_params": {"n_components": 2, "window_size": 1},
        "adapter": None,
        "python": BENCHMARK_PYTHON,
        "environment_name": "tods_legacy",
        "dependency_note": "tods_legacy",
    },
    {
        "paper_name": "HBOS",
        "framework_model_name": "tods.hbosski",
        "script_name": "hbosski",
        "result_name": "HBOS",
        "implementation_module": "ts_benchmark.baselines.tods.tods_models",
        "implementation_class": "TodsModelAdapter(HBOSSKI)",
        "source_existing_script": "scripts/multivariate_detection/detect_score/PSM_script/hbosski.sh",
        "model_hyper_params": {"alpha": 0.000001, "n_bins": 3, "tol": 0.9},
        "adapter": None,
        "python": BENCHMARK_PYTHON,
        "environment_name": "tods_legacy",
        "dependency_note": "tods_legacy",
    },
)


def command_for(task: str, spec: dict) -> str:
    """Return one independent, foreground benchmark command."""
    params = json.dumps(spec["model_hyper_params"], separators=(",", ":"), sort_keys=True)
    command = (
        f'{spec["python"]} ./scripts/run_benchmark.py '
        f'--config-path "{CONFIG_PATH}" '
        '--data-set-name "external_detect" '
        f'--data-name-list "{task}.csv" '
        f'--model-name "{spec["framework_model_name"]}" '
        f"--model-hyper-params '{params}' "
        '--seed 2021 --gpus 0 --num-workers 1 --timeout 60000 '
        f'--save-path "score/external_validation/{task}/{spec["result_name"]}"'
    )
    if spec["adapter"]:
        command += f' --adapter "{spec["adapter"]}"'
    return command + "\n"


def script_path(task: str, spec: dict) -> Path:
    return PROJECT_ROOT / "scripts" / "multivariate_detection" / "detect_score" / f"{task}_script" / f'{spec["script_name"]}.sh'


def write_external_baseline_assets() -> None:
    """Materialize static scripts, command list, and registry without running them."""
    for task in TASK_ORDER:
        for spec in BASELINE_SPECS:
            path = script_path(task, spec)
            path.write_text("#!/usr/bin/env sh\n" + command_for(task, spec), encoding="utf-8")
            path.chmod(0o755)

    command_path = PROJECT_ROOT / "EXTERNAL_BASELINE_COMMANDS.md"
    lines = [
        "# External Baseline Commands",
        "",
        "- Baseline models: 15",
        "- Tasks: 20",
        "- Planned commands: 300",
        "- Executable commands: 300",
        "- Every command is foreground-only and operates on one model and one task.",
        "- Activate `catch_env` for all models except OCSVM, PCA, and HBOS; activate `tods_legacy` for those three TODS models.",
        "- Shell scripts intentionally use `python`, matching the existing CATCH/MSD external scripts.",
        "- Each Baseline uses its own frozen PSM detect_score template on every task; any future CUDA OOM override must be recorded per model and task.",
        "",
    ]
    for spec in BASELINE_SPECS:
        lines.extend([f"## {spec['paper_name']}", ""])
        for task in TASK_ORDER:
            lines.extend([
                f"### {task}",
                "",
                "```bash",
                f"sh ./scripts/multivariate_detection/detect_score/{task}_script/{spec['script_name']}.sh",
                "```",
                "",
            ])
    command_path.write_text("\n".join(lines), encoding="utf-8")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    registry_path = RESULT_ROOT / "external_baseline_registry.csv"
    fields = [
        "paper_name", "framework_model_name", "implementation_module", "implementation_class",
        "source_existing_script", "config_template", "model_hyper_params", "adapter", "python_environment",
        "supports_detect_score", "supports_multivariate", "train_label_usage", "test_label_usage",
        "pretrained_checkpoint", "status", "reason",
    ]
    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in BASELINE_SPECS:
            writer.writerow(
                {
                    "paper_name": spec["paper_name"],
                    "framework_model_name": spec["framework_model_name"],
                    "implementation_module": spec["implementation_module"],
                    "implementation_class": spec["implementation_class"],
                    "source_existing_script": spec["source_existing_script"],
                    "config_template": "frozen PSM detect_score template; one fixed external configuration for all 20 tasks",
                    "model_hyper_params": json.dumps(spec["model_hyper_params"], sort_keys=True),
                    "adapter": spec["adapter"] or "",
                    "python_environment": spec["environment_name"],
                    "supports_detect_score": True,
                    "supports_multivariate": True,
                    "train_label_usage": "framework passes train label; implementation does not use test label",
                    "test_label_usage": "none",
                    "pretrained_checkpoint": "none",
                    "status": "executable",
                    "reason": spec["dependency_note"],
                }
            )

    overrides_path = RESULT_ROOT / "external_baseline_batch_overrides.csv"
    with overrides_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["task", "paper_name", "original_batch_size", "final_batch_size", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()


if __name__ == "__main__":
    write_external_baseline_assets()
