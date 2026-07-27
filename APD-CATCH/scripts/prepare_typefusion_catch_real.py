#!/usr/bin/env python3
"""Audit CATCH-master real-task provenance and prepare TypeFusion-CATCH scripts.

This script intentionally performs no model fitting.  It reads the immutable
CATCH-master archive, validates the shared anomaly-data directory, writes small
registries, and creates manual single-task scripts only in this repository.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shlex
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATCH_ROOT = ROOT.parent / "CATCH-master"
RESULT_ROOT = ROOT / "result/typefusion_catch_main"

TASKS: Tuple[str, ...] = (
    "ASD_dataset_1",
    "ASD_dataset_2",
    "ASD_dataset_3",
    "ASD_dataset_4",
    "ASD_dataset_5",
    "ASD_dataset_6",
    "ASD_dataset_7",
    "ASD_dataset_8",
    "ASD_dataset_9",
    "ASD_dataset_10",
    "ASD_dataset_11",
    "ASD_dataset_12",
    "CICIDS",
    "CalIt2",
    "Creditcard",
    "GECCO",
    "Genesis",
    "MSL",
    "NYC",
    "PSM",
    "SMAP",
    "SMD",
    "SWAT",
)
ASD_TASKS = tuple(task for task in TASKS if task.startswith("ASD_dataset_"))
PAPER_DATASET = {task: "ASD" if task in ASD_TASKS else task for task in TASKS}

COMMON_TYPEFUSION_PARAMS = (
    "batch_size",
    "cf_dim",
    "d_ff",
    "d_model",
    "dropout",
    "e_layers",
    "head_dim",
    "lr",
    "n_heads",
    "patience",
    "seq_len",
    "patch_size",
    "patch_stride",
)
TYPEFUSION_FIXED_PARAMS: Dict[str, Any] = {
    "seed": 2021,
    "fit_mode": "three_stage",
    "training_budget_mode": "equal_total_steps",
    "joint_finetune_lr_scale": 0.1,
    "lambda_freq": 0.1,
    "lambda_mask": 0.1,
    "temporal_hidden_dim": 128,
    "temporal_layers": 3,
    "memory_size": 32,
    "memory_topk": 4,
    "branch_dim": 128,
    "fusion_layers": 2,
    "fusion_heads": 4,
    "relation_mask_groups": 4,
    "pattern_mask_ratio": 0.25,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def load_catch_defaults(catch_root: Path) -> Dict[str, Any]:
    """Read the immutable CATCH adapter defaults without importing its model."""

    source = catch_root / "ts_benchmark/baselines/catch/CATCH.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS"
            for target in node.targets
        ):
            defaults = ast.literal_eval(node.value)
            if not isinstance(defaults, dict):
                break
            return defaults
    raise ValueError(f"Could not parse CATCH defaults from {source}")


def ensure_shared_data(catch_root: Path) -> Tuple[Path, str]:
    """Confirm existing data share a realpath, creating a link only when absent."""

    source = catch_root / "dataset/anomaly_detect"
    if not source.exists():
        raise FileNotFoundError(f"CATCH-master anomaly data are missing: {source}")
    source_real = source.resolve(strict=True)
    target = ROOT / "dataset/anomaly_detect"
    if target.exists() or target.is_symlink():
        if not target.exists() or not any(target.iterdir()):
            raise RuntimeError(
                f"Refusing to replace existing APD-CATCH anomaly data path: {target}"
            )
        if target.resolve(strict=True) != source_real:
            raise RuntimeError(
                "APD-CATCH anomaly data do not resolve to the CATCH-master source: "
                f"{target.resolve(strict=True)} != {source_real}"
            )
        return source_real, "existing_shared_symlink" if target.is_symlink() else "existing_shared_directory"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source_real, target_is_directory=True)
    return source_real, "created_read_only_source_symlink"


def parse_command(path: Path) -> Dict[str, Any]:
    """Parse the sole benchmark invocation stored in an archived CATCH script."""

    command_line = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if "run_benchmark.py" in line and "--model-name" in line:
            command_line = line.strip()
            break
    if command_line is None:
        raise ValueError(f"No benchmark command found in {path}")
    tokens = shlex.split(command_line)

    def value(flag: str, required: bool = True) -> str | None:
        if flag not in tokens:
            if required:
                raise ValueError(f"{path} is missing {flag}")
            return None
        return tokens[tokens.index(flag) + 1]

    hyper_params = json.loads(value("--model-hyper-params"))
    return {
        "config_path": value("--config-path"),
        "data_name": value("--data-name-list"),
        "model_name": value("--model-name"),
        "model_hyper_params": hyper_params,
        "seed_flag": value("--seed", required=False),
        "gpus": value("--gpus", required=False),
        "num_workers": value("--num-workers", required=False),
        "timeout": value("--timeout", required=False),
    }


def resolve_command_script(command_path: Path, catch_root: Path) -> Path:
    """Resolve a run wrapper's `exec bash .../CATCH.sh` provenance target."""

    text = command_path.read_text(encoding="utf-8")
    match = re.search(r"(?:exec\s+)?bash\s+([^\s]+CATCH\.sh)", text)
    if match is None:
        return command_path
    target = Path(match.group(1))
    return target if target.is_absolute() else catch_root / target


def command_signature(command: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Compare provenance commands without run-specific save paths."""

    return (
        command["config_path"],
        command["data_name"],
        command["model_name"],
        canonical_json(command["model_hyper_params"]),
        command["seed_flag"],
        command["gpus"],
        command["num_workers"],
        command["timeout"],
    )


def parse_report(path: Path) -> Dict[str, Any]:
    report = pd.read_csv(path)
    if len(report) != 1:
        raise ValueError(f"Expected one metric row in {path}, found {len(report)}")
    row = report.iloc[0]
    metric_columns = [
        column for column in report.columns if column not in {"strategy_args", "metric_name"}
    ]
    if len(metric_columns) != 1:
        raise ValueError(f"Expected one metric value column in {path}")
    metric_column = metric_columns[0]
    model_prefix, report_params_text = metric_column.split(";", 1)
    report_params = json.loads(report_params_text)
    strategy = json.loads(row["strategy_args"])
    value = float(row[metric_column])
    return {
        "model_prefix": model_prefix,
        "model_hyper_params": report_params,
        "strategy": strategy,
        "metric_name": str(row["metric_name"]),
        "metric_value": value,
    }


def archived_commit(run_dir: Path, fallback_commit: str) -> str:
    metadata_path = run_dir / "metadata.txt"
    if not metadata_path.is_file():
        return fallback_commit
    for line in metadata_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("catch_master_commit="):
            return line.split("=", 1)[1].strip() or fallback_commit
    return fallback_commit


def load_source_registry(catch_root: Path) -> List[Dict[str, Any]]:
    summary_path = catch_root / "result/CATCH_reproduction_summary/physical_task_metrics.csv"
    summary = pd.read_csv(summary_path)
    by_task = {str(row["physical_task"]): row for _, row in summary.iterrows()}
    missing = [task for task in TASKS if task not in by_task]
    if missing:
        raise ValueError(f"Reproduction summary is missing fixed tasks: {missing}")

    catch_defaults = load_catch_defaults(catch_root)
    current_catch_commit = git_commit(catch_root)
    registry: List[Dict[str, Any]] = []
    for task in TASKS:
        summary_row = by_task[task]
        report_path = catch_root / str(summary_row["score_report"])
        run_dir = report_path.parent
        official = run_dir / "official_CATCH.sh"
        wrapper = run_dir / "command.sh"
        fallback = catch_root / "scripts/multivariate_detection/detect_score" / f"{task}_script/CATCH.sh"
        report_valid = False
        report_detail: Dict[str, Any] = {}
        errors: List[str] = []
        try:
            report_detail = parse_report(report_path)
            strategy = report_detail["strategy"]
            report_valid = (
                report_path.is_file()
                and report_path.stat().st_size > 0
                and strategy.get("strategy_name") == "unfixed_detect_score"
                and int(strategy.get("seed")) == 2021
                and np.isfinite(report_detail["metric_value"])
            )
            if not report_valid:
                errors.append("invalid_baseline_report")
        except Exception as exc:  # registry must preserve all audit failures
            errors.append(f"report_parse_error:{exc}")

        parsed: Dict[str, Dict[str, Any]] = {}
        for name, path in (("official", official), ("command", wrapper), ("fallback", fallback)):
            try:
                source_path = resolve_command_script(path, catch_root) if name == "command" else path
                parsed[name] = parse_command(source_path)
                parsed[name]["resolved_path"] = source_path
            except Exception as exc:
                errors.append(f"{name}_parse_error:{exc}")

        source_conflict = False
        selected = parsed.get("official")
        if selected is None:
            source_conflict = True
            errors.append("missing_official_source")
        else:
            signature = command_signature(selected)
            for name in ("command", "fallback"):
                if name not in parsed or command_signature(parsed[name]) != signature:
                    source_conflict = True
                    errors.append(f"source_conflict:{name}")
            if report_detail and canonical_json(selected["model_hyper_params"]) != canonical_json(
                report_detail["model_hyper_params"]
            ):
                source_conflict = True
                errors.append("source_conflict:report_hyper_params")

        if selected is None:
            selected = {
                "config_path": "",
                "data_name": f"{task}.csv",
                "model_name": "",
                "model_hyper_params": {},
                "seed_flag": None,
                "gpus": None,
                "num_workers": None,
                "timeout": None,
            }

        raw_hyper_params = dict(selected["model_hyper_params"])
        effective_hyper_params = dict(catch_defaults)
        effective_hyper_params.update(raw_hyper_params)
        typefusion_params = {
            key: effective_hyper_params[key]
            for key in COMMON_TYPEFUSION_PARAMS
            if key in effective_hyper_params
        }
        if "num_epochs" in effective_hyper_params:
            typefusion_params["catch_train_epochs"] = effective_hyper_params["num_epochs"]
        typefusion_params.update(TYPEFUSION_FIXED_PARAMS)
        compatibility_override = ""
        if int(typefusion_params["cf_dim"]) % int(typefusion_params["n_heads"]) != 0:
            source_heads = int(typefusion_params["n_heads"])
            compatible_heads = math.gcd(int(typefusion_params["cf_dim"]), source_heads)
            if compatible_heads < 1:
                raise ValueError(f"Cannot derive a valid TypeFusion n_heads for {task}")
            typefusion_params["n_heads"] = compatible_heads
            compatibility_override = (
                f"n_heads:{source_heads}->{compatible_heads};"
                f"cf_dim={effective_hyper_params['cf_dim']};"
                "TypeFusion MultiheadAttention divisibility"
            )
        gecco_override = task == "GECCO"
        if gecco_override:
            typefusion_params["seq_len"] = 192

        source_complete = bool(
            official.is_file()
            and wrapper.is_file()
            and fallback.is_file()
            and "num_epochs" in effective_hyper_params
            and not source_conflict
        )
        registry.append(
            {
                "task": task,
                "paper_dataset": PAPER_DATASET[task],
                "dataset_file": selected["data_name"],
                "source_complete": source_complete,
                "source_conflict": source_conflict,
                "source_conflict_detail": "|".join(errors),
                "official_source_script": str(official),
                "command_wrapper_script": str(wrapper),
                "command_resolved_script": str(parsed.get("command", {}).get("resolved_path", "")),
                "fallback_source_script": str(fallback),
                "source_script_sha256": sha256_file(official) if official.is_file() else "",
                "fallback_script_sha256": sha256_file(fallback) if fallback.is_file() else "",
                "config_path": selected["config_path"],
                "data_name": selected["data_name"],
                "model_name": selected["model_name"],
                "seed": report_detail.get("strategy", {}).get("seed", ""),
                "gpus": selected["gpus"] or "",
                "num_workers": selected["num_workers"] or "",
                "timeout": selected["timeout"] or "",
                **{
                    key: effective_hyper_params.get(key, "")
                    for key in (*COMMON_TYPEFUSION_PARAMS, "num_epochs")
                },
                "parameter_origin_json": canonical_json(
                    {
                        key: "official_CATCH.sh"
                        if key in raw_hyper_params
                        else "CATCH.py default"
                        for key in (*COMMON_TYPEFUSION_PARAMS, "num_epochs")
                    }
                ),
                "catch_model_hyper_params_json": canonical_json(raw_hyper_params),
                "effective_catch_model_hyper_params_json": canonical_json(
                    effective_hyper_params
                ),
                "typefusion_hyper_params_json": canonical_json(typefusion_params),
                "typefusion_compatibility_override": compatibility_override,
                "baseline_report": str(report_path),
                "source_catch_master_commit": archived_commit(run_dir, current_catch_commit),
                "baseline_report_valid": report_valid,
                "baseline_metric_available": bool(report_detail),
                "baseline_metric_name": report_detail.get("metric_name", ""),
                "baseline_metric_value": report_detail.get("metric_value", ""),
                "baseline_requires_fair_rerun": gecco_override,
                "gecco_fairness_override": gecco_override,
                "direct_baseline_comparable": not gecco_override and report_valid,
                "original_batch_size": effective_hyper_params.get("batch_size", ""),
                "final_batch_size": effective_hyper_params.get("batch_size", ""),
                "oom_log": "",
                "catch_rerun_required": False,
            }
        )
    return registry


def inspect_data(path: Path, metadata: pd.DataFrame, dataset_file: str, min_seq_len: int) -> Dict[str, Any]:
    """Stream a long-format source CSV without materialising its wide frame."""

    metadata_discoverable = bool(
        dataset_file in metadata.index and str(metadata.loc[dataset_file, "size"]) in {"large", "small"}
    )
    feature_names = set()
    normal_count = 0
    anomaly_count = 0
    label_rows = 0
    errors: List[str] = []
    try:
        for chunk in pd.read_csv(path, usecols=["data", "cols"], chunksize=250_000):
            names = set(chunk["cols"].astype(str))
            feature_names.update(names.difference({"label"}))
            labels = chunk.loc[chunk["cols"].astype(str) == "label", "data"]
            if not labels.empty:
                numeric_labels = pd.to_numeric(labels, errors="coerce")
                if numeric_labels.isna().any():
                    errors.append("non_numeric_label")
                label_rows += int(numeric_labels.notna().sum())
                normal_count += int((numeric_labels == 0).sum())
                anomaly_count += int((numeric_labels == 1).sum())
                if int((~numeric_labels.isin([0, 1])).sum()) > 0:
                    errors.append("non_binary_label")
    except Exception as exc:
        errors.append(f"csv_read_error:{exc}")

    if "label" not in feature_names and label_rows == 0:
        errors.append("missing_label")
    if normal_count == 0 or anomaly_count == 0:
        errors.append("label_does_not_contain_0_and_1")
    if len(feature_names) < 2:
        errors.append("fewer_than_two_features")
    if label_rows <= min_seq_len:
        errors.append("insufficient_rows_for_seq_len")
    if not metadata_discoverable:
        errors.append("not_large_detect_discoverable")
    return {
        "resolved_path": str(path.resolve(strict=True)),
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": label_rows,
        "feature_count": len(feature_names),
        "normal_count": normal_count,
        "anomaly_count": anomaly_count,
        "metadata_discoverable": metadata_discoverable,
        "integrity_status": "ok" if not errors else "|".join(sorted(set(errors))),
    }


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    frame = pd.DataFrame(list(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def json_template(params: Mapping[str, Any]) -> str:
    batch_size = params["batch_size"]
    template = canonical_json(params)
    return template.replace(f'"batch_size":{batch_size}', '"batch_size":__BATCH_SIZE__')


def write_typefusion_script(row: Mapping[str, Any], audit_date: str) -> Path:
    task = str(row["task"])
    script_path = ROOT / "scripts/multivariate_detection/detect_score" / f"{task}_script/TypeFusionCATCH.sh"
    if str(row["source_conflict"]).strip().lower() == "true":
        return script_path
    params = json.loads(str(row["typefusion_hyper_params_json"]))
    data_name = str(row["data_name"])
    config_path = str(row["config_path"])
    gecco_fairness_override = str(row["gecco_fairness_override"]).strip().lower() == "true"
    body = f'''#!/usr/bin/env bash
set -euo pipefail

# Source CATCH official_CATCH.sh: {row["official_source_script"]}
# Source CATCH test report: {row["baseline_report"]}
# Source CATCH archive commit: {row["source_catch_master_commit"]}
# Configuration audit date (UTC): {audit_date}
# GECCO fairness override: {str(gecco_fairness_override).lower()}
# TypeFusion compatibility override: {row["typefusion_compatibility_override"] or 'none'}

ROOT_DIR="$(
  cd "$(dirname "${{BASH_SOURCE[0]}}")/../../../.." || exit 1
  pwd
)"
cd "$ROOT_DIR"

GPU_ID="${{GPU_ID:-0}}"
BATCH_SIZE="${{BATCH_SIZE:-{int(params["batch_size"])}}}"
SAVE_PATH="${{TYPEFUSION_SAVE_PATH:-score/TypeFusion-CATCH/{task}/run-$(date -u +%Y%m%dT%H%M%SZ)-$$}}"
MODEL_HYPER_PARAMS='{json_template(params)}'
MODEL_HYPER_PARAMS="${{MODEL_HYPER_PARAMS/__BATCH_SIZE__/${{BATCH_SIZE}}}}"
RUN_CONFIG_PATH="$ROOT_DIR/result/$SAVE_PATH/typefusion_run_config.json"
mkdir -p "$(dirname "$RUN_CONFIG_PATH")"
printf '{{"data_name":"{data_name}","model_name":"typefusion_catch.TypeFusionCATCH","seed":2021,"model_hyper_params":%s}}\\n' "$MODEL_HYPER_PARAMS" > "$RUN_CONFIG_PATH"
export TYPEFUSION_RUN_CONFIG_PATH="$RUN_CONFIG_PATH"

exec python ./scripts/run_benchmark.py --config-path "{config_path}" --data-name-list "{data_name}" --model-name "typefusion_catch.TypeFusionCATCH" --model-hyper-params "$MODEL_HYPER_PARAMS" --seed 2021 --gpus "$GPU_ID" --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH"
'''
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(body, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script_path


def write_gecco_fair_script(row: Mapping[str, Any], audit_date: str) -> Path:
    """Prepare but never execute the required seq_len=192 CATCH paired baseline."""

    script_path = ROOT / "scripts/multivariate_detection/detect_score/GECCO_script/CATCH_GECCO_FAIR.sh"
    params = json.loads(str(row["catch_model_hyper_params_json"]))
    params["seq_len"] = 192
    body = f'''#!/usr/bin/env bash
set -euo pipefail

# Prepared only: paired CATCH baseline for TypeFusion GECCO seq_len=192 fairness.
# Source CATCH official_CATCH.sh: {row["official_source_script"]}
# Source CATCH archive commit: {row["source_catch_master_commit"]}
# Configuration audit date (UTC): {audit_date}

ROOT_DIR="$(
  cd "$(dirname "${{BASH_SOURCE[0]}}")/../../../.." || exit 1
  pwd
)"
cd "$ROOT_DIR"

GPU_ID="${{GPU_ID:-0}}"
BATCH_SIZE="${{BATCH_SIZE:-{int(params["batch_size"])}}}"
SAVE_PATH="${{CATCH_SAVE_PATH:-score/CATCH/GECCO_fair_seq192/run-$(date -u +%Y%m%dT%H%M%SZ)-$$}}"
MODEL_HYPER_PARAMS='{json_template(params)}'
MODEL_HYPER_PARAMS="${{MODEL_HYPER_PARAMS/__BATCH_SIZE__/${{BATCH_SIZE}}}}"

exec python ./scripts/run_benchmark.py --config-path "{row["config_path"]}" --data-name-list "{row["data_name"]}" --model-name "catch.CATCH" --model-hyper-params "$MODEL_HYPER_PARAMS" --seed 2021 --gpus "$GPU_ID" --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH"
'''
    script_path.write_text(body, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script_path


def write_commands_document(rows: List[Mapping[str, Any]]) -> None:
    groups = (
        ("ASD", ASD_TASKS),
        ("Industrial / Cyber-Physical", ("CICIDS", "SWAT", "GECCO", "Genesis")),
        ("General Real Datasets", ("CalIt2", "Creditcard", "MSL", "NYC", "PSM", "SMAP", "SMD")),
    )
    lines = [
        "# TypeFusion-CATCH Real-Task Manual Commands",
        "",
        "These 23 single-task commands are prepared only and have not been run.",
        "Run PSM first; after it is valid, run subsequent tasks manually one at a time.",
        "Do not start all tasks in the background. Set `GPU_ID` explicitly when needed.",
        "After a real CUDA OOM, set `BATCH_SIZE` manually to an allowed halved value,",
        "inspect the completed result before proceeding, and prepare the paired CATCH rerun",
        "before any comparison. The PSM command below can be manually prefixed with",
        "`BATCH_SIZE=64` only after an actual OOM; no script retries or changes it automatically.",
        "",
    ]
    for title, tasks in groups:
        lines.extend((f"## {title}", ""))
        for task in tasks:
            lines.append(
                f"GPU_ID=0 sh ./scripts/multivariate_detection/detect_score/{task}_script/TypeFusionCATCH.sh"
            )
        lines.append("")
    (ROOT / "TYPEFUSION_CATCH_ALL_REAL_COMMANDS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catch-root", type=Path, default=DEFAULT_CATCH_ROOT)
    parser.add_argument(
        "--reuse-data-audit",
        action="store_true",
        help="reuse the existing 23-row data registry after a completed hash audit",
    )
    args = parser.parse_args()
    catch_root = args.catch_root.resolve(strict=True)
    catch_commit = git_commit(catch_root)
    audit_date = datetime.now(timezone.utc).date().isoformat()
    data_root, data_link_status = ensure_shared_data(catch_root)

    source_rows = load_source_registry(catch_root)
    data_metadata = pd.read_csv(data_root / "DETECT_META.csv").set_index("file_name", drop=False)
    data_registry_path = RESULT_ROOT / "typefusion_real_data_registry.csv"
    data_rows: List[Dict[str, Any]] = []
    source_by_task = {str(row["task"]): row for row in source_rows}
    if args.reuse_data_audit:
        if not data_registry_path.is_file():
            raise FileNotFoundError("--reuse-data-audit requires an existing data registry")
        data_rows = pd.read_csv(data_registry_path).to_dict(orient="records")
        if tuple(str(row["task"]) for row in data_rows) != TASKS:
            raise ValueError("existing data registry does not contain the fixed ordered 23 tasks")
    else:
        for task in TASKS:
            source_row = source_by_task[task]
            dataset_file = str(source_row["dataset_file"])
            data_row = {
                "task": task,
                "paper_dataset": PAPER_DATASET[task],
                "dataset_file": dataset_file,
                "data_link_status": data_link_status,
                **inspect_data(
                    data_root / "data" / dataset_file,
                    data_metadata,
                    dataset_file,
                    int(json.loads(source_row["typefusion_hyper_params_json"])["seq_len"]),
                ),
            }
            data_rows.append(data_row)

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(data_registry_path, data_rows)
    write_csv(RESULT_ROOT / "typefusion_catch_source_registry.csv", source_rows)
    for row in source_rows:
        if bool(row["source_complete"]) and not bool(row["source_conflict"]):
            write_typefusion_script(row, audit_date)
    gecco = source_by_task["GECCO"]
    if bool(gecco["baseline_requires_fair_rerun"]):
        write_gecco_fair_script(gecco, audit_date)
    write_commands_document(source_rows)

    unresolved = [row["task"] for row in source_rows if bool(row["source_conflict"])]
    print(f"catch_commit={catch_commit}")
    print(f"data_root={data_root}")
    print(f"prepared_tasks={len(source_rows) - len(unresolved)}")
    print(f"source_conflicts={','.join(unresolved) if unresolved else 'none'}")


if __name__ == "__main__":
    main()
