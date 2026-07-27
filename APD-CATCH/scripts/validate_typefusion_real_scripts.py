#!/usr/bin/env python3
"""Read-only CPU preflight for the prepared TypeFusion-CATCH real-task scripts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.typefusion_catch.config import TypeFusionConfig
from ts_benchmark.baselines.typefusion_catch.typefusion_catch import TypeFusionCATCHModel


RESULT_ROOT = ROOT / "result/typefusion_catch_main"
TEMPLATE_PATTERN = re.compile(r"^MODEL_HYPER_PARAMS='([^']+)'$", re.MULTILINE)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def script_params(script_path: Path, batch_size: int) -> Dict[str, Any]:
    text = script_path.read_text(encoding="utf-8")
    match = TEMPLATE_PATTERN.search(text)
    if match is None:
        raise ValueError("MODEL_HYPER_PARAMS template is missing")
    return json.loads(match.group(1).replace("__BATCH_SIZE__", str(batch_size)))


def preflight_task(source: Dict[str, str], data: Dict[str, str]) -> Dict[str, Any]:
    task = source["task"]
    script_path = ROOT / "scripts/multivariate_detection/detect_score" / f"{task}_script/TypeFusionCATCH.sh"
    result: Dict[str, Any] = {
        "task": task,
        "dataset_file": data["dataset_file"],
        "script_path": str(script_path),
        "data_exists": Path(data["resolved_path"]).is_file(),
        "metadata_discoverable": data["metadata_discoverable"],
        "rows": int(data["rows"]),
        "c_in": int(data["feature_count"]),
        "effective_relation_mask_groups": "",
        "bash_syntax_ok": False,
        "json_parse_ok": False,
        "config_validate_ok": False,
        "window_length_ok": False,
        "forward_ok": False,
        "output_shape": "",
        "finite_output": False,
        "status": "failed",
        "error": "",
    }
    try:
        if not script_path.is_file():
            raise FileNotFoundError("missing TypeFusionCATCH.sh")
        bash_check = subprocess.run(["bash", "-n", str(script_path)], check=False)
        result["bash_syntax_ok"] = bash_check.returncode == 0
        if not result["bash_syntax_ok"]:
            raise ValueError("bash -n failed")

        params = script_params(script_path, int(source["original_batch_size"]))
        result["json_parse_ok"] = True
        if params.get("seed") != 2021:
            raise ValueError("model hyper-parameters do not explicitly use seed=2021")
        if params.get("fit_mode") != "three_stage":
            raise ValueError("fit_mode is not three_stage")
        if params.get("training_budget_mode") != "equal_total_steps":
            raise ValueError("training_budget_mode is not equal_total_steps")

        config = TypeFusionConfig.from_kwargs(**params, c_in=result["c_in"])
        config.validate()
        result["config_validate_ok"] = True
        result["window_length_ok"] = result["rows"] > config.seq_len
        if not result["window_length_ok"]:
            raise ValueError("data rows do not exceed seq_len")

        # The actual adapter changes stages during fitting.  Preflight uses the
        # final stage only to exercise the complete joint reconstruction path.
        config.training_stage = "joint_finetune"
        config.validate()
        torch.manual_seed(2021)
        model = TypeFusionCATCHModel(config).eval()
        with torch.no_grad():
            output = model(torch.randn(1, config.seq_len, config.c_in), compute_joint=True)
        x_hat = output["x_hat_joint"]
        result["effective_relation_mask_groups"] = model.relation_branch.max_groups
        if model.relation_branch.max_groups != min(config.relation_mask_groups, config.c_in):
            raise ValueError("relation_mask_groups did not resolve safely")
        if tuple(x_hat.shape) != (1, config.seq_len, config.c_in):
            raise ValueError(f"unexpected joint reconstruction shape {tuple(x_hat.shape)}")
        result["output_shape"] = str(tuple(x_hat.shape))
        result["finite_output"] = bool(
            torch.isfinite(x_hat).all() and torch.isfinite(output["total_score"]).all()
        )
        if not result["finite_output"]:
            raise ValueError("non-finite joint output")
        result["forward_ok"] = True
        result["status"] = "ok"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.set_num_threads(1)
    source_rows = read_csv(RESULT_ROOT / "typefusion_catch_source_registry.csv")
    data_rows = {
        row["task"]: row for row in read_csv(RESULT_ROOT / "typefusion_real_data_registry.csv")
    }
    if len(source_rows) != 23 or set(data_rows) != {row["task"] for row in source_rows}:
        raise ValueError("registries must contain the same fixed 23 real tasks")
    results = [preflight_task(source, data_rows[source["task"]]) for source in source_rows]
    output_path = RESULT_ROOT / "typefusion_script_preflight.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    failures = [row["task"] for row in results if row["status"] != "ok"]
    print(f"preflight_tasks={len(results)}")
    print(f"preflight_failures={','.join(failures) if failures else 'none'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
