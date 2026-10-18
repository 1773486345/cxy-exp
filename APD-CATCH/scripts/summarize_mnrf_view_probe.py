#!/usr/bin/env python
"""Read-only CSV/Markdown summary for independent MNRF view-probe runs."""

import argparse
import json
from pathlib import Path

import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.mnrf_view_probe import load_probe
from ts_benchmark.baselines.mnrf_view_probe.probe import ARTIFACT_RESULTS, _validate_result_bundle


def _metadata_shape_error(manifest):
    """Return a precise structural error before any branch-specific access."""
    config = manifest.get("config", {})
    if not isinstance(config, dict):
        return "probe_manifest.json: config must be an object"
    layout = manifest.get("layout")
    if "layout" in manifest and not isinstance(layout, dict):
        return "probe_manifest.json: layout must be an object"
    retention = manifest.get("retention_after")
    if "retention_after" in manifest and not isinstance(retention, dict):
        return "probe_manifest.json: retention_after must be an object"
    if "selected_views" in manifest:
        selected = manifest["selected_views"]
    elif "selected_views" in config:
        selected = config["selected_views"]
    else:
        selected = None
    if selected is not None:
        if isinstance(selected, (str, bytes)) or not isinstance(selected, (list, tuple)):
            return "probe_manifest.json: selected_views must be a sequence of strings"
        if any(not isinstance(value, str) for value in selected):
            return "probe_manifest.json: selected_views contains a non-string value"
    elif "selected_views" in manifest or "selected_views" in config:
        return "probe_manifest.json: selected_views must be a sequence of strings"
    return None


def collect(root):
    rows = []
    for manifest_path in sorted(Path(root).glob("**/probe_manifest.json")):
        run = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as error:
            rows.append({"run": str(run), "validation_status": "invalid",
                         "validation_reason": f"invalid probe_manifest.json: {error}"})
            continue
        if not isinstance(manifest, dict):
            rows.append({"run": str(run), "validation_status": "invalid",
                         "validation_reason": "probe_manifest.json root must be an object"})
            continue
        status_path = run / "run_status.json"
        try:
            run_status = json.loads(status_path.read_text()) if status_path.is_file() else {"status": "missing_status"}
        except Exception as error:
            run_status = {"status": "invalid_status", "reason": str(error)}
        if not isinstance(run_status, dict):
            run_status = {"status": "invalid_status", "reason": "run_status.json root must be an object"}
        shape_error = _metadata_shape_error(manifest)
        if shape_error is not None:
            rows.append({"run": str(run), "task": manifest.get("task_name"),
                         "run_status": run_status.get("status"), "validation_status": "invalid",
                         "validation_reason": shape_error})
            continue
        config = manifest.get("config", {}) or {}
        result_path = run / "results" / "metrics.json"
        if manifest.get("artifact_type") == ARTIFACT_RESULTS or manifest.get("artifact_policy") == "results":
            validation_status = "invalid"
            validation_reason = ""
            metrics = {}
            try:
                checked = _validate_result_bundle(
                    run, allow_in_progress=manifest.get("result_cleanup_status") != "complete"
                )
                validation_status = ("results_valid" if checked.get("cleanup_status") == "complete"
                                     else "results_valid_cleanup_failed")
                metrics = checked["metrics"]
            except Exception as error:
                validation_reason = str(error)
            rows.append({
                "run": str(run), "task": manifest.get("task_name", config.get("task_name")),
                "seed": config.get("seed"), "seq_len": config.get("seq_len"),
                "mode": manifest.get("mode"), "head": metrics.get("head"),
                "selected_views": ",".join(manifest.get("selected_views", config.get("selected_views", []))),
                "selected_dim": manifest.get("selected_dim", config.get("selected_dim")),
                "layout_F": manifest.get("layout", {}).get("F"),
                "run_status": run_status.get("status"), "artifact_type": ARTIFACT_RESULTS,
                "artifact_policy": "results", "validation_status": validation_status,
                "validation_reason": validation_reason, "status": metrics.get("status", run_status.get("status")),
                "failure_reason": run_status.get("reason"), "auc_roc": metrics.get("auc_roc"),
                "auc_pr": metrics.get("auc_pr"), "threshold": metrics.get("threshold"),
                "valid_points": metrics.get("valid_points"), "coverage": metrics.get("coverage"),
                "excluded_points": metrics.get("excluded_points"),
                "fit_time_seconds": metrics.get("fit_time_seconds"),
                "inference_time_seconds": metrics.get("inference_time_seconds"),
                "result_cleanup_status": manifest.get("result_cleanup_status"),
                "cleanup_error": manifest.get("cleanup_error"),
                "retention_after_bytes": manifest.get("retention_after", {}).get("bytes"),
                "retention_released_bytes": manifest.get("retention_released_bytes"),
            })
            continue
        if not result_path.is_file():
            completed_stage1 = (run_status.get("status") == "stage1_complete"
                                and manifest.get("artifact_type") == "source_encoder"
                                and manifest.get("stage_status") == "stage1_complete")
            validation_status = "valid" if completed_stage1 else "not_applicable"
            validation_reason = ""
            if completed_stage1:
                try:
                    load_probe(run, device="cpu")
                except Exception as error:
                    validation_status = "invalid"
                    validation_reason = str(error)
            rows.append({"run": str(run), "task": config.get("task_name"),
                         "mode": manifest.get("mode"), "run_status": run_status.get("status"),
                         "artifact_type": manifest.get("artifact_type"),
                         "artifact_policy": manifest.get("artifact_policy", "full"),
                         "validation_status": validation_status, "validation_reason": validation_reason,
                         "status": "stage1_complete" if validation_status == "valid" else "missing_metrics"})
            continue
        try:
            metrics = json.loads(result_path.read_text())
        except Exception as error:
            rows.append({"run": str(run), "task": config.get("task_name") if isinstance(config, dict) else None,
                         "run_status": run_status.get("status"), "validation_status": "invalid",
                         "validation_reason": f"invalid metrics.json: {error}"})
            continue
        if not isinstance(metrics, dict):
            rows.append({"run": str(run), "task": None, "run_status": run_status.get("status"),
                         "validation_status": "invalid",
                         "validation_reason": "metrics.json root must be an object"})
            continue
        gmm_status_path = run / "heads" / "gmm_status.json"
        gmm_status = {}
        if gmm_status_path.is_file():
            try:
                gmm_status = json.loads(gmm_status_path.read_text())
                if not isinstance(gmm_status, dict):
                    raise ValueError("gmm_status.json root must be an object")
            except Exception as error:
                rows.append({"run": str(run), "task": config.get("task_name") if isinstance(config, dict) else None,
                             "run_status": run_status.get("status"), "validation_status": "invalid",
                             "validation_reason": f"invalid gmm_status.json: {error}"})
                continue
        validation_status = "not_applicable"
        validation_reason = ""
        if run_status.get("status") in {"complete", "stage1_complete"}:
            try:
                load_probe(run, device="cpu")
                validation_status = "valid"
            except Exception as error:
                validation_status = "invalid"
                validation_reason = str(error)
        rows.append({
            "run": str(run), "task": config.get("task_name"), "seed": config.get("seed"),
            "seq_len": config.get("seq_len"), "mode": manifest.get("mode"),
            "head": metrics.get("head"),
            "selected_views": ",".join(manifest.get("selected_views", config.get("selected_views", []))),
            "selected_dim": manifest.get("selected_dim", manifest.get("config", {}).get("selected_dim")),
            "layout_F": manifest.get("layout", {}).get("F"),
            "source_feature_fingerprint": manifest.get("source_feature_fingerprint", manifest.get("feature_fingerprint")),
            "gmm_components": config.get("gmm_components"), "gmm_covariance_type": config.get("gmm_covariance_type"),
            "gmm_fit_endpoint_count": metrics.get("gmm_fit_endpoint_count", gmm_status.get("fit_endpoint_count")),
            "run_status": run_status.get("status"), "head_status": manifest.get("head_status"),
            "artifact_type": manifest.get("artifact_type"), "head_type": manifest.get("head_type"),
            "artifact_policy": manifest.get("artifact_policy", "full"),
            "validation_status": validation_status, "validation_reason": validation_reason,
            "status": metrics.get("status"), "failure_reason": run_status.get("reason"),
            "auc_roc": metrics.get("auc_roc"), "auc_pr": metrics.get("auc_pr"),
            "threshold": metrics.get("threshold"), "valid_points": metrics.get("valid_points"),
            "head_parameter_count": metrics.get("head_parameter_count"),
            "fit_time_seconds": metrics.get("fit_time_seconds"),
            "inference_time_seconds": metrics.get("inference_time_seconds"),
            "source_identity": manifest.get("source_identity"),
            "protocol": manifest.get("temporal_rule"),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="result/score/MNRF-ViewProbe")
    parser.add_argument("--output", default="result/score/MNRF-ViewProbe/summary/mnrf_view_probe_summary.csv")
    args = parser.parse_args()
    frame = collect(args.root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    markdown_path = output.with_suffix(".md")
    columns = list(frame.columns)
    with open(markdown_path, "w", encoding="utf-8") as handle:
        if columns:
            handle.write("| " + " | ".join(columns) + " |\n")
            handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
            for row in frame.itertuples(index=False, name=None):
                handle.write("| " + " | ".join("" if value is None else str(value) for value in row) + " |\n")
        else:
            handle.write("No probe runs found.\n")
    print(f"summarized {len(frame)} probe runs to {output}")


if __name__ == "__main__":
    main()
