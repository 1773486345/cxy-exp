#!/usr/bin/env python
"""Read-only CSV/Markdown summary for independent MNRF view-probe runs."""

import argparse
import json
from pathlib import Path

import pandas as pd


def collect(root):
    rows = []
    for manifest_path in sorted(Path(root).glob("**/probe_manifest.json")):
        run = manifest_path.parent
        result_path = run / "results" / "metrics.json"
        if not result_path.is_file():
            rows.append({"run": str(run), "status": "missing_metrics"})
            continue
        manifest = json.loads(manifest_path.read_text())
        metrics = json.loads(result_path.read_text())
        config = manifest.get("config", {})
        gmm_status_path = run / "heads" / "gmm_status.json"
        gmm_status = json.loads(gmm_status_path.read_text()) if gmm_status_path.is_file() else {}
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
            "status": metrics.get("status"), "auc_roc": metrics.get("auc_roc"), "auc_pr": metrics.get("auc_pr"),
            "threshold": metrics.get("threshold"), "valid_points": metrics.get("valid_points"),
            "head_parameter_count": metrics.get("head_parameter_count"),
            "fit_time_seconds": metrics.get("fit_time_seconds"),
            "inference_time_seconds": metrics.get("inference_time_seconds"),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="result/score/MNRF-ViewProbe")
    parser.add_argument("--output", default="result/typefusion_catch_main/mnrf_view_probe_summary.csv")
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
