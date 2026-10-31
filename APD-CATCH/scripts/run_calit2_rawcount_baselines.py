#!/usr/bin/env python
"""Run the four CalIt2 RawCount methods on one isolated common contract."""
import argparse
import json
import time
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.export_mnrf_view_probe_common import export as export_contract
from scripts.run_mnrf_common_baselines import (
    _full_axis_metrics, _method_row, _read_contract, _verify_frame_for_contract,
    _fit_light_method, _sha256, _load_json,
)
from scripts.run_mnrf_catch_common import run as run_catch_common
from scripts.register_mnrf_catch_common import register as register_catch_common


def _catch_args(config, data_csv, contract, output):
    import argparse
    def g(name, default=None):
        return config[name] if name in config else default
    return argparse.Namespace(
        data_csv=str(data_csv), train_indices=contract["catch_splits"]["train"]["path"],
        validation_indices=contract["catch_splits"]["validation"]["path"],
        calibration_indices=contract["catch_splits"]["calibration"]["path"],
        train_rows=contract["train_rows"]["path"], output=str(output),
        common_contract=str(contract["common_contract_path"]), task_name="CalIt2", test_start=None,
        device="cpu", seed=2021, seq_len=int(g("seq_len", 192)),
        patch_size=int(g("patch_size", 16)), patch_stride=int(g("patch_stride", 16)),
        d_model=int(g("d_model", 128)), cf_dim=int(g("cf_dim", 128)),
        d_ff=int(g("d_ff", 128)), e_layers=int(g("e_layers", 2)),
        n_heads=int(g("n_heads", 8)), head_dim=int(g("head_dim", 32)),
        dropout=float(g("dropout", 0.2)), learning_rate=float(g("lr", 0.0001)),
        epochs=int(g("num_epochs", 10)), batch_size=int(g("batch_size", 128)),
        patience=int(g("patience", 3)), calibration_quantile=0.995,
        score_lambda=float(g("score_lambda", 0.5)), mask_learning_rate=float(g("Mlr", 1e-5)),
        auxi_lambda=float(g("auxi_lambda", 0.1)), dc_lambda=float(g("dc_lambda", 0.5)),
        temperature=float(g("temperature", 0.07)), head_dropout=float(g("head_dropout", 0.1)),
        anomaly_ratio=g("anomaly_ratio", [0.1, 0.5, 1, 2, 3, 5, 10, 15, 20, 25]),
        use_multi_scale=g("use_multi_scale", "False"), small_kernel_merged=g("small_kernel_merged", "False"),
        itr=int(g("itr", 1)), inference_patch_size=int(g("inference_patch_size", 16)),
        inference_patch_stride=int(g("inference_patch_stride", 1)), pct_start=float(g("pct_start", 0.3)),
        model_config=dict(config),
    )


def run(source_dir, data_root, output_dir):
    source_dir = Path(source_dir).resolve(); data_root = Path(data_root).resolve(); output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    data_csv = data_root / "data" / "CalIt2.csv"
    contract_dir = output_dir / "common_contract"
    contract = export_contract(source_dir, data_csv, contract_dir, catch_seq_len=192)
    contract["common_contract_path"] = str((contract_dir / "common_contract.json").resolve())
    # export() writes the contract before returning; bind the path for the
    # shared baseline helper without changing the source run.
    contract_path = contract_dir / "common_contract.json"
    contract = _read_contract(contract_path)
    contract["common_contract_path"] = str(contract_path.resolve())
    frame, values, labels, times = _verify_frame_for_contract("CalIt2", data_csv, contract)
    source_manifest = _load_json(source_dir / "probe_manifest.json")
    identity = source_manifest["source_identity"]
    test_indices = np.load(contract["catch_test"]["path"], allow_pickle=False)
    boundary_ids = None
    test_segments = identity.get("test_segments", [])
    rows = []
    # MNRF RawCount Full is the separately trained reference, not a rerun here.
    with np.load(source_dir / "results" / "aligned_scores.npz", allow_pickle=False) as scored:
        rows.append(_method_row("CalIt2", "MNRF Full RawCount", source_dir.name,
                                _full_axis_metrics(labels, test_indices, scored["scores"],
                                                   float(np.asarray(scored["threshold"])), test_segments, boundary_ids),
                                protocol="CalIt2 RawCount common protocol"))
    config = json.loads((ROOT / "result/analysis/MNRF-ViewProbe-Common/common-baseline-20261029T-newrun/archived_score_configs.json").read_text())["CalIt2"]["resolved_common_run_model_config"]
    catch_output = output_dir / "CATCH-common"
    started = time.perf_counter()
    catch_run = run_catch_common(_catch_args(config, data_csv, contract, catch_output))
    catch_seconds = time.perf_counter() - started
    catch_score = catch_output / "results" / "aligned_scores.npz"
    common_manifest = catch_output / "results" / "common_manifest.json"
    register_catch_common(__import__("argparse").Namespace(
        probe=str(source_dir / "results" / "aligned_scores.npz"), catch=str(catch_score),
        labels=contract["labels"]["path"], train_indices=contract["catch_splits"]["train"]["path"],
        validation_indices=contract["catch_splits"]["validation"]["path"], calibration_indices=contract["catch_splits"]["calibration"]["path"],
        training_evidence=str(catch_output / "evidence" / "training.json"), calibration_evidence=str(catch_output / "evidence" / "calibration.json"),
        task_name="CalIt2", output=str(common_manifest)))
    with np.load(catch_score, allow_pickle=False) as scored:
        rows.append(_method_row("CalIt2", "CATCH RawCount", source_dir.name,
                                _full_axis_metrics(labels, test_indices, scored["scores"], float(np.asarray(scored["threshold"])), test_segments, boundary_ids),
                                fit_seconds=catch_seconds, protocol="CalIt2 RawCount common protocol"))
    for method in ("IsolationForest", "PCA"):
        path, evidence = _fit_light_method(method, output_dir / method, values, labels, times, contract)
        with np.load(path, allow_pickle=False) as scored:
            rows.append(_method_row("CalIt2", method + " RawCount", source_dir.name,
                                    _full_axis_metrics(labels, test_indices, scored["scores"], float(np.asarray(scored["threshold"])), test_segments, boundary_ids),
                                    fit_seconds=evidence["fit_seconds"], inference_seconds=evidence["inference_seconds"],
                                    protocol="CalIt2 RawCount common protocol"))
    (output_dir / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True); parser.add_argument("--data-root", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); print(json.dumps(run(args.source_dir, args.data_root, args.output_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
