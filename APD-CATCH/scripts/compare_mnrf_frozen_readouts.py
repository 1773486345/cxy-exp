#!/usr/bin/env python
"""Compare three native MNRF-Lite V2 readouts without model forward or fit.

The input must be one completed *native Flow* benchmark run. ViewProbe GMM
or auxiliary decoder artifacts are deliberately not accepted as Flow scores.
S is the current input-standardized observation, not a learned latent. Thus
test S can be read from the bound CSV without running the encoder.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits, threadpool_info

from scripts.analyze_mnrf_r_residual_readout import budget_point, stats, sha, check, strict_axis, deciles
from scripts.run_mnrf_view_probe import _read_task
from scripts.validate_mnrf_lite_v2_run import main as validate_run
from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2, _hash_array
from ts_benchmark.baselines.mnrf_lite_v2.layout import FeatureLayout
from ts_benchmark.baselines.mnrf_lite_v2.readout import (
    SCORE_READOUTS, flow_scores, fit_amplitude_calibration, combine_scores,
)
from ts_benchmark.baselines.mnrf_lite_v2.windows import continuous_segments, window_endpoints

RTOL, ATOL = 1e-5, 1e-6


def agree(actual, expected, name):
    check(np.shape(actual) == np.shape(expected) and np.isfinite(actual).all()
          and np.isfinite(expected).all()
          and np.allclose(actual, expected, rtol=RTOL, atol=ATOL), name + " mismatch")


def compare(run_dir, task, frame, train_length, output_dir):
    """Read-only source, one native validation per call, exclusive new output."""
    check(not os.path.lexists(output_dir), "output already exists; refusing overwrite")
    root, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    check(root != output and root not in output.parents and output not in root.parents,
          "source and output must be separate non-overlapping directories")
    check(not os.path.lexists(output), "output already exists; refusing overwrite")
    # This validator loads/checks packages and caches, but never runs a model.
    validate_run(str(root), task)
    metadata = json.loads((root / "run_metadata.json").read_text())
    identity = json.loads((root / "run_identity.json").read_text())
    context = json.loads((root / "benchmark_context.json").read_text())
    config = metadata["config"]
    check(config["seed"] == 2021, "only the fixed seed=2021 protocol is supported")
    check(float(config["calibration_quantile"]) == .995, "expected calibration q=.995")
    check(metadata.get("score_readout", "flow") == "flow", "saved test scores must be Flow readout")
    check(identity["config"]["seq_len"] == config["seq_len"]
          and identity["variable_names"] == metadata["variable_names"], "run/config identity mismatch")
    layout = FeatureLayout.from_dict(metadata["feature_layout"])
    state_index = layout.slices["S"][0]
    features = frame.loc[:, frame.columns != "label"]
    values, columns = MNRFLiteV2._features_from_frame(features)
    time = MNRFLiteV2._time_index(features)
    full_labels = frame["label"].to_numpy()
    check(full_labels.ndim == 1 and full_labels.dtype.kind in "iufb"
          and np.isfinite(full_labels).all() and np.isin(full_labels, [0, 1]).all(),
          "full labels must be exact finite binary values")
    check(list(map(str, columns)) == metadata["variable_names"], "input field order mismatch")
    check(context["train_length"] == train_length and context["test_length"] == len(frame)-train_length,
          "official train/test split mismatch")
    check(_hash_array(values[:train_length]) == identity["input_sha256"], "training input identity mismatch")
    check(_hash_array(MNRFLiteV2._time_index(features.iloc[:train_length])) == identity["time_index_sha256"],
          "training time identity mismatch")
    # Old benchmark metadata binds training bytes, but not test observations.
    # Never infer the latter from matching lengths/labels. New launch captures
    # this small contract from the input actually supplied to that invocation.
    contract_path = root / "readout_data_contract.json"
    check(contract_path.is_file(), "missing same-run full-input binding: " + str(contract_path))
    contract = json.loads(contract_path.read_text())
    check(contract["source_run_id"] == root.name and contract["task"] == task
          and contract["train_length"] == train_length
          and contract["variable_names"] == list(map(str, columns)), "data contract run/task mismatch")
    check(contract["full_input_sha256"] == _hash_array(values)
          and contract["full_time_sha256"] == _hash_array(time)
          and contract["full_labels_sha256"] == _hash_array(full_labels.astype(np.int8)),
          "full input/time/label identity mismatch")
    files = [root / p for p in ("run_metadata.json", "run_identity.json", "cache_manifest.json",
                              "stage2_binding.json", "input_statistics.npz", "deployment.pt",
                              "benchmark_context.json")]
    score_path = root / "results" / (task + ".csv_aligned_scores.npz")
    files.append(score_path)
    files.append(contract_path)
    for split in ("validation", "calibration"):
        files.extend([Path(metadata["feature_cache_paths"][split]),
                      Path(metadata["feature_cache_window_end_indices"][split])])
        if (root / "cache" / (split + "_nll.npy")).is_file():
            files.append(root / "cache" / (split + "_nll.npy"))
    hashes = {str(p): sha(p) for p in files}
    with np.load(root / "input_statistics.npz", allow_pickle=False) as z:
        norm_mean, norm_scale = z["mean"], z["scale"]
        center, scale = z["calibration_center"], z["calibration_scale"]
        flow_threshold = float(z["threshold"])
    components, missing = {}, {}
    for split in ("validation", "calibration"):
        raw_path = Path(metadata["feature_cache_paths"][split])
        endpoints_path = Path(metadata["feature_cache_window_end_indices"][split])
        nll_path = root / "cache" / (split + "_nll.npy")
        raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        nll = np.load(nll_path, mmap_mode="r", allow_pickle=False) if nll_path.is_file() else None
        indices = strict_axis(np.load(endpoints_path, mmap_mode="r"))
        if nll is not None:
            check(nll.shape == raw.shape[:2] and nll.dtype == np.float32, split + " NLL shape/dtype mismatch")
        else:
            missing[split+"_flow_component"] = str(nll_path)
        component = np.empty((len(raw), 2), dtype=np.float32)
        component[:, 0] = np.nan
        for start in range(0, len(raw), 4096):
            block = slice(start, start+4096)
            check(nll is None or np.isfinite(nll[block]).all(), split + " nonfinite NLL")
            # Cross-check S against original normalized observations at these endpoints.
            expected_s = (values[indices[block]] - norm_mean) / norm_scale
            agree(raw[block, :, state_index], expected_s, split + " raw S/input binding")
            if nll is not None:
                component[block, 0] = flow_scores(nll[block], center, scale)
            component[block, 1] = np.abs(raw[block, :, state_index]).max(axis=1)
        components[split] = component
        if split == "validation" and nll is not None:
            # Match the original float32 median/MAD implementation, variable by variable.
            replay_center = np.asarray([np.median(nll[:, d]) for d in range(nll.shape[1])], dtype=np.float32)
            mad = np.asarray([np.median(np.abs(nll[:, d]-replay_center[d])) for d in range(nll.shape[1])], dtype=np.float32)
            agree(replay_center, center, "Flow validation center")
            agree(np.maximum(1.4826*mad, config["score_mad_floor"]), scale, "Flow validation scale")
    if "calibration_flow_component" not in missing:
        agree(np.asarray(np.quantile(components["calibration"][:, 0], .995)),
              np.asarray(flow_threshold), "Flow saved threshold replay")
    amplitude_state = None
    if not missing:
        amplitude_state = fit_amplitude_calibration(components["validation"], components["calibration"],
                                                   .995, config["score_mad_floor"])
    # New checkpoints already bind this state; old Flow-only checkpoints need no mutation.
    deployment = torch.load(root / "deployment.pt", map_location="cpu", weights_only=False)
    if "amplitude_calibration" in deployment:
        from ts_benchmark.baselines.mnrf_lite_v2.readout import validate_amplitude_calibration
        saved_state = validate_amplitude_calibration(deployment["amplitude_calibration"])
        if amplitude_state is not None:
            for key in ("center", "scale"):
                agree(np.asarray(amplitude_state[key]), np.asarray(saved_state[key]), key)
            for name in SCORE_READOUTS[1:]:
                agree(np.asarray(amplitude_state["thresholds"][name]), np.asarray(saved_state["thresholds"][name]), name+" threshold")
        amplitude_state = saved_state
    with np.load(score_path, allow_pickle=False) as pack:
        saved = {key: pack[key] for key in ("scores", "predictions", "time_indices", "local_time_indices",
                 "time_index_values", "local_valid_mask", "valid_mask", "actual", "threshold")}
    local = strict_axis(saved["local_time_indices"])
    global_indices = strict_axis(saved["time_indices"])
    check(np.array_equal(global_indices, local+train_length), "global/local coordinate mismatch")
    check(saved["local_valid_mask"].dtype == bool and saved["local_valid_mask"].shape == (len(frame)-train_length,)
          and np.array_equal(np.flatnonzero(saved["local_valid_mask"]), local)
          and saved["valid_mask"].dtype == bool and saved["valid_mask"].shape == (len(frame),)
          and np.array_equal(np.flatnonzero(saved["valid_mask"]), global_indices), "saved mask mapping mismatch")
    check(saved["scores"].shape == local.shape and np.isfinite(saved["scores"]).all(), "invalid saved Flow scores")
    with np.load(score_path, allow_pickle=False) as pack:
        check("score_readout" not in pack or str(pack["score_readout"]) == "flow", "test artifact is not Flow")
        if "per_variable_nll" in pack:
            agree(flow_scores(pack["per_variable_nll"], center, scale), saved["scores"], "test Flow NLL reaggregation")
    labels = frame["label"].to_numpy()[train_length:]
    check(labels.ndim == 1 and labels.dtype.kind in "iufb" and np.isfinite(labels).all()
          and np.isin(labels, [0, 1]).all() and np.array_equal(labels, saved["actual"]), "full test labels mismatch")
    test_frame = features.iloc[train_length:]
    rule = identity["temporal_rule"]
    boundaries = MNRFLiteV2._boundary_ids(test_frame, rule["entity_boundary_columns"])
    test_segments, _ = continuous_segments(MNRFLiteV2._time_index(test_frame), len(test_frame),
        expected_interval=rule["expected_interval_ns"], tolerance=rule["tolerance"], boundary_ids=boundaries)
    check(np.array_equal(local, window_endpoints(test_segments, int(config["seq_len"]))), "valid endpoints differ from frozen windows")
    check(np.array_equal(saved["time_index_values"], time.values_ns[global_indices]), "test timestamps mismatch")
    agree(np.asarray(flow_threshold), saved["threshold"], "saved test Flow threshold")
    check(np.array_equal(saved["predictions"], saved["scores"] > flow_threshold), "saved Flow predictions mismatch")
    test_s = ((values[global_indices]-norm_mean)/norm_scale)
    amplitude = np.abs(test_s).max(axis=1).astype(np.float32)
    all_scores = {"flow": saved["scores"], "state_amplitude": amplitude}
    thresholds = {"flow": flow_threshold, "state_amplitude": float(np.quantile(components["calibration"][:, 1], .995))}
    if amplitude_state is not None:
        agree(np.asarray(thresholds["state_amplitude"]), np.asarray(amplitude_state["thresholds"]["state_amplitude"]), "S threshold")
        thresholds.update(amplitude_state["thresholds"])
        all_scores["flow_plus_amplitude"] = combine_scores(saved["scores"], amplitude, amplitude_state).astype(np.float32)
    selected_labels = labels[local].astype(np.int8)
    range_strata = {"all": np.ones(len(local), dtype=bool)}
    range_limits = {}
    if task == "PSM" and {"feature_1", "feature_3"}.issubset(columns):
        out_of_range = np.zeros(len(local), dtype=bool)
        for field in ("feature_1", "feature_3"):
            d = list(columns).index(field)
            limit = float(np.max((values[:train_length, d]-norm_mean[d])/norm_scale[d]))
            range_limits[field] = limit
            out_of_range |= test_s[:, d] > limit
        range_strata.update(outside_official_train_max=out_of_range,
                            within_official_train_max=~out_of_range)
    rows = []
    cap = int(np.sum(selected_labels == 0)) // 100
    predictions = {}
    for name in all_scores:
        predictions[name] = all_scores[name] > thresholds[name]
    flow_budget, flow_prediction = budget_point(all_scores["flow"], selected_labels, cap)
    for name in all_scores:
        for scope in ("saved_q995", "posthoc_fpr_le_1pct"):
            threshold, pred = ((thresholds[name], predictions[name]) if scope == "saved_q995"
                               else budget_point(all_scores[name], selected_labels, cap))
            base = predictions["flow"] if scope == "saved_q995" else flow_prediction
            groups, _ = deciles(np.arange(train_length, len(frame)), global_indices)
            for segment in ([0]+list(range(1, 11)) if task == "PSM" else [0]):
                section = np.ones(len(pred), dtype=bool) if segment == 0 else groups == segment
                for stratum, range_mask in range_strata.items():
                    mask = section & range_mask
                    row = dict(dataset=task, source_run_id=root.name, readout=name, scope=scope,
                               official_decile=segment, range_stratum=stratum,
                               threshold=threshold, fp_cap=cap if scope != "saved_q995" else None,
                               calibration_windows=len(components["calibration"]),
                               threshold_source="original saved Flow" if name == "flow" else "same-run independent calibration q=.995",
                               **stats(selected_labels[mask], pred[mask], all_scores[name][mask]))
                    row.update(added_tp=int(np.sum(mask & (selected_labels == 1) & pred & ~base)),
                               lost_tp=int(np.sum(mask & (selected_labels == 1) & ~pred & base)),
                               added_fp=int(np.sum(mask & (selected_labels == 0) & pred & ~base)),
                               removed_fp=int(np.sum(mask & (selected_labels == 0) & ~pred & base)))
                    rows.append(row)
    if amplitude_state is None:
        rows.append(dict(dataset=task, source_run_id=root.name, readout="flow_plus_amplitude",
                         scope="not_computable", status="missing_artifacts", reason=json.dumps(missing)))
    # Only exclusive new directories; no files in the source are ever changed.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        with tempfile.TemporaryDirectory(prefix=".readout-", dir=output) as temporary:
            staging = Path(temporary)
            pd.DataFrame(rows).to_csv(staging / "results.csv", index=False)
            np.savez_compressed(staging / "scores.npz", **saved, selected_actual=selected_labels,
                                **{name+"_scores": all_scores[name] for name in all_scores},
                                **{name+"_predictions": predictions[name] for name in all_scores},
                                **{name+"_threshold": np.asarray(thresholds[name]) for name in all_scores})
            manifest = dict(status="complete" if len(all_scores)==3 else "partial", source_path=str(root), source_run_id=root.name, task=task,
                source_resume_id=identity["resume_id"], feature_fingerprint=metadata["feature_fingerprint"],
                variable_names=metadata["variable_names"], feature_layout=layout.to_dict(), seq_len=config["seq_len"],
                calibration_quantile=.995, calibration_windows=len(components["calibration"]),
                threshold_rule="strict score > threshold", score_readouts=list(all_scores), missing_artifacts=missing,
                data_version=contract["data_version"], data_contract_path=str(contract_path),
                full_input_sha256=_hash_array(values), full_time_sha256=_hash_array(time),
                psm_range_stratification_limits=range_limits,
                amplitude_calibration=amplitude_state, source_file_sha256=hashes,
                native_validation="once in this call; no encoder/Flow forward", prediction_scope="same complete-window endpoints",
                cpu_threads=torch.get_num_threads(), numerical_libraries=threadpool_info(),
                exploratory_budget="test-label posthoc whole-tie scan; not a deployment threshold")
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
            for path, digest in hashes.items():
                check(sha(Path(path)) == digest, "source changed during export: "+path)
            for path in sorted(staging.iterdir(), key=lambda p: p.name == "manifest.json"):
                os.link(path, output / path.name)
    except BaseException:
        # Keep any published partial output recognizable; retry must use a new path.
        raise
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-source", choices=("large_detect", "external_detect"), default="large_detect")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    check(args.cpu_threads > 0, "cpu threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    with threadpool_limits(limits=args.cpu_threads):
        frame, train, *_ = _read_task(args.task, args.data_source, args.data_root)
        rows = compare(args.run_dir, args.task, frame, train, args.output_dir)
    status = json.loads((args.output_dir / "manifest.json").read_text())["status"]
    print(json.dumps(dict(status=status, dataset=args.task, results=str(args.output_dir), rows=len(rows))))
    if status != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
