"""One fixed, posthoc PSM validation-tail fusion; saved scores only, no calibration."""

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_mnrf_r_residual_readout import budget_point, deciles
from scripts.run_mnrf_view_probe import _read_task
from ts_benchmark.baselines.mnrf_lite_v2.readout import flow_scores

BATCH = ROOT / "result/analysis/MNRF-Readout-Readiness/readout-readiness-20261110T084455Z"
RUN_ID = "run-20261110T105920Z-3229230"
# Locked before inspecting this rule's test results. No alternate fusion is tried.
RULE = "p_j(s)=(1+count(validation_j>=s))/(n+1); score=-log(min(p_flow,p_amplitude))"


def validation_tail(reference, scores):
    reference = np.asarray(reference)
    scores = np.asarray(scores)
    if reference.ndim != 1 or scores.ndim != 1 or not len(reference):
        raise ValueError("tail reference and scores must be nonempty-reference vectors")
    if not np.isfinite(reference).all() or not np.isfinite(scores).all():
        raise ValueError("nonfinite validation reference or scores")
    # left preserves >= and moves every tied validation score together.
    ge = len(reference) - np.searchsorted(np.sort(reference), scores, side="left")
    return (1.0 + ge) / (len(reference) + 1.0)


def tail_fusion(validation_flow, validation_amplitude, flow, amplitude):
    return -np.log(np.minimum(validation_tail(validation_flow, flow),
                              validation_tail(validation_amplitude, amplitude)))


def normal_window_groups(labels, endpoints, window):
    labels = np.asarray(labels)
    endpoints = np.asarray(endpoints)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("invalid full labels")
    if endpoints.ndim != 1 or endpoints.dtype.kind not in "iu" or np.any(endpoints < window-1):
        raise ValueError("invalid complete-window endpoints")
    prefix = np.r_[0, np.cumsum(labels, dtype=np.int64)]
    context_positive = prefix[endpoints+1] - prefix[endpoints-window+1]
    normal = labels[endpoints] == 0
    return {"entire_window_normal": normal & (context_positive == 0),
            "normal_endpoint_anomalous_context": normal & (context_positive > 0)}


def analyze():
    source = json.loads((BATCH / "formal_flow_sources.json").read_text())["runs"]["PSM"]
    assert source["source_run_id"] == RUN_ID
    work, comparison = Path(source["source_path"]), ROOT / source["comparison_path"]
    assert work.name == RUN_ID
    consumed = [comparison / name for name in ("manifest.json", "scores.npz", "results.csv")]
    consumed += [work / name for name in ("run_metadata.json", "input_statistics.npz",
                                        "readout_data_contract.json", "cache/validation_nll.npy",
                                        "cache/validation_window_end_indices.npy")]
    digests = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in consumed}
    metadata = json.loads((work / "run_metadata.json").read_text())
    manifest = json.loads((comparison / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["source_run_id"] == RUN_ID
    assert metadata["config"]["seq_len"] == 192
    validation_nll = np.load(work / "cache/validation_nll.npy", mmap_mode="r")
    features = np.load(metadata["feature_cache_paths"]["validation"], mmap_mode="r")
    validation_ids = np.load(work / "cache/validation_window_end_indices.npy", mmap_mode="r")
    with np.load(work / "input_statistics.npz", allow_pickle=False) as stats:
        vf = flow_scores(validation_nll, stats["calibration_center"], stats["calibration_scale"])
    state_index = metadata["feature_layout"]["slices"]["S"][0]
    va = np.max(np.abs(features[:, :, state_index]), axis=1)
    assert len(vf) == len(va) == len(validation_ids) == 13018
    with np.load(comparison / "scores.npz", allow_pickle=False) as saved:
        axis, local = saved["time_indices"], saved["local_time_indices"]
        labels, actual = saved["selected_actual"], saved["actual"]
        assert axis.dtype.kind in "iu" and np.all(np.diff(axis) > 0)
        assert np.array_equal(local, np.flatnonzero(saved["local_valid_mask"]))
        assert np.array_equal(labels, actual[local])
        assert len(axis) == 87650 and int((labels == 0).sum()) == 63272
        scores = {name: saved[name + "_scores"] for name in
                  ("flow", "state_amplitude", "flow_plus_amplitude")}
        threshold = float(saved["flow_threshold"])
        assert np.array_equal(scores["flow"] > threshold, saved["flow_predictions"])
    # This is the only new rule. It reads validation references, not test labels.
    scores["validation_tail_minp"] = tail_fusion(vf, va, scores["flow"], scores["state_amplitude"])
    amplitude_threshold, amplitude_pred = budget_point(scores["state_amplitude"], labels, 632)
    assert int(np.sum(amplitude_pred & (labels == 1))) == 388
    assert int(np.sum(amplitude_pred & (labels == 0))) == 631
    rows, summary = [], {"rule_locked_before_test_computation": RULE, "source_run_id": RUN_ID}
    for name, score in scores.items():
        operating_threshold, pred = budget_point(score, labels, 632)
        top = score == score.max()
        positive = labels == 1
        row = dict(dataset="PSM", source_run_id=RUN_ID, analysis="budget_632fp", readout=name,
                   points=len(labels), normal_points=63272, anomaly_points=24378,
                   threshold=operating_threshold, threshold_rule="posthoc test-label strict > whole-tie scan",
                   tp=int((pred & positive).sum()), fp=int((pred & ~positive).sum()),
                   ap=float(average_precision_score(labels, score)),
                   highest_score_ties=int(top.sum()), highest_tie_tp=int((top & positive).sum()),
                   highest_tie_fp=int((top & ~positive).sum()),
                   added_tp_relative_to_amplitude=int((pred & ~amplitude_pred & positive).sum()),
                   lost_tp_relative_to_amplitude=int((~pred & amplitude_pred & positive).sum()))
        rows.append(row)
    frame, train_length, *_ = _read_task("PSM", "large_detect", None)
    assert np.array_equal(axis, local + train_length)
    full_labels = frame["label"].to_numpy().astype(np.int8)
    assert np.array_equal(actual, full_labels[train_length:])
    assert np.array_equal(labels, full_labels[axis])
    segment_id, _ = deciles(np.arange(train_length, len(frame)), axis)
    groups = normal_window_groups(full_labels, axis, 192)
    assert np.array_equal(groups["entire_window_normal"] | groups["normal_endpoint_anomalous_context"], labels == 0)
    flow_pred = scores["flow"] > threshold
    assert (int((flow_pred & (labels == 1)).sum()), int((flow_pred & (labels == 0)).sum())) == (11722, 13937)
    for part in (0, 4, 6, 7, 8):
        for name, mask in groups.items():
            selected = mask if part == 0 else mask & (segment_id == part)
            n = int(selected.sum())
            fp = int((flow_pred & selected).sum())
            rows.append(dict(dataset="PSM", source_run_id=RUN_ID, analysis="saved_flow_context",
                             readout="flow", group=name, official_decile=part, points=n, normal_points=n,
                             threshold=threshold, threshold_rule="original saved q=.995 strict >",
                             fp=fp, fpr=fp/n if n else None,
                             flow_p99=float(np.quantile(scores["flow"][selected], .99)) if n else None))
    exceed_f = scores["flow"] > vf.max()
    exceed_a = scores["state_amplitude"] > va.max()
    for name, mask in (("flow", exceed_f), ("state_amplitude", exceed_a), ("either", exceed_f | exceed_a)):
        rows.append(dict(dataset="PSM", source_run_id=RUN_ID, analysis="above_validation_max", readout=name,
                         points=int(mask.sum()), tp=int((mask & (labels==1)).sum()), fp=int((mask & (labels==0)).sum()),
                         validation_max=float(vf.max()) if name=="flow" else float(va.max()) if name=="state_amplitude" else None))
    for path, digest in digests.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, str(path)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    output = BATCH / "psm_validation_tail_comparison.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary.update(rows=rows, validation_points=len(vf), flow_validation_max=float(vf.max()),
                   amplitude_validation_max=float(va.max()), consumed_files_unchanged=len(digests))
    return summary


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        print(json.dumps(analyze(), indent=2, allow_nan=False))
