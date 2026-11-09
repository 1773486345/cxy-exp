#!/usr/bin/env python
"""PSM saved Full/R-residual false-positive overlap; no model execution."""
import csv
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from analyze_mnrf_r_residual_readout import ROOT, Inputs, check, deciles, load_task


OUT = ROOT / "result/analysis/MNRF-R-Residual-Readout/readout-20261110"


def weights(matrix):
    """Match the existing PSM attribution tolerance; split each tied maximum."""
    check(np.isfinite(matrix).all(), "nonfinite attribution matrix")
    tied = np.isclose(matrix, matrix.max(axis=1, keepdims=True), rtol=0, atol=1e-7)
    result = tied / tied.sum(axis=1, keepdims=True)
    check(np.allclose(result.sum(axis=1), 1, rtol=0, atol=1e-14), "tie weights")
    return result


def window_clean(labels, segment_ids, endpoints, length):
    starts = endpoints - length + 1
    check(np.all(starts >= 0), "window outside full input")
    # A complete window must stay within one frozen segment, not just have
    # normal endpoints. Source segment IDs are canonical contiguous runs.
    transitions = np.r_[0, np.cumsum(segment_ids[1:] != segment_ids[:-1])]
    check(np.all(transitions[endpoints] == transitions[starts]), "window crosses boundary")
    cumulative = np.r_[0, np.cumsum(labels, dtype=np.int64)]
    return cumulative[endpoints + 1] == cumulative[starts]


def summary(values, prefix):
    a = np.asarray(values)
    return {prefix + name: float(np.quantile(a, q)) if len(a) else None
            for name, q in (("p01", .01), ("p25", .25), ("p50", .5),
                            ("p75", .75), ("p90", .9), ("p99", .99))}


def dominant_summary(raw, z, contribution, values, pick, j):
    selected = pick & (contribution[:, j] > 0)
    return dict(dominant_points=int(selected.sum()),
                dominant_raw_median=float(np.median(raw[selected, j])) if selected.any() else None,
                dominant_z_median=float(np.median(z[selected, j])) if selected.any() else None,
                dominant_input_median=float(np.median(values[selected, j])) if selected.any() else None)


def save_csv(path, rows):
    check(not path.exists(), f"preserve existing attribution: {path}")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    # Small invariant checks exercise the exact helpers used for real arrays.
    assert np.allclose(weights(np.array([[2., 2., 0.], [4., 4., 4.]])),
                       [[.5, .5, 0], [1/3, 1/3, 1/3]])
    assert window_clean(np.array([0, 1, 0, 0]), np.zeros(4), np.array([2, 3]), 2).tolist() == [False, True]
    try:
        window_clean(np.zeros(4), np.array([0, 0, 1, 1]), np.array([2]), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("boundary-crossing window accepted")
    with threadpool_limits(limits=1):
        inputs = Inputs()
        task = load_task("PSM", inputs)
        source = ROOT / "result/score/MNRF-ViewProbe/PSM" / task["run_id"]
        diagnostic = ROOT / "result/analysis/MNRF-FrozenDecoder/PSM" / task["run_id"]
        manifest = task["manifest"]
        endpoints = task["coordinates"]
        names = manifest["variable_names"]
        n, c = len(endpoints), len(names)
        check(n == 87650 and c == 25, "unexpected PSM axis")
        with np.load(source / "analysis_inputs.npz", allow_pickle=False) as a:
            labels_all, ids_all = a["all_labels"], a["all_segment_ids"]
            check(np.array_equal(a["all_global_indices"], np.arange(len(labels_all))), "non-row-index global axis")
        seq_len = manifest["config"]["seq_len"]
        clean = window_clean(labels_all, ids_all, endpoints, seq_len)
        normal = task["labels"] == 0
        methods = ["Full-GMM", "R-residual"]
        fp = [task["methods"][m]["scores"] > task["methods"][m]["threshold"] for m in methods]
        classes = {"both_fp": fp[0] & fp[1], "only_full_fp": fp[0] & ~fp[1],
                   "only_r_fp": ~fp[0] & fp[1], "neither_fp": ~fp[0] & ~fp[1]}
        segment, edges = deciles(task["official"], endpoints)
        raw = {"Full-GMM": np.load(source / "results/test_nll.npy", mmap_mode="r", allow_pickle=False)}
        with np.load(diagnostic / "R_test_prediction_evidence.npz", allow_pickle=False) as a:
            check(np.array_equal(a["endpoints"], endpoints), "R residual endpoints")
            raw["R-residual"] = np.square(a["residual"])
        center, scale, z, w, raw_w = {}, {}, {}, {}, {}
        for m, p in [("Full-GMM", source / "heads/calibration.npz"),
                     ("R-residual", diagnostic / "R_squared_residual_test_scores.npz")]:
            with np.load(p, allow_pickle=False) as a:
                center[m], scale[m] = a["center"], a["scale"]
            check(raw[m].shape == (n, c) and np.isfinite(raw[m]).all(), "raw score matrix")
            z[m] = (raw[m] - center[m]) / scale[m]
            check(np.array_equal(z[m].max(axis=1), task["methods"][m]["scores"]), "max reproduction")
            w[m], raw_w[m] = weights(z[m]), weights(raw[m])
        inputs.record(source / "cache/normalized_values.npy", manifest["normalized_cache_sha256"])
        values = np.load(source / "cache/normalized_values.npy", mmap_mode="r", allow_pickle=False)
        check(values.shape == (len(labels_all), c) and values.dtype == np.float32, "normalized input shape/dtype")
        x_test = values[endpoints]
        check(np.isfinite(x_test).all(), "nonfinite normalized test inputs")
        inputs.record(source / "heads/calibration_nll.npy")
        inputs.record(source / "cache/calibration_window_end_indices.npy", manifest["endpoint_sha256"]["calibration"])
        cal_end = np.load(source / "cache/calibration_window_end_indices.npy", allow_pickle=False)
        cal_raw = np.load(source / "heads/calibration_nll.npy", mmap_mode="r", allow_pickle=False)
        check(cal_raw.shape == (len(cal_end), c) and np.isfinite(cal_raw).all(), "calibration NLL dimensions")
        lo, hi = manifest["source_identity"]["ranges"]["calibration"]
        check(np.all((cal_end >= lo) & (cal_end < hi)), "calibration outside frozen range")
        cal_clean = window_clean(labels_all, ids_all, cal_end, seq_len)
        cal_normal = labels_all[cal_end] == 0
        cal_z = (cal_raw - center["Full-GMM"]) / scale["Full-GMM"]
        cal_w, cal_raw_w = weights(cal_z), weights(cal_raw)
        x_cal = values[cal_end]
        check(np.isfinite(x_cal).all(), "nonfinite normalized calibration inputs")
        intersections, evidence = [], []
        for scope, eligible in [("endpoint_normal", normal), ("whole_window_normal", normal & clean)]:
            cal_pick = cal_normal & (cal_clean if scope == "whole_window_normal" else True)
            for part in [0] + list(range(1, 11)):
                use = eligible & ((segment == part) if part else True)
                groups = {name: use & mask for name, mask in classes.items()}
                row = dict(scope=scope, decile=part or "global", normal_points=int(use.sum()),
                           global_start=int(task["official"][edges[part-1]]) if part else int(task["official"][0]),
                           global_stop_exclusive=int(task["official"][edges[part]-1])+1 if part else int(task["official"][-1])+1)
                row.update({name: int(p.sum()) for name, p in groups.items()})
                check(sum(row[g] for g in groups) == row["normal_points"], "overlap partition")
                both = groups["both_fp"]
                row["full_fp"] = row["both_fp"] + row["only_full_fp"]
                row["r_fp"] = row["both_fp"] + row["only_r_fp"]
                row["overlap_over_union"] = row["both_fp"] / (row["full_fp"] + row["only_r_fp"]) if (row["full_fp"] + row["only_r_fp"]) else None
                row["same_dominant_weight_at_both_fp"] = float((w[methods[0]][both] * w[methods[1]][both]).sum())
                row["any_shared_dominant_both_fp"] = int(np.any((w[methods[0]][both] > 0) & (w[methods[1]][both] > 0), axis=1).sum())
                for j in (1, 3):
                    row[f"feature_{j}_joint_dominant_weight_at_both_fp"] = float((w[methods[0]][both,j] * w[methods[1]][both,j]).sum())
                intersections.append(row)
                for group, pick in groups.items():
                    for m in methods:
                        for j, name in enumerate(names):
                            r = dict(scope=scope, decile=part or "global", group=group, method=m,
                                     variable=name, points=int(pick.sum()),
                                     max_dominant_weight=float(w[m][pick,j].sum()),
                                     max_dominant_share=float(w[m][pick,j].mean()) if pick.any() else None,
                                     raw_max_dominant_share=float(raw_w[m][pick,j].mean()) if pick.any() else None,
                                     validation_center=float(center[m][j]), scale=float(scale[m][j]),
                                     score_mad_floor=float(manifest["config"]["score_mad_floor"]),
                                     scale_floor_active=bool(scale[m][j] <= np.asarray(
                                         manifest["config"]["score_mad_floor"], dtype=scale[m].dtype)),
                                     calibration_raw_available=m == "Full-GMM", calibration_normal_points=int(cal_pick.sum()))
                            r.update(summary(raw[m][pick,j], "raw_"))
                            r.update(summary(z[m][pick,j], "z_"))
                            r.update(summary(x_test[pick,j], "input_"))
                            r.update(summary(np.abs(x_test[pick,j]), "abs_input_"))
                            r.update(dominant_summary(raw[m], z[m], w[m], x_test, pick, j))
                            if cal_pick.any():
                                cal_x = x_cal[cal_pick,j]
                                q01, q99 = np.quantile(cal_x, [.01, .99])
                                r.update(summary(cal_x, "cal_input_"))
                                r["input_outside_cal_p01_p99_fraction"] = float(np.mean((x_test[pick,j] < q01) | (x_test[pick,j] > q99))) if pick.any() else None
                                if m == "Full-GMM":
                                    r.update(summary(cal_raw[cal_pick,j], "cal_raw_"))
                                    r.update(summary(cal_z[cal_pick,j], "cal_z_"))
                                    r["raw_above_cal_p99_fraction"] = float(np.mean(raw[m][pick,j] > np.quantile(cal_raw[cal_pick,j], .99))) if pick.any() else None
                            evidence.append(r)
            # Calibration reference rows retain all variable attribution, too.
            for j, name in enumerate(names):
                r = dict(scope=scope, decile="calibration", group="normal_reference", method="Full-GMM",
                         variable=name, points=int(cal_pick.sum()),
                         max_dominant_weight=float(cal_w[cal_pick,j].sum()),
                         max_dominant_share=float(cal_w[cal_pick,j].mean()) if cal_pick.any() else None,
                         raw_max_dominant_share=float(cal_raw_w[cal_pick,j].mean()) if cal_pick.any() else None,
                         validation_center=float(center["Full-GMM"][j]), scale=float(scale["Full-GMM"][j]))
                r.update(summary(cal_raw[cal_pick,j], "raw_"))
                r.update(summary(cal_z[cal_pick,j], "z_"))
                r.update(summary(x_cal[cal_pick,j], "input_"))
                evidence.append(r)
        for scope in ("endpoint_normal", "whole_window_normal"):
            total, *parts = [r for r in intersections if r["scope"] == scope]
            for key in ("normal_points", "both_fp", "only_full_fp", "only_r_fp", "neither_fp"):
                check(sum(r[key] for r in parts) == total[key], f"decile accounting {key}")
        check(intersections[0]["full_fp"] == 17305 and intersections[0]["r_fp"] == 14826, "saved FP totals")
        inputs.unchanged()
        save_csv(OUT / "psm_fp_intersections.csv", intersections)
        save_csv(OUT / "psm_fp_variable_evidence.csv", evidence)
        for row in intersections:
            if row["decile"] in ("global", 4, 6, 7, 8):
                print(row)
        print("read_only_inputs", len(inputs.hashes), "evidence_rows", len(evidence))


if __name__ == "__main__":
    main()
