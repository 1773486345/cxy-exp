"""Offline operating-point tests; no model imports, fitting or forwards."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[4] / "scripts/analyze_mnrf_r_residual_readout.py"
SPEC = importlib.util.spec_from_file_location("r_readout_analysis", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_whole_ties_and_minimum_false_positives():
    # Can't take only the positive in the tied top group at zero FP.
    scores = np.array([5, 5, 4, 3, 2], dtype=np.float32)
    labels = np.array([1, 0, 1, 0, 0])
    threshold, pred = MODULE.budget_point(scores, labels, 0)
    assert not pred.any() and threshold == 5
    threshold, pred = MODULE.budget_point(scores, labels, 2)
    assert pred.tolist() == [True, True, True, False, False]
    assert np.array_equal(scores.astype(float) > threshold, pred)


def test_exhaustive_operating_points_match_brute_force():
    rng = np.random.default_rng(2021)
    for _ in range(20):
        scores = rng.integers(-2, 4, 30).astype(np.float32)
        labels = rng.integers(0, 2, 30)
        for budget in (0, 1, 4, 30):
            threshold, pred = MODULE.budget_point(scores, labels, budget)
            candidates = []
            for t in [*np.unique(scores).astype(float), np.nextafter(float(scores.min()), -np.inf)]:
                p = scores.astype(float) > t
                tp, fp = int((p & (labels == 1)).sum()), int((p & (labels == 0)).sum())
                if fp <= budget:
                    candidates.append((tp, -fp))
            assert ((pred & (labels == 1)).sum(), -(pred & (labels == 0)).sum()) == max(candidates)


def test_official_deciles_not_compact_reindexing():
    full = np.arange(100, 203)
    compact = full[13:]
    parts, edges = MODULE.deciles(full, compact)
    assert parts[0] == 2 and parts[-1] == 10
    assert np.array_equal(edges, np.arange(11) * 103 // 10)
    with pytest.raises(ValueError, match="sorted unique"):
        MODULE.deciles(full, np.array([115.5, 116.5]))
    with pytest.raises(ValueError, match="outside"):
        MODULE.deciles(full, np.array([99, 110]))


def test_reselect_sensitivity_and_segment_accounting():
    # Exercise actual row generation: the remaining normal count fixes a new
    # budget, while all decile rows reuse the selected global threshold.
    axis = np.arange(1000)
    labels = (axis % 13 == 0).astype(np.int8)
    task = dict(task="PSM", run_id="synthetic", official=axis, coordinates=axis,
                labels=labels, methods={"Full-GMM": dict(scores=axis.astype(float),
                                                        threshold=500, path=MODULE.ROOT / "synthetic.npz")},
                gaps=[], metrics=[])
    rows = MODULE.analyze(task)
    for scope in ("all_common", "exclude_deciles_7_8"):
        for rule in ("saved_q995", "posthoc_fpr_1pct"):
            selected = [r for r in rows if r["scope"] == scope and r["operating_point"] == rule]
            total, *parts = selected
            assert sum(p["fp"] for p in parts) == total["fp"]
            assert sum(p["tp"] for p in parts) == total["tp"]
            assert all(p["threshold"] == total["threshold"] for p in parts)
            if rule.startswith("posthoc"):
                assert total["global_fp_budget"] == total["normal_points"] // 100
                assert total["fp"] <= total["global_fp_budget"]
            else:
                assert total["threshold"] == 500


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_scores_rejected(bad):
    with pytest.raises(ValueError, match="invalid"):
        MODULE.budget_point([1, bad], [0, 1], 1)


@pytest.mark.parametrize("method", list(MODULE.SAVED_THRESHOLD_SOURCES))
def test_actual_rows_distinguish_saved_threshold_provenance(method):
    axis = np.arange(1000)
    task = dict(task="PSM", run_id="synthetic", official=axis, coordinates=axis,
                labels=(axis % 13 == 0).astype(np.int8), gaps=[], metrics=[],
                methods={method: dict(scores=axis.astype(float), threshold=500,
                                      path=MODULE.ROOT / "synthetic.npz")})
    rows = MODULE.analyze(task)
    saved = [r for r in rows if r["operating_point"] == "saved_q995"]
    assert all(r["threshold"] == 500 for r in saved)
    assert all(r["threshold_source"] == MODULE.SAVED_THRESHOLD_SOURCES[method] for r in saved)
    if method in ("CATCH-common", "IsolationForest", "PCA-residual"):
        assert all(r["threshold_source"].startswith("direct calibration q=.995")
                   and "no validation median/MAD" in r["threshold_source"] for r in saved)
    else:
        assert all(r["threshold_source"].startswith("saved validation median/MAD") for r in saved)
    assert all(r["threshold_source"] == "post-hoc test-label ranking diagnostic; no calibration"
               for r in rows if r["operating_point"].startswith("posthoc"))
    global_saved, = [r for r in saved if r["scope"] == "all_common" and r["decile"] == "global"]
    assert global_saved["tp"] == int(np.sum((axis % 13 == 0) & (axis > 500)))
    assert global_saved["fp"] == int(np.sum((axis % 13 != 0) & (axis > 500)))
