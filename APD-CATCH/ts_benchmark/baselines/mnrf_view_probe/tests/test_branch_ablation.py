import json

import pytest
import hashlib
import json

import numpy as np
import pandas as pd

from ts_benchmark.baselines.mnrf_view_probe import MNRFViewProbe, load_probe
from scripts.validate_mnrf_view_probe import validate

from scripts.run_mnrf_branch_ablation import COMBINATIONS, EXPECTED_LAYOUT, build_plan, selected_indices
from scripts.summarize_mnrf_branch_ablation import contribution_deltas, segment_counts_from_axis


def test_matrix_has_all_nine_combinations_and_dimensions():
    assert [name for name, _ in COMBINATIONS] == [
        "S", "SE", "SP", "SR", "FULL", "WITHOUT_S", "WITHOUT_E", "WITHOUT_P", "WITHOUT_R"
    ]
    assert [len(selected_indices(views)) for _, views in COMBINATIONS] == [1, 9, 9, 9, 25, 24, 17, 17, 17]
    assert EXPECTED_LAYOUT["slices"] == {"S": [0, 1], "E": [1, 9], "P": [9, 17], "R": [17, 25]}


def test_plan_keeps_blocked_dataset_matrix_without_starting_work():
    plan = build_plan(["PSM"], validate_sources=False)
    assert plan["positions"] == 9
    assert len(plan["rows"]) == 9
    assert all(row["status"] == "blocked" for row in plan["rows"])
    assert all(row["selected_dim"] in {1, 9, 17, 24, 25} for row in plan["rows"])


def test_contribution_signs_match_removal_definition():
    values = {
        "S": {"average_precision": 0.40, "auc_roc": 0.60},
        "SE": {"average_precision": 0.50, "auc_roc": 0.70},
        "SP": {"average_precision": 0.39, "auc_roc": 0.59},
        "SR": {"average_precision": 0.40, "auc_roc": 0.60},
        "FULL": {"average_precision": 0.55, "auc_roc": 0.75},
        "WITHOUT_S": {"average_precision": 0.30, "auc_roc": 0.50},
        "WITHOUT_E": {"average_precision": 0.56, "auc_roc": 0.76},
        "WITHOUT_P": {"average_precision": 0.50, "auc_roc": 0.70},
        "WITHOUT_R": {"average_precision": 0.55, "auc_roc": 0.75},
    }
    deltas = contribution_deltas(values)
    assert deltas["E"]["added"]["average_precision"] == pytest.approx(0.10)
    assert deltas["E"]["deleted"]["average_precision"] == pytest.approx(-0.01)
    assert deltas["S"]["added"]["average_precision"] is None
    assert deltas["S"]["deleted"]["average_precision"] == pytest.approx(0.25)


def test_segment_counts_preserve_unscored_anomaly_and_segment_boundaries():
    all_global = list(range(20))
    all_labels = [0] * 20
    all_labels[10:15] = [1] * 5
    all_labels[17:19] = [1] * 2
    all_segments = [0] * 10 + [1] * 5 + [2] * 5
    test_global = list(range(10, 20))
    details = segment_counts_from_axis(all_labels, all_global, all_segments, test_global,
                                       [17], [1])
    assert details["total"] == 2
    assert details["scored"] == 1
    assert details["details"][0]["valid_points"] == 0
    assert details["details"][0]["no_valid_scores"] is True
    assert details["details"][1]["hit"] is True


def test_real_tiny_select_gmm_uses_frozen_cache_for_four_widths(tmp_path, monkeypatch):
    rng = np.random.default_rng(716)
    total, train_length = 520, 430
    frame = pd.DataFrame(rng.normal(size=(total, 2)).astype(np.float32), columns=("x", "y"),
                         index=pd.date_range("2026-01-01", periods=total, freq="min"))
    labels = np.zeros(total - train_length, dtype=np.int64)
    labels[31:36] = 1
    frame["label"] = np.r_[np.zeros(train_length, dtype=np.int64), labels]
    common = dict(seq_len=32, pattern_len=32, batch_size=64, encoding_batch_size=64,
                  stage1_epochs=1, stage2_epochs=1, patience=1, device="cpu",
                  gmm_components=1, gmm_fit_max_windows=8, task_name="SYN.csv",
                  artifact_policy="full")
    source_dir = tmp_path / "frozen-full"
    MNRFViewProbe(**dict(common, output_dir=str(source_dir), mode="full_gmm")).fit(frame, train_length, labels)
    source_manifest = json.loads((source_dir / "probe_manifest.json").read_text())
    source_hashes = {}
    for rel in ("cache/normalized_values.npy", "cache/train_features.npy",
                "cache/validation_features.npy", "cache/calibration_features.npy",
                "cache/test_features.npy", "heads/gmm_fit_endpoints.npy"):
        path = source_dir / rel
        if path.exists():
            source_hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(MNRFViewProbe, "_cache_all", lambda *a, **k: pytest.fail("derived run re-encoded cache"))
    monkeypatch.setattr(MNRFViewProbe, "_normalized_memmap", lambda *a, **k: pytest.fail("derived run rebuilt normalized values"))
    monkeypatch.setattr(MNRFViewProbe, "_stage1_with_progress", lambda *a, **k: pytest.fail("derived run entered Stage 1"))
    monkeypatch.setattr(MNRFViewProbe, "_fit_flow", lambda *a, **k: pytest.fail("derived run entered Flow"))
    combos = (("S", ("S",)), ("FULL", ("S", "E", "P", "R")),
              ("WITHOUT_E", ("S", "P", "R")), ("WITHOUT_S", ("E", "P", "R")))
    for name, views in combos:
        output = tmp_path / ("head-" + name)
        derived = MNRFViewProbe(**dict(common, output_dir=str(output), mode="select_gmm",
                                       source_dir=str(source_dir), selected_views=views))
        derived.fit(frame, train_length, labels)
        manifest = json.loads((output / "probe_manifest.json").read_text())
        assert manifest["selected_dim"] == len(selected_indices(views))
        assert manifest["source_feature_fingerprint"] == source_manifest["feature_fingerprint"]
        assert hashlib.sha256((output / "heads/gmm_fit_endpoints.npy").read_bytes()).hexdigest() == source_hashes["heads/gmm_fit_endpoints.npy"]
        assert np.array_equal(np.load(output / "heads/fixed_noise.npy"), np.load(source_dir / "heads/fixed_noise.npy"))
        validation = validate(output)
        assert validation["status"] == "complete", validation
        loaded = load_probe(output, device="cpu")
        cached = loaded.score_cached_test()
        with np.load(output / "results/aligned_scores.npz", allow_pickle=False) as aligned:
            np.testing.assert_allclose(cached["scores"], aligned["scores"], rtol=1e-6, atol=1e-7)
            assert np.array_equal(cached["predictions"], aligned["predictions"])
    for rel, expected in source_hashes.items():
        assert hashlib.sha256((source_dir / rel).read_bytes()).hexdigest() == expected
