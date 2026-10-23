import json

import pytest
import hashlib
import json

import numpy as np
import pandas as pd

from ts_benchmark.baselines.mnrf_view_probe import MNRFViewProbe, load_probe
from scripts.validate_mnrf_view_probe import validate

from scripts.run_mnrf_branch_ablation import COMBINATIONS, EXPECTED_LAYOUT, build_plan, selected_indices
from scripts.summarize_mnrf_branch_ablation import (
    _evaluation, _point_and_segment_changes, completion_counts,
    contribution_deltas, paired_seed_contribution_rows, segment_counts_from_axis,
)


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


def test_completion_requires_every_requested_cell_and_counts_failed():
    tasks, seeds = ["PSM"], [2021]
    rows = [{"dataset": "PSM", "combination": combo, "head_seed": 2021,
             "status": "reused" if combo == "FULL" else "planned/pending"}
            for combo, _ in COMBINATIONS]
    result = completion_counts(tasks, seeds, rows)
    assert result["datasets_with_results"] == 1
    assert result["datasets_complete"] == 0
    assert result["positions_expected"] == 9
    assert result["batch_status"] == "partial"
    failed_rows = [dict(row, status="failed", reason="injected failure") if row["combination"] == "S"
                   else row for row in rows]
    failed = completion_counts(tasks, seeds, failed_rows)
    assert failed["failed"] == 1
    assert failed["batch_status"] == "failed"
    assert failed["datasets_complete"] == 0


def test_paired_changes_report_added_and_lost_even_when_net_is_zero():
    base = {"global_indices": np.arange(4), "local_indices": np.arange(4),
            "labels": np.array([1, 1, 0, 0]), "scores": np.array([.9, .1, .9, .1]),
            "predictions": np.array([1, 0, 1, 0]), "threshold": .5,
            "segment_details": [
                {"segment_id": 4, "start_global": 0, "end_global": 0, "hit": True,
                 "points": 1, "valid_points": 1, "hit_points": 1},
                {"segment_id": 4, "start_global": 1, "end_global": 1, "hit": False,
                 "points": 1, "valid_points": 1, "hit_points": 0},
            ]}
    changed = {**base, "scores": np.array([.1, .9, .1, .9]),
               "predictions": np.array([0, 1, 0, 1]), "threshold": .4,
               "segment_details": [
                   {"segment_id": 4, "start_global": 0, "end_global": 0, "hit": False,
                    "points": 1, "valid_points": 1, "hit_points": 0},
                   {"segment_id": 4, "start_global": 1, "end_global": 1, "hit": True,
                    "points": 1, "valid_points": 1, "hit_points": 1},
               ]}
    changes, added_segments, lost_segments = _point_and_segment_changes(base, changed, "source")
    assert len(changes["added_tp"]) == len(changes["lost_tp"]) == 1
    assert len(changes["added_fp"]) == len(changes["removed_fp"]) == 1
    assert len(added_segments) == len(lost_segments) == 1


def test_repeat_contribution_pairs_same_seeds_and_reports_missing_pairs():
    evals = {
        ("PSM", "S", 2021): {"average_precision": .4, "auc_roc": .6},
        ("PSM", "SE", 2021): {"average_precision": .5, "auc_roc": .7},
        ("PSM", "S", 2022): {"average_precision": .5, "auc_roc": .7},
        ("PSM", "SE", 2022): {"average_precision": .4, "auc_roc": .6},
        ("PSM", "SE", 2023): {"average_precision": .3, "auc_roc": .5},
    }
    rows = paired_seed_contribution_rows(["PSM"], [2021, 2022, 2023], evals)
    row = next(item for item in rows if item["branch"] == "E" and item["direction"] == "added"
               and item["metric"] == "average_precision")
    assert row["n_requested"] == 3 and row["n_pairs"] == 2
    assert json.loads(row["participating_seeds"]) == [2021, 2022]
    assert json.loads(row["missing_seeds"]) == [2023]
    assert row["mean"] == pytest.approx(0.0)
    assert row["std"] == pytest.approx(np.std([.1, -.1], ddof=1))
    s_added = next(item for item in rows if item["branch"] == "S" and item["direction"] == "added")
    assert s_added["not_applicable"] is True and s_added["mean"] is None


def test_summary_native_validation_rejects_tampered_scores_and_changed_point_set(tmp_path, monkeypatch):
    rng = np.random.default_rng(716)
    total, train_length = 520, 430
    frame = pd.DataFrame(rng.normal(size=(total, 2)).astype(np.float32), columns=("x", "y"),
                         index=pd.date_range("2026-01-01", periods=total, freq="min"))
    labels = np.zeros(total - train_length, dtype=np.int64)
    labels[31:36] = 1
    frame["label"] = np.r_[np.zeros(train_length, dtype=np.int64), labels]
    common = dict(seq_len=32, pattern_len=32, batch_size=64, encoding_batch_size=64,
                  stage1_epochs=1, stage2_epochs=1, patience=1, device="cpu", gmm_components=1,
                  gmm_fit_max_windows=8, task_name="SYN.csv", artifact_policy="full")
    source = tmp_path / "source"
    MNRFViewProbe(**dict(common, output_dir=str(source), mode="full_gmm")).fit(frame, train_length, labels)
    # Exercise the same bound full-label/time evidence contract used by the
    # CLI, using this isolated synthetic source's own bytes and manifest.
    from scripts.run_mnrf_view_probe import _write_run_supporting_evidence, _sha256_file
    source_manifest = json.loads((source / "probe_manifest.json").read_text())
    data_path = tmp_path / "SYN.csv"
    frame.to_csv(data_path, index_label="date")
    metadata_path = tmp_path / "DETECT_META.csv"
    metadata_path.write_text("file_name,train_lens\nSYN.csv,430\n")
    identity = source_manifest["source_identity"]
    source_data = {"task": "SYN", "data_source": source_manifest["config"]["data_source"],
                   "rows": total, "train_length": train_length, "variable_names": ["x", "y"],
                   "time_kind": identity["time_index"]["kind"],
                   "time_values_sha256": identity["time_index"]["values_sha256"],
                   "feature_data_sha256": identity["feature_data_sha256"],
                   "boundary_columns": identity.get("boundary_columns", []),
                   "boundary_values_sha256": identity.get("boundary_values_sha256"),
                   "selected_metadata_row": {"file_name": "SYN.csv", "train_lens": "430"},
                   "data_file": str(data_path), "data_file_sha256": _sha256_file(data_path),
                   "metadata_file": str(metadata_path), "metadata_file_sha256": _sha256_file(metadata_path)}
    provenance = {"provenance_type": "recovered_partial", "source_data": source_data}
    _write_run_supporting_evidence(source, frame, train_length, labels, data_path, metadata_path, provenance)
    derived = tmp_path / "derived-s"
    MNRFViewProbe(**dict(common, output_dir=str(derived), mode="select_gmm", source_dir=str(source),
                         selected_views=("S",))).fit(frame, train_length, labels)
    # Both source and candidate take the production validator path before metric inclusion.
    _, source_bundle = _evaluation(source, source, "SYN", "FULL", 2021)
    assert len(source_bundle["global_indices"]) > 0
    _, candidate_bundle = _evaluation(derived, source, "SYN", "S", 2021)
    assert np.array_equal(candidate_bundle["global_indices"], source_bundle["global_indices"])
    assert np.array_equal(candidate_bundle["valid_mask"], source_bundle["valid_mask"])
    original = np.load(derived / "results/aligned_scores.npz", allow_pickle=False)
    data = {key: np.asarray(original[key]).copy() for key in original.files}
    original.close()
    metrics_path = derived / "results/metrics.json"
    original_metrics = metrics_path.read_bytes()
    original_manifest = json.loads((derived / "probe_manifest.json").read_text())
    data["scores"][0] += 1.0
    data["aligned_scores"][data["valid_mask"]] = data["scores"]
    data["predictions"] = (data["scores"] > float(data["threshold"])).astype(np.int64)
    data["aligned_predictions"][data["valid_mask"]] = data["predictions"]
    np.savez_compressed(derived / "results/aligned_scores.npz", **data)
    with pytest.raises((ValueError, AssertionError)):
        _evaluation(derived, source, "SYN", "S", 2021)
    # Even a self-consistent file/manifest/metrics digest cannot rebind a score
    # vector that no longer reproduces from the saved test NLL and calibration.
    from sklearn.metrics import average_precision_score, roc_auc_score
    metrics = json.loads(metrics_path.read_text())
    metrics["auc_pr"] = float(average_precision_score(data["actual"], data["scores"]))
    metrics["auc_roc"] = float(roc_auc_score(data["actual"], data["scores"]))
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    manifest_path = derived / "probe_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["aligned_scores_sha256"] = hashlib.sha256(
        (derived / "results/aligned_scores.npz").read_bytes()).hexdigest()
    manifest["metrics_sha256"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="reproduce from calibration|test_nll and calibration"):
        _evaluation(derived, source, "SYN", "S", 2021)
    # A locally self-consistent mask/coordinate edit is still rejected because
    # the candidate must use exactly the pinned Full scoring point set.
    clean = data
    np.savez_compressed(derived / "results/aligned_scores.npz", **clean)
    metrics_path.write_bytes(original_metrics)
    restored_manifest = dict(original_manifest)
    restored_manifest["aligned_scores_sha256"] = hashlib.sha256(
        (derived / "results/aligned_scores.npz").read_bytes()).hexdigest()
    restored_manifest["metrics_sha256"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(restored_manifest, indent=2, sort_keys=True) + "\n")
    with np.load(derived / "results/aligned_scores.npz", allow_pickle=False) as payload:
        clean = {key: np.asarray(payload[key]).copy() for key in payload.files}
    clean["local_indices"] = clean["local_indices"].copy()
    clean["local_indices"][0] += 1
    clean["valid_mask"] = clean["valid_mask"].copy()
    clean["valid_mask"][0] = False
    clean["valid_mask"][1] = True
    clean["time_indices"] = clean["time_indices"].copy()
    clean["time_indices"][0], clean["time_indices"][1] = clean["time_indices"][1], clean["time_indices"][0]
    np.savez_compressed(derived / "results/aligned_scores.npz", **clean)
    manifest = json.loads(manifest_path.read_text())
    manifest["aligned_scores_sha256"] = hashlib.sha256(
        (derived / "results/aligned_scores.npz").read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises((ValueError, AssertionError)):
        _evaluation(derived, source, "SYN", "S", 2021)


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
