import json
import subprocess
import sys
from pathlib import Path

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
    contribution_deltas, segment_counts_from_axis, summarize,
)
import scripts.mnrf_branch_source_registry as source_registry
import scripts.register_mnrf_branch_source as source_registration
from scripts.run_mnrf_branch_ablation import (
    TASKS, _manifest_may_match, _manifest_task, _prune_superseded_inventories,
)


def test_matrix_has_all_nine_combinations_and_dimensions():
    assert [name for name, _ in COMBINATIONS] == [
        "S", "SE", "SP", "SR", "FULL", "WITHOUT_S", "WITHOUT_E", "WITHOUT_P", "WITHOUT_R"
    ]
    assert [len(selected_indices(views)) for _, views in COMBINATIONS] == [1, 9, 9, 9, 25, 24, 17, 17, 17]
    assert EXPECTED_LAYOUT["slices"] == {"S": [0, 1], "E": [1, 9], "P": [9, 17], "R": [17, 25]}


@pytest.mark.parametrize("manifest,task", [
    ({"config": {"task_name": "PSM.csv"}}, "PSM"),
    ({"task_name": "MSL.csv"}, "MSL"),
    ({"config": {"task_name": "detect_score/CICIDS_script/CICIDS.csv"}}, "CICIDS"),
])
def test_candidate_manifest_task_supports_current_and_legacy_locations(manifest, task):
    assert _manifest_task(manifest) == task


def test_candidate_manifest_prefilter_only_allows_exact_source_contract(tmp_path):
    source_path = tmp_path / "source"
    candidate_path = tmp_path / "candidate"
    source = {
        "path": str(source_path), "source_identity": {"id": "source"},
        "feature_fingerprint": "feature", "cache_sha256": {"train": "cache"},
        "endpoint_sha256": {"train": "endpoint"}, "gmm_fit_endpoints_sha256": "fit",
        "input_statistics_sha256": "stats", "fixed_noise_sha256": "noise",
        "config": {"seq_len": 192, "pattern_len": 32, "hidden_dim": 32,
                   "evolution_dim": 8, "pattern_dim": 8, "relation_dim": 8,
                   "view_dim": 8, "gmm_components": 4, "gmm_covariance_type": "full",
                   "gmm_reg_covar": 0.001, "gmm_n_init": 1, "gmm_max_iter": 100,
                   "gmm_fit_max_windows": 20000, "train_noise_std": 0.01,
                   "noise_seed": 20261013, "feature_std_floor": 0.05,
                   "score_mad_floor": 0.001, "calibration_quantile": 0.995},
    }
    manifest = {
        "selected_views": ["S"], "selected_indices": [0], "config": {**source["config"], "seed": 2021},
        "source_identity": source["source_identity"], "source_feature_fingerprint": "feature",
        "source_dir": str(source_path), "source_cache_sha256": source["cache_sha256"],
        "source_endpoint_sha256": source["endpoint_sha256"], "gmm_fit_endpoints_sha256": "fit",
        "raw_input_statistics_sha256": "stats", "fixed_noise_sha256": "noise",
        "head_type": "gmm", "head_status": "complete",
    }
    assert _manifest_may_match(manifest, candidate_path, source, "S", 2021)
    manifest["head_status"] = "in_progress"
    assert _manifest_may_match(manifest, candidate_path, source, "S", 2021,
                               require_complete=False)
    assert not _manifest_may_match(manifest, candidate_path, source, "S", 2021)
    manifest["head_status"] = "complete"
    manifest["source_identity"] = {"id": "other"}
    assert not _manifest_may_match(manifest, candidate_path, source, "S", 2021)


def test_plan_keeps_blocked_dataset_matrix_without_starting_work():
    plan = build_plan(["PSM"], validate_sources=False)
    assert plan["positions"] == 9
    assert len(plan["rows"]) == 9
    assert all(row["status"] == "blocked" for row in plan["rows"])
    assert all(row["selected_dim"] in {1, 9, 17, 24, 25} for row in plan["rows"])


@pytest.mark.parametrize("task", ["PSM", "Creditcard", "CICIDS"])
def test_retrained_full_tasks_require_explicit_source_registration(task, tmp_path, monkeypatch):
    monkeypatch.setattr(source_registry, "REGISTRY_DIR", tmp_path / "registry")
    assert TASKS[task][0] is None
    path, registration = source_registry.selected_source(task, TASKS[task][0])
    assert path is None and registration is None
    target = tmp_path / task / "new-full-run"
    record = {"path": str(target), "run_id": target.name,
              "source_identity_sha256": "identity", "feature_fingerprint": "features",
              "manifest_sha256": "manifest"}
    source_registry.write_registry(task, record)
    path, loaded = source_registry.selected_source(task, None)
    assert path == target.resolve()
    assert loaded == record


@pytest.mark.parametrize("task,old_id", [
    ("PSM", "full_gmm-full-20261021T112121Z-2466820"),
    ("Creditcard", "full_gmm-full-20261021T112159Z-2472093"),
    ("CICIDS", "full_gmm-full-20261021T112642Z-2510633"),
])
def test_retired_residual_removed_only_after_explicit_replacement(task, old_id, tmp_path, monkeypatch):
    monkeypatch.setattr(source_registration, "ROOT", tmp_path)
    task_root = tmp_path / "result/score/MNRF-ViewProbe" / task
    old = task_root / old_id
    new = task_root / "new-full-run"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "retained-until-replacement.txt").write_text("residual")
    (new / "probe_manifest.json").write_text("new")
    outcome = source_registration.retire_old_residual(task, new)
    assert outcome["status"] == "removed_after_new_source_registered"
    assert not old.exists()
    assert (new / "probe_manifest.json").read_text() == "new"


@pytest.mark.parametrize("task", ["PSM", "Creditcard", "CICIDS"])
def test_source_registration_passes_dataset_named_task_report(task, tmp_path, monkeypatch):
    monkeypatch.setattr(source_registration, "ROOT", tmp_path)
    monkeypatch.setattr(source_registry, "REGISTRY_DIR", tmp_path / "registry")
    task_root = tmp_path / "result/score/MNRF-ViewProbe" / task
    run_dir = task_root / "new-full-run"
    run_dir.mkdir(parents=True)
    expected_seq_len = TASKS[task][1]
    manifest = {
        "artifact_policy": "full", "mode": "pretrained", "stage_status": "complete",
        "head_type": "gmm", "selected_views": ["S", "E", "P", "R"],
        "config": {"seq_len": expected_seq_len, "seed": 2021},
        "source_identity_sha256": "identity", "feature_fingerprint": "features",
        "cache_sha256": {}, "endpoint_sha256": {},
    }
    (run_dir / "probe_manifest.json").write_text(json.dumps(manifest))
    (run_dir / "run_status.json").write_text(json.dumps({"status": "complete"}))
    observed = []

    def fake_validate(source, loaded_manifest, task_report, device):
        observed.append((source, task_report.name, device))
        return {"native_validation": {"status": "complete"}, "source_preflight_device": device}

    import scripts.recover_mnrf_view_probe_cache as recovery
    monkeypatch.setattr(recovery, "_validate_complete_source", fake_validate)
    source_registration.register(task, run_dir)
    assert observed == [(run_dir, task, "cpu")]


def test_refresh_prunes_only_old_inventory_batches_and_matching_review_zip(tmp_path):
    analysis = tmp_path / "analysis"
    review = tmp_path / "review"
    current = analysis / "branch-ablation-current"
    stale = analysis / "branch-ablation-stale"
    executed = analysis / "branch-ablation-executed"
    for path in (current, stale, executed):
        path.mkdir(parents=True)
    current_plan = {"batch_id": current.name, "mode": "first_round", "rows": []}
    stale_plan = {"batch_id": stale.name, "mode": "first_round",
                  "rows": [{"status": "reused"}, {"status": "planned/pending"}]}
    executed_plan = {"batch_id": executed.name, "mode": "first_round",
                     "rows": [{"status": "complete"}]}
    for path, plan in ((current, current_plan), (stale, stale_plan), (executed, executed_plan)):
        (path / "plan.json").write_text(json.dumps(plan))
    review.mkdir()
    archive = review / "MNRF_BRANCH_ABLATION_{}.zip".format(stale.name)
    with __import__("zipfile").ZipFile(archive, "w") as handle:
        handle.writestr("plan.json", json.dumps(stale_plan))
    removed = _prune_superseded_inventories(current.name, analysis, review)
    assert removed == [stale.name]
    assert current.is_dir() and executed.is_dir()
    assert not stale.exists() and not archive.exists()


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


def test_repeat_plan_is_rejected_instead_of_shrinking_to_single_seed(tmp_path):
    plan = {
        "tasks": ["PSM"], "seeds": [2021, 2022, 2023], "source_records": {},
        "rows": [], "batch_id": "obsolete-repeat",
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed head seed"):
        summarize(path)


def test_runner_rejects_removed_repeat_cli_without_creating_a_batch(tmp_path):
    runner = Path(__file__).resolve().parents[4] / "scripts" / "run_mnrf_branch_ablation.py"
    result = subprocess.run(
        [sys.executable, str(runner), "--task", "PSM", "--mode", "repeat"],
        cwd=str(runner.parents[1]), text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert not list((runner.parents[1] / "result/analysis/MNRF-ViewProbe-Ablation").glob("*repeat*"))


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
