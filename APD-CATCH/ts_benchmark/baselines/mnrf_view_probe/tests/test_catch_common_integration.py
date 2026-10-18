import argparse
import json

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_mnrf_view_probe_common import evaluate
from scripts.export_mnrf_view_probe_common import export
from scripts.register_mnrf_catch_common import register
from scripts.run_mnrf_catch_common import run as run_catch
from ts_benchmark.baselines.mnrf_view_probe import MNRFViewProbe
from ts_benchmark.baselines.mnrf_view_probe.evaluation import calibrated_quantile


def test_float32_and_float64_calibration_reproduction():
    for dtype in (np.float32, np.float64):
        values = np.asarray([0.1, 0.3, 1.2, 2.6, 5.1], dtype=dtype)
        threshold = calibrated_quantile(values, 0.8)
        assert threshold == float(np.quantile(values, 0.8, method="linear"))
        assert abs(threshold - 999.0) > 1.0


def test_real_probe_and_catch_common_train_register_strict_evaluate(tmp_path):
    rng = np.random.default_rng(7103)
    n, train_length = 520, 430
    timestamps = pd.date_range("2026-05-01", periods=n, freq="min")
    values = rng.normal(size=(n, 2)).astype(np.float32)
    labels = np.zeros(n, dtype=np.int64)
    labels[450::3] = 1
    frame = pd.DataFrame(values, columns=("x", "y"), index=timestamps)
    frame["label"] = labels

    probe_dir = tmp_path / "probe"
    probe = MNRFViewProbe(seq_len=32, pattern_len=32, batch_size=64, stage1_epochs=1,
                          stage2_epochs=1, patience=1, device="cpu", output_dir=str(probe_dir),
                          task_name="PSM", mode="full_gmm", gmm_components=1,
                          gmm_fit_max_windows=32, cpu_threads=1)
    probe.fit(frame, train_length, labels[train_length:])
    data_path = tmp_path / "PSM.csv"
    csv_frame = frame.reset_index(names="timestamp")
    csv_frame.to_csv(data_path, index=False)
    contract = export(probe_dir, data_path, tmp_path / "contract")
    with pytest.raises(ValueError, match="catch_seq_len=64"):
        export(probe_dir, data_path, tmp_path / "contract-catch64", catch_seq_len=64)
    split_paths = {name: tmp_path / "contract" / f"catch_{name}_indices.npy"
                   for name in ("train", "validation", "calibration")}
    fit_rows_path = tmp_path / "contract" / "train_rows.npy"
    labels_path = tmp_path / "contract" / "labels.npy"

    catch_dir = run_catch(argparse.Namespace(
        data_csv=str(data_path), train_indices=str(split_paths["train"]),
        validation_indices=str(split_paths["validation"]),
        calibration_indices=str(split_paths["calibration"]), train_rows=str(fit_rows_path),
        output=str(tmp_path / "catch"), task_name="PSM", test_start=train_length,
        common_contract=str(tmp_path / "contract" / "common_contract.json"),
        device="cpu", seed=2021, seq_len=32, patch_size=8, patch_stride=4,
        d_model=8, cf_dim=4, d_ff=16, e_layers=1, n_heads=2, head_dim=4,
        dropout=0.0, learning_rate=1e-4, epochs=1, batch_size=64, patience=1,
        calibration_quantile=0.995, score_lambda=0.05, mask_learning_rate=1e-5,
        inference_patch_size=8, inference_patch_stride=1, pct_start=0.3,
    ))
    probe_scores = probe_dir / "results" / "aligned_scores.npz"
    catch_scores = catch_dir / "results" / "aligned_scores.npz"
    training_evidence = catch_dir / "evidence" / "training.json"
    calibration_evidence = catch_dir / "evidence" / "calibration.json"
    common_manifest = tmp_path / "common.json"
    register(argparse.Namespace(
        probe=str(probe_scores), catch=str(catch_scores), labels=str(labels_path),
        train_indices=str(split_paths["train"]), validation_indices=str(split_paths["validation"]),
        calibration_indices=str(split_paths["calibration"]),
        training_evidence=str(training_evidence), calibration_evidence=str(calibration_evidence),
        task_name="PSM", output=str(common_manifest),
    ))
    result = evaluate(str(probe_scores), str(catch_scores), str(labels_path),
                      split_paths=(str(split_paths["train"]), str(split_paths["validation"]),
                                   str(split_paths["calibration"])),
                      common_manifest=str(common_manifest), strict_common=True)
    assert result["protocol"] == "CATCH-common"
    assert result["common_count"] > 0
    assert result["catch"]["auc_pr"] is not None
    assert result["probe"]["threshold"] == float(np.load(probe_scores)["threshold"])
    registered = json.loads(common_manifest.read_text())
    assert registered["protocol"] == "CATCH-common"
    assert contract["raw_training_range"] == list(probe.source_identity["ranges"]["train"])
    with pytest.raises(ValueError, match="requires a frozen common contract"):
        evaluate(str(probe_scores), str(catch_scores), str(labels_path),
                 split_paths=(str(split_paths["train"]), str(split_paths["validation"]),
                              str(split_paths["calibration"])), strict_common=True)
    tampered_manifest = tmp_path / "tampered-common.json"
    tampered = dict(registered)
    tampered["calibration_identity"] = dict(registered["calibration_identity"])
    embedded = dict(registered["calibration_identity"]["evidence"])
    embedded["threshold"] = 999.0
    tampered["calibration_identity"]["evidence"] = embedded
    tampered_manifest.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="embedded evidence differs"):
        evaluate(str(probe_scores), str(catch_scores), str(labels_path),
                 split_paths=(str(split_paths["train"]), str(split_paths["validation"]),
                              str(split_paths["calibration"])),
                 common_manifest=str(tampered_manifest), strict_common=True)

    # The producer receipt, not a newly calculated hash, owns the test score.
    replaced_scores = tmp_path / "replaced.npz"
    with np.load(catch_scores, allow_pickle=False) as payload:
        values = {name: np.asarray(payload[name]) for name in payload.files}
    values["scores"] = np.zeros_like(values["scores"])
    np.savez_compressed(replaced_scores, **values)
    with pytest.raises(ValueError, match="production score"):
        register(argparse.Namespace(
            probe=str(probe_scores), catch=str(replaced_scores), labels=str(labels_path),
            train_indices=str(split_paths["train"]), validation_indices=str(split_paths["validation"]),
            calibration_indices=str(split_paths["calibration"]),
            training_evidence=str(training_evidence), calibration_evidence=str(calibration_evidence),
            task_name="PSM", output=str(tmp_path / "replaced-manifest.json"),
        ))

    reference = evaluate(str(probe_scores), str(catch_scores), str(labels_path),
                         split_paths=(str(split_paths["train"]), str(split_paths["validation"]),
                                      str(split_paths["calibration"])),
                         common_manifest=str(common_manifest), strict_common=False)
    assert reference["protocol"] == "index_intersection_reference"


def test_entity_boundary_common_contract_is_consumed_end_to_end(tmp_path):
    rng = np.random.default_rng(7110)
    n, train_length = 520, 430
    # Each entity has an independent clock; no window may cross the entity
    # boundary even though the rows share one table.
    timestamps = list(pd.date_range("2026-06-01", periods=260, freq="min"))
    timestamps += list(pd.date_range("2026-06-01", periods=260, freq="min"))
    frame = pd.DataFrame({"timestamp": timestamps,
                          "entity_id": ["a"] * 260 + ["b"] * 260,
                          "x": rng.normal(size=n).astype(np.float32),
                          "y": rng.normal(size=n).astype(np.float32),
                          "label": np.r_[np.zeros(train_length, dtype=np.int64), np.arange(n - train_length) % 2]})
    probe_dir = tmp_path / "probe-entities"
    probe = MNRFViewProbe(seq_len=32, pattern_len=32, batch_size=64, stage1_epochs=1,
                          stage2_epochs=1, patience=1, device="cpu", output_dir=str(probe_dir),
                          task_name="PSM", mode="full_gmm", gmm_components=1,
                          gmm_fit_max_windows=32, cpu_threads=1)
    probe.fit(frame.set_index("timestamp"), train_length, frame["label"].to_numpy()[train_length:])
    data_path = tmp_path / "PSM-entities.csv"
    frame.to_csv(data_path, index=False)
    contract = export(probe_dir, data_path, tmp_path / "contract-entities")
    assert contract["boundary_columns"] == ["entity_id"]
    split_paths = {name: tmp_path / "contract-entities" / f"catch_{name}_indices.npy"
                   for name in ("train", "validation", "calibration")}
    catch_dir = run_catch(argparse.Namespace(
        data_csv=str(data_path), train_indices=str(split_paths["train"]),
        validation_indices=str(split_paths["validation"]), calibration_indices=str(split_paths["calibration"]),
        train_rows=str(tmp_path / "contract-entities" / "train_rows.npy"),
        common_contract=str(tmp_path / "contract-entities" / "common_contract.json"),
        output=str(tmp_path / "catch-entities"), task_name="PSM", test_start=None,
        device="cpu", seed=2021, seq_len=32, patch_size=8, patch_stride=4,
        d_model=8, cf_dim=4, d_ff=16, e_layers=1, n_heads=2, head_dim=4,
        dropout=0.0, learning_rate=1e-4, epochs=1, batch_size=64, patience=1,
        calibration_quantile=0.995, score_lambda=0.05, mask_learning_rate=1e-5,
        inference_patch_size=8, inference_patch_stride=1, pct_start=0.3,
    ))
    labels_path = tmp_path / "contract-entities" / "labels.npy"
    register(argparse.Namespace(
        probe=str(probe_dir / "results" / "aligned_scores.npz"),
        catch=str(catch_dir / "results" / "aligned_scores.npz"), labels=str(labels_path),
        train_indices=str(split_paths["train"]), validation_indices=str(split_paths["validation"]),
        calibration_indices=str(split_paths["calibration"]),
        training_evidence=str(catch_dir / "evidence" / "training.json"),
        calibration_evidence=str(catch_dir / "evidence" / "calibration.json"),
        task_name="PSM", output=str(tmp_path / "entities-common.json"),
    ))
    result = evaluate(str(probe_dir / "results" / "aligned_scores.npz"),
                      str(catch_dir / "results" / "aligned_scores.npz"), str(labels_path),
                      split_paths=tuple(str(split_paths[name]) for name in ("train", "validation", "calibration")),
                      common_manifest=str(tmp_path / "entities-common.json"), strict_common=True)
    assert result["protocol"] == "CATCH-common"
