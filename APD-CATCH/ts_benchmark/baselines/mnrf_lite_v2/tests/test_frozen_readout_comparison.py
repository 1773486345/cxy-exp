"""A real tiny native run, followed by comparison with all forwards forbidden."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.compare_mnrf_frozen_readouts import compare, sha
from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2, _hash_array
from ts_benchmark.baselines.mnrf_lite_v2.views import FourViewEncoder
from ts_benchmark.baselines.mnrf_lite_v2.fusion import AlternatingAffineFlow


@pytest.fixture
def native_run(tmp_path, monkeypatch):
    from ts_benchmark.evaluation.evaluator import Evaluator
    import ts_benchmark.evaluation.strategy.mnrf_valid_detect as strategy_module
    torch.set_num_threads(1)
    rng = np.random.default_rng(912)
    frame = pd.DataFrame(rng.normal(size=(520, 2)).astype(np.float32), columns=["9", "7"],
                         index=pd.date_range("2025-01-01", periods=520, freq="min"))
    frame["label"] = 0
    frame.iloc[478:482, -1] = 1

    class Pool:
        def get_series(self, name):
            return frame

        def get_series_meta_info(self, name):
            return pd.Series({"train_lens": 430})

    class DataPool:
        def get_pool(self):
            return Pool()

    monkeypatch.setattr(strategy_module, "DataPool", DataPool)
    run = tmp_path / "native-flow"
    model = MNRFLiteV2(seq_len=32, batch_size=64, stage1_epochs=1, stage2_epochs=1,
                       patience=1, device="cpu", task_name="toy.csv", output_dir=str(run))
    evaluator = Evaluator([{"name": "auc_roc"}, {"name": "auc_pr"}])
    strategy = strategy_module.MNRFValidDetect({"strategy_name": "mnrf_valid_detect"}, evaluator)
    strategy.execute("toy.csv", lambda: model)
    metrics = json.loads((run / "results/toy.csv_metrics.json").read_text())
    report = run / "test_report_synthetic.csv"
    pd.DataFrame({"metric_name": ["auc_roc", "auc_pr"], "MNRFLiteV2;synthetic":
                  [metrics["auc_roc"], metrics["auc_pr"]]}).to_csv(report, index=False)
    context = json.loads((run / "benchmark_context.json").read_text())
    (run / "report_reference.json").write_text(json.dumps(dict(
        task="toy.csv", model_name="MNRFLiteV2", run_id=run.name,
        benchmark_invocation_id=context["benchmark_invocation_id"], report_path=str(report),
        report_sha256=sha(report), model_column="MNRFLiteV2;synthetic",
        metrics={key: metrics[key] for key in ("auc_roc", "auc_pr")},
    )))
    contract = json.loads((run / "readout_data_contract.json").read_text())
    assert contract["captured"] == "actual benchmark pool input, before fit"
    return run, frame, model


def forbid(*args, **kwargs):
    raise AssertionError("offline comparison must not train or run any model forward")


def test_native_old_flow_three_readouts_no_forward(native_run, tmp_path, monkeypatch):
    run, frame, model = native_run
    # Independently obtain new-checkpoint reference states using the original trainer
    # calibration, before forbidding forward. This is synthetic data only.
    validation = np.load(run / "cache/validation_features.npy", mmap_mode="r")
    calibration = np.load(run / "cache/calibration_features.npy", mmap_mode="r")
    model.trainer.calibrate(validation, calibration, run / "cache", with_amplitude=True)
    expected = {name: model.score_with_metadata(frame.iloc[430:].drop(columns="label"), 430,
                 score_readout=name) for name in ("flow", "state_amplitude", "flow_plus_amplitude")}
    # The actual source is still its old-style Flow-only deployment.
    assert "amplitude_calibration" not in torch.load(run / "deployment.pt", map_location="cpu", weights_only=False)
    before = {str(p.relative_to(run)): sha(p) for p in run.rglob("*") if p.is_file()}
    monkeypatch.setattr(FourViewEncoder, "encode_views", forbid)
    monkeypatch.setattr(AlternatingAffineFlow, "forward", forbid)
    monkeypatch.setattr(MNRFLiteV2, "detect_fit", forbid)
    output = tmp_path / "comparison"
    rows = compare(run, "toy", frame, 430, output)
    assert len(rows) == 6
    with np.load(output / "scores.npz", allow_pickle=False) as pack:
        for name, reference in expected.items():
            np.testing.assert_array_equal(pack[name+"_scores"], reference["scores"])
            np.testing.assert_array_equal(pack[name+"_predictions"], reference["predictions"])
            assert float(pack[name+"_threshold"]) == reference["threshold"]
            np.testing.assert_array_equal(pack["time_indices"], reference["time_indices"])
    assert before == {str(p.relative_to(run)): sha(p) for p in run.rglob("*") if p.is_file()}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["cpu_threads"] == 1
    assert manifest["variable_names"] == ["9", "7"]
    with pytest.raises(ValueError, match="output already exists"):
        compare(run, "toy", frame, 430, output)


@pytest.mark.parametrize("change", ["label", "time", "field_order", "source_input", "test_input", "nll", "mask", "readout", "missing_data_contract"])
def test_binding_rejection_before_output(native_run, tmp_path, change):
    run, frame, _ = native_run
    frame = frame.copy()
    if change == "label":
        frame.iloc[478, -1] = 0
    elif change == "time":
        index = frame.index.to_numpy().copy()
        index[-1] += np.timedelta64(1, "m")
        frame.index = index
    elif change == "field_order":
        frame = frame[["7", "9", "label"]]
    elif change == "source_input":
        frame.iloc[0, 0] += 1
    elif change == "test_input":
        frame.iloc[-1, 0] += 1
    elif change == "missing_data_contract":
        (run / "readout_data_contract.json").unlink()
    elif change == "nll":
        nll = np.load(run / "cache/calibration_nll.npy", mmap_mode="r+")
        nll[:] += 1e8
        nll.flush()
    else:
        path = run / "results/toy.csv_aligned_scores.npz"
        with np.load(path, allow_pickle=False) as z:
            pack = dict(z)
        if change == "mask":
            pack["local_valid_mask"] = np.roll(pack["local_valid_mask"], 1)
        else:
            pack["score_readout"] = np.asarray("state_amplitude")
        np.savez_compressed(path, **pack)
    output = tmp_path / "rejected"
    with pytest.raises((ValueError, SystemExit)):
        compare(run, "toy", frame, 430, output)
    assert not output.exists()


def test_missing_nll_preserves_available_readouts(native_run, tmp_path, monkeypatch):
    run, frame, _ = native_run
    (run / "cache/validation_nll.npy").unlink()
    monkeypatch.setattr(FourViewEncoder, "encode_views", forbid)
    monkeypatch.setattr(AlternatingAffineFlow, "forward", forbid)
    output = tmp_path / "partial"
    rows = compare(run, "toy", frame, 430, output)
    assert {r["readout"] for r in rows if r.get("scope") != "not_computable"} == {"flow", "state_amplitude"}
    missing = next(r for r in rows if r.get("status") == "missing_artifacts")
    assert "validation_nll.npy" in missing["reason"]
    assert json.loads((output / "manifest.json").read_text())["status"] == "partial"


def test_real_parent_checkpoint_compatibility(tmp_path):
    """Parent code really trains/scores in an isolated subprocess, not a fake payload."""
    import io
    import os
    import subprocess
    import sys
    import tarfile
    repo = Path(__file__).resolve().parents[5]
    parent = subprocess.check_output(["git", "rev-parse", "4c264c9^"], cwd=repo, text=True).strip()
    archive = subprocess.check_output(["git", "archive", parent, "APD-CATCH/ts_benchmark"], cwd=repo)
    checkout = tmp_path / "parent"
    checkout.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(checkout)
    source = checkout / "APD-CATCH"
    driver = '''
import sys
import numpy as np
import pandas as pd
import torch
from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2
torch.set_num_threads(1)
rng = np.random.default_rng(2021)
frame = pd.DataFrame(rng.normal(size=(430, 2)), columns=["x", "y"])
test = pd.DataFrame(rng.normal(size=(75, 2)), columns=["x", "y"])
frame["entity_id"] = "a"
test["entity_id"] = ["a"]*36 + ["b"]*39
model = MNRFLiteV2(seq_len=32, batch_size=64, stage1_epochs=1, stage2_epochs=1,
                   patience=1, device="cpu", output_dir=sys.argv[1])
model.detect_fit(frame)
s = model.score_with_metadata(test, 430)
np.savez(sys.argv[2], values=test[["x", "y"]].to_numpy(), scores=s["scores"],
         predictions=s["predictions"], threshold=s["threshold"],
         time_indices=s["time_indices"], valid_mask=s["valid_mask"])
'''
    run, reference = tmp_path / "old-run", tmp_path / "old-reference.npz"
    result = subprocess.run([sys.executable, "-c", driver, str(run), str(reference)], cwd=source,
        env={**os.environ, "PYTHONPATH": str(source), "CUDA_VISIBLE_DEVICES": "",
             "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    deployment = run / "deployment.pt"
    assert "amplitude_calibration" not in torch.load(deployment, map_location="cpu", weights_only=False)
    model = MNRFLiteV2(device="cpu").load_checkpoint(deployment)
    with np.load(reference, allow_pickle=False) as expected:
        test = pd.DataFrame(expected["values"], columns=["x", "y"])
        test["entity_id"] = ["a"]*36 + ["b"]*39
        actual = model.score_with_metadata(test, 430)
        for key in ("scores", "predictions", "threshold", "time_indices", "valid_mask"):
            np.testing.assert_array_equal(actual[key], expected[key])
    assert model.score_readout == "flow"


def test_new_checkpoint_calibration_and_cli(native_run, tmp_path):
    import os
    import subprocess
    import sys
    run, frame, model = native_run
    model.trainer.calibrate(np.load(run / "cache/validation_features.npy", mmap_mode="r"),
                            np.load(run / "cache/calibration_features.npy", mmap_mode="r"),
                            run / "cache", with_amplitude=True)
    model._save_artifacts()
    data = tmp_path / "external"
    (data / "data").mkdir(parents=True)
    frame.rename_axis("timestamp").reset_index().to_csv(data / "data/toy.csv", index=False)
    pd.DataFrame([dict(file_name="toy.csv", train_lens=430)]).to_csv(
        data / "EXTERNAL_DETECT_META.csv", index=False)
    output = tmp_path / "from-cli"
    root = Path(__file__).resolve().parents[4]
    command = [sys.executable, "scripts/compare_mnrf_frozen_readouts.py", "--task", "toy",
               "--run-dir", str(run), "--data-source", "external_detect", "--data-root", str(data),
               "--output-dir", str(output)]
    result = subprocess.run(command, cwd=root, env={**os.environ, "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout+result.stderr
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"
    again = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=45)
    assert again.returncode != 0
