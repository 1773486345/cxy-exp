"""Exercise the real formal launcher with an isolated tiny benchmark dataset."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.compare_mnrf_frozen_readouts import compare
from ts_benchmark.baselines.mnrf_lite_v2.MNRFLiteV2 import MNRFLiteV2, _hash_array
from ts_benchmark.baselines.mnrf_lite_v2.training import MNRFTrainer


ROOT = Path(__file__).resolve().parents[4]


def test_real_launcher_two_files_and_offline_three_readouts(tmp_path, monkeypatch):
    import torch
    torch.set_num_threads(1)
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    for name in ("run_mnrf_lite_v2_task.sh", "run_benchmark.py", "validate_mnrf_lite_v2_run.py",
                 "publish_mnrf_benchmark_outputs.py", "preflight_mnrf_lite_v2.py"):
        shutil.copy2(ROOT / "scripts" / name, project / "scripts" / name)
    (project / "ts_benchmark").symlink_to(ROOT / "ts_benchmark", target_is_directory=True)
    (project / "config").symlink_to(ROOT / "config", target_is_directory=True)
    data = project / "data_root"
    (data / "data").mkdir(parents=True)
    rng = np.random.default_rng(912)
    frame = pd.DataFrame(rng.normal(size=(520, 2)).astype(np.float32), columns=["9", "7"],
                         index=pd.date_range("2025-01-01", periods=520, freq="min"))
    frame["label"] = 0
    frame.iloc[478:482, -1] = 1
    long = frame.rename_axis("date").reset_index().melt(id_vars="date", var_name="cols", value_name="data")
    long[["date", "data", "cols"]].to_csv(data / "data/toy.csv", index=False)
    pd.DataFrame([dict(file_name="toy.csv", train_lens=430)]).to_csv(data / "DETECT_META.csv", index=False)
    # Metadata must remain read-only even with an unregistered adjacent CSV.
    (data / "data/unregistered.csv").write_text("not a registered dataset\n")
    before_meta = (data / "DETECT_META.csv").read_bytes()
    env = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
           "GPU_ID": "cpu", "MNRF_SEQ_LEN": "32", "BATCH_SIZE": "64",
           "ENCODING_BATCH_SIZE": "64", "STAGE1_EPOCHS": "1", "STAGE2_EPOCHS": "1",
           "MNRF_DATA_ROOT": str(data), "MNRF_ARTIFACT_POLICY": "readout",
           "MNRF_DATA_VERSION": "tiny-only", "MNRF_SAVE_PATH": "score/tiny/readout",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    for key in ("MNRF_RESUME_FROM", "MNRF_EXPECTED_FIELDS", "MNRF_EXPECTED_TEST", "MNRF_EXPECTED_CALIBRATION"):
        env.pop(key, None)
    command = ["bash", "scripts/run_mnrf_lite_v2_task.sh", "toy", "large_detect"]
    result = subprocess.run(command, cwd=project, env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    published = project / "result/score/tiny/readout"
    files = list(published.iterdir())
    assert len(files) == 2
    assert len(list(published.glob("*.tar.gz"))) == 1
    assert len(list(published.glob("test_report*.csv"))) == 1
    work = project / "result/.mnrf-work/score/tiny/readout"
    contract = json.loads((work / "readout_data_contract.json").read_text())
    assert contract["source_run_id"] == "readout" and contract["task"] == "toy"
    assert contract["data_version"] == "tiny-only"
    assert contract["data_root"] == str(data)
    assert contract["full_input_sha256"] == _hash_array(MNRFLiteV2._features_from_frame(frame)[0])
    assert (data / "DETECT_META.csv").read_bytes() == before_meta
    # No hand-written contract or report reference is added to make comparison work.
    rows = compare(work, "toy", frame, 430, tmp_path / "comparison")
    assert {r["readout"] for r in rows} == {"flow", "state_amplitude", "flow_plus_amplitude"}
    assert json.loads((tmp_path / "comparison/manifest.json").read_text())["status"] == "complete"
    # Publication refuses to overwrite a completed output, no implicit rerun.
    again = subprocess.run(command, cwd=project, env=env, text=True, capture_output=True, timeout=15)
    assert again.returncode != 0 and "destination already exists" in again.stderr


def test_contract_persisted_before_training_and_resume_rejects_changes(tmp_path, monkeypatch):
    frame = pd.DataFrame(np.random.default_rng(21).normal(size=(520, 2)).astype(np.float32),
                         columns=["9", "7"], index=pd.date_range("2025", periods=520, freq="min"))
    frame["label"] = 0
    run = tmp_path / "run"
    model = MNRFLiteV2(seq_len=32, device="cpu", output_dir=str(run))
    model.bind_readout_data(frame, 430, "toy.csv", "tiny")

    def interrupt(*args, **kwargs):
        assert (run / "readout_data_contract.json").is_file()
        raise RuntimeError("controlled early interruption")

    monkeypatch.setattr(MNRFTrainer, "stage1", interrupt)
    with pytest.raises(RuntimeError, match="controlled early interruption"):
        model.detect_fit(frame.iloc[:430].drop(columns="label"))
    original = (run / "readout_data_contract.json").read_bytes()
    resume = MNRFLiteV2(seq_len=32, device="cpu", output_dir=str(run), resume_from=str(run))
    resume.bind_readout_data(frame, 430, "toy.csv", "tiny")
    changed = frame.copy()
    changed.iloc[-1, 0] += 1
    with pytest.raises(ValueError, match="binding mismatch"):
        resume.bind_readout_data(changed, 430, "toy.csv", "tiny")
    assert (run / "readout_data_contract.json").read_bytes() == original
    (run / "readout_data_contract.json").unlink()
    with pytest.raises(ValueError, match="no retrospective binding"):
        resume.bind_readout_data(frame, 430, "toy.csv", "tiny")
    assert not (run / "readout_data_contract.json").exists()
