"""Static and capture tests for independent ViewProbe task launchers."""

import json
import os
from pathlib import Path
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

from scripts.mnrf_lite_v2_preflight import _disk_estimate
from ts_benchmark.baselines.mnrf_lite_v2.training import purged_ranges


ROOT = Path(__file__).resolve().parents[4]
SCRIPT_ROOT = ROOT / "scripts" / "multivariate_detection" / "detect_score"
SCRIPTS = {
    "PSM": SCRIPT_ROOT / "PSM_script" / "MNRFViewProbe.sh",
    "MSL": SCRIPT_ROOT / "MSL_script" / "MNRFViewProbe.sh",
    "GECCO": SCRIPT_ROOT / "GECCO_script" / "MNRFViewProbe.sh",
    "Genesis": SCRIPT_ROOT / "Genesis_script" / "MNRFViewProbe.sh",
    "NYC": SCRIPT_ROOT / "NYC_script" / "MNRFViewProbe.sh",
    "CalIt2": SCRIPT_ROOT / "CalIt2_script" / "MNRFViewProbe.sh",
    "Creditcard": SCRIPT_ROOT / "Creditcard_script" / "MNRFViewProbe.sh",
    "CICIDS": SCRIPT_ROOT / "CICIDS_script" / "MNRFViewProbe.sh",
}
LEGACY_SMD = SCRIPT_ROOT / "SMD_script" / "MNRFViewProbe.sh"
REFERENCE = {
    "PSM": (192, 128), "MSL": (192, 128), "GECCO": (96, 128),
    "Genesis": (192, 4), "NYC": (192, 128), "CalIt2": (192, 128),
    "Creditcard": (192, 128), "CICIDS": (192, 128),
}


def _capture_script(tmp_path, script, preflight_fail=False, python_status=0, tee_status=0, export_status=0):
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True)
    capture = tmp_path / "calls.log"
    fake_python = bindir / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'CUDA_VISIBLE_DEVICES=%s ' \"${CUDA_VISIBLE_DEVICES-<unset>}\" >> \"$CAPTURE\"\n"
        "printf '<%s> ' \"$@\" >> \"$CAPTURE\"\n"
        "printf '\\n' >> \"$CAPTURE\"\n"
        "if [[ \"${1##*/}\" == 'mnrf_lite_v2_preflight.py' ]]; then exit \"${PYTHON_STATUS:-0}\"; fi\n"
        "if [[ \"${1##*/}\" == 'export_mnrf_view_probe_analysis.py' ]]; then exit \"${EXPORT_STATUS:-0}\"; fi\n"
    )
    fake_python.chmod(0o755)
    fake_tee = bindir / "tee"
    fake_tee.write_text("#!/usr/bin/env bash\ncat >/dev/null\nexit \"${TEE_STATUS:-0}\"\n")
    fake_tee.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["CAPTURE"] = str(capture)
    env["PYTHON_STATUS"] = str(python_status if python_status else (23 if preflight_fail else 0))
    env["TEE_STATUS"] = str(tee_status)
    env["EXPORT_STATUS"] = str(export_status)
    env["CUDA_VISIBLE_DEVICES"] = "9"
    env.update({
        "MODE": "stage1", "SEED": "7", "SEQ_LEN": "32", "BATCH_SIZE": "999",
        "EVOLUTION_DIM": "3", "VIEW_PROBE_SOURCE_DIR": "/old/source",
        "VIEW_PROBE_RESUME_FROM": "/old/resume", "VIEW_PROBE_OUTPUT": "/old/output",
        "DEVICE": "cpu",
    })
    result = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    return result, capture.read_text().splitlines()


def test_all_task_scripts_are_executable_and_valid_bash():
    for script in (*SCRIPTS.values(), LEGACY_SMD):
        assert script.exists()
        assert os.access(script, os.X_OK)
        result = subprocess.run(["bash", "-n", str(script)], check=False)
        assert result.returncode == 0


def test_scripts_capture_literal_configuration_without_environment_override(tmp_path):
    for task, script in SCRIPTS.items():
        result, lines = _capture_script(tmp_path / task, script)
        assert result.returncode == 0, result.stderr
        assert len(lines) == 3
        preflight, run, export = lines
        seq_len, batch_size = REFERENCE[task]
        assert f"CUDA_VISIBLE_DEVICES=9 <{ROOT}/scripts/mnrf_lite_v2_preflight.py>" in preflight
        assert f"<--task> <{task}>" in preflight
        assert f"<--seq-len> <{seq_len}>" in preflight
        assert f"<--batch-size> <{batch_size}>" in preflight
        assert f"<--task> <{task}>" in run
        assert "<--mode> <full_gmm>" in run
        assert "<--seed> <2021>" in run
        assert f"<--seq-len> <{seq_len}>" in run
        assert f"<--batch-size> <{batch_size}>" in run
        assert "<--pattern-len> <32>" in run
        assert "<--hidden-dim> <32>" in run
        assert "<--evolution-dim> <8>" in run
        assert "<--pattern-dim> <8>" in run
        assert "<--relation-dim> <8>" in run
        assert "<--artifact-policy> <full>" in run
        assert "<--startup-script>" in run
        assert "<--log-every-steps> <100>" in run
        assert "<--log-every-seconds> <30>" in run
        assert "<--gmm-log-every-iters> <10>" in run
        assert "<--learning-rate> <0.001>" in run
        assert "<--weight-decay> <0.0001>" in run
        assert "<--source-dir> <>" in run
        assert "<--resume-from> <>" in run
        assert "<--gmm-components> <4>" in run
        assert "<--gmm-covariance-type> <full>" in run
        assert "<--gmm-fit-max-windows> <20000>" in run
        assert "/old/source" not in run and "/old/resume" not in run and "/old/output" not in run
        assert f"MNRF-ViewProbe/{task}/full_gmm-" in run
        assert "export_mnrf_view_probe_analysis.py" in export


def test_preflight_failure_stops_before_training(tmp_path):
    result, lines = _capture_script(tmp_path, SCRIPTS["PSM"], preflight_fail=True)
    assert result.returncode == 23
    assert len(lines) == 1
    assert "mnrf_lite_v2_preflight.py" in lines[0]


def test_preflight_python_and_tee_failures_return_nonzero_and_stop(tmp_path):
    cases = ((23, 0, 23), (0, 17, 17), (23, 17, 23))
    for index, (python_status, tee_status, expected) in enumerate(cases):
        result, lines = _capture_script(tmp_path / str(index), SCRIPTS["NYC"],
                                        python_status=python_status, tee_status=tee_status)
        assert result.returncode == expected
        assert len(lines) == 1
        assert "mnrf_lite_v2_preflight.py" in lines[0]
        assert "export_mnrf_view_probe_analysis.py" not in result.stderr


def test_analysis_export_failure_is_nonzero_and_reports_both_paths(tmp_path):
    result, lines = _capture_script(tmp_path, SCRIPTS["NYC"], export_status=19)
    assert result.returncode == 19
    assert len(lines) == 3
    assert "analysis_export=failed" in result.stderr
    assert "output_dir=" in result.stderr and "analysis_dir=" in result.stderr
    assert "analysis_export=complete" not in result.stdout


def test_two_invocations_get_distinct_run_and_analysis_paths(tmp_path):
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(_capture_script, tmp_path / "first", SCRIPTS["PSM"])
        second_future = executor.submit(_capture_script, tmp_path / "second", SCRIPTS["PSM"])
        first, first_lines = first_future.result()
        second, second_lines = second_future.result()
    assert first.returncode == second.returncode == 0
    run_outputs = []
    analysis_outputs = []
    for lines in (first_lines, second_lines):
        run_tokens = lines[1].split()
        run_outputs.append(next(run_tokens[i + 1][1:-1] for i, token in enumerate(run_tokens) if token == "<--output-dir>"))
        analysis_tokens = lines[2].split()
        analysis_outputs.append(next(analysis_tokens[i + 1][1:-1] for i, token in enumerate(analysis_tokens) if token == "<--output-dir>"))
    assert run_outputs[0] != run_outputs[1]
    assert analysis_outputs[0] != analysis_outputs[1]


def test_calit2_purged_counts_and_preflight_storage_formula(tmp_path):
    split = purged_ranges(2520, 192)
    counts = dict(split["window_counts"])
    counts["test"] = 5040 - 2520 - 192 + 1
    assert counts == {"train": 1519, "validation": 22, "calibration": 24, "test": 2329}
    estimate = _disk_estimate(counts, 2, 25, 5040, str(tmp_path / "not-created" / "run"), tmp_path,
                              hidden_dim=32, evolution_dim=8, pattern_dim=8, relation_dim=8)
    expected_features = sum(counts.values()) * 2 * 25 * 4
    assert estimate["feature_cache_bytes"] == expected_features
    assert estimate["normalized_values_bytes"] == 5040 * 2 * 4
    assert estimate["endpoint_index_bytes"] == sum(counts.values()) * 8
    assert estimate["validation_calibration_test_nll_bytes"] == (22 + 24 + 2329) * 2 * 4
    assert estimate["target_volume_path"] == str(tmp_path)
    assert estimate["not_peak_ram_or_vram"] is True


def test_independent_scripts_do_not_select_environment_or_device():
    forbidden = ("conda activate", "conda run", "conda.sh", "CUDA_VISIBLE_DEVICES=", "${BATCH_SIZE:-")
    for script in (*SCRIPTS.values(), LEGACY_SMD):
        text = script.read_text()
        assert not any(item in text for item in forbidden)
        assert 'DEVICE="cuda:0"' in text


def test_public_catch_window_and_batch_values_are_copied():
    for task, script in SCRIPTS.items():
        catch = (SCRIPT_ROOT / f"{task}_script" / "CATCH.sh").read_text()
        match = re.search(r"--model-hyper-params\s+'([^']+)'", catch)
        assert match, task
        params = json.loads(match.group(1))
        seq_len, batch_size = REFERENCE[task]
        text = script.read_text()
        assert re.search(rf'(?:^|; )SEQ_LEN="{seq_len}"', text, re.MULTILINE)
        assert re.search(rf'(?:^|; )BATCH_SIZE="{batch_size}"', text, re.MULTILINE)
        assert params["seq_len"] == seq_len
        assert params["batch_size"] == batch_size
