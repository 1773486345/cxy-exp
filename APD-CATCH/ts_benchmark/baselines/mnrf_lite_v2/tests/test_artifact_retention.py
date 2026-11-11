"""Storage policy checks without fitting or invoking any model."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.publish_mnrf_benchmark_outputs import publish


ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("task", ["CalIt2", "PSM"])
@pytest.mark.parametrize("requested", [None, "readout", "full"])
def test_formal_wrapper_retention_default_and_explicit_override(tmp_path, task, requested):
    relative = Path("scripts/multivariate_detection/detect_score") / (task + "_script/MNRFLiteV2.sh")
    wrapper = tmp_path / relative
    wrapper.parent.mkdir(parents=True)
    shutil.copy2(ROOT / relative, wrapper)
    # Exercise the actual wrapper, intercept only the generic launcher so that
    # no preflight, data load, training, forward, or publication can run.
    (tmp_path / "scripts/run_mnrf_lite_v2_task.sh").write_text(
        'set -eu\n'
        'printf "%s|%s|%s|%s|%s|%s|%s\\n" "$MNRF_ARTIFACT_POLICY" '
        '"$MNRF_SEQ_LEN" "$MNRF_SCORE_READOUT" "$BATCH_SIZE" '
        '"$ENCODING_BATCH_SIZE" "$CUDA_VISIBLE_DEVICES" "$1"\n'
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="7,3")
    env.pop("MNRF_ARTIFACT_POLICY", None)
    if requested is not None:
        env["MNRF_ARTIFACT_POLICY"] = requested
    result = subprocess.run(["bash", str(wrapper)], env=env, text=True,
                            capture_output=True, check=True, timeout=10)
    assert result.stdout.strip().split("|") == [requested or "benchmark", "192", "flow",
                                               "128", "128", "7,3", task]


@pytest.mark.parametrize("keep_work", [False, True])
def test_publication_retains_only_explicit_readout_work(tmp_path, keep_work):
    relative = Path("score/MNRF-Lite-V2/toy/run-1")
    work = tmp_path / ".mnrf-work" / relative
    work.mkdir(parents=True)
    report, record = work / "test_report_toy.csv", work / "native.tar.gz"
    report.write_bytes(b"metric,value\nauc,0.5\n")
    record.write_bytes(b"synthetic native record")
    (work / "training-cache.npy").write_bytes(b"synthetic work cache")
    (work / "report_reference.json").write_text(json.dumps(
        dict(run_id=work.name, report_path=str(report), record_paths=[str(record)])))
    other = tmp_path / ".mnrf-work/other-run"
    other.mkdir()
    (other / "evidence").write_bytes(b"keep")
    published = publish(tmp_path, relative, keep_work=keep_work)
    assert sorted(p.name for p in published.iterdir()) == [record.name, report.name]
    assert (published / report.name).read_bytes() == b"metric,value\nauc,0.5\n"
    assert work.exists() == keep_work
    if keep_work:
        assert (work / "training-cache.npy").read_bytes() == b"synthetic work cache"
    assert (other / "evidence").read_bytes() == b"keep"
    with pytest.raises(FileExistsError):
        publish(tmp_path, relative, keep_work=keep_work)
