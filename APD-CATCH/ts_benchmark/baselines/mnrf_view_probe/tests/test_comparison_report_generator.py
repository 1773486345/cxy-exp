import json

from scripts.generate_mnrf_comparison_diagnosis import (
    REQUIRED_SOURCE_FILES,
    REQUIRED_SPLIT_FILES,
    source_contract,
)


def test_source_contract_files_present_but_failed_status_is_not_verified(tmp_path):
    source = tmp_path / "failed-full"
    source.mkdir()
    for relative in REQUIRED_SOURCE_FILES + REQUIRED_SPLIT_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    identity = {
        "schema": "test-source-identity",
        "task_name": "PSM.csv",
        "seq_len": 32,
        "variable_names": ["x"],
    }
    (source / "run_status.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    (source / "probe_manifest.json").write_text(json.dumps({
        "source_identity": identity,
        "feature_fingerprint": "feature-hash",
        "fixed_noise_sha256": "noise-hash",
        "config": {"task_name": "PSM.csv"},
    }), encoding="utf-8")
    reference = {
        "source_run_id": "failed-source",
        "source_run": str(source),
        "source_identity": json.dumps(identity),
        "feature_fingerprint": "feature-hash",
        "fixed_noise_sha256": "noise-hash",
        "fit_endpoint_sha256": "fit-hash",
        "input_statistics_sha256": "stats-hash",
        "cache_sha256": "cache-hash",
    }
    full = {"dataset": "PSM", "source_run_id": "failed-source"}

    result = source_contract([full], [reference])

    assert result["PSM"]["presence_checked"] is True
    assert result["PSM"]["run_status"] == "failed"
    assert result["PSM"]["status"] == "presence_checked"
    assert "not complete" in result["PSM"]["reason"]
