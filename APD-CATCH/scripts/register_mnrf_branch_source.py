#!/usr/bin/env python
"""Register an explicitly completed replacement Full source for ablations."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mnrf_branch_source_registry import REGISTRY_DIR, write_registry
from scripts.run_mnrf_branch_ablation import ROOT, TASKS, _sha256_file, _json

RETIRED_RUNS = {
    "PSM": "full_gmm-full-20261021T112121Z-2466820",
    "Creditcard": "full_gmm-full-20261021T112159Z-2472093",
    "CICIDS": "full_gmm-full-20261021T112642Z-2510633",
}


def retire_old_residual(task, new_run):
    task_root = (ROOT / "result/score/MNRF-ViewProbe" / task).resolve()
    old = task_root / RETIRED_RUNS[task]
    new_run = Path(new_run).resolve()
    if old == new_run:
        raise ValueError("refusing to register the retired residual run as its replacement")
    if old.is_symlink():
        raise ValueError("refusing to remove a symlink at the retired source path")
    if old.exists():
        if not old.is_dir() or old.parent.resolve() != task_root:
            raise ValueError("retired source path is not the expected direct child directory")
        shutil.rmtree(old)
        return {"path": str(old), "status": "removed_after_new_source_registered"}
    return {"path": str(old), "status": "already_absent"}


def register(task, run_dir):
    run_dir = Path(run_dir).resolve(strict=True)
    expected_root = (ROOT / "result/score/MNRF-ViewProbe" / task).resolve()
    if run_dir.parent != expected_root:
        raise ValueError("run directory must be a direct child of the task source directory")
    manifest_path = run_dir / "probe_manifest.json"
    manifest = _json(manifest_path)
    run_status = _json(run_dir / "run_status.json")
    if (manifest.get("artifact_policy") != "full" or manifest.get("mode") != "pretrained"
            or manifest.get("stage_status") != "complete" or manifest.get("head_type") != "gmm"
            or run_status.get("status") != "complete"):
        raise ValueError("replacement source is not a completed full_gmm Full run")
    _, expected_seq_len, _ = TASKS[task]
    config = manifest.get("config", {})
    if config.get("seq_len") != expected_seq_len or config.get("seed") != 2021:
        raise ValueError("replacement source window/seed differs from the fixed task protocol")
    views = manifest.get("selected_views") or config.get("selected_views")
    if tuple(views or ()) != ("S", "E", "P", "R"):
        raise ValueError("replacement source is not the full S/E/P/R layout")

    # Run the same read-only native validator and select_gmm loader preflight
    # used when deciding whether a frozen source can be consumed.
    from scripts.recover_mnrf_view_probe_cache import _validate_complete_source
    with tempfile.TemporaryDirectory(prefix="mnrf-source-registration-") as temporary:
        validation = _validate_complete_source(run_dir, manifest, Path(temporary), "cpu")
    record = {
        "task": task, "path": str(run_dir), "run_id": run_dir.name,
        "manifest_sha256": _sha256_file(manifest_path),
        "source_identity_sha256": manifest.get("source_identity_sha256"),
        "feature_fingerprint": manifest.get("feature_fingerprint"),
        "cache_sha256": manifest.get("cache_sha256"),
        "endpoint_sha256": manifest.get("endpoint_sha256"),
        "gmm_fit_endpoints_sha256": manifest.get("gmm_fit_endpoints_sha256"),
        "registered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation": {"native_status": validation["native_validation"]["status"],
                       "select_gmm_source_preflight": "passed",
                       "device": validation["source_preflight_device"]},
    }
    write_registry(task, record)
    record["retired_residual_cleanup"] = retire_old_residual(task, run_dir)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("PSM", "Creditcard", "CICIDS"))
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    record = register(args.task, args.run_dir)
    print(json.dumps({"status": "registered", "registry": str(REGISTRY_DIR / (args.task + ".json")), "source": record},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
