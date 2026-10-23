#!/usr/bin/env python
"""Plan or run frozen-feature MNRF view ablations for one task or all tasks."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.mnrf_view_probe.probe import _sha256_file
from scripts.validate_mnrf_view_probe import validate


TASKS = {
    "PSM": ("full_gmm-full-20261021T112121Z-2466820", 192, 128),
    "MSL": ("full_gmm-full-20261021T112147Z-2470335", 192, 128),
    "GECCO": ("full_gmm-full-20261020T204951Z-1642355", 96, 128),
    "Genesis": ("full_gmm-full-20261020T205005Z-1645522", 192, 4),
    "NYC": ("full_gmm-full-20261020T211341Z-1931786", 192, 128),
    "CalIt2": ("full_gmm-full-20261021T112212Z-2474004", 192, 128),
    "Creditcard": ("full_gmm-full-20261021T112159Z-2472093", 192, 128),
    "CICIDS": ("full_gmm-full-20261021T112642Z-2510633", 192, 128),
}
COMBINATIONS = (
    ("S", ("S",)), ("SE", ("S", "E")), ("SP", ("S", "P")),
    ("SR", ("S", "R")), ("FULL", ("S", "E", "P", "R")),
    ("WITHOUT_S", ("E", "P", "R")), ("WITHOUT_E", ("S", "P", "R")),
    ("WITHOUT_P", ("S", "E", "R")), ("WITHOUT_R", ("S", "E", "P")),
)
SEEDS = (2021, 2022, 2023)
EXPECTED_LAYOUT = {
    "F": 25,
    "order": ["S", "E", "P", "R"],
    "dimensions": {"S": 1, "E": 8, "P": 8, "R": 8},
    "slices": {"S": [0, 1], "E": [1, 9], "P": [9, 17], "R": [17, 25]},
}
SEARCH_ROOTS = (
    "result/score/MNRF-ViewProbe",
    "result/score/MNRF-ViewProbe-SOnly",
    "result/score/MNRF-ViewProbe-Components",
    "result/score/MNRF-ViewProbe-Ablation",
)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-{}".format(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(path.name + ".tmp-{}".format(os.getpid()))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _views(manifest):
    return tuple(manifest.get("selected_views") or manifest.get("config", {}).get("selected_views") or ())


def _indices(manifest):
    values = manifest.get("selected_indices")
    if values is None:
        values = manifest.get("config", {}).get("selected_indices")
    if values is not None:
        return [int(value) for value in values]
    layout = manifest.get("layout") or manifest.get("config", {}).get("resolved_layout") or {}
    slices = layout.get("slices", {})
    return [index for view in _views(manifest) for index in range(*slices[view])]


def _manifest_run(path):
    try:
        manifest = _json(Path(path) / "probe_manifest.json")
        return manifest if isinstance(manifest, dict) else None
    except Exception:
        return None


def _scan_candidates(task):
    paths = set()
    for root_name in SEARCH_ROOTS:
        root = ROOT / root_name
        if root.is_dir():
            paths.update(p.parent.resolve() for p in root.rglob("probe_manifest.json"))
    for root_name in ("result/analysis/MNRF-ViewProbe", "result/review/MNRF-ViewProbe",
                      "result/review/MNRF-ViewProbe-SOnly"):
        root = ROOT / root_name
        if root.is_dir():
            paths.update(p.parent.resolve() for p in root.rglob("probe_manifest.json"))
    return sorted((path for path in paths if _manifest_run(path) is not None
                   and str(_manifest_run(path).get("task_name", "")).split("/")[-1].rsplit(".", 1)[0] == task),
                  key=lambda item: item.as_posix())


def _validate_run(path, validate_cache=None):
    result = validate(path)
    if result.get("status") not in {"complete", "stage1_complete"}:
        raise ValueError("existing validator did not return complete: {}".format(result.get("status")))
    manifest = _json(Path(path) / "probe_manifest.json")
    metrics_path = Path(path) / "results" / "metrics.json"
    metrics = _json(metrics_path) if metrics_path.is_file() else {}
    aligned_path = Path(path) / "results" / "aligned_scores.npz"
    nll_path = Path(path) / "results" / "test_nll.npy"
    for filename, declared in (("results/metrics.json", manifest.get("metrics_sha256")),
                               ("results/aligned_scores.npz", manifest.get("aligned_scores_sha256")),
                               ("results/test_nll.npy", manifest.get("test_nll_sha256"))):
        target = Path(path) / filename
        if declared and (not target.is_file() or _sha256_file(target) != declared):
            raise ValueError("{} digest differs from manifest".format(filename))
    return {"path": str(Path(path).resolve()), "manifest": manifest, "metrics": metrics,
            "validation": result, "aligned_path": str(aligned_path), "nll_path": str(nll_path)}


def _source_record(task, validate_cache):
    run_id, seq_len, batch_size = TASKS[task]
    path = ROOT / "result/score/MNRF-ViewProbe" / task / run_id
    record = {"task": task, "run_id": run_id, "path": str(path.resolve()),
              "seq_len_expected": seq_len, "batch_size_reference": batch_size,
              "status": "blocked", "reason": None}
    if not path.is_dir():
        record["reason"] = "pinned source directory is missing"
        return record
    try:
        checked = validate_cache.get(path.resolve())
        if checked is None:
            checked = _validate_run(path)
            validate_cache[path.resolve()] = checked
        manifest = checked["manifest"]
        config = manifest.get("config", {})
        identity = manifest.get("source_identity") or {}
        if manifest.get("artifact_type") != "probe_head" or manifest.get("head_type") != "gmm":
            raise ValueError("pinned source is not a GMM probe head")
        if manifest.get("artifact_policy") != "full" or manifest.get("stage_status") != "complete":
            raise ValueError("pinned source is not a complete full-artifact run")
        if config.get("seq_len") != seq_len:
            raise ValueError("source seq_len={} expected {}".format(config.get("seq_len"), seq_len))
        if config.get("seed") != 2021:
            raise ValueError("pinned source encoder/head seed is not 2021")
        if manifest.get("layout") != EXPECTED_LAYOUT or _views(manifest) != ("S", "E", "P", "R"):
            raise ValueError("pinned source does not have the frozen 1/8/8/8 S/E/P/R layout")
        if (config.get("gmm_components") != 4 or config.get("gmm_covariance_type") != "full"
                or float(config.get("gmm_reg_covar", -1)) != 0.001
                or config.get("gmm_n_init") != 1 or config.get("gmm_max_iter") != 100
                or config.get("gmm_fit_max_windows") != 20000
                or float(config.get("train_noise_std", -1)) != 0.01
                or config.get("noise_seed") != 20261013
                or float(config.get("feature_std_floor", -1)) != 0.05
                or float(config.get("score_mad_floor", -1)) != 0.001
                or float(config.get("calibration_quantile", -1)) != 0.995):
            raise ValueError("pinned source numerical protocol differs from requested ablation contract")
        if len(identity.get("variable_names", ())) < 1:
            raise ValueError("source identity has no variable order")
        for name in ("train", "validation", "calibration", "test"):
            if name not in manifest.get("cache_sha256", {}) or name not in manifest.get("endpoint_sha256", {}):
                raise ValueError("source cache/endpoint binding missing for {}".format(name))
            cache = path / "cache" / (name + "_features.npy")
            endpoints = path / "cache" / (name + "_window_end_indices.npy")
            if not cache.is_file() or not endpoints.is_file():
                raise ValueError("source cache files missing for {}".format(name))
        fit_path = path / "heads/gmm_fit_endpoints.npy"
        noise_path = path / "heads/fixed_noise.npy"
        if not fit_path.is_file() or not noise_path.is_file():
            raise ValueError("source GMM fit endpoints or fixed noise missing")
        record.update({"status": "complete", "feature_fingerprint": manifest.get("feature_fingerprint"),
                       "source_identity": identity, "source_identity_sha256": manifest.get("source_identity_sha256"),
                       "cache_sha256": manifest.get("cache_sha256"),
                       "endpoint_sha256": manifest.get("endpoint_sha256"),
                       "gmm_fit_endpoints_sha256": manifest.get("gmm_fit_endpoints_sha256"),
                       "input_statistics_sha256": manifest.get("raw_input_statistics_sha256",
                                                                manifest.get("input_statistics_sha256")),
                       "normalized_cache_sha256": manifest.get("normalized_cache_sha256"),
                       "fixed_noise_sha256": manifest.get("fixed_noise_sha256"),
                       "gmm_fit_endpoints_path": str(fit_path.resolve()),
                       "fixed_noise_path": str(noise_path.resolve()),
                       "layout": manifest.get("layout"), "config": config,
                       "metrics": checked["metrics"], "source_manifest_sha256": _sha256_file(path / "probe_manifest.json"),
                       "validated_status": checked["validation"].get("status")})
    except Exception as error:
        record["reason"] = "{}: {}".format(type(error).__name__, error)
    return record


def _same_conditions(candidate, source, combination, seed):
    manifest = candidate["manifest"]
    config = manifest.get("config", {})
    views = _views(manifest)
    expected_views = dict(COMBINATIONS)[combination]
    if views != expected_views or _indices(manifest) != selected_indices(expected_views):
        return False, "selected view/index identity differs"
    if int(config.get("seed", -1)) != seed:
        return False, "head seed differs"
    if manifest.get("source_identity") != source["source_identity"]:
        return False, "frozen source data/time/split identity differs"
    if manifest.get("source_feature_fingerprint", manifest.get("feature_fingerprint")) != source["feature_fingerprint"]:
        return False, "source encoder feature_fingerprint differs"
    if Path(manifest.get("source_dir", "")).resolve() != Path(source["path"]).resolve():
        return False, "source_dir does not point to the pinned Full source"
    for field in ("source_cache_sha256", "source_endpoint_sha256"):
        source_field = "cache_sha256" if field == "source_cache_sha256" else "endpoint_sha256"
        if manifest.get(field) != source[source_field]:
            return False, "{} differs".format(field)
    if manifest.get("gmm_fit_endpoints_sha256") != source["gmm_fit_endpoints_sha256"]:
        return False, "GMM fitting endpoints differ from source"
    candidate_path = Path(candidate["path"])
    candidate_fit = candidate_path / "heads/gmm_fit_endpoints.npy"
    candidate_noise = candidate_path / "heads/fixed_noise.npy"
    if not candidate_fit.is_file() or _sha256_file(candidate_fit) != source["gmm_fit_endpoints_sha256"]:
        return False, "saved fitting endpoint bytes differ from pinned source"
    if (not candidate_noise.is_file() or not source.get("fixed_noise_sha256")
            or manifest.get("fixed_noise_sha256") != source["fixed_noise_sha256"]
            or _sha256_file(candidate_noise) != source["fixed_noise_sha256"]):
        return False, "saved full-layout fixed noise differs from pinned source"
    import numpy as np
    source_noise = np.load(source["fixed_noise_path"], mmap_mode="r", allow_pickle=False)
    candidate_noise_array = np.load(candidate_noise, mmap_mode="r", allow_pickle=False)
    if source_noise.shape != candidate_noise_array.shape or source_noise.dtype != candidate_noise_array.dtype:
        return False, "fixed noise shape/dtype differs from pinned source"
    if not np.array_equal(source_noise, candidate_noise_array):
        return False, "fixed noise values differ from pinned source"
    source_fit = np.load(source["gmm_fit_endpoints_path"], mmap_mode="r", allow_pickle=False)
    candidate_fit_array = np.load(candidate_fit, mmap_mode="r", allow_pickle=False)
    if source_fit.shape != candidate_fit_array.shape or not np.array_equal(source_fit, candidate_fit_array):
        return False, "fitting endpoint values differ from pinned source"
    if manifest.get("raw_input_statistics_sha256") != source["input_statistics_sha256"]:
        return False, "input normalization identity differs"
    if manifest.get("fixed_noise_sha256") is None:
        return False, "fixed noise binding is missing"
    numeric = source["config"]
    for key in ("seq_len", "pattern_len", "hidden_dim", "evolution_dim", "pattern_dim", "relation_dim",
                "view_dim", "gmm_components", "gmm_covariance_type", "gmm_reg_covar", "gmm_n_init",
                "gmm_max_iter", "gmm_fit_max_windows", "train_noise_std", "noise_seed",
                "feature_std_floor", "score_mad_floor", "calibration_quantile"):
        if config.get(key) != numeric.get(key):
            return False, "numerical config {} differs".format(key)
    if manifest.get("head_type") != "gmm" or manifest.get("head_status") != "complete":
        return False, "candidate GMM head is not complete"
    return True, ""


def selected_indices(views):
    slices = EXPECTED_LAYOUT["slices"]
    return [i for view in views for i in range(*slices[view])]


def _candidate_for(task, source, combination, seed, validate_cache, inventory):
    paths = inventory.setdefault(task, _scan_candidates(task))
    valid = []
    rejected = []
    expected_views = dict(COMBINATIONS)[combination]
    pinned = Path(source["path"]).resolve()
    if combination == "FULL" and seed == 2021:
        return {"status": "reused", "path": str(pinned), "reason": "pinned full source"}, None
    for path in paths:
        if path.resolve() == pinned:
            continue
        manifest = _manifest_run(path)
        if not manifest or _views(manifest) != expected_views:
            continue
        try:
            checked = validate_cache.get(path.resolve())
            if checked is None:
                checked = _validate_run(path)
                validate_cache[path.resolve()] = checked
            ok, reason = _same_conditions(checked, source, combination, seed)
            if ok:
                valid.append(checked)
            else:
                rejected.append({"path": str(path), "reason": reason})
        except Exception as error:
            rejected.append({"path": str(path), "reason": "{}: {}".format(type(error).__name__, error)})
    if not valid:
        return {"status": "planned/pending", "path": "", "reason": "no validated same-source candidate",
                "excluded_candidates": rejected}, None
    valid.sort(key=lambda item: item["path"])
    first = valid[0]
    if len(valid) > 1:
        # Same-condition duplicate runs are reusable only if the saved scored
        # vectors and thresholds agree. Never choose by AP/AUC.
        import numpy as np
        base = np.load(first["aligned_path"], allow_pickle=False)
        conflicts = []
        for other in valid[1:]:
            current = np.load(other["aligned_path"], allow_pickle=False)
            if (not np.array_equal(base["time_indices"], current["time_indices"])
                    or not np.array_equal(base["valid_mask"], current["valid_mask"])
                    or not np.array_equal(base["predictions"], current["predictions"])
                    or not np.array_equal(base["scores"], current["scores"])
                    or float(base["threshold"]) != float(current["threshold"])):
                conflicts.append(other["path"])
        if conflicts:
            return {"status": "conflict", "path": "", "reason": "same-condition candidates disagree",
                    "conflict_paths": [first["path"]] + conflicts}, None
    return {"status": "reused", "path": first["path"],
            "reason": "validated candidate; lexical path tie-break, not score-based"}, first


def _row(task, combo, views, seed, source, status, candidate=None, output_path="", source_encoder_seed=2021):
    config = source.get("config") or {}
    return {
        "dataset": task, "combination": combo, "head_seed": seed,
        "source_encoder_seed": source_encoder_seed, "source_run_id": source.get("run_id"),
        "source_identity": source.get("source_identity_sha256"),
        "feature_fingerprint": source.get("feature_fingerprint"),
        "selected_views": ",".join(views), "selected_indices": json.dumps(selected_indices(views)),
        "selected_dim": len(selected_indices(views)),
        "gmm_components": config.get("gmm_components"),
        "gmm_covariance_type": config.get("gmm_covariance_type"),
        "gmm_reg_covar": config.get("gmm_reg_covar"), "gmm_n_init": config.get("gmm_n_init"),
        "gmm_max_iter": config.get("gmm_max_iter"), "gmm_fit_max_windows": config.get("gmm_fit_max_windows"),
        "fit_endpoint_hash": source.get("gmm_fit_endpoints_sha256"),
        "candidate_or_output_path": (candidate or {}).get("path") or output_path,
        "candidate_audit": json.dumps((candidate or {}).get("excluded_candidates", []),
                                       sort_keys=True, separators=(",", ":")),
        "status": status, "reason": (candidate or {}).get("reason", source.get("reason")),
        "auc_roc": None, "average_precision": None, "precision": None, "recall": None,
        "point_f1": None, "fp": None, "fn": None, "tp": None, "tn": None,
        "normal_fpr": None, "coverage": None, "anomaly_segments_total": None,
        "anomaly_segments_scored": None, "anomaly_segments_hit": None,
        "fit_time_seconds": None, "inference_time_seconds": None,
        "elapsed_seconds_kind": "pending",
    }


def build_plan(tasks, repeat=False, batch_id=None, validate_sources=True):
    batch_id = batch_id or "branch-ablation-{}-{}".format(time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), os.getpid())
    seeds = SEEDS if repeat else (2021,)
    validate_cache = {}
    inventory = {}
    source_records = {task: _source_record(task, validate_cache) if validate_sources else {
        "task": task, "run_id": TASKS[task][0], "path": str((ROOT / "result/score/MNRF-ViewProbe" / task / TASKS[task][0]).resolve()),
        "status": "unverified", "reason": "plan metadata only"} for task in tasks}
    rows = []
    for task in tasks:
        source = source_records[task]
        for seed in seeds:
            for combo, views in COMBINATIONS:
                output = ROOT / "result/score/MNRF-ViewProbe-Ablation" / task / (
                    "select_gmm-{}-{}-{}-{}".format(combo, seed, batch_id, os.getpid()))
                if source["status"] != "complete":
                    status, candidate = "blocked", {"reason": source.get("reason")}
                else:
                    candidate, _ = _candidate_for(task, source, combo, seed, validate_cache, inventory)
                    status = candidate["status"]
                rows.append(_row(task, combo, views, seed, source, status, candidate,
                                 str(output.resolve())))
    return {"batch_id": batch_id, "mode": "repeat" if repeat else "first_round",
            "tasks": list(tasks), "seeds": list(seeds), "source_records": source_records,
            "candidate_paths": {task: [str(path) for path in paths]
                                for task, paths in inventory.items()},
            "rows": rows, "positions": len(rows),
            "pending_count": sum(row["status"] == "planned/pending" for row in rows),
            "reused_count": sum(row["status"] == "reused" for row in rows),
            "blocked_count": sum(row["status"] == "blocked" for row in rows),
            "conflict_count": sum(row["status"] == "conflict" for row in rows)}


def _save_batch(plan, summary_root=None):
    batch_dir = ROOT / "result/analysis/MNRF-ViewProbe-Ablation" / plan["batch_id"]
    batch_dir.mkdir(parents=True, exist_ok=False)
    _write_json(batch_dir / "plan.json", plan)
    _write_csv(batch_dir / "matrix.csv", plan["rows"])
    note = {
        "batch_id": plan["batch_id"], "positions": plan["positions"],
        "reused": plan["reused_count"], "pending": plan["pending_count"],
        "blocked": plan["blocked_count"], "conflict": plan["conflict_count"],
        "notice": "Plan only; no encoder, GMM fit, calibration or training was run.",
    }
    _write_json(batch_dir / "status.json", note)
    return batch_dir


def _command(task, combo, views, seed, source, output):
    config = source["config"]
    command = [sys.executable, str(ROOT / "scripts/run_mnrf_view_probe.py"),
               "--task", task, "--data-source", config.get("data_source", "large_detect"),
               "--mode", "select_gmm", "--views", ",".join(views),
               "--source-dir", source["path"], "--output-dir", str(output),
               "--seed", str(seed), "--seq-len", str(config["seq_len"]),
               "--pattern-len", str(config["pattern_len"]), "--hidden-dim", str(config["hidden_dim"]),
               "--batch-size", str(config["batch_size"]), "--stage1-epochs", str(config["stage1_epochs"]),
               "--stage2-epochs", str(config["stage2_epochs"]), "--patience", str(config["patience"]),
               "--learning-rate", str(config["learning_rate"]), "--weight-decay", str(config["weight_decay"]),
               "--feature-std-floor", str(config["feature_std_floor"]),
               "--score-mad-floor", str(config["score_mad_floor"]),
               "--calibration-quantile", str(config["calibration_quantile"]),
               "--device", "cpu", "--cache-chunk-size", str(config["cache_chunk_size"]),
               "--encoding-batch-size", str(min(config.get("encoding_batch_size") or config["batch_size"], config["batch_size"])),
               "--view-dim", str(config["view_dim"]), "--evolution-dim", str(config["evolution_dim"]),
               "--pattern-dim", str(config["pattern_dim"]), "--relation-dim", str(config["relation_dim"]),
               "--gmm-components", str(config["gmm_components"]), "--gmm-covariance-type", config["gmm_covariance_type"],
               "--gmm-reg-covar", str(config["gmm_reg_covar"]), "--gmm-n-init", str(config["gmm_n_init"]),
               "--gmm-max-iter", str(config["gmm_max_iter"]),
               "--gmm-fit-max-windows", str(config["gmm_fit_max_windows"]),
               "--cpu-threads", "1", "--train-noise-std", str(config["train_noise_std"]),
               "--noise-seed", str(config["noise_seed"]), "--artifact-policy", "full"]
    return command


def execute_task(task, repeat=False):
    if task not in TASKS:
        raise ValueError("unknown task: {}".format(task))
    plan = build_plan([task], repeat=repeat)
    batch_dir = _save_batch(plan)
    sources = plan["source_records"]
    source = sources[task]
    if source["status"] != "complete":
        print("blocked {}: {}".format(task, source.get("reason")), file=sys.stderr)
        return 1
    rows_by_key = {(row["combination"], int(row["head_seed"])): row for row in plan["rows"]}
    failures = 0
    for seed in plan["seeds"]:
        for combo, views in COMBINATIONS:
            row = rows_by_key[(combo, seed)]
            if row["status"] in {"reused", "complete"}:
                continue
            if row["status"] == "conflict":
                failures += 1
                continue
            output = Path(row["candidate_or_output_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                row["status"] = "conflict"
                row["reason"] = "refusing to overwrite an existing run directory"
                failures += 1
                continue
            row["status"] = "running"
            _write_csv(batch_dir / "matrix.csv", plan["rows"])
            started = time.time()
            log_path = output.parent / (output.name + ".launcher.log")
            try:
                command = _command(task, combo, views, seed, source, output)
                print("MNRF ablation task={} combo={} seed={} output_dir={}".format(task, combo, seed, output), flush=True)
                with log_path.open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True, bufsize=1)
                    for line in process.stdout:
                        sys.stdout.write(line)
                        log.write(line)
                        log.flush()
                    status = process.wait()
                if status != 0:
                    raise subprocess.CalledProcessError(status, command)
                checked = _validate_run(output)
                row.update(_metrics_row(task, combo, views, seed, source, output, checked))
                row["status"] = "complete"
                row["elapsed_seconds_kind"] = "this_run"
            except Exception as error:
                row["status"] = "failed"
                row["reason"] = "{}: {}".format(type(error).__name__, error)
                failures += 1
            row["elapsed_seconds"] = time.time() - started
            _write_csv(batch_dir / "matrix.csv", plan["rows"])
            _write_json(batch_dir / "plan.json", plan)
    _write_json(batch_dir / "status.json", {"batch_id": plan["batch_id"], "task": task,
                "positions": plan["positions"], "complete": sum(r["status"] in {"complete", "reused"} for r in plan["rows"]),
                "failed_or_conflict": failures, "model_training": "not run; frozen-cache select_gmm only"})
    return 1 if failures else 0


def _metrics_row(task, combo, views, seed, source, run, checked):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    path = Path(run)
    with np.load(path / "results/aligned_scores.npz", allow_pickle=False) as aligned:
        scores = np.asarray(aligned["scores"], dtype=np.float64)
        predictions = np.asarray(aligned["predictions"], dtype=np.int64)
        endpoints = np.asarray(aligned["time_indices"], dtype=np.int64)
        valid = np.asarray(aligned["valid_mask"], dtype=bool)
        actual = np.asarray(aligned["actual"], dtype=np.int8)
        threshold = float(aligned["threshold"])
    if actual.shape != scores.shape or not np.isin(actual, (0, 1)).all():
        raise ValueError("candidate actual labels are not aligned binary labels")
    if not np.array_equal(predictions, scores > threshold):
        raise ValueError("candidate predictions do not match its saved calibrated threshold")
    tp = int(np.sum((actual == 1) & (predictions == 1)))
    fp = int(np.sum((actual == 0) & (predictions == 1)))
    fn = int(np.sum((actual == 1) & (predictions == 0)))
    tn = int(np.sum((actual == 0) & (predictions == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-15)
    from scripts.summarize_mnrf_branch_ablation import segment_counts
    segment = segment_counts(source["path"], endpoints, predictions)
    metrics = checked["metrics"]
    manifest = _json(path / "probe_manifest.json")
    config = manifest.get("config", {})
    gmm_status_path = path / "heads/gmm_status.json"
    gmm_status = _json(gmm_status_path) if gmm_status_path.is_file() else {}
    parameter_count = metrics.get("head_parameter_count", gmm_status.get("head_parameter_count"))
    return {
        "dataset": task, "combination": combo, "head_seed": seed,
        "source_encoder_seed": 2021, "source_run_id": Path(source["path"]).name,
        "source_identity": source.get("source_identity_sha256"),
        "feature_fingerprint": source.get("feature_fingerprint"),
        "selected_views": ",".join(views), "selected_indices": json.dumps(selected_indices(views)),
        "selected_dim": len(selected_indices(views)),
        "gmm_components": config.get("gmm_components"), "gmm_covariance_type": config.get("gmm_covariance_type"),
        "gmm_reg_covar": config.get("gmm_reg_covar"), "gmm_n_init": config.get("gmm_n_init"),
        "gmm_max_iter": config.get("gmm_max_iter"),
        "gmm_fit_max_windows": config.get("gmm_fit_max_windows"),
        "head_parameter_count": parameter_count,
        "fit_endpoint_hash": manifest.get("gmm_fit_endpoints_sha256"),
        "candidate_or_output_path": str(path.resolve()), "status": "complete", "reason": "",
        "auc_roc": float(roc_auc_score(actual, scores)) if np.unique(actual).size == 2 else None,
        "average_precision": float(average_precision_score(actual, scores)),
        "precision": precision, "recall": recall, "point_f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "normal_fpr": fp / max(fp + tn, 1), "coverage": float(np.sum(valid) / max(len(valid), 1)),
        "anomaly_segments_total": segment["total"], "anomaly_segments_scored": segment["scored"],
        "anomaly_segments_hit": segment["hit"],
        "fit_time_seconds": metrics.get("fit_time_seconds"),
        "inference_time_seconds": metrics.get("inference_time_seconds"),
        "elapsed_seconds_kind": "validated candidate or run; see source", "threshold": threshold,
        "independent_recheck": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mode", choices=("plan", "execute", "repeat"), default="plan")
    parser.add_argument("--repeat-plan", action="store_true",
                        help="plan all three head seeds without fitting; valid with --mode plan")
    parser.add_argument("--validate-sources", action="store_true", default=True)
    args = parser.parse_args()
    tasks = list(TASKS) if args.all else ([args.task] if args.task else [])
    if not tasks:
        parser.error("provide --task TASK or --all")
    if args.mode == "plan":
        plan = build_plan(tasks, repeat=args.repeat_plan, validate_sources=args.validate_sources)
        batch_dir = _save_batch(plan)
        print("plan_dir={}".format(batch_dir.resolve()))
        print("positions={} reused={} pending={} blocked={} conflict={}".format(
            plan["positions"], plan["reused_count"], plan["pending_count"],
            plan["blocked_count"], plan["conflict_count"]))
        return 1 if plan["conflict_count"] or plan["blocked_count"] else 0
    if args.mode == "execute":
        if len(tasks) != 1:
            parser.error("execute accepts exactly one --task; no cross-dataset batch scheduling")
        return execute_task(tasks[0], repeat=False)
    if len(tasks) != 1:
        parser.error("repeat accepts exactly one --task; no cross-dataset batch scheduling")
    return execute_task(tasks[0], repeat=True)


if __name__ == "__main__":
    raise SystemExit(main())
