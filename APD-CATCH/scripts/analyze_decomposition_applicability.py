"""Read-only applicability analysis for frozen CATCH and MSD-CATCH results.

This script never imports a benchmark runner, trains a model, or scores a model.
It reads frozen archives, raw data, metadata, and existing score/checkpoint files
to describe where the fixed three-scale decomposition was helpful or harmful.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import pickle
import tarfile
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_functional


TASKS = [
    "PSM",
    "Genesis",
    "GECCO",
    "CalIt2",
    "NYC",
    "MSL",
    *[f"ASD_dataset_{index}" for index in range(1, 13)],
    "CICIDS",
    "Creditcard",
    "SMAP",
    "SMD",
    "SWAT",
]
PAPER_ORDER = [
    "PSM",
    "Genesis",
    "GECCO",
    "CalIt2",
    "NYC",
    "MSL",
    "ASD",
    "CICIDS",
    "Creditcard",
    "SMAP",
    "SMD",
    "SWAT",
]
FORMAL_MSD_JSON = {"CalIt2", "NYC", "MSL"}
FORMAL_MSD_LOG = {"PSM", "Genesis", "GECCO"}
# Frozen formal sources. Paths are relative to the cxy workspace root, never selected by mtime.
FORMAL_RESULT_SOURCES = {
    "PSM": ("CATCH-master/result/score/CATCH/PSM/run-20260716T222700Z-660928-22488/CATCH.1784241452.h3c-R5500-G5.661628.csv.tar.gz", "APD-CATCH/result/msd_catch_screen/PSM.log"),
    "Genesis": ("CATCH-master/result/score/CATCH/Genesis/run-20260716T213208Z-6088-16524/CATCH.1784239890.h3c-R5500-G5.6426.csv.tar.gz", "APD-CATCH/result/msd_catch_screen/Genesis.log"),
    "GECCO": ("APD-CATCH/result/score/CATCH_RSA_GECCO/CATCH.1784315768.h3c-R5500-G5.3595187.csv.tar.gz", "APD-CATCH/result/msd_catch_screen/GECCO.log"),
    "CalIt2": ("CATCH-master/result/score/CATCH/CalIt2/run-20260715T183704Z-2374669-5874/CATCH.1784140875.h3c-R5500-G5.2376944.csv.tar.gz", "APD-CATCH/result/msd_catch_total_screen/CalIt2.json"),
    "NYC": ("CATCH-master/result/score/CATCH/NYC/run-20260716T222643Z-657156-32728/CATCH.1784241041.h3c-R5500-G5.657503.csv.tar.gz", "APD-CATCH/result/msd_catch_total_screen/NYC.json"),
    "MSL": ("CATCH-master/result/score/CATCH/MSL/run-20260716T213635Z-58595-28964/CATCH.1784244404.h3c-R5500-G5.59126.csv.tar.gz", "APD-CATCH/result/msd_catch_total_screen/MSL.json"),
    "ASD_dataset_1": ("CATCH-master/result/score/CATCH/ASD_dataset_1/run-20260715T120733Z-1347582-29581/CATCH.1784117687.h3c-R5500-G5.1349346.csv.tar.gz", "APD-CATCH/result/score/by_dataset/ASD_dataset_1/MSDCATCH/MSDCATCH.1784390185.h3c-R5500-G5.3190810.csv.tar.gz"),
    "ASD_dataset_2": ("CATCH-master/result/score/CATCH/ASD_dataset_2/run-20260715T123513Z-1655293-4562/CATCH.1784119679.h3c-R5500-G5.1657441.csv.tar.gz", "APD-CATCH/result/score/by_dataset/ASD_dataset_2/MSDCATCH/MSDCATCH.1784390351.h3c-R5500-G5.3230432.csv.tar.gz"),
    "ASD_dataset_3": ("CATCH-master/result/score/CATCH/ASD_dataset_3/run-20260715T130403Z-1970702-20553/CATCH.1784121522.h3c-R5500-G5.1974015.csv.tar.gz", "APD-CATCH/result/score/by_dataset/ASD_dataset_3/MSDCATCH/MSDCATCH.1784390812.h3c-R5500-G5.3274854.csv.tar.gz"),
    "ASD_dataset_4": ("CATCH-master/result/score/CATCH/ASD_dataset_4/run-20260715T134133Z-2385568-17192/CATCH.1784123438.h3c-R5500-G5.2387988.csv.tar.gz", "APD-CATCH/result/score/by_dataset/ASD_dataset_4/MSDCATCH/MSDCATCH.1784391366.h3c-R5500-G5.3400081.csv.tar.gz"),
    "ASD_dataset_5": ("CATCH-master/result/score/CATCH/ASD_dataset_5/run-20260715T141230Z-2968914-11771/CATCH.1784125388.h3c-R5500-G5.2976094.csv.tar.gz", "APD-CATCH/result/score/by_dataset/ASD_dataset_5/MSDCATCH/MSDCATCH.1784391918.h3c-R5500-G5.3568954.csv.tar.gz"),
    "ASD_dataset_6": ("CATCH-master/result/score/CATCH/ASD_dataset_6/run-20260715T143325Z-3669526-27356/CATCH.1784126625.h3c-R5500-G5.3676661.csv.tar.gz", "APD-CATCH/result/score/by_dataset/ASD_dataset_6/MSDCATCH/MSDCATCH.1784392471.h3c-R5500-G5.3714799.csv.tar.gz"),
    "ASD_dataset_7": ("CATCH-master/result/score/CATCH/ASD_dataset_7/run-20260715T150742Z-85080-11778/CATCH.1784128749.h3c-R5500-G5.87689.csv.tar.gz", "APD-CATCH/result/score/by_dataset/ASD_dataset_7/MSDCATCH/MSDCATCH.1784392989.h3c-R5500-G5.3864449.csv.tar.gz"),
    "ASD_dataset_8": ("CATCH-master/result/score/CATCH/ASD_dataset_8/run-20260715T153406Z-377465-29501/CATCH.1784130361.h3c-R5500-G5.379417.csv.tar.gz", "APD-CATCH/result/score/by_dataset/ASD_dataset_8/MSDCATCH/MSDCATCH.1784393473.h3c-R5500-G5.4028866.csv.tar.gz"),
    "ASD_dataset_9": ("CATCH-master/result/score/CATCH/ASD_dataset_9/run-20260715T155915Z-646428-29264/CATCH.1784131876.h3c-R5500-G5.649174.csv.tar.gz", "APD-CATCH/result/score/by_dataset/ASD_dataset_9/MSDCATCH/MSDCATCH.1784393964.h3c-R5500-G5.4172863.csv.tar.gz"),
    "ASD_dataset_10": ("CATCH-master/result/score/CATCH/ASD_dataset_10/run-20260715T120921Z-1368429-738/CATCH.1784118012.h3c-R5500-G5.1369978.csv.tar.gz", "APD-CATCH/result/score/by_dataset/ASD_dataset_10/MSDCATCH/MSDCATCH.1784394511.h3c-R5500-G5.126083.csv.tar.gz"),
    "ASD_dataset_11": ("CATCH-master/result/score/CATCH/ASD_dataset_11/run-20260715T122849Z-1587674-11538/CATCH.1784119230.h3c-R5500-G5.1589378.csv.tar.gz", "APD-CATCH/result/score/by_dataset/ASD_dataset_11/MSDCATCH/MSDCATCH.1784395056.h3c-R5500-G5.287470.csv.tar.gz"),
    "ASD_dataset_12": ("CATCH-master/result/score/CATCH/ASD_dataset_12/run-20260715T125526Z-1885032-7036/CATCH.1784120935.h3c-R5500-G5.1887948.csv.tar.gz", "APD-CATCH/result/score/by_dataset/ASD_dataset_12/MSDCATCH/MSDCATCH.1784395511.h3c-R5500-G5.464310.csv.tar.gz"),
    "CICIDS": ("CATCH-master/result/score/CATCH/CICIDS/run-20260715T120722Z-1345491-4045/CATCH.1784122606.h3c-R5500-G5.1353374.csv.tar.gz", "APD-CATCH/result/score/by_dataset/CICIDS/MSDCATCH/MSDCATCH.1784402663.h3c-R5500-G5.3309038.csv.tar.gz"),
    "Creditcard": ("CATCH-master/result/score/CATCH/Creditcard/run-20260716T202925Z-3426561-275/CATCH.1784236079.h3c-R5500-G5.3427278.csv.tar.gz", "APD-CATCH/result/score/by_dataset/Creditcard/MSDCATCH/MSDCATCH.1784402131.h3c-R5500-G5.1040497.csv.tar.gz"),
    "SMAP": ("CATCH-master/result/score/CATCH/SMAP/run-20260717T070416Z-2731832-15614/CATCH.1784278165.h3c-R5500-G5.2732611.csv.tar.gz", "APD-CATCH/result/score/by_dataset/SMAP/MSDCATCH/MSDCATCH.1784403162.h3c-R5500-G5.1079965.csv.tar.gz"),
    "SMD": ("CATCH-master/result/score/CATCH/SMD/run-20260716T225633Z-1008371-29270/CATCH.1784301239.h3c-R5500-G5.1011549.csv.tar.gz", "APD-CATCH/result/score/by_dataset/SMD/MSDCATCH/MSDCATCH.1784559943.h3c-R5500-G5.179946.csv.tar.gz"),
    "SWAT": ("CATCH-master/result/score/CATCH/SWAT/run-20260715T184743Z-2499166-18894/CATCH.1784210821.h3c-R5500-G5.2521250.csv.tar.gz", "APD-CATCH/result/score/by_dataset/SWAT/MSDCATCH/MSDCATCH.1784547386.h3c-R5500-G5.233296.csv.tar.gz"),
}
EPSILON = 1e-8


def numeric(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def finite(value) -> bool:
    return numeric(value) is not None


def config_integer(params: Optional[dict], key: str) -> Optional[int]:
    value = numeric(params.get(key)) if isinstance(params, dict) else None
    if value is None or value < 1 or not value.is_integer():
        return None
    return int(value)


def paper_dataset(task: str) -> str:
    return "ASD" if task.startswith("ASD_dataset_") else task


def read_tar_row(path: Path) -> dict:
    with tarfile.open(path, "r:gz") as archive:
        member = next(member for member in archive.getmembers() if member.isfile())
        return next(csv.DictReader(io.TextIOWrapper(archive.extractfile(member), encoding="utf-8")))


def choose_valid_archive(paths: Iterable[Path]) -> tuple[Optional[Path], Optional[dict]]:
    del paths
    raise RuntimeError("Automatic result selection is prohibited; use FORMAL_RESULT_SOURCES.")


def msd_json_row(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    score = payload["scores"]["total_score"]
    return {
        "auc_pr": score["auc_pr"],
        "auc_roc": score["auc_roc"],
        "model_params": payload.get("config"),
        "strategy_args": None,
        "payload": payload,
    }


def msd_log_row(path: Path) -> Optional[dict]:
    """Parse the terminal formal result emitted by the three first-screen runs."""
    for line in reversed(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        scores = payload.get("scores")
        total = scores.get("total_score") if isinstance(scores, dict) else None
        if isinstance(total, dict) and finite(total.get("auc_pr")) and finite(total.get("auc_roc")):
            return {
                "auc_pr": total["auc_pr"],
                "auc_roc": total["auc_roc"],
                "model_params": None,
                "strategy_args": None,
                "payload": payload,
            }
    return None


def metric_group(delta: Optional[float]) -> str:
    if delta is None:
        return "N/A"
    if delta > 0.01:
        return "gain"
    if delta < -0.01:
        return "loss"
    return "neutral"


def _deprecated_mtime_discovery_do_not_use(repo: Path, catch_root: Path) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    sources = []
    for task in TASKS:
        catch_path, catch_row = choose_valid_archive(
            (catch_root / "result" / "score" / "CATCH" / task).rglob("CATCH*.tar.gz")
        )
        if catch_row is None:
            catch_pr = catch_roc = None
            catch_params = None
            catch_source = None
        else:
            catch_pr = numeric(catch_row["auc_pr"])
            catch_roc = numeric(catch_row["auc_roc"])
            catch_params = json.loads(catch_row["model_params"])
            catch_source = str(catch_path)
            sources.append(
                {
                    "task": task,
                    "artifact": "CATCH detect_score archive",
                    "path": catch_source,
                    "status": "used",
                    "notes": "Frozen score archive",
                }
            )

        msd_payload = None
        if task in FORMAL_MSD_JSON:
            msd_path = repo / "result" / "msd_catch_total_screen" / f"{task}.json"
            if msd_path.exists():
                msd_row = msd_json_row(msd_path)
                msd_source = str(msd_path)
                msd_kind = "formal total-screen JSON"
            else:
                msd_row = None
                msd_source = None
                msd_kind = "N/A"
        elif task in FORMAL_MSD_LOG:
            msd_path = repo / "result" / "msd_catch_screen" / f"{task}.log"
            if msd_path.exists():
                msd_row = msd_log_row(msd_path)
                msd_source = str(msd_path) if msd_row is not None else None
                msd_kind = "formal total-screen log" if msd_row is not None else "N/A"
            else:
                msd_row = None
                msd_source = None
                msd_kind = "N/A"
        else:
            msd_path, msd_tar_row = choose_valid_archive(
                (repo / "result" / "score" / "by_dataset" / task / "MSDCATCH").rglob(
                    "MSDCATCH*.tar.gz"
                )
            )
            if msd_tar_row is None:
                msd_row = None
                msd_source = None
                msd_kind = "N/A"
            else:
                msd_row = {
                    "auc_pr": msd_tar_row["auc_pr"],
                    "auc_roc": msd_tar_row["auc_roc"],
                    "model_params": json.loads(msd_tar_row["model_params"]),
                    "strategy_args": json.loads(msd_tar_row["strategy_args"]),
                    "payload": None,
                }
                msd_source = str(msd_path)
                msd_kind = "formal by-dataset archive"

        if msd_row is None:
            msd_pr = msd_roc = delta_pr = delta_roc = None
            parameter_match = "N/A"
            primary_score = "N/A"
            notes = "No valid formal MSD archive found."
        else:
            msd_pr = numeric(msd_row["auc_pr"])
            msd_roc = numeric(msd_row["auc_roc"])
            delta_pr = msd_pr - catch_pr if msd_pr is not None and catch_pr is not None else None
            delta_roc = msd_roc - catch_roc if msd_roc is not None and catch_roc is not None else None
            msd_payload = msd_row["payload"]
            msd_params = msd_row["model_params"]
            parameter_match = (
                "N/A" if msd_params is None else str(catch_params == msd_params).lower()
            )
            primary_score = (
                msd_payload.get("primary_score", "total_score") if msd_payload else "total_score"
            )
            notes = ""
            sources.append(
                {
                    "task": task,
                    "artifact": "MSD formal result",
                    "path": msd_source,
                    "status": "used",
                    "notes": msd_kind,
                }
            )
        rows.append(
            {
                "task": task,
                "paper_dataset": paper_dataset(task),
                "formal_seq_len": config_integer(catch_params, "seq_len"),
                "formal_patch_size": config_integer(catch_params, "patch_size"),
                "catch_auc_pr": catch_pr,
                "msd_auc_pr": msd_pr,
                "delta_auc_pr": delta_pr,
                "pr_group": metric_group(delta_pr),
                "catch_auc_roc": catch_roc,
                "msd_auc_roc": msd_roc,
                "delta_auc_roc": delta_roc,
                "roc_group": metric_group(delta_roc),
                "catch_source": catch_source,
                "msd_source": msd_source,
                "msd_source_kind": msd_kind,
                "parameter_match": parameter_match,
                "primary_score": primary_score,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows), sources


def parsed_mapping(value) -> dict:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    return json.loads(value)


def record_config(row: dict) -> dict:
    params = parsed_mapping(row.get("model_params"))
    strategy = parsed_mapping(row.get("strategy_args"))
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    dataset = row.get("file_name") or payload.get("dataset")
    return {
        "dataset": Path(str(dataset)).stem if dataset else None,
        "seed": numeric(strategy.get("seed")),
        "strategy_name": strategy.get("strategy_name"),
        "seq_len": config_integer(params, "seq_len"),
        "patch_size": config_integer(params, "patch_size"),
    }


def fixed_msd_row(path: Path) -> tuple[Optional[dict], str]:
    if path.suffix == ".json":
        return msd_json_row(path), "formal total-screen JSON"
    if path.suffix == ".log":
        return msd_log_row(path), "formal total-screen log"
    row = read_tar_row(path)
    if row.get("log_info") or not finite(row.get("auc_pr")) or not finite(row.get("auc_roc")):
        return None, "invalid fixed archive"
    return {
        "auc_pr": row["auc_pr"],
        "auc_roc": row["auc_roc"],
        "model_params": parsed_mapping(row.get("model_params")),
        "strategy_args": parsed_mapping(row.get("strategy_args")),
        "file_name": row.get("file_name"),
        "payload": None,
    }, "formal by-dataset archive"


def compare_formal_configs(task: str, catch_config: dict, msd_config: dict) -> tuple[str, str, str]:
    conflicts = []
    missing = []
    for field in ("dataset", "seed", "seq_len", "patch_size", "strategy_name"):
        catch_value = catch_config.get(field)
        msd_value = msd_config.get(field)
        if catch_value is None or msd_value is None:
            missing.append(field)
        elif catch_value != msd_value:
            conflicts.append(field)
    if catch_config.get("dataset") not in (None, task):
        conflicts.append("catch_dataset")
    if msd_config.get("dataset") not in (None, task):
        conflicts.append("msd_dataset")
    if conflicts:
        return "source/config conflict", ",".join(sorted(set(conflicts))), ",".join(missing)
    if missing:
        return "partially_verified_not_persisted", "", ",".join(missing)
    return "verified", "", ""


def source_audit_row(task: str, model: str, path: Path, kind: str, config: dict, status: str, conflicts: str, missing: str) -> dict:
    return {
        "task": task,
        "model": model,
        "artifact": f"{model} formal result",
        "path": str(path),
        "status": "fixed_source_used",
        "source_selection": "explicit FORMAL_RESULT_SOURCES mapping",
        "source_kind": kind,
        "record_dataset": config.get("dataset"),
        "seed": config.get("seed"),
        "seq_len": config.get("seq_len"),
        "patch_size": config.get("patch_size"),
        "evaluation_strategy": config.get("strategy_name"),
        "primary_score": "detect_score" if model == "CATCH" else "total_score",
        "config_verification": status,
        "config_conflict_fields": conflicts,
        "config_not_persisted_fields": missing,
        "notes": "GECCO uses the fixed fair seq_len=192 CATCH archive." if task == "GECCO" else "",
    }


def discover_metrics(repo: Path, catch_root: Path) -> tuple[pd.DataFrame, list[dict]]:
    del catch_root
    if set(FORMAL_RESULT_SOURCES) != set(TASKS):
        raise ValueError("FORMAL_RESULT_SOURCES must contain exactly the 23 formal tasks")
    workspace = repo.parent
    rows = []
    sources = []
    for task in TASKS:
        catch_relative, msd_relative = FORMAL_RESULT_SOURCES[task]
        catch_path = workspace / catch_relative
        msd_path = workspace / msd_relative
        if not catch_path.exists() or not msd_path.exists():
            raise FileNotFoundError(f"Fixed formal source missing for {task}: {catch_path}, {msd_path}")
        catch_row = read_tar_row(catch_path)
        if catch_row.get("log_info") or not finite(catch_row.get("auc_pr")) or not finite(catch_row.get("auc_roc")):
            raise ValueError(f"Invalid fixed CATCH archive for {task}: {catch_path}")
        msd_row, msd_kind = fixed_msd_row(msd_path)
        if msd_row is None:
            raise ValueError(f"Invalid fixed MSD source for {task}: {msd_path}")

        catch_config = record_config(catch_row)
        msd_config = record_config(msd_row)
        config_status, conflicts, missing = compare_formal_configs(task, catch_config, msd_config)
        catch_pr = numeric(catch_row["auc_pr"])
        catch_roc = numeric(catch_row["auc_roc"])
        msd_pr = numeric(msd_row["auc_pr"])
        msd_roc = numeric(msd_row["auc_roc"])
        sources.append(source_audit_row(task, "CATCH", catch_path, "fixed archive", catch_config, config_status, conflicts, missing))
        sources.append(source_audit_row(task, "MSDCATCH", msd_path, msd_kind, msd_config, config_status, conflicts, missing))
        rows.append(
            {
                "task": task,
                "paper_dataset": paper_dataset(task),
                "formal_source_selection": "explicit_fixed_mapping",
                "formal_seq_len": catch_config["seq_len"],
                "formal_patch_size": catch_config["patch_size"],
                "catch_record_dataset": catch_config["dataset"],
                "catch_seed": catch_config["seed"],
                "catch_strategy": catch_config["strategy_name"],
                "catch_primary_score": "detect_score",
                "msd_record_dataset": msd_config["dataset"],
                "msd_seed": msd_config["seed"],
                "msd_seq_len": msd_config["seq_len"],
                "msd_patch_size": msd_config["patch_size"],
                "msd_strategy": msd_config["strategy_name"],
                "msd_primary_score": "total_score",
                "source_config_status": config_status,
                "source_config_conflict_fields": conflicts,
                "source_config_not_persisted_fields": missing,
                "catch_auc_pr": catch_pr,
                "msd_auc_pr": msd_pr,
                "delta_auc_pr": msd_pr - catch_pr,
                "pr_group": metric_group(msd_pr - catch_pr),
                "catch_auc_roc": catch_roc,
                "msd_auc_roc": msd_roc,
                "delta_auc_roc": msd_roc - catch_roc,
                "roc_group": metric_group(msd_roc - catch_roc),
                "catch_source": str(catch_path),
                "msd_source": str(msd_path),
                "msd_source_kind": msd_kind,
                "notes": "fixed fair GECCO source" if task == "GECCO" else "",
            }
        )
    return pd.DataFrame(rows), sources


def metadata_lookup(repo: Path) -> dict[str, dict]:
    metadata = pd.read_csv(repo / "dataset" / "anomaly_detect" / "DETECT_META.csv")
    return {Path(row.file_name).stem: row.to_dict() for _, row in metadata.iterrows()}


def read_train_and_labels(path: Path, train_length: int, test_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Stream the same contiguous blocks selected by process_data_df in the formal loader."""
    series_length = train_length + test_length
    total_rows = sum(len(chunk) for chunk in pd.read_csv(path, usecols=["data"], chunksize=200_000))
    if series_length < 1 or total_rows % series_length:
        raise ValueError(
            f"{path.name} has {total_rows} rows, incompatible with formal series length {series_length}"
        )
    block_count = total_rows // series_length
    if block_count < 2:
        raise ValueError(f"{path.name} does not contain feature and label blocks")

    feature_chunks: list[list[np.ndarray]] = [[] for _ in range(block_count - 1)]
    label_chunks: list[np.ndarray] = []
    row_offset = 0
    for chunk in pd.read_csv(path, usecols=["data"], chunksize=200_000):
        values = pd.to_numeric(chunk["data"], errors="coerce").to_numpy(dtype=np.float64)
        offset = 0
        while offset < len(values):
            absolute = row_offset + offset
            block = absolute // series_length
            within_block = absolute % series_length
            count = min(len(values) - offset, series_length - within_block)
            segment = values[offset : offset + count]
            if block < block_count - 1 and within_block < train_length:
                feature_chunks[block].append(segment[: min(count, train_length - within_block)])
            elif block == block_count - 1:
                label_start = max(train_length - within_block, 0)
                if label_start < count:
                    label_chunks.append(segment[label_start:])
            offset += count
        row_offset += len(values)

    features = []
    for values in feature_chunks:
        feature = np.concatenate(values)[:train_length]
        if len(feature) != train_length:
            raise ValueError(f"{path.name} has a feature block with {len(feature)} train points, expected {train_length}")
        median = np.nanmedian(feature)
        features.append(np.nan_to_num(feature, nan=median if math.isfinite(median) else 0.0))
    labels = np.concatenate(label_chunks)[:test_length] if label_chunks else np.empty(0)
    if len(labels) != test_length:
        raise ValueError(f"{path.name} has {len(labels)} formal test labels, expected {test_length}")
    return np.column_stack(features), labels


def formal_loader_test_labels(repo: Path, path: Path, train_length: int) -> np.ndarray:
    """Use the benchmark's data-only long-format loader; no strategy or model is constructed."""
    project_root = str(repo)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from ts_benchmark.data.utils import process_data_df

    loaded = process_data_df(pd.read_csv(path))
    return np.ascontiguousarray(loaded["label"].iloc[train_length:].to_numpy())


def checksum(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def asd_parity_row(task: str, analysis_labels: np.ndarray, loader_labels: np.ndarray, expected_length: int) -> dict:
    analysis_count = int(np.count_nonzero(analysis_labels > 0))
    loader_count = int(np.count_nonzero(loader_labels > 0))
    exact_match = bool(
        analysis_labels.shape == loader_labels.shape
        and analysis_labels.dtype == loader_labels.dtype
        and np.array_equal(analysis_labels, loader_labels)
    )
    analysis_checksum = checksum(analysis_labels)
    loader_checksum = checksum(loader_labels)
    return {
        "asd_subset": task,
        "analysis_length": len(analysis_labels),
        "loader_length": len(loader_labels),
        "expected_formal_test_length": expected_length,
        "analysis_dtype": str(analysis_labels.dtype),
        "loader_dtype": str(loader_labels.dtype),
        "analysis_anomaly_count": analysis_count,
        "loader_anomaly_count": loader_count,
        "anomaly_count_match": analysis_count == loader_count,
        "exact_match": exact_match,
        "analysis_checksum": analysis_checksum,
        "loader_checksum": loader_checksum,
        "checksum_match": analysis_checksum == loader_checksum,
        "formal_length_match": len(loader_labels) == expected_length,
    }
def anomaly_segments(labels: np.ndarray) -> np.ndarray:
    starts = np.flatnonzero(np.diff(np.r_[0, labels, 0]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[0, labels, 0]) == -1)
    return ends - starts


def legal_odd_kernel(kernel: int, sequence_length: int) -> int:
    maximum = sequence_length if sequence_length % 2 else sequence_length - 1
    return max(1, min(max(1, kernel), maximum))


def moving_average(values: np.ndarray, kernel: int) -> np.ndarray:
    """Exact replicate-padded centered average used by frozen MSDCATCH_model.py."""
    padding = kernel // 2
    padded = np.pad(values, ((padding, padding), (0, 0)), mode="edge")
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(padded, axis=0)))
    return (cumulative[kernel:] - cumulative[:-kernel]) / kernel


def checkpoint_gate(path: Path) -> Optional[dict[str, torch.Tensor]]:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") or payload.get("model_state")
    if not isinstance(state, dict):
        return None
    expected = ["scale_gate.net.0.weight", "scale_gate.net.0.bias", "scale_gate.net.2.weight", "scale_gate.net.2.bias"]
    extracted = {}
    for name in expected:
        source = next((key for key in state if key.endswith(name)), None)
        if source is None:
            return None
        extracted[name] = state[source].detach().cpu()
    return extracted


def gate_weights(windows: np.ndarray, state: dict[str, torch.Tensor]) -> np.ndarray:
    values = torch.from_numpy(windows.astype(np.float32, copy=False))
    means = values.mean(dim=1)
    stds = values.std(dim=1, unbiased=False)
    differences = values.diff(dim=1).abs().mean(dim=1) if values.shape[1] > 1 else torch.zeros_like(means)
    features = torch.stack((means, stds, differences), dim=-1)
    with torch.no_grad():
        hidden = torch_functional.gelu(
            torch_functional.linear(features, state["scale_gate.net.0.weight"], state["scale_gate.net.0.bias"])
        )
        logits = torch_functional.linear(hidden, state["scale_gate.net.2.weight"], state["scale_gate.net.2.bias"])
        return torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)


def describe_training_data(
    train: np.ndarray,
    labels: np.ndarray,
    formal_test_length: int,
    sequence_length: int,
    patch_size: int,
    gate_state: Optional[dict[str, torch.Tensor]],
) -> tuple[dict, dict]:
    channels = train.shape[1]
    means = train.mean(axis=0)
    stds = np.maximum(train.std(axis=0), EPSILON)
    standardized = (train - means) / stds
    window_count = len(standardized) // sequence_length
    usable = window_count * sequence_length
    standardized = standardized[:usable]
    raw = train[:usable]
    global_correlation = None
    if channels > 1:
        global_correlation = np.nan_to_num(np.corrcoef(standardized, rowvar=False))

    drift_mean_sum = drift_std_sum = drift_count = 0.0
    legacy_mean_drift_sum = legacy_variance_drift_sum = 0.0
    correlation_sum = correlation_count = 0.0
    low_sum = np.zeros(channels)
    entropy_sum = np.zeros(channels)
    concentration_sum = np.zeros(channels)
    spectral_count = 0
    previous_mean = previous_std = None
    raw_energy = trend_energy = residual_energy = 0.0
    identity_error = 0.0
    scale_sum = np.zeros(3)
    scale_entropy_sum = 0.0
    scale_count = 0
    dominant_count = np.zeros(3, dtype=np.int64)
    kernels = tuple(legal_odd_kernel(value, sequence_length) for value in (patch_size - 1, 2 * patch_size - 1, 4 * patch_size - 1))

    for start in range(0, usable, sequence_length * 32):
        stop = min(start + sequence_length * 32, usable)
        raw_block = raw[start:stop].reshape(-1, sequence_length, channels)
        standard_block = standardized[start:stop].reshape(-1, sequence_length, channels)
        learned_weights = gate_weights(standard_block, gate_state) if gate_state is not None else None
        for index, (raw_window, standard_window) in enumerate(zip(raw_block, standard_block)):
            window_mean = raw_window.mean(axis=0)
            window_std = raw_window.std(axis=0)
            if previous_mean is not None:
                normalized_mean_change = np.abs(window_mean - previous_mean) / (stds + EPSILON)
                normalized_std_change = np.abs(window_std - previous_std) / (stds + EPSILON)
                drift_mean_sum += float(np.mean(normalized_mean_change))
                drift_std_sum += float(np.mean(normalized_std_change))
                legacy_mean_drift_sum += float(np.abs(window_mean - previous_mean).sum() / stds.sum())
                legacy_variance_drift_sum += float(np.abs(window_std - previous_std).sum() / stds.sum())
                drift_count += 1
            previous_mean, previous_std = window_mean, window_std

            spectrum = np.abs(np.fft.rfft(standard_window, axis=0)) ** 2
            nonzero = spectrum[1:]
            if len(nonzero):
                total = np.maximum(nonzero.sum(axis=0), EPSILON)
                low_count = max(1, math.ceil(len(nonzero) * 0.1))
                power = nonzero / total
                low_sum += nonzero[:low_count].sum(axis=0) / total
                entropy_sum += -(power * np.log(np.maximum(power, EPSILON))).sum(axis=0) / math.log(len(nonzero))
                concentration_sum += np.sort(nonzero, axis=0)[-min(3, len(nonzero)):].sum(axis=0) / total
                spectral_count += 1

            if global_correlation is not None:
                window_correlation = np.nan_to_num(np.corrcoef(standard_window, rowvar=False))
                correlation_sum += float(np.linalg.norm(window_correlation - global_correlation, ord="fro"))
                correlation_count += 1

            candidates = np.stack([moving_average(standard_window, kernel) for kernel in kernels], axis=-1)
            if learned_weights is None:
                trend = candidates.mean(axis=-1)
            else:
                weights = learned_weights[index]
                trend = (candidates * weights[None, :, :]).sum(axis=-1)
                scale_sum += weights.sum(axis=0)
                entropy = -(weights * np.log(np.maximum(weights, EPSILON))).sum(axis=-1) / math.log(3)
                scale_entropy_sum += float(entropy.sum())
                scale_count += weights.shape[0]
                dominant_count += np.bincount(weights.argmax(axis=-1), minlength=3)
            residual = standard_window - trend
            raw_energy += float(np.square(standard_window).sum())
            trend_energy += float(np.square(trend).sum())
            residual_energy += float(np.square(residual).sum())
            identity_error = max(identity_error, float(np.abs(trend + residual - standard_window).max()))

    labels_complete = len(labels) == formal_test_length
    segments = anomaly_segments(labels) if labels_complete else None
    base = {
        "channel_count": channels,
        "train_length": len(train),
        "test_length": formal_test_length,
        "label_length_observed": len(labels),
        "label_coverage_complete": labels_complete,
        "anomaly_ratio": float(labels.mean()) if labels_complete else None,
        "anomaly_segment_count": len(segments) if labels_complete else None,
        "median_anomaly_segment_length": float(np.median(segments)) if labels_complete and len(segments) else None,
        "seq_len": sequence_length,
        "patch_size": patch_size,
        "train_windows": window_count,
        "mean_drift": drift_mean_sum / drift_count if drift_count else None,
        "variance_drift": drift_std_sum / drift_count if drift_count else None,
        "mean_drift_pre_correction": legacy_mean_drift_sum / drift_count if drift_count else None,
        "variance_drift_pre_correction": legacy_variance_drift_sum / drift_count if drift_count else None,
        "low_frequency_energy_ratio_mean": float((low_sum / spectral_count).mean()) if spectral_count else None,
        "low_frequency_energy_ratio_median": float(np.median(low_sum / spectral_count)) if spectral_count else None,
        "spectral_entropy": float((entropy_sum / spectral_count).mean()) if spectral_count else None,
        "periodicity_top3_ratio": float((concentration_sum / spectral_count).mean()) if spectral_count else None,
        "correlation_drift": correlation_sum / correlation_count if correlation_count else None,
    }
    decomp = {
        "trend_energy_over_raw": trend_energy / max(raw_energy, EPSILON),
        "residual_energy_over_raw": residual_energy / max(raw_energy, EPSILON),
        "trend_over_residual_energy": trend_energy / max(residual_energy, EPSILON),
        "decomposition_identity_max_error": identity_error,
        "kernels": json.dumps(kernels),
        "decomposition_mode": "checkpoint_scale_gate" if gate_state is not None else "deterministic_equal_weight_no_checkpoint",
        "scale_weight_mean": json.dumps((scale_sum / scale_count).tolist()) if scale_count else None,
        "scale_weight_entropy": scale_entropy_sum / scale_count if scale_count else None,
        "dominant_scale": int(dominant_count.argmax()) if scale_count else None,
        "dominant_scale_frequency": float(dominant_count.max() / scale_count) if scale_count else None,
    }
    return base, decomp


def quantile_diagnostics(labels: np.ndarray, scores: np.ndarray) -> dict:
    normal = scores[labels == 0]
    anomalous = scores[labels == 1]
    normal_iqr = np.subtract(*np.percentile(normal, [75, 25])) if len(normal) else np.nan
    def top_rate(fraction: float) -> float:
        count = max(1, math.ceil(len(scores) * fraction))
        top = np.argpartition(scores, -count)[-count:]
        return float(labels[top].mean())
    return {
        "normal_score_median": float(np.median(normal)) if len(normal) else None,
        "anomaly_score_median": float(np.median(anomalous)) if len(anomalous) else None,
        "normal_score_iqr": float(normal_iqr),
        "anomaly_score_iqr": float(np.subtract(*np.percentile(anomalous, [75, 25]))) if len(anomalous) else None,
        "median_separation_over_normal_iqr": float((np.median(anomalous) - np.median(normal)) / max(normal_iqr, EPSILON)) if len(anomalous) and len(normal) else None,
        "top_1pct_anomaly_fraction": top_rate(0.01),
        "top_5pct_anomaly_fraction": top_rate(0.05),
    }


def formal_msd_payload(repo: Path, task: str) -> Optional[dict]:
    json_path = repo / "result" / "msd_catch_total_screen" / f"{task}.json"
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    log_path = repo / "result" / "msd_catch_screen" / f"{task}.log"
    if task in FORMAL_MSD_LOG and log_path.exists():
        for line in reversed(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload.get("scores"), dict):
                return payload
    return None


def score_diagnostics(repo: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        source = repo / "result" / "msd_catch_total_screen" / f"{task}_scores.npz"
        metric = metrics.loc[metrics.task == task].iloc[0]
        payload = formal_msd_payload(repo, task)
        scores = payload.get("scores", {}) if payload else {}
        common = {
            "task": task,
            "paper_dataset": paper_dataset(task),
            "catch_msd_total_score_spearman": None,
            "catch_score_source": None,
            "msd_score_source": str(source) if source.exists() else None,
            "trend_auc_pr": scores.get("trend_score", {}).get("auc_pr"),
            "residual_auc_pr": scores.get("residual_score", {}).get("auc_pr"),
            "total_auc_pr": metric.msd_auc_pr,
        }
        if not source.exists():
            rows.append({**common, "model": "CATCH", "status": "N/A_no_continuous_score_archive"})
            rows.append(
                {
                    **common,
                    "model": "MSDCATCH",
                    "status": "component_metrics_only" if payload else "N/A_no_continuous_score_archive",
                }
            )
            continue
        if payload is None:
            raise ValueError(f"Missing formal MSD payload for continuous score archive: {task}")
        with np.load(source, allow_pickle=False) as archive:
            labels = archive["labels"].astype(np.int8)
            total = archive["total_score"].astype(np.float64)
        rows.append({**common, "model": "CATCH", "status": "N/A_no_continuous_score_archive"})
        rows.append(
            {
                **common,
                **quantile_diagnostics(labels, total),
                "model": "MSDCATCH",
                "status": "available",
                "trend_auc_pr": scores.get("trend_score", {}).get("auc_pr"),
                "residual_auc_pr": scores.get("residual_score", {}).get("auc_pr"),
                "total_auc_pr": scores.get("total_score", {}).get("auc_pr"),
            }
        )
    return pd.DataFrame(rows)


def paper_metrics(task_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "catch_auc_pr", "msd_auc_pr", "delta_auc_pr", "catch_auc_roc", "msd_auc_roc", "delta_auc_roc"
    ]
    rows = []
    for paper in PAPER_ORDER:
        subset = task_metrics[task_metrics.paper_dataset == paper]
        row = {"paper_dataset": paper, "task_count": len(subset)}
        for column in numeric_columns:
            row[column] = subset[column].dropna().mean() if subset[column].notna().any() else None
        row["pr_group"] = metric_group(row["delta_auc_pr"])
        row["roc_group"] = metric_group(row["delta_auc_roc"])
        row["formal_msd_task_count"] = int(subset.msd_auc_pr.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def spearman(values: pd.DataFrame, descriptor: str, target: str) -> tuple[Optional[float], int, Optional[str], Optional[float], bool]:
    subset = values[["task", descriptor, target]].dropna()
    if len(subset) < 3:
        return None, len(subset), None, None, False
    baseline = subset[descriptor].rank(method="average").corr(subset[target].rank(method="average"))
    shifts = []
    for index, row in subset.iterrows():
        remaining = subset.drop(index)
        value = remaining[descriptor].rank(method="average").corr(remaining[target].rank(method="average"))
        shifts.append((abs(value - baseline), row.task, value))
    shift, task, leave_one_out = max(shifts)
    return float(baseline), len(subset), task, float(shift), bool(np.sign(leave_one_out) != np.sign(baseline))


def correlations(task_values: pd.DataFrame, paper_values: pd.DataFrame) -> pd.DataFrame:
    descriptors = [
        "channel_count", "mean_drift", "variance_drift", "low_frequency_energy_ratio_mean",
        "spectral_entropy", "periodicity_top3_ratio", "correlation_drift", "trend_energy_over_raw",
        "residual_energy_over_raw", "trend_over_residual_energy",
    ]
    rows = []
    for level, values in (("task", task_values), ("paper", paper_values)):
        for descriptor in descriptors:
            for target in ("delta_auc_pr", "delta_auc_roc"):
                rho, count, influential, shift, sign_flip = spearman(values, descriptor, target)
                rows.append(
                    {
                        "level": level,
                        "descriptor": descriptor,
                        "target": target,
                        "spearman_rho": rho,
                        "n": count,
                        "most_influential_leave_one_out": influential,
                        "max_leave_one_out_shift": shift,
                        "leave_one_out_sign_flip": sign_flip,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_paper_descriptors(task_descriptors: pd.DataFrame, paper_metric_rows: pd.DataFrame) -> pd.DataFrame:
    numeric = task_descriptors.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for paper in PAPER_ORDER:
        subset = task_descriptors[task_descriptors.paper_dataset == paper]
        row = {"task": paper, "paper_dataset": paper}
        for column in numeric:
            row[column] = subset[column].dropna().mean() if subset[column].notna().any() else None
        metric = paper_metric_rows[paper_metric_rows.paper_dataset == paper].iloc[0]
        row["delta_auc_pr"] = metric.delta_auc_pr
        row["delta_auc_roc"] = metric.delta_auc_roc
        row["pr_group"] = metric.pr_group
        row["roc_group"] = metric.roc_group
        rows.append(row)
    return pd.DataFrame(rows)



CORRELATION_DESCRIPTORS = [
    "channel_count", "mean_drift", "variance_drift", "low_frequency_energy_ratio_mean",
    "spectral_entropy", "periodicity_top3_ratio", "correlation_drift", "trend_energy_over_raw",
    "residual_energy_over_raw", "trend_over_residual_energy",
]


def rank_spearman(values: pd.DataFrame, descriptor: str, target: str) -> Optional[float]:
    subset = values[[descriptor, target]].dropna()
    if len(subset) < 3:
        return None
    return numeric(subset[descriptor].rank(method="average").corr(subset[target].rank(method="average")))


def has_sign_flip(full_rho: Optional[float], candidate_rho: Optional[float]) -> bool:
    return full_rho is not None and candidate_rho is not None and np.sign(candidate_rho) != np.sign(full_rho)


def row_leave_one_out(values: pd.DataFrame, level: str, descriptor: str, target: str) -> tuple[dict, list[dict]]:
    subset = values[["task", "paper_dataset", descriptor, target]].dropna()
    full_rho = rank_spearman(subset, descriptor, target)
    rows = []
    for index, row in subset.iterrows():
        loo_rho = rank_spearman(subset.drop(index), descriptor, target)
        rows.append(
            {
                "level": level,
                "descriptor": descriptor,
                "target": target,
                "removed_item": row.task,
                "removed_group": row.paper_dataset,
                "n": len(subset),
                "full_rho": full_rho,
                "loo_rho": loo_rho,
                "absolute_shift": abs(loo_rho - full_rho) if loo_rho is not None and full_rho is not None else None,
                "sign_flip": has_sign_flip(full_rho, loo_rho),
            }
        )
    valid = [row for row in rows if row["loo_rho"] is not None]
    influential = max(valid, key=lambda row: row["absolute_shift"]) if valid else None
    return {
        "spearman_rho": full_rho,
        "n": len(subset),
        "most_influential_leave_one_out": influential["removed_item"] if influential else None,
        "max_leave_one_out_shift": influential["absolute_shift"] if influential else None,
        "minimum_loo_rho": min(row["loo_rho"] for row in valid) if valid else None,
        "maximum_loo_rho": max(row["loo_rho"] for row in valid) if valid else None,
        "any_loo_sign_flip": any(row["sign_flip"] for row in rows),
    }, rows


def grouped_leave_out(values: pd.DataFrame, descriptor: str, target: str) -> tuple[dict, list[dict]]:
    subset = values[["task", "paper_dataset", descriptor, target]].dropna()
    full_rho = rank_spearman(subset, descriptor, target)
    rows = []
    for group in PAPER_ORDER:
        remaining = subset[subset.paper_dataset != group]
        group_rho = rank_spearman(remaining, descriptor, target)
        if len(remaining) == len(subset):
            continue
        rows.append(
            {
                "level": "task",
                "descriptor": descriptor,
                "target": target,
                "removed_group": group,
                "n": len(subset),
                "remaining_n": len(remaining),
                "full_rho": full_rho,
                "group_loo_rho": group_rho,
                "absolute_shift": abs(group_rho - full_rho) if group_rho is not None and full_rho is not None else None,
                "sign_flip": has_sign_flip(full_rho, group_rho),
            }
        )
    valid = [row for row in rows if row["group_loo_rho"] is not None]
    asd = next((row for row in rows if row["removed_group"] == "ASD"), None)
    return {
        "rho_without_asd_group": asd["group_loo_rho"] if asd else None,
        "minimum_group_loo_rho": min(row["group_loo_rho"] for row in valid) if valid else None,
        "maximum_group_loo_rho": max(row["group_loo_rho"] for row in valid) if valid else None,
        "any_group_loo_sign_flip": any(row["sign_flip"] for row in rows),
    }, rows


def descriptor_group_differences(values: pd.DataFrame, level: str, correlations_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for descriptor in CORRELATION_DESCRIPTORS:
        for target, group_column in (("delta_auc_pr", "pr_group"), ("delta_auc_roc", "roc_group")):
            subset = values[[descriptor, group_column]].dropna()
            gains = subset.loc[subset[group_column] == "gain", descriptor]
            other = subset.loc[subset[group_column].isin(["loss", "neutral"]), descriptor]
            rho = correlations_df.loc[
                (correlations_df.level == level)
                & (correlations_df.descriptor == descriptor)
                & (correlations_df.target == target),
                "spearman_rho",
            ].iloc[0]
            gain_median = numeric(gains.median()) if len(gains) else None
            other_median = numeric(other.median()) if len(other) else None
            difference = gain_median - other_median if gain_median is not None and other_median is not None else None
            direction_consistent = bool(
                difference is not None
                and rho is not None
                and ((rho > 0 and difference > 0) or (rho < 0 and difference < 0))
            )
            rows.append(
                {
                    "level": level,
                    "descriptor": descriptor,
                    "target": target,
                    "gain_count": len(gains),
                    "loss_or_neutral_count": len(other),
                    "gain_median": gain_median,
                    "loss_or_neutral_median": other_median,
                    "gain_minus_loss_or_neutral_median": difference,
                    "direction_consistent_with_rho": direction_consistent,
                }
            )
    return pd.DataFrame(rows)


def correlations(task_values: pd.DataFrame, paper_values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    row_loo_rows = []
    group_loo_rows = []
    for level, values in (("task", task_values), ("paper", paper_values)):
        for descriptor in CORRELATION_DESCRIPTORS:
            for target in ("delta_auc_pr", "delta_auc_roc"):
                row_summary, rows = row_leave_one_out(values, level, descriptor, target)
                row_loo_rows.extend(rows)
                summary_rows.append({"level": level, "descriptor": descriptor, "target": target, **row_summary})
                if level == "task":
                    group_summary, groups = grouped_leave_out(values, descriptor, target)
                    group_loo_rows.extend(groups)
                    summary_rows[-1].update(group_summary)
                else:
                    summary_rows[-1].update(
                        {
                            "rho_without_asd_group": None,
                            "minimum_group_loo_rho": None,
                            "maximum_group_loo_rho": None,
                            "any_group_loo_sign_flip": None,
                        }
                    )
    summary = pd.DataFrame(summary_rows)
    differences = pd.concat(
        [
            descriptor_group_differences(task_values, "task", summary),
            descriptor_group_differences(paper_values, "paper", summary),
        ],
        ignore_index=True,
    )
    qualifications = []
    for descriptor in CORRELATION_DESCRIPTORS:
        for target in ("delta_auc_pr", "delta_auc_roc"):
            task_row = summary[(summary.level == "task") & (summary.descriptor == descriptor) & (summary.target == target)].iloc[0]
            paper_row = summary[(summary.level == "paper") & (summary.descriptor == descriptor) & (summary.target == target)].iloc[0]
            group_diff = differences[(differences.level == "task") & (differences.descriptor == descriptor) & (differences.target == target)].iloc[0]
            same_direction = (
                task_row.spearman_rho is not None
                and paper_row.spearman_rho is not None
                and np.sign(task_row.spearman_rho) == np.sign(paper_row.spearman_rho)
            )
            qualified = bool(
                same_direction
                and abs(task_row.spearman_rho) >= 0.35
                and abs(paper_row.spearman_rho) >= 0.35
                and not task_row.any_loo_sign_flip
                and not paper_row.any_loo_sign_flip
                and not task_row.any_group_loo_sign_flip
                and task_row.rho_without_asd_group is not None
                and np.sign(task_row.rho_without_asd_group) == np.sign(task_row.spearman_rho)
                and group_diff.gain_count >= 3
                and group_diff.loss_or_neutral_count >= 3
                and group_diff.direction_consistent_with_rho
            )
            qualifications.append(
                {
                    "descriptor": descriptor,
                    "target": target,
                    "qualified_for_pr": qualified if target == "delta_auc_pr" else False,
                    "qualified_for_roc": qualified if target == "delta_auc_roc" else False,
                }
            )
    summary = summary.merge(pd.DataFrame(qualifications), on=["descriptor", "target"], how="left")
    return summary, pd.DataFrame(row_loo_rows), pd.DataFrame(group_loo_rows), differences


def paper_reference(repo: Path, catch_root: Path) -> pd.DataFrame:
    documents = list(repo.glob("README*")) + list(catch_root.glob("README*"))
    documents += list(repo.rglob("*.md")) + list(catch_root.rglob("*.md"))
    candidates = []
    for path in documents:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "CATCH" in text and ("AUC-PR" in text or "AUC-ROC" in text):
            candidates.append(str(path))
    source = "No reliable structured original CATCH paper table found in repository."
    if candidates:
        source += " Candidate documents were not treated as a numeric paper table: " + "; ".join(sorted(set(candidates))[:3])
    return pd.DataFrame(
        {
            "dataset": PAPER_ORDER,
            "baseline": "N/A",
            "paper_pr_roc": "N/A",
            "catch": "N/A",
            "msd": "N/A",
            "source": source,
            "status": "Paper-reported reference not rerun under current commit; no reliable table parsed.",
        }
    )


def report(output: Path, task_metrics: pd.DataFrame, paper: pd.DataFrame, correlations_df: pd.DataFrame) -> None:
    available = task_metrics.dropna(subset=["delta_auc_pr"])
    gains = int((available.delta_auc_pr > 0.01).sum())
    losses_or_neutral = int((available.delta_auc_pr <= 0.01).sum())
    qualified = []
    for descriptor in correlations_df.descriptor.unique():
        task = correlations_df[(correlations_df.level == "task") & (correlations_df.descriptor == descriptor) & (correlations_df.target == "delta_auc_pr")].iloc[0]
        paper_row = correlations_df[(correlations_df.level == "paper") & (correlations_df.descriptor == descriptor) & (correlations_df.target == "delta_auc_pr")].iloc[0]
        if (
            task.spearman_rho is not None and paper_row.spearman_rho is not None
            and np.sign(task.spearman_rho) == np.sign(paper_row.spearman_rho)
            and abs(task.spearman_rho) >= 0.35 and abs(paper_row.spearman_rho) >= 0.35
            and not task.leave_one_out_sign_flip and not paper_row.leave_one_out_sign_flip
            and gains >= 3 and losses_or_neutral >= 3
        ):
            qualified.append(descriptor)
    if qualified:
        conclusion = "A: at least one descriptor meets the numerical cross-level screen: " + ", ".join(qualified)
    elif gains and losses_or_neutral:
        conclusion = "B: performance has mixed task groups, but no descriptor meets all cross-level hypothesis criteria."
    else:
        conclusion = "C: no repeated gain/loss grouping is available in the current formal records."
    lines = [
        "# Decomposition Applicability Report",
        "",
        "This report is read-only: it does not call a training, scoring, or benchmark entry point.",
        "",
        "## Formal Performance Coverage",
        f"- Formal MSD delta available for {len(available)}/{len(task_metrics)} execution tasks.",
        "- PSM, Genesis, and GECCO use the terminal formal total-screen JSON line in their MSD logs.",
        "- ASD paper-level values are equal-weight macro means of its 12 execution tasks.",
        "",
        "## Descriptor Method",
        "- Raw descriptors use normal training prefixes and non-overlapping formal seq_len windows.",
        "- Spectral quantities are per-channel training-window summaries; correlation drift is N/A for one channel.",
        "- Decomposition uses replicate-padded moving averages and verifies trend + residual = input.",
        "",
        "## Task-Level Performance",
        "```text",
        task_metrics[["task", "catch_auc_pr", "msd_auc_pr", "delta_auc_pr", "pr_group", "catch_auc_roc", "msd_auc_roc", "delta_auc_roc", "roc_group", "msd_source_kind"]].to_string(index=False),
        "```",
        "",
        "## Paper-Level Performance",
        "```text",
        paper[["paper_dataset", "task_count", "catch_auc_pr", "msd_auc_pr", "delta_auc_pr", "catch_auc_roc", "msd_auc_roc", "delta_auc_roc"]].to_string(index=False),
        "```",
        "",
        "## Score Diagnostics",
        "- CATCH continuous scores are not stored in the formal archives, so CATCH-versus-MSD Spearman score correlation is N/A without a rerun.",
        "- MSD quantile diagnostics are available only where a frozen `*_scores.npz` exists; component AUCs from formal logs are still retained.",
        "",
        "## Descriptor Correlations",
        "```text",
        correlations_df[["level", "descriptor", "target", "spearman_rho", "n", "most_influential_leave_one_out", "max_leave_one_out_shift", "leave_one_out_sign_flip"]].to_string(index=False),
        "```",
        "",
        "## Paper Baseline Reference",
        "- No reliable structured original CATCH paper table was present locally; paper-reported values remain N/A rather than being substituted by current reruns.",
        "",
        "## Conclusion",
        f"- {conclusion}",
        "- This is descriptive evidence from frozen runs, not a model-selection rule or a new implementation proposal.",
    ]
    (output / "DECOMPOSITION_APPLICABILITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def drift_correction_summary(descriptors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for corrected, legacy in (("mean_drift", "mean_drift_pre_correction"), ("variance_drift", "variance_drift_pre_correction")):
        subset = descriptors[["task", corrected, legacy]].dropna()
        difference = (subset[corrected] - subset[legacy]).abs()
        corrected_rank = subset[corrected].rank(method="min")
        legacy_rank = subset[legacy].rank(method="min")
        rows.append(
            {
                "descriptor": corrected,
                "maximum_absolute_difference": float(difference.max()) if len(difference) else None,
                "tasks_with_rank_change": int((corrected_rank != legacy_rank).sum()),
                "maximum_rank_change": int((corrected_rank - legacy_rank).abs().max()) if len(subset) else None,
            }
        )
    return pd.DataFrame(rows)


def candidate_screen(correlations_df: pd.DataFrame, differences: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for descriptor in CORRELATION_DESCRIPTORS:
        for target in ("delta_auc_pr", "delta_auc_roc"):
            task_row = correlations_df[(correlations_df.level == "task") & (correlations_df.descriptor == descriptor) & (correlations_df.target == target)].iloc[0]
            paper_row = correlations_df[(correlations_df.level == "paper") & (correlations_df.descriptor == descriptor) & (correlations_df.target == target)].iloc[0]
            difference = differences[(differences.level == "task") & (differences.descriptor == descriptor) & (differences.target == target)].iloc[0]
            rows.append(
                {
                    "descriptor": descriptor,
                    "target": target,
                    "task_rho": task_row.spearman_rho,
                    "paper_rho": paper_row.spearman_rho,
                    "rho_without_asd_group": task_row.rho_without_asd_group,
                    "task_any_loo_sign_flip": task_row.any_loo_sign_flip,
                    "paper_any_loo_sign_flip": paper_row.any_loo_sign_flip,
                    "task_any_group_loo_sign_flip": task_row.any_group_loo_sign_flip,
                    "gain_median": difference.gain_median,
                    "loss_or_neutral_median": difference.loss_or_neutral_median,
                    "median_direction_consistent": difference.direction_consistent_with_rho,
                    "qualified_for_pr": task_row.qualified_for_pr,
                    "qualified_for_roc": task_row.qualified_for_roc,
                }
            )
    return pd.DataFrame(rows)


def report(
    output: Path,
    task_metrics: pd.DataFrame,
    paper: pd.DataFrame,
    descriptors: pd.DataFrame,
    correlations_df: pd.DataFrame,
    differences: pd.DataFrame,
    parity: pd.DataFrame,
) -> None:
    available = task_metrics.dropna(subset=["delta_auc_pr"])
    current_pr_groups = task_metrics.pr_group.value_counts().to_dict()
    previous_delta = task_metrics[["task", "delta_auc_pr"]].copy()
    previous_delta.loc[previous_delta.task == "GECCO", "delta_auc_pr"] = 0.40649507103606486 - 0.4174229654567119
    previous_groups = previous_delta.delta_auc_pr.map(metric_group).value_counts().to_dict()
    drift_summary = drift_correction_summary(descriptors)
    screen = candidate_screen(correlations_df, differences)
    qualified = screen[(screen.qualified_for_pr) | (screen.qualified_for_roc)]
    cross_level_signal = screen[(screen.task_rho.abs() >= 0.35) & (screen.paper_rho.abs() >= 0.35) & (np.sign(screen.task_rho) == np.sign(screen.paper_rho))]
    if len(qualified):
        clauses = []
        for _, row in qualified.iterrows():
            label = "Delta AUC-PR" if row.qualified_for_pr else "Delta AUC-ROC"
            clauses.append(f"{row.descriptor} is a candidate association for {label}")
        conclusion = "A: " + "; ".join(clauses) + "."
    elif len(cross_level_signal):
        conclusion = "B: candidate stratification signals exist, but the complete robustness and grouped-LOO criteria are not met."
    else:
        conclusion = "C: the corrected formal sources and descriptors do not support a repeatable explanatory condition for fixed decomposition benefit."
    conflict_tasks = task_metrics.loc[task_metrics.source_config_status == "source/config conflict", "task"].tolist()
    partial_tasks = task_metrics.loc[task_metrics.source_config_status == "partially_verified_not_persisted", "task"].tolist()
    parity_ok = bool(len(parity) == 12 and parity.exact_match.all() and parity.checksum_match.all() and parity.anomaly_count_match.all() and parity.formal_length_match.all())
    lines = [
        "# Decomposition Applicability Report",
        "",
        "This report is read-only: it does not train, infer, rescore, or invoke a benchmark runner.",
        "",
        "## Fixed Formal Sources",
        f"- Explicit frozen source mapping used for {len(available)}/{len(task_metrics)} execution tasks; no archive was selected by mtime.",
        "- GECCO CATCH uses CATCH_RSA_GECCO seq_len=192: PR 0.409311912, ROC 0.963459932; MSD uses the seq_len=192 total_score: PR 0.406495071, ROC 0.964381743.",
        f"- Source/config conflicts: {', '.join(conflict_tasks) if conflict_tasks else 'none'}.",
        f"- Required fields not persisted by the frozen log/JSON sources: {', '.join(partial_tasks) if partial_tasks else 'none'}; these are reported as partial verification, not silently substituted.",
        "",
        "## ASD Loader Parity",
        f"- ASD parity exact match: {int(parity.exact_match.sum())}/12; overall status: {'pass' if parity_ok else 'fail'}.",
        "- Analysis labels and formal loader labels are compared for shape, dtype, anomaly count, element equality, and SHA-256 checksum.",
        "",
        "## Drift Correction",
        "```text",
        drift_summary.to_string(index=False),
        "```",
        "- mean_drift and variance_drift now use per-channel normalized changes before averaging adjacent-window pairs; legacy values are retained only for this comparison.",
        "",
        "## Performance Groups",
        f"- Task PR groups before GECCO fair-source correction: {previous_groups}; after correction: {current_pr_groups}.",
        "- ASD paper-level values are equal-weight macro means of its 12 execution tasks.",
        "```text",
        paper[["paper_dataset", "task_count", "delta_auc_pr", "pr_group", "delta_auc_roc", "roc_group"]].to_string(index=False),
        "```",
        "",
        "## Candidate Correlations",
        "```text",
        screen.to_string(index=False),
        "```",
        "- Row LOO evaluates every removed execution/paper record. Grouped LOO removes all ASD subsets together and each remaining paper dataset once.",
        "",
        "## Score Diagnostics",
        "- CATCH continuous scores are absent from frozen formal archives; score-vector Spearman remains N/A without prohibited rescoring.",
        "",
        "## Conclusion",
        f"- {conclusion}",
        "- Any qualified association is descriptive, non-causal, not a direct model-selection rule, and not externally validated on independent datasets.",
    ]
    (output / "DECOMPOSITION_APPLICABILITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    catch_root = repo.parent / "CATCH-master"
    output = repo / "result" / "decomposition_applicability"
    output.mkdir(parents=True, exist_ok=True)

    task_metrics, sources = discover_metrics(repo, catch_root)
    metadata = metadata_lookup(repo)
    descriptor_rows = []
    parity_rows = []
    decomposition_rows = []

    for task in TASKS:
        metric = task_metrics.loc[task_metrics.task == task].iloc[0]
        meta = metadata.get(task)
        raw_path = repo / "dataset" / "anomaly_detect" / "data" / f"{task}.csv"
        sources.append(
            {
                "task": task,
                "artifact": "raw long-format dataset",
                "path": str(raw_path),
                "status": "used" if raw_path.exists() else "missing",
                "notes": "Normal training prefix only for descriptors",
            }
        )
        sources.append(
            {
                "task": task,
                "artifact": "DETECT_META entry",
                "path": str(repo / "dataset" / "anomaly_detect" / "DETECT_META.csv"),
                "status": "used" if meta else "missing",
                "notes": "Recorded train_lens and test_lens",
            }
        )
        bhd_json = repo / "result" / "bhd_msd_catch_screen" / f"{task}.json"
        bhd_archives = list((repo / "result" / "score" / "by_dataset" / task).rglob("*BHD*.tar.gz"))
        bhd_path = bhd_json if bhd_json.exists() else (bhd_archives[0] if bhd_archives else None)
        sources.append(
            {
                "task": task,
                "artifact": "BHD frozen full-line result context",
                "path": str(bhd_path) if bhd_path else None,
                "status": "available_not_compared" if bhd_path else "missing",
                "notes": "Full-line BHD evidence retained as background; not used in CATCH-versus-MSD deltas",
            }
        )
        if meta is None or not raw_path.exists() or not finite(metric.formal_seq_len) or not finite(metric.formal_patch_size):
            descriptor_rows.append(
                {
                    "task": task,
                    "paper_dataset": paper_dataset(task),
                    "descriptor_status": "N/A_missing_raw_metadata_or_formal_config",
                }
            )
            decomposition_rows.append(
                {
                    "task": task,
                    "paper_dataset": paper_dataset(task),
                    "decomposition_status": "N/A_missing_raw_metadata_or_formal_config",
                }
            )
            continue

        train_length = config_integer(meta, "train_lens")
        test_length = config_integer(meta, "test_lens")
        if train_length is None or test_length is None:
            raise ValueError(f"Invalid DETECT_META lengths for {task}")
        checkpoint_path = repo / "result" / "msd_catch_total_screen" / f"{task}.pt"
        gate_state = checkpoint_gate(checkpoint_path)
        sources.append(
            {
                "task": task,
                "artifact": "MSD scale-gate checkpoint",
                "path": str(checkpoint_path) if checkpoint_path.exists() else None,
                "status": "used" if gate_state is not None else "not_available",
                "notes": "Weights inspected only; no model forward, scoring, or training",
            }
        )
        print(f"[descriptor] {task}", flush=True)
        train, labels = read_train_and_labels(raw_path, train_length, test_length)
        if task.startswith("ASD_dataset_"):
            loader_labels = formal_loader_test_labels(repo, raw_path, train_length)
            parity_rows.append(asd_parity_row(task, labels, loader_labels, test_length))
            sources.append(
                {
                    "task": task,
                    "artifact": "formal benchmark data-only loader",
                    "path": str(raw_path),
                    "status": "used_for_asd_label_parity",
                    "notes": "process_data_df only; no strategy, score, or model invocation",
                }
            )
        base, decomp = describe_training_data(
            train=train,
            labels=labels,
            formal_test_length=test_length,
            sequence_length=int(metric.formal_seq_len),
            patch_size=int(metric.formal_patch_size),
            gate_state=gate_state,
        )
        descriptor_rows.append(
            {
                "task": task,
                "paper_dataset": paper_dataset(task),
                "descriptor_status": "available",
                **base,
            }
        )
        decomposition_rows.append(
            {
                "task": task,
                "paper_dataset": paper_dataset(task),
                "decomposition_status": "available",
                **decomp,
            }
        )

    descriptors = pd.DataFrame(descriptor_rows)
    decomposition = pd.DataFrame(decomposition_rows)
    parity = pd.DataFrame(parity_rows).sort_values("asd_subset").reset_index(drop=True)
    task_metrics.to_csv(output / "task_level_metrics.csv", index=False)
    descriptors.to_csv(output / "dataset_descriptors.csv", index=False)
    decomposition.to_csv(output / "decomposition_descriptors.csv", index=False)
    pd.DataFrame(sources).to_csv(output / "analysis_sources.csv", index=False)
    parity.to_csv(output / "asd_label_parity.csv", index=False)
    parity_pass = bool(
        len(parity) == 12
        and parity.exact_match.all()
        and parity.checksum_match.all()
        and parity.anomaly_count_match.all()
        and parity.formal_length_match.all()
    )
    if not parity_pass:
        raise RuntimeError("ASD label parity failed; descriptor association outputs were not regenerated.")

    diagnostics = score_diagnostics(repo, task_metrics)
    paper = paper_metrics(task_metrics)
    task_descriptor_values = descriptors.merge(decomposition, on=["task", "paper_dataset"], how="outer")
    task_values = task_metrics.merge(task_descriptor_values, on=["task", "paper_dataset"], how="left")
    paper_values = aggregate_paper_descriptors(task_descriptor_values, paper)
    correlations_df, row_loo, group_loo, differences = correlations(task_values, paper_values)
    baseline_reference = paper_reference(repo, catch_root)

    paper.to_csv(output / "paper_level_metrics.csv", index=False)
    diagnostics.to_csv(output / "score_distribution_diagnostics.csv", index=False)
    correlations_df.to_csv(output / "descriptor_delta_correlations.csv", index=False)
    differences.to_csv(output / "descriptor_group_differences.csv", index=False)
    row_loo.to_csv(output / "descriptor_leave_one_out.csv", index=False)
    group_loo.to_csv(output / "descriptor_group_leave_out.csv", index=False)
    baseline_reference.to_csv(output / "paper_baseline_reference.csv", index=False)
    report(output, task_metrics, paper, descriptors, correlations_df, differences, parity)
    print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
