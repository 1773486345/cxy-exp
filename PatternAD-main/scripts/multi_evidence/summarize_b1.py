#!/usr/bin/env python3
"""Summarize fixed-seed Direction B1 results without reselecting any run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return value


def _protocol_hash(run_dir: Path) -> str:
    config = _load_json(run_dir / "synthetic_suite" / "resolved_config.json")
    config = dict(config)
    config.pop("seed", None)
    return _canonical_hash(config)


def _gate(row: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    try:
        return row["gates"][name]
    except KeyError as error:
        raise ValueError(f"B1 output is missing required gate {name!r}.") from error


def _per_seed_row(result: Mapping[str, Any], run_dir: Path) -> Dict[str, Any]:
    background = _gate(result, "background_normal_fpr")
    gates = result["gates"]
    return {
        "seed": int(result["seed"]),
        "status": str(result["status"]),
        "all_gates_pass": bool(all(value["pass"] for value in gates.values())),
        "cross_relative_improvement": float(
            result["cross_reference_skill"]["relative_improvement"]
        ),
        "coherent_exceedance_count": int(_gate(result, "coherent_control")["count"]),
        "background_max_group_fpr": float(
            max(
                value
                for component in background["components"].values()
                for value in component["by_reliability_stratum"].values()
            )
        ),
        "background_disagreement_fpr_gap": float(background["disagreement_fpr_gap"]),
        "unsupported_cross_median_tail_delta": float(
            _gate(result, "unsupported_target_break_cross_residual_tail")["median_delta"]
        ),
        "unsupported_disagreement_median_tail_delta": float(
            _gate(result, "unsupported_target_break_disagreement_tail")["median_delta"]
        ),
        "omission_cross_median_tail_delta": float(
            _gate(result, "target_omission_break_cross_residual_tail")["median_delta"]
        ),
        "omission_disagreement_median_tail_delta": float(
            _gate(result, "target_omission_break_disagreement_tail")["median_delta"]
        ),
        "target_spike_success_count": int(
            _gate(result, "target_spike_residuals")["success_count"]
        ),
        "result_dir": str(run_dir),
    }


def _range(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def summarize(input_dirs: List[Path], output_dir: Path) -> Dict[str, Any]:
    if not input_dirs:
        raise ValueError("At least one B1 input directory is required.")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite summary directory: {output_dir}")
    results = []
    protocol_hashes = set()
    provenance_hashes: Dict[str, set[str]] = {
        "generator_sha256": set(),
        "runner_sha256": set(),
        "model_sha256": set(),
        "reliability_calibration_sha256": set(),
    }
    for run_dir in input_dirs:
        result_path = run_dir / "b1_evaluation.json"
        result = _load_json(result_path)
        if result.get("phase") != "B1":
            raise ValueError(f"{result_path} is not a B1 result.")
        if result.get("calibration", {}).get("mode") != "input_energy_stratified":
            raise ValueError(f"{result_path} does not use B1 reliability calibration.")
        protocol_hashes.add(_protocol_hash(run_dir))
        for name, hashes in provenance_hashes.items():
            hashes.add(str(result["provenance"][name]))
        results.append(_per_seed_row(result, run_dir))
    if len(protocol_hashes) != 1:
        raise ValueError("B1 runs do not share one frozen protocol hash.")
    inconsistent = [name for name, hashes in provenance_hashes.items() if len(hashes) != 1]
    if inconsistent:
        raise ValueError(f"B1 runs use inconsistent source hashes: {inconsistent}")
    results.sort(key=lambda row: int(row["seed"]))
    if len({int(row["seed"]) for row in results}) != len(results):
        raise ValueError("B1 input directories contain duplicate seeds.")
    output_dir.mkdir(parents=True, exist_ok=False)
    fields = list(results[0])
    with (output_dir / "b1_multiseed_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "phase": "B1",
        "seed_count": len(results),
        "seeds": [int(row["seed"]) for row in results],
        "all_seed_gates_pass": bool(all(row["all_gates_pass"] for row in results)),
        "protocol_hash_without_seed": next(iter(protocol_hashes)),
        "source_hashes": {name: next(iter(hashes)) for name, hashes in provenance_hashes.items()},
        "metrics": {
            name: _range(float(row[name]) for row in results)
            for name in (
                "cross_relative_improvement",
                "background_max_group_fpr",
                "background_disagreement_fpr_gap",
                "unsupported_cross_median_tail_delta",
                "unsupported_disagreement_median_tail_delta",
                "omission_cross_median_tail_delta",
                "omission_disagreement_median_tail_delta",
            )
        },
        "integer_metrics": {
            name: [int(row[name]) for row in results]
            for name in ("coherent_exceedance_count", "target_spike_success_count")
        },
        "runs": results,
    }
    (output_dir / "b1_multiseed_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    summary = summarize(arguments.input_dir, arguments.output_dir)
    print(
        "B1 multi-seed status: "
        f"{'passed' if summary['all_seed_gates_pass'] else 'failed_gates'}; "
        f"seeds={summary['seeds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
