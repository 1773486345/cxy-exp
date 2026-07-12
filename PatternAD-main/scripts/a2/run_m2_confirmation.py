#!/usr/bin/env python3
"""Run A2-M2's frozen confirmation seeds after its event-pre control passes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a2.generate_transition_contract import _load_json, generate_suite
from scripts.a2.audit_transition_contract import audit_suite
from scripts.a2.run_m1b_confirmation import _seed_pairs
from scripts.a2.run_transition_compatibility import _canonical_hash, run_experiment, write_result
from ts_benchmark.baselines.A2TransitionCompatibility import A2ContrastiveCompatibility


DEFAULT_CONTRACT_CONFIG = REPO_ROOT / "config" / "a2" / "transition_contract_v2.json"
DEFAULT_CONFIRMATION_CONFIG = REPO_ROOT / "config" / "a2" / "m2_v2_confirmation_v1.json"
DEFAULT_M2_CONFIG = REPO_ROOT / "config" / "a2" / "contrastive_energy_m2_v2.json"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _require_passing_ablation(ablation_path: Path, base_contract: Mapping[str, Any]) -> None:
    ablation = _load_json(ablation_path)
    if not bool(ablation.get("gate", {}).get("passed")):
        raise RuntimeError("A2-M2 confirmation requires a passing event-pre ablation artifact.")
    expected_contract_hash = _canonical_hash(base_contract)
    if str(ablation.get("contract_config_hash")) != expected_contract_hash:
        raise RuntimeError(
            "A2-M2 ablation artifact was generated from a different contract configuration."
        )


def _preflight_contract_seeds(
    base_contract: Mapping[str, Any], seed_pairs: List[tuple[int, int]]
) -> List[Dict[str, Any]]:
    """Audit every frozen contract before creating any model result directory."""
    audits = []
    for contract_seed, model_seed in seed_pairs:
        contract = dict(base_contract, seed=contract_seed)
        audit = audit_suite(contract, generate_suite(contract))
        audits.append(
            {
                "contract_seed": contract_seed,
                "model_seed": model_seed,
                "passed": bool(audit["passed"]),
                "cue_observability_accuracy": float(audit["event_pre_cue_observability_accuracy"]),
                "cue_observability_margin": float(audit["event_pre_cue_observability_margin"]),
                "cue_maximum_amplitude_error": float(
                    audit.get("event_pre_cue_maximum_amplitude_error", 0.0)
                ),
                "violations": list(audit["violations"]),
            }
        )
    failed = [item for item in audits if not item["passed"]]
    if failed:
        raise RuntimeError(f"A2-M2 confirmation contract preflight failed: {failed}")
    return audits


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-config", type=Path, default=DEFAULT_CONTRACT_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_M2_CONFIG)
    parser.add_argument("--confirmation-config", type=Path, default=DEFAULT_CONFIRMATION_CONFIG)
    parser.add_argument("--ablation-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=1)
    arguments = parser.parse_args(argv)
    if arguments.torch_threads < 1:
        raise ValueError("--torch-threads must be positive.")
    if arguments.output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing confirmation root: {arguments.output_root}"
        )
    torch.set_num_threads(arguments.torch_threads)
    base_contract = _load_json(arguments.contract_config)
    base_experiment = _load_json(arguments.experiment_config)
    confirmation = _load_json(arguments.confirmation_config)
    seed_pairs = _seed_pairs(confirmation)
    _require_passing_ablation(arguments.ablation_result, base_contract)
    contract_preflight = _preflight_contract_seeds(base_contract, seed_pairs)
    arguments.output_root.mkdir(parents=True, exist_ok=False)
    index_rows = []
    for contract_seed, model_seed in seed_pairs:
        contract = dict(base_contract, seed=contract_seed)
        experiment = dict(base_experiment, seed=model_seed)
        summary, checkpoint = run_experiment(
            contract, experiment, model_factory=A2ContrastiveCompatibility
        )
        output_dir = arguments.output_root / f"contract{contract_seed}_model{model_seed}"
        write_result(output_dir, summary, checkpoint)
        index_rows.append(
            {
                "contract_seed": contract_seed,
                "model_seed": model_seed,
                "all_gates_passed": bool(summary["all_gates_passed"]),
                "summary": str(output_dir / "summary.json"),
            }
        )
    index = {
        "confirmation_id": str(confirmation["confirmation_id"]),
        "experiment_id": str(base_experiment["experiment_id"]),
        "ablation_result": str(arguments.ablation_result),
        "contract_preflight": contract_preflight,
        "runs": index_rows,
        "passed_runs": int(sum(row["all_gates_passed"] for row in index_rows)),
        "run_count": len(index_rows),
    }
    _write_json(arguments.output_root / "confirmation_index.json", index)
    print(
        f"A2-M2 confirmation complete: {index['passed_runs']}/{index['run_count']} passed "
        f"index={arguments.output_root / 'confirmation_index.json'}"
    )
    return 0 if index["passed_runs"] == index["run_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
