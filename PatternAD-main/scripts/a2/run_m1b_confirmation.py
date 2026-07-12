#!/usr/bin/env python3
"""Run A2-M1b's frozen confirmation seeds sequentially on one process."""

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

from scripts.a2.generate_transition_contract import DEFAULT_CONFIG as DEFAULT_CONTRACT_CONFIG
from scripts.a2.generate_transition_contract import _load_json
from scripts.a2.run_transition_compatibility import (
    run_experiment,
    write_result,
)


DEFAULT_CONFIRMATION_CONFIG = REPO_ROOT / "config" / "a2" / "m1b_confirmation_v1.json"
DEFAULT_M1B_CONFIG = REPO_ROOT / "config" / "a2" / "trajectory_gru_m1b_alpha05.json"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _seed_pairs(config: Mapping[str, Any]) -> List[tuple[int, int]]:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("A2 confirmation config must use schema_version=1.")
    contract_seeds = [int(seed) for seed in config["contract_seeds"]]
    model_seeds = [int(seed) for seed in config["model_seeds"]]
    if len(contract_seeds) != len(model_seeds) or not contract_seeds:
        raise ValueError("A2 confirmation requires equally sized, non-empty seed lists.")
    if len(set(contract_seeds)) != len(contract_seeds):
        raise ValueError("A2 confirmation contract seeds must be unique.")
    if len(set(model_seeds)) != len(model_seeds):
        raise ValueError("A2 confirmation model seeds must be unique.")
    return list(zip(contract_seeds, model_seeds))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-config", type=Path, default=DEFAULT_CONTRACT_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_M1B_CONFIG)
    parser.add_argument("--confirmation-config", type=Path, default=DEFAULT_CONFIRMATION_CONFIG)
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
    arguments.output_root.mkdir(parents=True, exist_ok=False)
    index_rows = []
    for contract_seed, model_seed in seed_pairs:
        contract = dict(base_contract, seed=contract_seed)
        experiment = dict(base_experiment, seed=model_seed)
        summary, checkpoint = run_experiment(contract, experiment)
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
        "runs": index_rows,
        "passed_runs": int(sum(row["all_gates_passed"] for row in index_rows)),
        "run_count": len(index_rows),
    }
    _write_json(arguments.output_root / "confirmation_index.json", index)
    print(
        f"A2-M1b confirmation complete: {index['passed_runs']}/{index['run_count']} passed "
        f"index={arguments.output_root / 'confirmation_index.json'}"
    )
    return 0 if index["passed_runs"] == index["run_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
