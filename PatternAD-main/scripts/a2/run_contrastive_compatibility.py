#!/usr/bin/env python3
"""Run A2-M2's normal-pair contrastive compatibility-energy model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a2.generate_transition_contract import DEFAULT_CONFIG as DEFAULT_CONTRACT_CONFIG
from scripts.a2.generate_transition_contract import _load_json
from scripts.a2.run_transition_compatibility import run_experiment, write_result
from ts_benchmark.baselines.A2TransitionCompatibility import A2ContrastiveCompatibility


DEFAULT_EXPERIMENT_CONFIG = REPO_ROOT / "config" / "a2" / "contrastive_energy_m2_v1.json"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-config", type=Path, default=DEFAULT_CONTRACT_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--contract-seed", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    arguments = parser.parse_args(argv)
    if arguments.torch_threads < 1:
        raise ValueError("--torch-threads must be positive.")
    torch.set_num_threads(arguments.torch_threads)
    contract_config = _load_json(arguments.contract_config)
    experiment_config = _load_json(arguments.experiment_config)
    if arguments.contract_seed is not None:
        contract_config["seed"] = int(arguments.contract_seed)
    if arguments.seed is not None:
        experiment_config["seed"] = int(arguments.seed)
    summary, checkpoint = run_experiment(
        contract_config, experiment_config, model_factory=A2ContrastiveCompatibility
    )
    write_result(arguments.output_dir, summary, checkpoint)
    print(
        f"A2-M2 complete: gates_passed={summary['all_gates_passed']} "
        f"summary={arguments.output_dir / 'summary.json'}"
    )
    return 0 if summary["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
