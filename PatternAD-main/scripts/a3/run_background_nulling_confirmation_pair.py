#!/usr/bin/env python3
"""Run one pre-registered A3-N1 confirmation pair on its frozen CUDA setup."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a3.n1_confirmation import DEFAULT_CONFIRMATION_CONFIG, prepared_confirmation_pair
from scripts.a3.run_background_nulling_route_graph import run_experiment, write_result


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-index", type=int, required=True)
    parser.add_argument("--confirmation-config", type=Path, default=DEFAULT_CONFIRMATION_CONFIG)
    parser.add_argument("--torch-threads", type=int, default=1)
    arguments = parser.parse_args(argv)
    if arguments.torch_threads < 1:
        raise ValueError("--torch-threads must be positive.")
    torch.set_num_threads(arguments.torch_threads)
    plan, contract, background, preflight, experiment, output_dir = prepared_confirmation_pair(
        arguments.pair_index, arguments.confirmation_config
    )
    summary, checkpoint, factor = run_experiment(contract, background, preflight, experiment)
    write_result(output_dir, summary, checkpoint, factor)
    pair = plan["pairs"][arguments.pair_index]
    print(
        "A3-N1 confirmation complete: "
        f"pair={arguments.pair_index} contract_seed={pair['contract_seed']} model_seed={pair['model_seed']} "
        f"gates_passed={summary['all_gates_passed']} summary={output_dir / 'summary.json'}"
    )
    return 0 if summary["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
