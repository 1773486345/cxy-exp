"""Load and validate the frozen A3-N1 confirmation plan."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from scripts.a3.generate_trigger_response_contract import _load_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIRMATION_CONFIG = REPO_ROOT / "config" / "a3" / "background_nulling_n1_confirmation_v1.json"


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repo_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def load_confirmation_plan(
    config_path: Path = DEFAULT_CONFIRMATION_CONFIG,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    plan = _load_json(_repo_path(config_path))
    if int(plan.get("schema_version", 0)) != 1:
        raise ValueError("A3-N1 confirmation plan must use schema_version=1.")
    if str(plan.get("confirmation_id", "")) != "a3_n1_background_nulling_route_graph_confirmation_v1":
        raise ValueError("Unexpected A3-N1 confirmation ID.")
    pairs = plan.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 4:
        raise ValueError("A3-N1 confirmation requires exactly four frozen seed pairs.")
    if int(plan.get("required_complete_passes", 0)) != 4:
        raise ValueError("A3-N1 confirmation requires four complete passes.")
    required_paths = ("contract_config", "background_protocol", "preflight_config", "experiment_config")
    if any(not isinstance(plan.get(name), str) for name in required_paths):
        raise ValueError("A3-N1 confirmation plan has incomplete input paths.")
    contract = _load_json(_repo_path(plan["contract_config"]))
    background = _load_json(_repo_path(plan["background_protocol"]))
    preflight = _load_json(_repo_path(plan["preflight_config"]))
    experiment = _load_json(_repo_path(plan["experiment_config"]))
    if str(experiment.get("experiment_id", "")) != "a3_n1_background_nulling_route_graph_development_v1":
        raise ValueError("A3-N1 confirmation must use the frozen development configuration.")
    if experiment.get("model", {}).get("condition_on_event_pre") is not True:
        raise ValueError("A3-N1 confirmation must retain event-pre conditioning.")
    contract_seeds = set()
    model_seeds = set()
    output_dirs = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("A3-N1 confirmation pair must be an object.")
        try:
            contract_seed = int(pair["contract_seed"])
            model_seed = int(pair["model_seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("A3-N1 confirmation pair has invalid seeds.") from error
        output_dir = pair.get("output_dir")
        if contract_seed <= 0 or model_seed <= 0 or not isinstance(output_dir, str) or not output_dir:
            raise ValueError("A3-N1 confirmation pair has an invalid output directory or seed.")
        contract_seeds.add(contract_seed)
        model_seeds.add(model_seed)
        output_dirs.add(output_dir)
    if len(contract_seeds) != 4 or len(model_seeds) != 4 or len(output_dirs) != 4:
        raise ValueError("A3-N1 confirmation seeds and output directories must be unique.")
    if int(contract["seed"]) in contract_seeds or int(experiment["seed"]) in model_seeds:
        raise ValueError("A3-N1 confirmation may not reuse its development seed pair.")
    return plan, contract, background, preflight, experiment


def prepared_confirmation_pair(
    pair_index: int, config_path: Path = DEFAULT_CONFIRMATION_CONFIG
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Path]:
    plan, contract, background, preflight, experiment = load_confirmation_plan(config_path)
    if pair_index < 0 or pair_index >= len(plan["pairs"]):
        raise ValueError(f"A3-N1 confirmation pair index must be in [0, {len(plan['pairs']) - 1}].")
    pair = plan["pairs"][pair_index]
    contract = copy.deepcopy(contract)
    experiment = copy.deepcopy(experiment)
    contract["seed"] = int(pair["contract_seed"])
    experiment["seed"] = int(pair["model_seed"])
    return plan, contract, background, preflight, experiment, _repo_path(pair["output_dir"])
