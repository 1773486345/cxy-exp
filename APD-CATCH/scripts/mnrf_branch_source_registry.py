"""Explicit source registry shared by branch planning, execution and summary."""

import json
import os
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "result/analysis/MNRF-ViewProbe-Ablation/current_sources"
RETRAINED_TASKS = {"PSM", "Creditcard", "CICIDS"}


def load_registry():
    sources = {}
    if REGISTRY_DIR.is_dir():
        for path in REGISTRY_DIR.glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            task = path.stem
            if value.get("schema") != 1 or value.get("task") != task or not isinstance(value.get("source"), dict):
                raise ValueError("MNRF ablation source registration is invalid: {}".format(path))
            sources[task] = value["source"]
    return {"schema": 1, "sources": sources}


def selected_source(task, legacy_run_id):
    """Return the explicitly registered source for rerun tasks, otherwise legacy pin."""
    if task in RETRAINED_TASKS:
        entry = load_registry()["sources"].get(task)
        if not isinstance(entry, dict) or not entry.get("path"):
            return None, None
        path = Path(entry["path"]).resolve()
        if path.name != entry.get("run_id"):
            raise ValueError("registered source path and Run ID disagree for {}".format(task))
        return path, entry
    return (ROOT / "result/score/MNRF-ViewProbe" / task / legacy_run_id).resolve(), None


def write_registry(task, source):
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    path = REGISTRY_DIR / (task + ".json")
    temporary = path.with_name(path.name + ".tmp-{}-{}".format(os.getpid(), uuid.uuid4().hex))
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": 1, "task": task, "source": source},
                                sort_keys=True, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
