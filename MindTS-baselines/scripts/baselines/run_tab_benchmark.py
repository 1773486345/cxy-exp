#!/usr/bin/env python
import importlib.util
import runpy
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[2]
    optimized_vus_path = (
        project_root
        / "ts_benchmark"
        / "evaluation"
        / "metrics"
        / "vus_metrics.py"
    )
    target_script = Path.cwd() / "scripts" / "run_benchmark.py"
    if not target_script.is_file():
        raise FileNotFoundError(f"TAB benchmark entry not found: {target_script}")

    module_name = "ts_benchmark.evaluation.metrics.vus_metrics"
    spec = importlib.util.spec_from_file_location(module_name, optimized_vus_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load optimized VUS module: {optimized_vus_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module

    sys.argv[0] = str(target_script)
    runpy.run_path(str(target_script), run_name="__main__")


if __name__ == "__main__":
    main()
