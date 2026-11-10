"""Publish the validated single-task MNRF benchmark in its native two-file format."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


def publish(result_root: Path, run_rel: Path, keep_work: bool = False) -> Path:
    if (run_rel.is_absolute() or not run_rel.parts
            or ".." in run_rel.parts or ".mnrf-work" in run_rel.parts):
        raise ValueError("Expected a relative benchmark result path outside .mnrf-work.")
    result_root = result_root.resolve()
    work_dir = result_root / ".mnrf-work" / run_rel
    destination = result_root / run_rel
    if work_dir.resolve() != work_dir or destination.resolve() != destination:
        raise ValueError("Benchmark publication does not accept symlinked run paths.")
    if os.path.lexists(destination):
        raise FileExistsError(destination)

    reference = json.loads((work_dir / "report_reference.json").read_text(encoding="utf-8"))
    records = reference.get("record_paths", [])
    if reference.get("run_id") != work_dir.name or len(records) != 1:
        raise ValueError("The validated invocation must reference exactly one native benchmark record.")
    report = Path(reference["report_path"])
    record = Path(records[0])
    if not (report.name.startswith("test_report") and report.suffix == ".csv"
            and record.name.endswith(".tar.gz")):
        raise ValueError("Expected a native test_report CSV and a benchmark record archive.")
    for artifact in (report, record):
        if artifact.parent != work_dir or artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"Benchmark output is missing or outside the work directory: {artifact}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mnrf-publish-", dir=destination.parent) as staging:
        bundle = Path(staging) / destination.name
        bundle.mkdir()
        for artifact in (report, record):
            shutil.copy2(artifact, bundle / artifact.name)
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        bundle.rename(destination)

    if not keep_work:
        shutil.rmtree(work_dir)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_rel", type=Path)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "result"
    print(f"Native benchmark outputs: {publish(root, args.run_rel, args.keep_work)}")
