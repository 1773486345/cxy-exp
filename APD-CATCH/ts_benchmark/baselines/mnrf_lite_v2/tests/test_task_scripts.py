import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
INTERNAL = {
    *(f"ASD_dataset_{index}" for index in range(1, 13)),
    "CICIDS", "CalIt2", "Creditcard", "GECCO", "Genesis", "MSL", "NYC",
    "PSM", "SMAP", "SMD", "SWAT",
}
EXTERNAL = {
    "BATADAL", "HAI20_07", "MTSB_METRO",
    *(f"MTSB_OPPORTUNITY_{index:02d}" for index in range(1, 14)),
    "MTSB_SWAN_SF", "MetroPT3",
}


def test_task_wrappers_are_explicit_executable_and_valid_bash():
    base = ROOT / "scripts/multivariate_detection/detect_score"
    scripts = sorted(base.glob("*_script/MNRFLiteV2.sh"))
    assert len(INTERNAL) == 23
    assert len(EXTERNAL) == 18
    assert len(scripts) == 41
    for path in scripts:
        task = path.parent.name[:-7] if path.parent.name.endswith("_script") else path.parent.name
        source = "large_detect" if task in INTERNAL else "external_detect"
        assert task in INTERNAL | EXTERNAL
        assert path.stat().st_mode & 0o111
        subprocess.run(["bash", "-n", str(path)], check=True)
        text = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in text
        assert f"{task} {source}" in text
        assert "run_mnrf_lite_v2_task.sh" in text
    assert "MTSB_OCCUPANCY_01" not in INTERNAL | EXTERNAL
    assert "MTSB_OCCUPANCY_02" not in INTERNAL | EXTERNAL
