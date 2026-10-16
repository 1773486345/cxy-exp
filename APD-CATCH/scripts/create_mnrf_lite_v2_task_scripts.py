#!/usr/bin/env python
"""Generate one explicit MNRF-Lite V2 command wrapper per supported task."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERNAL = (
    "ASD_dataset_1", "ASD_dataset_2", "ASD_dataset_3", "ASD_dataset_4",
    "ASD_dataset_5", "ASD_dataset_6", "ASD_dataset_7", "ASD_dataset_8",
    "ASD_dataset_9", "ASD_dataset_10", "ASD_dataset_11", "ASD_dataset_12",
    "CICIDS", "CalIt2", "Creditcard", "GECCO", "Genesis", "MSL", "NYC",
    "PSM", "SMAP", "SMD", "SWAT",
)
EXTERNAL = (
    "BATADAL", "HAI20_07", "MTSB_METRO", "MTSB_OPPORTUNITY_01",
    "MTSB_OPPORTUNITY_02", "MTSB_OPPORTUNITY_03", "MTSB_OPPORTUNITY_04",
    "MTSB_OPPORTUNITY_05", "MTSB_OPPORTUNITY_06", "MTSB_OPPORTUNITY_07",
    "MTSB_OPPORTUNITY_08", "MTSB_OPPORTUNITY_09", "MTSB_OPPORTUNITY_10",
    "MTSB_OPPORTUNITY_11", "MTSB_OPPORTUNITY_12", "MTSB_OPPORTUNITY_13",
    "MTSB_SWAN_SF", "MetroPT3",
)


def main():
    for task, source in [(name, "large_detect") for name in INTERNAL] + [
        (name, "external_detect") for name in EXTERNAL
    ]:
        path = ROOT / "scripts/multivariate_detection/detect_score" / f"{task}_script" / "MNRFLiteV2.sh"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite an existing task script: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "SCRIPT_DIR=\"$(cd -- \"$(dirname -- \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
            "ROOT=\"$(cd -- \"${SCRIPT_DIR}/../../../..\" && pwd)\"\n"
            f"exec bash \"${{ROOT}}/scripts/run_mnrf_lite_v2_task.sh\" {task} {source}\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
    print(f"Generated {len(INTERNAL) + len(EXTERNAL)} task scripts.")


if __name__ == "__main__":
    main()
