#!/usr/bin/env bash
# Fill only missing original-CATCH reports without changing its commands or seed.
set -euo pipefail

original_root="${1:-../CATCH-master}"
mode="${2:-all}"
dataset_spec="${3:-Creditcard GECCO Genesis MSL NYC PSM SMD SMAP}"
read -r -a datasets <<< "${dataset_spec}"

if [[ ! -d "${original_root}" ]]; then
  echo "missing original CATCH checkout: ${original_root}" >&2
  exit 1
fi
if [[ "${mode}" != "score" && "${mode}" != "label" && "${mode}" != "all" ]]; then
  echo "mode must be score, label, or all" >&2
  exit 1
fi

for protocol in score label; do
  if [[ "${mode}" != "all" && "${mode}" != "${protocol}" ]]; then
    continue
  fi
  for dataset in "${datasets[@]}"; do
    report_root="${original_root}/result/${protocol}/CATCH/${dataset}"
    if find "${report_root}" -type f -name 'test_report.*.csv' -print -quit 2>/dev/null | grep -q .; then
      echo "skip existing original CATCH ${protocol} ${dataset}"
      continue
    fi
    script="scripts/multivariate_detection/detect_${protocol}/${dataset}_script/CATCH.sh"
    if [[ ! -f "${original_root}/${script}" ]]; then
      echo "missing original CATCH command: ${original_root}/${script}" >&2
      exit 1
    fi
    echo "run original CATCH ${protocol} ${dataset}"
    (
      cd "${original_root}"
      bash "${script}"
    )
  done
done
