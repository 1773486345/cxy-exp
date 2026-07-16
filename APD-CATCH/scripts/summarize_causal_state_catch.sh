#!/usr/bin/env bash
set -euo pipefail

output_root="${1:-result/causal_state_catch_v2_screen}"
python scripts/analysis/summarize_apd_catch_results.py "${output_root}"
