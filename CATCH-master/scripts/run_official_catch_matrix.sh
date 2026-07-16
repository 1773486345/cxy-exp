#!/usr/bin/env bash
# Sequentially preflight or run the complete official CATCH matrix in manifest order.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MANIFEST="${SCRIPT_DIR}/official_catch_manifest.tsv"
RUNNER="${SCRIPT_DIR}/run_official_catch.sh"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_official_catch_matrix.sh [score|label|all] [--run]

Without --run the complete selected matrix is only preflighted.  The default protocol
selection is all, which checks all 70 official protocol-task combinations.  --run is
required before any training command is launched.
EOF
}

protocols=all
run_args=()
for arg in "$@"; do
    case "$arg" in
        score|label|all)
            [[ $protocols == all ]] || { usage >&2; exit 2; }
            protocols=$arg
            ;;
        --run)
            [[ ${#run_args[@]} -eq 0 ]] || { usage >&2; exit 2; }
            run_args=(--run)
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

if [[ $protocols == all ]]; then
    selected_protocols=(score label)
else
    selected_protocols=($protocols)
fi

count=0
while IFS=$'\t' read -r task _; do
    [[ -n ${task:-} && ${task:0:1} != '#' ]] || continue
    for protocol in "${selected_protocols[@]}"; do
        printf '[%d] %s %s\n' "$((count + 1))" "$task" "$protocol"
        bash "$RUNNER" "$task" "$protocol" "${run_args[@]}"
        count=$((count + 1))
    done
done < "$MANIFEST"

if [[ ${#run_args[@]} -eq 1 ]]; then
    mode=run
else
    mode=preflight
fi
printf 'Completed %d %s task(s).\n' "$count" "$mode"
