#!/usr/bin/env bash
# Run exactly one official CATCH task with an isolated, traceable result directory.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MANIFEST="${SCRIPT_DIR}/official_catch_manifest.tsv"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_official_catch.sh <physical-task> <score|label> [--run]

Without --run this performs a non-training preflight: manifest lookup, script parsing,
data-file existence, and planned output path.  --run is required to launch CATCH.

Examples:
  bash scripts/run_official_catch.sh Genesis score
  bash scripts/run_official_catch.sh Genesis score --run
  bash scripts/run_official_catch.sh ASD_dataset_1 label --run
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage >&2
    exit 2
fi

TASK=$1
PROTOCOL=$2
RUN=false
if [[ $# -eq 3 ]]; then
    [[ $3 == "--run" ]] || { usage >&2; exit 2; }
    RUN=true
fi
if [[ $PROTOCOL != "score" && $PROTOCOL != "label" ]]; then
    printf 'Protocol must be score or label, got: %s\n' "$PROTOCOL" >&2
    exit 2
fi

match=$(awk -F '\t' -v task="$TASK" '$1 == task { print; found = 1 } END { exit !found }' "$MANIFEST") || {
    printf 'Unknown physical task: %s\n' "$TASK" >&2
    printf 'Use a physical task from %s (for example Genesis or ASD_dataset_1).\n' "$MANIFEST" >&2
    exit 2
}
IFS=$'\t' read -r physical_task paper_dataset dataset_file kind script_dir <<< "$match"
script_dir=${script_dir# }

catch_script="${REPO_ROOT}/scripts/multivariate_detection/detect_${PROTOCOL}/${script_dir}/CATCH.sh"
data_file="${REPO_ROOT}/dataset/anomaly_detect/data/${dataset_file}"
config_file="${REPO_ROOT}/config/unfixed_detect_${PROTOCOL}_multi_config.json"
default_save_path="${PROTOCOL}/CATCH/${physical_task}"

[[ -f $catch_script ]] || { printf 'Missing official script: %s\n' "$catch_script" >&2; exit 1; }
[[ -f $data_file ]] || { printf 'Missing official data file: %s\n' "$data_file" >&2; exit 1; }
[[ -f $config_file ]] || { printf 'Missing official config: %s\n' "$config_file" >&2; exit 1; }

script_dataset=$(sed -n 's/.*--data-name-list "\([^"]*\)".*/\1/p' "$catch_script")
script_config=$(sed -n 's/.*--config-path "\([^"]*\)".*/\1/p' "$catch_script")
if [[ $script_dataset != "$dataset_file" ]]; then
    printf 'Manifest/script data mismatch: %s != %s\n' "$dataset_file" "$script_dataset" >&2
    exit 1
fi
if [[ $script_config != "unfixed_detect_${PROTOCOL}_multi_config.json" ]]; then
    printf 'Manifest/script protocol mismatch: %s\n' "$script_config" >&2
    exit 1
fi

if [[ $RUN == false ]]; then
    printf 'PRECHECK OK\n'
    printf 'physical_task=%s\n' "$physical_task"
    printf 'paper_dataset=%s\n' "$paper_dataset"
    printf 'protocol=%s\n' "$PROTOCOL"
    printf 'data_file=%s\n' "$(readlink -f "$data_file")"
    printf 'official_script=%s\n' "$catch_script"
    printf 'planned_result_root=%s\n' "${REPO_ROOT}/result/${default_save_path}/run-<utc-timestamp>-<pid>-<random>"
    printf 'Training was not started. Add --run to execute this task.\n'
    exit 0
fi

run_id="run-$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}"
save_path="${default_save_path}/${run_id}"
run_dir="${REPO_ROOT}/result/${save_path}"
mkdir -p "$run_dir"

cp "$catch_script" "${run_dir}/official_CATCH.sh"
sha256sum "$catch_script" > "${run_dir}/official_CATCH.sh.sha256"

cat > "${run_dir}/command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(printf '%q' "$REPO_ROOT")
export CATCH_SAVE_PATH=$(printf '%q' "$save_path")
exec bash $(printf '%q' "${catch_script#"${REPO_ROOT}/"}")
EOF
chmod +x "${run_dir}/command.sh"

{
    printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'physical_task=%s\n' "$physical_task"
    printf 'paper_dataset=%s\n' "$paper_dataset"
    printf 'kind=%s\n' "$kind"
    printf 'protocol=%s\n' "$PROTOCOL"
    printf 'dataset_file=%s\n' "$dataset_file"
    printf 'dataset_path=%s\n' "$(readlink -f "$data_file")"
    printf 'dataset_sha256=%s\n' "$(sha256sum "$data_file" | awk '{print $1}')"
    printf 'metadata_path=%s\n' "$(readlink -f "${REPO_ROOT}/dataset/anomaly_detect/DETECT_META.csv")"
    printf 'metadata_sha256=%s\n' "$(sha256sum "${REPO_ROOT}/dataset/anomaly_detect/DETECT_META.csv" | awk '{print $1}')"
    printf 'official_script=%s\n' "$catch_script"
    printf 'official_script_sha256=%s\n' "$(sha256sum "$catch_script" | awk '{print $1}')"
    printf 'official_config=%s\n' "$config_file"
    printf 'official_config_sha256=%s\n' "$(sha256sum "$config_file" | awk '{print $1}')"
    printf 'save_path_relative_to_result=%s\n' "$save_path"
    printf 'run_directory=%s\n' "$run_dir"
    printf 'git_toplevel=%s\n' "$(git -C "$REPO_ROOT" rev-parse --show-toplevel)"
    printf 'git_prefix=%s\n' "$(git -C "$REPO_ROOT" rev-parse --show-prefix)"
    printf 'catch_master_commit=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
} > "${run_dir}/metadata.txt"

{
    printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'uname=%s\n' "$(uname -a)"
    printf 'python=%s\n' "$(command -v python || true)"
    python --version || true
    python -m pip freeze || true
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi || true
    fi
} > "${run_dir}/environment.txt" 2>&1

{
    printf 'run_directory=%s\n' "$run_dir"
    printf 'save_path=%s\n' "$save_path"
    printf 'command_file=%s\n' "${run_dir}/command.sh"
    bash "${run_dir}/command.sh"
} 2>&1 | tee "${run_dir}/console.log"
