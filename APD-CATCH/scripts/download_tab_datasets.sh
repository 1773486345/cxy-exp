#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="${TAB_DATA_ARCHIVE:-/tmp/TAB_dataset.zip}"
DATA_DIR="${ROOT_DIR}/dataset/anomaly_detect"
URL="https://drive.usercontent.google.com/download?id=1V5BAHWBKU8uih3hE1R7WdF6_crZlIbQT&export=download&confirm=t"

if [[ -f "${DATA_DIR}/DETECT_META.csv" && "${FORCE:-0}" != "1" ]]; then
    echo "Dataset metadata already exists at ${DATA_DIR}/DETECT_META.csv"
    echo "Set FORCE=1 to download and extract again."
    exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required." >&2
    exit 1
fi
if ! command -v unzip >/dev/null 2>&1; then
    echo "unzip is required." >&2
    exit 1
fi

mkdir -p "$(dirname "${ARCHIVE}")"
if [[ -f "${ARCHIVE}" ]] && unzip -tq "${ARCHIVE}" >/dev/null 2>&1; then
    echo "Using complete archive ${ARCHIVE}"
else
    echo "Downloading the official TAB dataset archive (about 1.9 GB) to ${ARCHIVE}"
    curl -L --fail --continue-at - \
        --connect-timeout 30 \
        --retry 12 \
        --retry-delay 5 \
        --speed-limit 1024 \
        --speed-time 180 \
        -o "${ARCHIVE}" "${URL}"
fi

echo "Extracting dataset into ${ROOT_DIR}"
unzip -q -o "${ARCHIVE}" -d "${ROOT_DIR}"

if [[ ! -f "${DATA_DIR}/DETECT_META.csv" ]]; then
    echo "Extraction finished but ${DATA_DIR}/DETECT_META.csv is missing." >&2
    exit 1
fi

echo "Dataset is ready at ${ROOT_DIR}/dataset"
