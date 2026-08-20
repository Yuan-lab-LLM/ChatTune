#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILTIN_ROOT="$SCRIPT_DIR/builtin_datasets"
RAW_SOURCE_DIR="$BUILTIN_ROOT/medical-example"
RAW_DATASET_NAME="medical-example"
EVALUATION_SOURCE_DIR="$BUILTIN_ROOT/evaluation"
TARGET_ROOT="${1:-${MEDFLOW_HOST_WORKSPACE:-/home/workspace}}"
RAW_TARGET_DIR="$TARGET_ROOT/dataset/$RAW_DATASET_NAME"
EVALUATION_TARGET_DIR="$TARGET_ROOT/evaluation"

if [[ ! -d "$RAW_SOURCE_DIR" ]]; then
  echo "Built-in dataset source not found: $RAW_SOURCE_DIR" >&2
  exit 1
fi

mkdir -p "$RAW_TARGET_DIR"
find "$RAW_TARGET_DIR" -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' \) -delete
cp "$RAW_SOURCE_DIR"/*.json "$RAW_TARGET_DIR"/
echo "Installed built-in dataset: $RAW_TARGET_DIR"

if [[ -d "$EVALUATION_SOURCE_DIR" ]]; then
  mkdir -p "$EVALUATION_TARGET_DIR"
  cp -a "$EVALUATION_SOURCE_DIR"/. "$EVALUATION_TARGET_DIR"/
  echo "Installed built-in evaluation data: $EVALUATION_TARGET_DIR"
else
  echo "Warning: built-in evaluation source not found: $EVALUATION_SOURCE_DIR" >&2
fi