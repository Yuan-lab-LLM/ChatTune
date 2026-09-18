#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=medical-smoke-lib.sh
source "$SCRIPT_DIR/medical-smoke-lib.sh"

medical_smoke_init
BASE_URL=$(medical_smoke_base_url "${1:-}" "${2:-}" "${MEDFLOW_BASE_URL:-}")

medical_smoke_request \
    "quality inspection" \
    POST \
    "$BASE_URL/quality_inspect" \
    "$SCRIPT_DIR/data/quality_inspect_post_input.json" \
    quality

medical_smoke_request \
    "quality inspection with named configuration" \
    POST \
    "$BASE_URL/quality_inspect" \
    "$SCRIPT_DIR/data/quality_inspect_post_input_with_config_name.json" \
    quality

medical_smoke_request \
    "quality modification" \
    POST \
    "$BASE_URL/quality_modify" \
    "$SCRIPT_DIR/data/quality_modify_post_input.json" \
    quality

medical_smoke_request \
    "quality modification with conversation" \
    POST \
    "$BASE_URL/quality_modify" \
    "$SCRIPT_DIR/data/quality_modify_post_input_second.json" \
    quality

medical_smoke_request \
    "quality modification with multiple findings" \
    POST \
    "$BASE_URL/quality_modify" \
    "$SCRIPT_DIR/data/quality_modify_multi_post_input.json" \
    quality
