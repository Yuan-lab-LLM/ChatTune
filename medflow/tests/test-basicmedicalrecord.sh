#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=medical-smoke-lib.sh
source "$SCRIPT_DIR/medical-smoke-lib.sh"

medical_smoke_init
BASE_URL=$(medical_smoke_base_url "${1:-}" "${2:-}" "${MEDFLOW_BASE_URL:-}")

medical_smoke_request \
    "basic medical record (v2)" \
    POST \
    "$BASE_URL/inference?request_type=v2" \
    "$SCRIPT_DIR/data/basicmedicalrecord.json" \
    basic-medical-record
