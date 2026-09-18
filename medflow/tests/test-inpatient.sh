#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=medical-smoke-lib.sh
source "$SCRIPT_DIR/medical-smoke-lib.sh"

medical_smoke_init
BASE_URL=$(medical_smoke_base_url "${1:-}" "${2:-}" "${MEDFLOW_BASE_URL:-}")
URL="$BASE_URL/inference?request_type=inpatient&scheme=admission_record"

medical_smoke_request \
    "inpatient admission record generation" \
    POST \
    "$URL" \
    "$SCRIPT_DIR/data/inpatient-admission_record.json" \
    inpatient

medical_smoke_request \
    "inpatient admission record modification" \
    POST \
    "$URL" \
    "$SCRIPT_DIR/data/inpatient-admission_record-modify.json" \
    inpatient
