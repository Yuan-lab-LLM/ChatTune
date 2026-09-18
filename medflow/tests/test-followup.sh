#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=medical-smoke-lib.sh
source "$SCRIPT_DIR/medical-smoke-lib.sh"

medical_smoke_init
BASE_URL=$(medical_smoke_base_url "${1:-}" "${2:-}" "${MEDFLOW_BASE_URL:-}")

medical_smoke_request \
    "follow-up questionnaire" \
    POST \
    "$BASE_URL/follow_up/questionnaire" \
    "$SCRIPT_DIR/data/followup-generate-questions.json" \
    followup-questionnaire

medical_smoke_request \
    "follow-up chat" \
    POST \
    "$BASE_URL/follow_up/chat" \
    "$SCRIPT_DIR/data/followup-chat-first.json" \
    followup-chat

medical_smoke_request \
    "follow-up report" \
    POST \
    "$BASE_URL/follow_up/report" \
    "$SCRIPT_DIR/data/followup-report.json" \
    followup-report
