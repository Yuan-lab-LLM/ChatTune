#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=medical-smoke-lib.sh
source "$SCRIPT_DIR/medical-smoke-lib.sh"

medical_smoke_init
BASE_URL=$(medical_smoke_base_url "${1:-}" "${2:-}" "${CASE2CHAT_BASE_URL:-}")

medical_smoke_request \
    "case2chat compatibility" \
    GET \
    "$BASE_URL/train/123456?item_type=case2chat" \
    "$SCRIPT_DIR/data/case2chat.json" \
    case2chat
