#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOST=${1:-${MEDFLOW_SMOKE_HOST:-}}
PORT=${2:-${MEDFLOW_SMOKE_PORT:-}}
CASE2CHAT_HOST=${3:-${CASE2CHAT_SMOKE_HOST:-}}
CASE2CHAT_PORT=${4:-${CASE2CHAT_SMOKE_PORT:-}}

if [[ -z "${MEDFLOW_BASE_URL:-}" && ( -z "$HOST" || -z "$PORT" ) ]]; then
    echo "Usage: $0 <medical-host> <medical-port> [case2chat-host case2chat-port]" >&2
    echo "Or set MEDFLOW_BASE_URL. Set MEDFLOW_SMOKE_INCLUDE_CASE2CHAT=1 to include Case2Chat." >&2
    exit 1
fi

for test_script in \
    test-basicmedicalrecord.sh \
    test-diagnosis.sh \
    test-inpatient.sh \
    test-followup.sh \
    test-quality.sh; do
    bash "$SCRIPT_DIR/$test_script" "$HOST" "$PORT"
done

if [[ "${MEDFLOW_SMOKE_INCLUDE_CASE2CHAT:-0}" == "1" ]]; then
    if [[ -z "${CASE2CHAT_BASE_URL:-}" && ( -z "$CASE2CHAT_HOST" || -z "$CASE2CHAT_PORT" ) ]]; then
        echo "ERROR: Case2Chat was requested; provide its host/port or CASE2CHAT_BASE_URL" >&2
        exit 1
    fi
    bash "$SCRIPT_DIR/test-case2chat.sh" "$CASE2CHAT_HOST" "$CASE2CHAT_PORT"
else
    echo "[SKIP] case2chat compatibility (set MEDFLOW_SMOKE_INCLUDE_CASE2CHAT=1 to run)"
fi

echo "All requested MedFlow medical smoke tests passed."
