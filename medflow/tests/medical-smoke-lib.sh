#!/usr/bin/env bash

set -euo pipefail

MEDICAL_SMOKE_TEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MEDICAL_SMOKE_REQUEST_INDEX=0

medical_smoke_init() {
    for command_name in curl python3 mktemp; do
        if ! command -v "$command_name" >/dev/null 2>&1; then
            echo "ERROR: required command not found: $command_name" >&2
            exit 1
        fi
    done

    MEDICAL_SMOKE_TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/medflow-smoke.XXXXXX")
    trap 'rm -rf -- "$MEDICAL_SMOKE_TMP_DIR"' EXIT
}

medical_smoke_base_url() {
    local host=${1:-}
    local port=${2:-}
    local configured_url=${3:-}

    if [[ -n "$configured_url" ]]; then
        printf '%s\n' "${configured_url%/}"
        return
    fi
    if [[ -z "$host" || -z "$port" ]]; then
        echo "ERROR: provide <host> <port> or configure a base URL environment variable" >&2
        return 1
    fi
    printf 'http://%s:%s\n' "$host" "$port"
}

medical_smoke_request() {
    local label=$1
    local method=$2
    local url=$3
    local fixture=$4
    local validator=$5
    local response_file
    local status

    if [[ ! -f "$fixture" ]]; then
        echo "ERROR: fixture not found for $label: $fixture" >&2
        return 1
    fi

    MEDICAL_SMOKE_REQUEST_INDEX=$((MEDICAL_SMOKE_REQUEST_INDEX + 1))
    response_file="$MEDICAL_SMOKE_TMP_DIR/response-${MEDICAL_SMOKE_REQUEST_INDEX}.body"

    echo "[RUN] $label"
    if ! status=$(curl --silent --show-error \
        --connect-timeout "${MEDFLOW_SMOKE_CONNECT_TIMEOUT:-10}" \
        --max-time "${MEDFLOW_SMOKE_REQUEST_TIMEOUT:-600}" \
        --output "$response_file" \
        --write-out '%{http_code}' \
        --request "$method" \
        --header 'accept: application/json' \
        --header 'Content-Type: application/json' \
        --data-binary "@$fixture" \
        "$url"); then
        echo "ERROR: request failed for $label" >&2
        return 1
    fi

    if [[ ! "$status" =~ ^2[0-9][0-9]$ ]]; then
        echo "ERROR: $label returned HTTP $status" >&2
        head -c 2000 "$response_file" >&2 || true
        echo >&2
        return 1
    fi

    if ! python3 "$MEDICAL_SMOKE_TEST_DIR/validate_medical_smoke_response.py" \
        "$validator" "$response_file"; then
        echo "ERROR: response validation failed for $label" >&2
        head -c 2000 "$response_file" >&2 || true
        echo >&2
        return 1
    fi

    echo "[PASS] $label (HTTP $status)"
}
