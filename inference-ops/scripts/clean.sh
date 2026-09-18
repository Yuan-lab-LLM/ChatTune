#!/usr/bin/env bash
set -euo pipefail

PID_DIR="${1:-}"
if [[ -z "$PID_DIR" || ! -d "$PID_DIR" ]]; then
    echo "Usage: clean.sh PID_DIR" >&2
    echo "Refusing to kill processes by port without an instance PID directory." >&2
    exit 2
fi

for pid_file in "$PID_DIR"/*.pid; do
    [[ -f "$pid_file" ]] || continue
    pid="$(tr -dc '0-9' < "$pid_file")"
    run_id="$(basename "$(dirname "$PID_DIR")")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && \
       tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -Fxq "SERVICE_RUN_ID=$run_id"; then
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
done
