#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ACTION="${1:-}"
CONFIG_PROFILE="${2:-service}"
START_TIMEOUT_SECONDS="${INFERENCE_OPS_START_TIMEOUT_SECONDS:-900}"
STARTUP_ERROR=""
VLLM_EXIT_CODE=""

if [[ -z "$ACTION" ]]; then
    echo "Usage: bash start-service.sh start|stop [service|default|CONFIG_FILE]" >&2
    exit 1
fi

case "$CONFIG_PROFILE" in
    service|current|"") CONFIG_FILE="$OPS_ROOT/config/service.yaml" ;;
    default) CONFIG_FILE="$OPS_ROOT/config/service.default.yaml" ;;
    *) CONFIG_FILE="$CONFIG_PROFILE" ;;
esac
CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

for command_name in yq; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "Required command is not installed: $command_name" >&2
        exit 1
    }
done

PROVIDER="$(yq -r '.PROVIDER // "vllm"' "$CONFIG_FILE")"
[[ "$PROVIDER" == "vllm" || "$PROVIDER" == "medflow" ]] || {
    echo "Unsupported inference runtime provider: $PROVIDER" >&2
    exit 1
}
PROVIDER="vllm"

VLLM_OPENAI_PORT="$(yq -r '.PORTS.VLLM_OPENAI_PORT' "$CONFIG_FILE")"
HOST_IP="$(yq -r '.ENV.HOST_IP' "$CONFIG_FILE")"
MODEL_NAME="$(yq -r '.ENV.MODEL_NAME' "$CONFIG_FILE")"
MODEL_PATH="$(yq -r '.ENV.MODEL_PATH' "$CONFIG_FILE")${MODEL_NAME}"
CUDA_VISIBLE_DEVICES="$(yq -r '.ENV.CUDA_VISIBLE_DEVICES' "$CONFIG_FILE")"
MASTER_PORT="$(yq -r '.ENV.MASTER_PORT // 50121' "$CONFIG_FILE")"
TENSOR_PARALLEL_SIZE="$(yq -r '.RUNTIME.TENSOR_PARALLEL_SIZE' "$CONFIG_FILE")"
GPU_MEMORY_UTILIZATION="$(yq -r '.RUNTIME.GPU_MEMORY_UTILIZATION' "$CONFIG_FILE")"

if [[ -n "${SERVICE_RUN_ID:-}" ]]; then
    RUN_ID="$SERVICE_RUN_ID"
elif [[ "$(basename "$CONFIG_FILE")" == "service.runtime.yaml" ]]; then
    RUN_ID="$(basename "$(dirname "$CONFIG_FILE")")"
else
    RUN_ID="$(date +"%Y%m%d_%H%M%S")_$$"
fi
if [[ -n "${SERVICE_RUN_LOG_DIR:-}" ]]; then
    RUN_LOG_DIR="$SERVICE_RUN_LOG_DIR"
elif [[ "$(basename "$CONFIG_FILE")" == "service.runtime.yaml" ]]; then
    RUN_LOG_DIR="$(dirname "$CONFIG_FILE")"
else
    RUN_LOG_DIR="$(dirname "$CONFIG_FILE")/../runtime/services/runs/$RUN_ID"
fi
PID_DIR="${SERVICE_PID_DIR:-$(dirname "$CONFIG_FILE")/pids}"
STATUS_FILE="$RUN_LOG_DIR/status.json"
LOG_FILE="$RUN_LOG_DIR/start-service.log"
STARTED_AT="$(date +"%Y-%m-%d %H:%M:%S")"

write_status() {
    local status="$1"
    local error_json="${2:-null}"
    mkdir -p "$RUN_LOG_DIR"
    cat > "$STATUS_FILE" <<EOF
{
  "schema_version": 2,
  "provider": "$PROVIDER",
  "run_id": "$RUN_ID",
  "status": "$status",
  "script_pid": $$,
  "config_profile": "$CONFIG_PROFILE",
  "config_file": "$CONFIG_FILE",
  "log_dir": "$RUN_LOG_DIR",
  "started_at": "$STARTED_AT",
  "finished_at": $(if [[ "$status" == "starting" ]]; then echo null; else printf '"%s"' "$(date +"%Y-%m-%d %H:%M:%S")"; fi),
  "ports": {"vllm": $VLLM_OPENAI_PORT, "master": $MASTER_PORT},
  "error": $error_json
}
EOF
}

wait_for_port() {
    local port="$1" name="$2" process_name="${3:-}" deadline=$((SECONDS + START_TIMEOUT_SECONDS))
    while ! lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; do
        if [[ -n "$process_name" ]]; then
            local process_pid=""
            [[ -f "$PID_DIR/$process_name.pid" ]] && process_pid="$(tr -dc '0-9' < "$PID_DIR/$process_name.pid")"
            if [[ -z "$process_pid" ]] || ! process_is_running "$process_pid"; then
                local process_label="$process_name"
                [[ "$process_name" == "vllm" ]] && process_label="vLLM"
                if [[ -n "$process_pid" ]]; then
                    if wait "$process_pid" 2>/dev/null; then
                        VLLM_EXIT_CODE=0
                    else
                        VLLM_EXIT_CODE=$?
                    fi
                fi
                STARTUP_ERROR="$process_label exited before $name became ready (exit code ${VLLM_EXIT_CODE:-unknown}); see $process_name.log"
                echo "$STARTUP_ERROR" >&2
                return 1
            fi
        fi
        if (( SECONDS >= deadline )); then
            STARTUP_ERROR="Timed out waiting for $name on port $port"
            echo "Timed out waiting for $name on port $port" >&2
            return 1
        fi
        sleep 1
    done
}

wait_for_openai_api() {
    local host="$HOST_IP"
    local deadline=$((SECONDS + START_TIMEOUT_SECONDS))
    [[ "$host" == "0.0.0.0" || "$host" == "::" || -z "$host" ]] && host="127.0.0.1"
    local url="http://$host:$VLLM_OPENAI_PORT/v1/models"
    while ! curl --noproxy '*' --fail --silent --show-error --max-time 2 \
        -H "Authorization: Bearer EMPTY" "$url" >/dev/null 2>&1; do
        if [[ -f "$RUN_LOG_DIR/vllm.log" ]] && \
            tail -n 120 "$RUN_LOG_DIR/vllm.log" | \
            grep -Fq "AttributeError: '_IncludedRouter' object has no attribute 'path'"; then
            STARTUP_ERROR="vLLM OpenAI API failed: prometheus_fastapi_instrumentator cannot handle FastAPI _IncludedRouter; check that the bundled vLLM route compatibility patch loaded; see vllm.log"
            echo "$STARTUP_ERROR" >&2
            return 1
        fi
        local vllm_pid=""
        [[ -f "$PID_DIR/vllm.pid" ]] && vllm_pid="$(tr -dc '0-9' < "$PID_DIR/vllm.pid")"
        if [[ -z "$vllm_pid" ]] || ! process_is_running "$vllm_pid"; then
            if [[ -n "$vllm_pid" ]]; then
                if wait "$vllm_pid" 2>/dev/null; then
                    VLLM_EXIT_CODE=0
                else
                    VLLM_EXIT_CODE=$?
                fi
            fi
            STARTUP_ERROR="vLLM exited before its OpenAI API became ready (exit code ${VLLM_EXIT_CODE:-unknown}); see vllm.log"
            echo "vLLM exited before its OpenAI API became ready; see $RUN_LOG_DIR/vllm.log" >&2
            return 1
        fi
        if (( SECONDS >= deadline )); then
            STARTUP_ERROR="Timed out waiting for vLLM OpenAI API readiness at $url"
            echo "Timed out waiting for vLLM OpenAI API readiness at $url" >&2
            return 1
        fi
        sleep 1
    done
}

process_is_running() {
    local pid="$1"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null || return 1
    if [[ -r "/proc/$pid/stat" ]]; then
        local process_stat process_state
        process_stat="$(<"/proc/$pid/stat")"
        [[ -n "$process_stat" ]] || return 1
        process_state="${process_stat##*) }"
        process_state="${process_state%% *}"
        [[ "$process_state" != "Z" && "$process_state" != "X" ]] || return 1
    fi
    return 0
}

ensure_port_free() {
    local port="$1"
    if lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Refusing to start: port $port is already in use" >&2
        exit 1
    fi
}

start_process() {
    local name="$1" cwd="$2" log_file="$3"
    shift 3
    (
        cd "$cwd"
        exec nohup setsid "$@"
    ) > "$log_file" 2>&1 &
    echo $! > "$PID_DIR/$name.pid"
}

stop_recorded_processes() {
    local names=(start-service web case2chat inference vllm)
    for name in "${names[@]}"; do
        local pid_file="$PID_DIR/$name.pid"
        [[ -f "$pid_file" ]] || continue
        local pid
        pid="$(tr -dc '0-9' < "$pid_file")"
        if [[ -n "$pid" && "$pid" != "$$" ]] && kill -0 "$pid" 2>/dev/null && \
           tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -Fxq "SERVICE_RUN_ID=$RUN_ID"; then
            kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    for name in "${names[@]}"; do
        local pid_file="$PID_DIR/$name.pid"
        [[ -f "$pid_file" ]] || continue
        local pid
        pid="$(tr -dc '0-9' < "$pid_file")"
        if [[ -n "$pid" && "$pid" != "$$" ]] && kill -0 "$pid" 2>/dev/null && \
           tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -Fxq "SERVICE_RUN_ID=$RUN_ID"; then
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
    done
}

if [[ "$ACTION" == "stop" ]]; then
    stop_recorded_processes
    write_status "stopped" "null"
    exit 0
fi

if [[ "$ACTION" != "start" ]]; then
    echo "Unknown action: $ACTION" >&2
    exit 1
fi

mkdir -p "$RUN_LOG_DIR" "$PID_DIR"
echo $$ > "$PID_DIR/start-service.pid"
write_status "starting" "null"
trap 'code=$?; if [[ $code -ne 0 ]]; then stop_recorded_processes; failure_message="${STARTUP_ERROR:-startup failed with exit code $code}"; write_status "failed" "\"$failure_message\""; fi' EXIT

for command_name in curl lsof setsid vllm; do
    command -v "$command_name" >/dev/null 2>&1 || {
        STARTUP_ERROR="Required command is not installed: $command_name"
        echo "$STARTUP_ERROR" >&2
        exit 1
    }
done

for port in "$VLLM_OPENAI_PORT" "$MASTER_PORT"; do
    ensure_port_free "$port"
done

{
    echo "provider=$PROVIDER"
    echo "CONFIG_FILE=$CONFIG_FILE"
    echo "RUN_ID=$RUN_ID"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "MODEL_PATH=$MODEL_PATH"
} >> "$LOG_FILE"

start_process vllm "$OPS_ROOT" "$RUN_LOG_DIR/vllm.log" env \
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "VLLM_PORT=$MASTER_PORT" "MASTER_PORT=$MASTER_PORT" \
    "PYTHONPATH=$SCRIPT_DIR/vllm_compat${PYTHONPATH:+:$PYTHONPATH}" \
    "INFERENCE_OPS_VLLM_ROUTE_COMPAT=1" \
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL_NAME" --host "$HOST_IP" \
    --port "$VLLM_OPENAI_PORT" --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --enable-auto-tool-choice --tool-call-parser hermes
wait_for_port "$VLLM_OPENAI_PORT" "vLLM OpenAI API" "vllm"
wait_for_openai_api

write_status "finished" "null"
trap - EXIT
