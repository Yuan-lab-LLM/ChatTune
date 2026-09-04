#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${MEDFLOW_RUNTIME_ENV_FILE:-$ROOT_DIR/runtime.env}"
STATE_DIR="$ROOT_DIR/.runtime"
LOG_DIR="$STATE_DIR/logs"
DATA_DIR="$STATE_DIR/data"
ACTION="start"
ROLE_OVERRIDE=""
FOREGROUND=false
SKIP_STUDIO=false
SKIP_BRIDGE=false
LOG_SERVICE=""
INIT_PROFILE=""
INIT_NODES=""
INIT_NODE_ID=""
INIT_CENTER_URL=""
INIT_AGENT_URL=""
INIT_IPHOST=""
INIT_FORCE=false

usage() {
  cat <<'EOF'
Usage: bash runtime.sh [options] {init|start|stop|restart|status|logs|check|doctor}

Options:
  --env-file PATH   Load a specific runtime env file.
  --role ROLE       Override RUNTIME_ROLE for this invocation: center or node.
  --profile PROFILE Init profile: single, center, or node.
  --nodes LIST      Center init nodes, for example node-a=http://10.0.0.2:8099,node-b=http://10.0.0.3:8099.
  --node-id ID      Node init id. Defaults to node-a.
  --center-url URL  Node init Studio backend URL. Defaults to http://<center-host>:3000.
  --agent-url URL   Node init local Agent API URL. Defaults to http://<node-host>:8099.
  --iphost HOST     Single init host used for non-model local URLs. Defaults to 127.0.0.1.
  --force           Init: overwrite an existing env file.
  --foreground      Run the selected service in the foreground when possible.
  --no-studio       Center start: do not start Studio.
  --no-bridge       Center start: do not start Bridge.

Examples:
  bash runtime.sh init --profile single
  bash runtime.sh init --profile single --iphost 10.0.0.5
  bash runtime.sh --env-file runtime.env init --profile center --nodes node-a=http://10.0.0.2:8099
  bash runtime.sh --env-file runtime.node.env init --profile node --node-id node-a --center-url http://10.0.0.1:3000
  bash runtime.sh check
  bash runtime.sh --env-file runtime.node.env start
  bash runtime.sh logs agent
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --role)
      ROLE_OVERRIDE="$2"
      shift 2
      ;;
    --profile)
      INIT_PROFILE="$2"
      shift 2
      ;;
    --nodes)
      INIT_NODES="$2"
      shift 2
      ;;
    --node-id)
      INIT_NODE_ID="$2"
      shift 2
      ;;
    --center-url)
      INIT_CENTER_URL="$2"
      shift 2
      ;;
    --agent-url)
      INIT_AGENT_URL="$2"
      shift 2
      ;;
    --iphost)
      INIT_IPHOST="$2"
      shift 2
      ;;
    --force)
      INIT_FORCE=true
      shift
      ;;
    --foreground)
      FOREGROUND=true
      shift
      ;;
    --no-studio)
      SKIP_STUDIO=true
      shift
      ;;
    --no-bridge)
      SKIP_BRIDGE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    init|start|stop|restart|status|logs|check|doctor)
      ACTION="$1"
      shift
      if [[ "$ACTION" == "logs" && $# -gt 0 && "$1" != --* ]]; then
        LOG_SERVICE="$1"
        shift
      fi
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$ACTION" != "init" && ! -f "$ENV_FILE" ]]; then
  echo "Missing configuration: $ENV_FILE"
  echo "Create it first:"
  echo "  single machine: bash runtime.sh init --profile single"
  echo "  center server:  bash runtime.sh init --profile center --nodes node-a=http://<node-host>:8099"
  echo "  node server:    bash runtime.sh init --profile node --node-id node-a --center-url http://<center-host>:3000"
  exit 1
fi

load_env_file() {
  local source_file="$1" normalized_file=""
  set -a
  if grep -q $'\r' "$source_file" 2>/dev/null; then
    echo "Warning: $source_file uses CRLF line endings; loading it after stripping carriage returns."
    normalized_file="$(mktemp "${TMPDIR:-/tmp}/medflow-runtime-env.XXXXXX")"
    tr -d '\r' < "$source_file" > "$normalized_file"
    # shellcheck disable=SC1090
    source "$normalized_file"
    rm -f "$normalized_file"
  else
    # shellcheck disable=SC1090
    source "$source_file"
  fi
  set +a
}

if [[ "$ACTION" != "init" ]]; then
  load_env_file "$ENV_FILE"
fi

RUNTIME_ROLE="${ROLE_OVERRIDE:-${RUNTIME_ROLE:-center}}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for python_candidate in python python3 py; do
    if command -v "$python_candidate" >/dev/null 2>&1 && "$python_candidate" -c 'print(1)' >/dev/null 2>&1; then
      PYTHON_BIN="$python_candidate"
      break
    fi
  done
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
NPM_BIN="${NPM_BIN:-npm}"
STUDIO_PORT="${STUDIO_PORT:-3000}"
STUDIO_HOST="${STUDIO_HOST:-0.0.0.0}"
STUDIO_PUBLIC_URL="${STUDIO_PUBLIC_URL:-http://127.0.0.1:5173}"
BRIDGE_PORT="${BRIDGE_PORT:-3100}"
AGENT_API_HOST="${AGENT_API_HOST:-0.0.0.0}"
AGENT_API_PORT="${AGENT_API_PORT:-8099}"
START_LOCAL_AGENT="${START_LOCAL_AGENT:-false}"
RUNTIME_STOP_TIMEOUT_SECONDS="${RUNTIME_STOP_TIMEOUT_SECONDS:-5}"
MEDFLOW_STUDIO_RUNTIME_TOKEN="${MEDFLOW_STUDIO_RUNTIME_TOKEN:-}"
export MEDFLOW_STUDIO_RUNTIME_TOKEN

if [[ "$ACTION" != "init" ]]; then
  mkdir -p "$LOG_DIR" "$DATA_DIR"
fi

generate_token() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32
    return
  fi
  "$PYTHON_BIN" - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
}

json_from_runtime_pairs() {
  "$PYTHON_BIN" - "$@" <<'PY'
import json
import sys

pairs = sys.argv[1:]
result = {}
for pair in pairs:
    key, value = pair.split("=", 1)
    result[key] = value
print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
PY
}

json_nodes_from_runtime_pairs() {
  "$PYTHON_BIN" - "$@" <<'PY'
import json
import sys

nodes = []
for pair in sys.argv[1:]:
    node_id, base_url, resource_token = pair.split("=", 2)
    nodes.append({
        "id": node_id,
        "name": node_id,
        "baseUrl": base_url,
        "resourceApiToken": resource_token,
    })
print(json.dumps(nodes, ensure_ascii=False, separators=(",", ":")))
PY
}

ensure_init_target() {
  if [[ -f "$ENV_FILE" && "$INIT_FORCE" != "true" ]]; then
    echo "Refusing to overwrite existing env file: $ENV_FILE"
    echo "Use --force if you really want to replace it."
    exit 1
  fi
  mkdir -p "$(dirname "$ENV_FILE")"
}

write_common_runtime_defaults() {
  cat <<'EOF'

# Runtime defaults. Usually keep these values.
MEDFLOW_RESOURCE_TIMEOUT_MS=60000
MEDFLOW_RESOURCE_SNAPSHOT_TIMEOUT_MS=15000
MEDFLOW_RESOURCE_GPU_BUSY_MEMORY_MB=200
MEDFLOW_TRAINING_RESERVATION_TTL_SECONDS=300
MEDFLOW_GPU_RESERVATION_MAX_TTL_SECONDS=900
MEDFLOW_TRAINING_RESOURCE_LOCK_TTL_SECONDS=300
MEDFLOW_RESOURCE_GPU_QUERY_TIMEOUT_SECONDS=10
MEDFLOW_RESOURCE_GPU_REFRESH_INTERVAL_SECONDS=10
MEDFLOW_RESOURCE_GPU_SNAPSHOT_MAX_AGE_SECONDS=30
MEDFLOW_RESOURCE_GPU_CACHE_TTL_SECONDS=30
MEDFLOW_TRAINING_RESERVATION_HEARTBEAT_SECONDS=60
MEDFLOW_TRAINING_RESERVATION_MAX_HEARTBEAT_FAILURES=3
MEDFLOW_STUDIO_RESOURCE_REQUEST_TIMEOUT_SECONDS=90
MEDFLOW_REQUIRE_GPU_ASSIGNMENT=1
MEDFLOW_GPU_PREFLIGHT_TIMEOUT_SECONDS=5
MEDFLOW_GPU_PREFLIGHT_BUSY_MEMORY_MB=200
MEDFLOW_RESOURCE_GROUP_ID=
MEDFLOW_TRAINING_POOL_ID=
MEDFLOW_MULTINODE_NODE_COUNT=2
MEDFLOW_MULTINODE_GPUS_PER_NODE=1
MEDFLOW_DEFAULT_GPU_COUNT_BATCH_TRAIN_LORA=8
MEDFLOW_DEFAULT_GPU_COUNT_BATCH_TRAIN_FULL=8
MEDFLOW_DEFAULT_GPU_COUNT_BATCH_PRETRAIN_LORA=8
MEDFLOW_DEFAULT_GPU_COUNT_BATCH_PRETRAIN_FULL=8
MEDFLOW_DEFAULT_GPU_COUNT_DPO_TRAIN=8
MEDFLOW_DEFAULT_GPU_COUNT_GRPO_TRAIN=8
MEDFLOW_DEFAULT_GPU_COUNT_CKPT_EVAL=8
MEDFLOW_DEFAULT_GPU_COUNT_COMPARE_MODELS=8
MEDFLOW_DEFAULT_GPU_COUNT_SINGLE_MODEL_EVAL=4
MEDFLOW_RUNTIME_PROJECT=
MEDFLOW_RUNTIME_RUN_NAME=
MEDFLOW_STUDIO_REGISTRATION_TIMEOUT_SECONDS=10
AGENT_API_PROCESS_PATH=/runtime-process
AGENT3_WORKFLOW_DB_PATH=./data/workflows/workflows.db
AGENT3_WORKFLOW_POLL_INTERVAL=30
AGENT3_WORKFLOW_TRAIN_START_GRACE_SECONDS=180
AGENT3_WORKFLOW_EVENT_LEASE_SECONDS=300
AGENT3_WORKFLOW_EVENT_LEASE_RENEW_SECONDS=60
AGENT3_WORKFLOW_PUBLISH_DIR=/home/workspace/medical_models
AGENT3_SESSION_TIMEOUT=3600
AGENT3_MAX_CONCURRENT_SESSIONS=100
RUNTIME_STOP_TIMEOUT_SECONDS=5
# NVIDIA_SMI_PATH=/usr/bin/nvidia-smi
# PYTHON_BIN=python
# NPM_BIN=npm
EOF
}

init_single_env() {
  local studio_token agent_token resource_token node_token admin_password node_tokens single_host
  studio_token="$(generate_token)"
  agent_token="$(generate_token)"
  resource_token="$(generate_token)"
  node_token="$(generate_token)"
  admin_password="$(generate_token)"
  node_tokens="$(json_from_runtime_pairs "local=$node_token")"
  single_host="${INIT_IPHOST:-127.0.0.1}"
  ensure_init_target
  {
    cat <<EOF
# Generated by bash runtime.sh init --profile single
# Fill the business fields in this top section, then run bash runtime.sh check.

RUNTIME_ROLE=center
START_LOCAL_AGENT=true

# Local node identity and task containers.
MEDFLOW_RESOURCE_NODE_ID=local
MEDFLOW_RESOURCE_NODE_NAME=local
MEDFLOW_LOCAL_TRAINING_CONTAINER=replace-training-container
MULTINODE_DOCKER_CONTAINER=replace-multinode-container
MEDFLOW_LOCAL_EVALUATE_CONTAINER=replace-evaluate-container
MEDFLOW_LOCAL_GRPO_CONTAINER=replace-grpo-container

# Model and inference Agent services.
MODEL_NAME=replace-model-name
MODEL_API_KEY=empty
MODEL_BASE_URL=http://127.0.0.1:7000/v1
INFERENCE_AGENT_URL=http://127.0.0.1:8899/inference_agent

# Network entry points. Keep 127.0.0.1 only for same-host local debugging.
STUDIO_PUBLIC_URL=http://$single_host:5173
STUDIO_URL=http://$single_host:3000
MEDFLOW_RESOURCE_NODES='[{"id":"local","name":"local","baseUrl":"http://$single_host:8099"}]'
AGENT_API_BACKENDS=http://$single_host:8099

# Generated credentials. Do not reuse these values for other systems.
MEDFLOW_STUDIO_RUNTIME_TOKEN=$studio_token
MEDFLOW_AGENT_API_TOKEN=$agent_token
MEDFLOW_RESOURCE_API_TOKEN=$resource_token
MEDFLOW_RUNTIME_NODE_TOKEN=$node_token
MEDFLOW_STUDIO_RUNTIME_NODE_TOKENS='$node_tokens'
MEDFLOW_ADMIN_USERNAME=admin
MEDFLOW_ADMIN_PASSWORD=$admin_password

STUDIO_HOST=0.0.0.0
STUDIO_PORT=3000
BRIDGE_PORT=3100
AGENT_API_HOST=0.0.0.0
AGENT_API_PORT=8099
MEDFLOW_AUTH_COOKIE_SECURE=false
EOF
    write_common_runtime_defaults
  } >"$ENV_FILE"
  echo "Generated single-machine runtime env: $ENV_FILE"
  echo "Next: edit container/model URLs, then run bash runtime.sh --env-file \"$ENV_FILE\" check"
}

init_center_env() {
  local studio_token agent_token center_resource_token center_node_token admin_password node_pairs=() token_pairs=() backend_urls=() node_specs node_spec node_id node_url resource_token node_token nodes_json tokens_json backends
  if [[ -z "$INIT_NODES" ]]; then
    echo "Missing --nodes for center init."
    echo "Example: bash runtime.sh init --profile center --nodes node-a=http://10.0.0.2:8099,node-b=http://10.0.0.3:8099"
    exit 1
  fi
  studio_token="$(generate_token)"
  agent_token="$(generate_token)"
  center_resource_token="$(generate_token)"
  center_node_token="$(generate_token)"
  admin_password="$(generate_token)"
  node_pairs+=("center=http://127.0.0.1:8099=$center_resource_token")
  token_pairs+=("center=$center_node_token")
  backend_urls+=("http://127.0.0.1:8099")
  IFS=',' read -r -a node_specs <<<"$INIT_NODES"
  for node_spec in "${node_specs[@]}"; do
    node_id="${node_spec%%=*}"
    node_url="${node_spec#*=}"
    if [[ "$node_id" == "center" ]]; then
      echo "Reserved --nodes id: center. The center-local Agent node is generated automatically."
      exit 1
    fi
    if [[ -z "$node_id" || -z "$node_url" || "$node_id" == "$node_url" ]]; then
      echo "Invalid --nodes item: $node_spec"
      exit 1
    fi
    resource_token="$(generate_token)"
    node_token="$(generate_token)"
    node_pairs+=("$node_id=$node_url=$resource_token")
    token_pairs+=("$node_id=$node_token")
    backend_urls+=("$node_url")
  done
  nodes_json="$(json_nodes_from_runtime_pairs "${node_pairs[@]}")"
  tokens_json="$(json_from_runtime_pairs "${token_pairs[@]}")"
  backends="$(IFS=','; echo "${backend_urls[*]}")"
  ensure_init_target
  {
    cat <<EOF
# Generated by bash runtime.sh init --profile center
# Fill center URLs and model settings, then copy the printed per-node tokens to each node runtime.env.

RUNTIME_ROLE=center
START_LOCAL_AGENT=true

# Center-local Agent identity and task containers.
MEDFLOW_RESOURCE_NODE_ID=center
MEDFLOW_RESOURCE_NODE_NAME=center
MEDFLOW_LOCAL_TRAINING_CONTAINER=replace-training-container
MULTINODE_DOCKER_CONTAINER=replace-multinode-container
MEDFLOW_LOCAL_EVALUATE_CONTAINER=replace-evaluate-container
MEDFLOW_LOCAL_GRPO_CONTAINER=replace-grpo-container

# Center network entry points.
STUDIO_PUBLIC_URL=http://<center-host>:5173
STUDIO_URL=http://<center-host>:3000
MEDFLOW_RESOURCE_NODES='$nodes_json'
MEDFLOW_STUDIO_RUNTIME_NODE_TOKENS='$tokens_json'
AGENT_API_BACKENDS=$backends

# Model and inference Agent services used by the center-local Agent.
MODEL_NAME=replace-model-name
MODEL_API_KEY=empty
MODEL_BASE_URL=http://<llm-host>:<llm-port>/v1
INFERENCE_AGENT_URL=http://<inference-agent-host>:8899/inference_agent

# Generated shared credentials. Copy these to every node env where required.
MEDFLOW_STUDIO_RUNTIME_TOKEN=$studio_token
MEDFLOW_AGENT_API_TOKEN=$agent_token
MEDFLOW_RESOURCE_API_TOKEN=$center_resource_token
MEDFLOW_RUNTIME_NODE_TOKEN=$center_node_token
MEDFLOW_ADMIN_USERNAME=admin
MEDFLOW_ADMIN_PASSWORD=$admin_password

STUDIO_HOST=0.0.0.0
STUDIO_PORT=3000
BRIDGE_PORT=3100
AGENT_API_HOST=0.0.0.0
AGENT_API_PORT=8099
MEDFLOW_AUTH_COOKIE_SECURE=false
EOF
    write_common_runtime_defaults
  } >"$ENV_FILE"
  echo "Generated center runtime env: $ENV_FILE"
  echo "Copy these shared values into node env files:"
  echo "  MEDFLOW_STUDIO_RUNTIME_TOKEN=$studio_token"
  echo "  MEDFLOW_AGENT_API_TOKEN=$agent_token"
  echo "Per-node generated values (center is already written to this env; copy compute-node values into matching node env files):"
  "$PYTHON_BIN" - "$nodes_json" "$tokens_json" <<'PY'
import json
import sys
nodes = json.loads(sys.argv[1])
tokens = json.loads(sys.argv[2])
for node in nodes:
    node_id = node["id"]
    print(f"  {node_id}: MEDFLOW_RESOURCE_API_TOKEN={node['resourceApiToken']} MEDFLOW_RUNTIME_NODE_TOKEN={tokens[node_id]}")
PY
}

init_node_env() {
  local node_id center_url agent_url agent_port
  node_id="${INIT_NODE_ID:-node-a}"
  center_url="${INIT_CENTER_URL:-http://<center-host>:3000}"
  agent_url="${INIT_AGENT_URL:-http://<node-host>:8099}"
  agent_port="${agent_url##*:}"
  agent_port="${agent_port%%/*}"
  [[ "$agent_port" =~ ^[0-9]+$ ]] || agent_port="8099"
  ensure_init_target
  {
    cat <<EOF
# Generated by bash runtime.sh init --profile node
# Copy shared and per-node credentials from the center init output.

RUNTIME_ROLE=node

# Node identity and task containers.
MEDFLOW_RESOURCE_NODE_ID=$node_id
MEDFLOW_RESOURCE_NODE_NAME=$node_id
MEDFLOW_LOCAL_TRAINING_CONTAINER=replace-training-container
MULTINODE_DOCKER_CONTAINER=replace-multinode-container
MEDFLOW_LOCAL_EVALUATE_CONTAINER=replace-evaluate-container
MEDFLOW_LOCAL_GRPO_CONTAINER=replace-grpo-container

# Credentials copied from center runtime.env / init output.
MEDFLOW_STUDIO_RUNTIME_TOKEN=replace-copy-from-center-studio-runtime-token
MEDFLOW_AGENT_API_TOKEN=replace-copy-from-center-agent-api-token
MEDFLOW_RESOURCE_API_TOKEN=replace-copy-from-center-node-resource-api-token
MEDFLOW_RUNTIME_NODE_TOKEN=replace-copy-from-center-node-runtime-token

# Network entry points.
STUDIO_URL=$center_url
# This node should be reachable from the center as: $agent_url
AGENT_API_HOST=0.0.0.0
AGENT_API_PORT=$agent_port

# Model and inference Agent services.
MODEL_NAME=replace-model-name
MODEL_API_KEY=empty
MODEL_BASE_URL=http://<llm-host>:<llm-port>/v1
INFERENCE_AGENT_URL=http://<inference-agent-host>:8899/inference_agent
EOF
    write_common_runtime_defaults
  } >"$ENV_FILE"
  echo "Generated node runtime env: $ENV_FILE"
  echo "Next: paste the shared and per-node token values printed by center init, then run bash runtime.sh --env-file \"$ENV_FILE\" check"
}

init_env() {
  case "$INIT_PROFILE" in
    single) init_single_env ;;
    center) init_center_env ;;
    node) init_node_env ;;
    "")
      echo "Missing --profile for init. Use single, center, or node."
      exit 1
      ;;
    *)
      echo "Unknown init profile: $INIT_PROFILE"
      exit 1
      ;;
  esac
}
prepare_studio_database() {
  local database_path legacy_path
  database_path="${MEDFLOW_DATABASE_PATH:-$DATA_DIR/database.sqlite}"
  legacy_path="${HOME:-}/Medflow-Studio/database.sqlite"

  mkdir -p "$(dirname "$database_path")"
  if [[ ! -f "$database_path" && -n "${HOME:-}" && -f "$legacy_path" ]]; then
    echo "Migrating Studio database: $legacy_path -> $database_path"
    cp -p "$legacy_path" "$database_path"
  fi
  export MEDFLOW_DATABASE_PATH="$database_path"
}

pid_file() {
  echo "$STATE_DIR/$1.pid"
}

is_running() {
  local file pid
  file="$(pid_file "$1")"
  [[ -f "$file" ]] || return 1
  pid="$(cat "$file" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

cleanup_stale_pid() {
  local name="$1" file
  file="$(pid_file "$name")"
  if [[ -f "$file" ]] && ! is_running "$name"; then
    echo "Removing stale pid file for $name."
    rm -f "$file"
  fi
}

start_process() {
  local name="$1"
  local workdir="$2"
  shift 2

  cleanup_stale_pid "$name"
  if is_running "$name"; then
    echo "$name is already running (PID $(cat "$(pid_file "$name")"))."
    return
  fi

  echo "Starting $name..."
  if [[ "$FOREGROUND" == "true" ]]; then
    cd "$workdir"
    exec "$@"
  fi

  (
    cd "$workdir"
    nohup setsid "$@" >>"$LOG_DIR/$name.log" 2>&1 &
    echo $! >"$(pid_file "$name")"
  )
  sleep 1
  if ! is_running "$name"; then
    echo "$name failed to start. Check $LOG_DIR/$name.log"
    exit 1
  fi
}

stop_process() {
  local name="$1"
  local file pid deadline
  file="$(pid_file "$name")"
  if ! is_running "$name"; then
    rm -f "$file"
    echo "$name is not running."
    return
  fi

  pid="$(cat "$file")"
  echo "Stopping $name (PID $pid)..."
  kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  deadline=$((SECONDS + RUNTIME_STOP_TIMEOUT_SECONDS))
  while kill -0 "$pid" 2>/dev/null && [[ "$SECONDS" -lt "$deadline" ]]; do
    sleep 0.25
  done
  kill -9 -- "-$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
  rm -f "$file"
}

require_value() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "$value" ]]; then
    echo "Missing required setting: $name"
    exit 1
  fi
  if [[ "$value" == replace-* || "$value" == *"<"* || "$value" == *">"* ]]; then
    echo "Setting still uses a placeholder: $name=$value"
    echo "Edit $ENV_FILE and replace it with your own value."
    exit 1
  fi
}

check_command() {
  local command_name="$1"
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "Missing required command: $command_name"
    return 1
  }
}

check_json_env() {
  local name="$1" value="${!1:-}"
  [[ -z "$value" ]] && return 0
  "$PYTHON_BIN" -c 'import json, os, sys; json.loads(os.environ[sys.argv[1]])' "$name" 2>/dev/null || {
    echo "Invalid JSON in $name"
    return 1
  }
}
normalize_alias_env() {
  local preferred_name="$1" legacy_name="$2" preferred_value legacy_value
  preferred_value="${!preferred_name:-}"
  legacy_value="${!legacy_name:-}"
  if [[ -n "$preferred_value" && -n "$legacy_value" && "$preferred_value" != "$legacy_value" ]]; then
    echo "$legacy_name must match $preferred_name. Keep only $preferred_name, or set both to the same value."
    return 1
  fi
  if [[ -z "$preferred_value" && -n "$legacy_value" ]]; then
    printf -v "$preferred_name" '%s' "$legacy_value"
    export "$preferred_name"
  elif [[ -n "$preferred_value" && -z "$legacy_value" ]]; then
    printf -v "$legacy_name" '%s' "$preferred_value"
    export "$legacy_name"
  fi
}

json_object_from_pair() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json
import sys

print(json.dumps({sys.argv[1]: sys.argv[2]}, ensure_ascii=False, separators=(",", ":")))
PY
}

normalize_runtime_env() {
  local failed=0
  normalize_alias_env MEDFLOW_RUNTIME_NODE_TOKEN MEDFLOW_STUDIO_NODE_TOKEN || failed=1
  normalize_alias_env MEDFLOW_LOCAL_TRAINING_CONTAINER AGENT3_DEFAULT_DOCKER_CONTAINER || failed=1
  normalize_alias_env MEDFLOW_LOCAL_EVALUATE_CONTAINER AGENT3_DEFAULT_EVALUATE_DOCKER_CONTAINER || failed=1
  normalize_alias_env MEDFLOW_LOCAL_GRPO_CONTAINER MEDFLOW_GRPO_DOCKER_CONTAINER || failed=1
  [[ "$failed" -eq 0 ]] || exit 1

  if [[ "$RUNTIME_ROLE" == "center" && "$START_LOCAL_AGENT" == "true" \
      && -z "${MEDFLOW_STUDIO_RUNTIME_NODE_TOKENS:-}" \
      && -n "${MEDFLOW_RESOURCE_NODE_ID:-}" \
      && -n "${MEDFLOW_RUNTIME_NODE_TOKEN:-}" ]]; then
    MEDFLOW_STUDIO_RUNTIME_NODE_TOKENS="$(json_object_from_pair "$MEDFLOW_RESOURCE_NODE_ID" "$MEDFLOW_RUNTIME_NODE_TOKEN")"
    export MEDFLOW_STUDIO_RUNTIME_NODE_TOKENS
  fi
}

validate_runtime_env_consistency() {
  "$PYTHON_BIN" <<'PY'
import json
import os
import sys

errors = []


def env(name: str) -> str:
    return os.environ.get(name, "").strip()


def parse_json_env(name: str, expected_type):
    raw = env(name)
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except Exception as exc:
        errors.append(f"Invalid JSON in {name}: {exc}")
        return None
    if not isinstance(value, expected_type):
        errors.append(f"{name} must be a {expected_type.__name__}.")
        return None
    return value


runtime_role = env("RUNTIME_ROLE")
start_local_agent = env("START_LOCAL_AGENT").lower() == "true"
has_local_agent = runtime_role == "node" or start_local_agent
node_id = env("MEDFLOW_RESOURCE_NODE_ID")
node_token = env("MEDFLOW_RUNTIME_NODE_TOKEN") or env("MEDFLOW_STUDIO_NODE_TOKEN")

tokens = parse_json_env("MEDFLOW_STUDIO_RUNTIME_NODE_TOKENS", dict)
if has_local_agent and tokens is not None and node_id and node_token:
    expected = str(tokens.get(node_id, "")).strip()
    if expected and expected != node_token:
        errors.append(
            "MEDFLOW_STUDIO_RUNTIME_NODE_TOKENS"
            f"[{node_id!r}] must match MEDFLOW_RUNTIME_NODE_TOKEN."
        )
    elif runtime_role == "center" and start_local_agent and not expected:
        errors.append(
            "MEDFLOW_STUDIO_RUNTIME_NODE_TOKENS must include the local "
            f"MEDFLOW_RESOURCE_NODE_ID {node_id!r} when START_LOCAL_AGENT=true."
        )

nodes = parse_json_env("MEDFLOW_RESOURCE_NODES", list)
if nodes is not None:
    matching_nodes = []
    for item in nodes:
        if not isinstance(item, dict):
            continue
        if node_id and str(item.get("id", "")).strip() == node_id:
            matching_nodes.append(item)

    if runtime_role == "center" and start_local_agent and node_id:
        if not matching_nodes:
            errors.append(
                f"MEDFLOW_RESOURCE_NODES must include local node id {node_id!r} "
                "when START_LOCAL_AGENT=true."
            )
        else:
            node = matching_nodes[0]
            resource_token = env("MEDFLOW_RESOURCE_API_TOKEN")
            node_resource_token = str(node.get("resourceApiToken", "") or "").strip()
            if resource_token and node_resource_token and resource_token != node_resource_token:
                errors.append(
                    "MEDFLOW_RESOURCE_API_TOKEN must match "
                    f"MEDFLOW_RESOURCE_NODES[{node_id!r}].resourceApiToken."
                )

            checks = (
                ("defaultContainer", env("MEDFLOW_LOCAL_TRAINING_CONTAINER") or env("AGENT3_DEFAULT_DOCKER_CONTAINER")),
                ("defaultEvaluateContainer", env("MEDFLOW_LOCAL_EVALUATE_CONTAINER") or env("AGENT3_DEFAULT_EVALUATE_DOCKER_CONTAINER")),
            )
            for key, expected_value in checks:
                configured_value = str(node.get(key, "") or "").strip()
                if configured_value and expected_value and configured_value != expected_value:
                    errors.append(
                        f"MEDFLOW_RESOURCE_NODES[{node_id!r}].{key} must match "
                        f"the local env value {expected_value!r}."
                    )

if errors:
    for error in errors:
        print(error)
    sys.exit(1)
PY
}



check_port_free() {
  local label="$1" host="$2" port="$3" bind_host
  bind_host="$host"
  [[ "$bind_host" == "0.0.0.0" || "$bind_host" == "::" ]] && bind_host="127.0.0.1"
  "$PYTHON_BIN" - "$bind_host" "$port" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((host, port))
finally:
    sock.close()
PY
  local result=$?
  if [[ "$result" -ne 0 ]]; then
    echo "$label port is already in use: $host:$port"
    return 1
  fi
}

validate_common() {
  normalize_runtime_env
  check_command setsid
  check_command curl
  if [[ "$RUNTIME_ROLE" == "node" || "$START_LOCAL_AGENT" == "true" || "$ACTION" == "check" || "$ACTION" == "doctor" ]]; then
    check_command "$PYTHON_BIN"
  fi
  if [[ "$RUNTIME_ROLE" == "center" && ( "$SKIP_STUDIO" != "true" || "$SKIP_BRIDGE" != "true" ) ]]; then
    check_command "$NPM_BIN"
  fi
  if [[ "$RUNTIME_ROLE" == "node" || "$START_LOCAL_AGENT" == "true" ]]; then
    require_value MEDFLOW_RESOURCE_API_TOKEN
  fi
  if [[ "${MEDFLOW_RESOURCE_API_TOKEN:-}" == "replace-with-a-long-random-token" || "${MEDFLOW_RESOURCE_API_TOKEN:-}" == "abcdefg" ]]; then
    echo "MEDFLOW_RESOURCE_API_TOKEN must be changed before startup."
    exit 1
  fi
  if [[ -z "${MEDFLOW_STUDIO_RUNTIME_TOKEN:-}" ]]; then
    echo "Missing required setting: MEDFLOW_STUDIO_RUNTIME_TOKEN"
    exit 1
  fi
  require_value MEDFLOW_AGENT_API_TOKEN
  if [[ -n "${MEDFLOW_RESOURCE_API_TOKEN:-}" && "$MEDFLOW_AGENT_API_TOKEN" == "$MEDFLOW_RESOURCE_API_TOKEN" ]]; then
    echo "MEDFLOW_AGENT_API_TOKEN must differ from MEDFLOW_RESOURCE_API_TOKEN."
    exit 1
  fi
  if [[ "$MEDFLOW_AGENT_API_TOKEN" == "$MEDFLOW_STUDIO_RUNTIME_TOKEN" ]]; then
    echo "MEDFLOW_AGENT_API_TOKEN must differ from MEDFLOW_STUDIO_RUNTIME_TOKEN."
    exit 1
  fi
  validate_runtime_env_consistency
}

doctor() {
  local failed=0
  echo "Runtime env: $ENV_FILE"
  validate_common || failed=1
  check_json_env MEDFLOW_RESOURCE_NODES || failed=1
  check_json_env MEDFLOW_STUDIO_RUNTIME_NODE_TOKENS || failed=1
  if [[ "$RUNTIME_ROLE" == "center" ]]; then
    [[ "$SKIP_STUDIO" == "true" ]] || check_port_free Studio "$STUDIO_HOST" "$STUDIO_PORT" || failed=1
    [[ "$SKIP_BRIDGE" == "true" ]] || check_port_free Bridge "127.0.0.1" "$BRIDGE_PORT" || failed=1
  fi
  if [[ "$RUNTIME_ROLE" == "node" || "$START_LOCAL_AGENT" == "true" ]]; then
    check_port_free Agent "$AGENT_API_HOST" "$AGENT_API_PORT" || failed=1
    check_command docker || failed=1
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "nvidia-smi: found"
    elif [[ -n "${NVIDIA_SMI_PATH:-}" && -x "$NVIDIA_SMI_PATH" ]]; then
      echo "nvidia-smi: using $NVIDIA_SMI_PATH"
    else
      echo "nvidia-smi: not found on host; Agent will try docker exec fallback"
    fi
    if [[ -n "${AGENT3_DEFAULT_DOCKER_CONTAINER:-}" ]]; then
      docker inspect "$AGENT3_DEFAULT_DOCKER_CONTAINER" >/dev/null 2>&1 || {
        echo "Docker container not found: $AGENT3_DEFAULT_DOCKER_CONTAINER"
        failed=1
      }
    fi
  fi
  [[ "$failed" -eq 0 ]] && echo "Runtime check passed."
  return "$failed"
}

wait_for_url() {
  local name="$1"
  local url="$2"
  echo "Waiting for $name at $url..."
  for _ in {1..60}; do
    if curl --connect-timeout 2 -sS -o /dev/null "$url" 2>/dev/null; then
      return
    fi
    sleep 1
  done
  echo "Timed out waiting for $name at $url"
  exit 1
}

start_agent() {
  require_value MEDFLOW_RESOURCE_NODE_ID
  require_value MEDFLOW_RESOURCE_NODE_NAME
  require_value AGENT3_DEFAULT_DOCKER_CONTAINER
  require_value STUDIO_URL
  wait_for_url Studio "$STUDIO_URL"
  export HOST="$AGENT_API_HOST"
  export PORT="$AGENT_API_PORT"
  start_process agent "$ROOT_DIR/agent" "$PYTHON_BIN" api_app.py
}

start_center() {
  require_value MEDFLOW_RESOURCE_NODES
  require_value AGENT_API_BACKENDS
  prepare_studio_database
  export AGENT_BRIDGE_BASE_URL="http://127.0.0.1:$BRIDGE_PORT"
  export HOST="$STUDIO_HOST"
  export STUDIO_HOST
  export STUDIO_PUBLIC_URL
  export PORT="$STUDIO_PORT"
  if [[ "$SKIP_STUDIO" != "true" ]]; then
    start_process studio "$ROOT_DIR/studio" "$NPM_BIN" run dev
    wait_for_url Studio "http://127.0.0.1:$STUDIO_PORT"
  fi

  if [[ "$SKIP_BRIDGE" != "true" ]]; then
    export PORT="$BRIDGE_PORT"
    start_process bridge "$ROOT_DIR/agent-studio-runtime-bridge" "$NPM_BIN" run dev
  fi

  if [[ "$START_LOCAL_AGENT" == "true" ]]; then
    start_agent
  fi
}

show_status() {
  for name in studio bridge agent; do
    cleanup_stale_pid "$name"
    if is_running "$name"; then
      echo "$name: running (PID $(cat "$(pid_file "$name")"))"
    else
      echo "$name: stopped"
    fi
  done
}

case "$ACTION" in
  init)
    init_env
    ;;
  start)
    validate_common
    case "$RUNTIME_ROLE" in
      center) start_center ;;
      node) start_agent ;;
      *) echo "RUNTIME_ROLE must be center or node"; exit 1 ;;
    esac
    show_status
    if [[ "$RUNTIME_ROLE" == "center" ]]; then
      echo "Studio UI: $STUDIO_PUBLIC_URL"
    fi
    echo "Logs: $LOG_DIR"
    ;;
  stop)
    stop_process agent
    stop_process bridge
    stop_process studio
    ;;
  restart)
    bash "$ROOT_DIR/runtime.sh" --env-file "$ENV_FILE" --role "$RUNTIME_ROLE" stop
    bash "$ROOT_DIR/runtime.sh" --env-file "$ENV_FILE" --role "$RUNTIME_ROLE" start
    ;;
  status)
    show_status
    ;;
  logs)
    if [[ -n "$LOG_SERVICE" ]]; then
      tail -n 100 -F "$LOG_DIR/$LOG_SERVICE.log"
    else
      tail -n 100 -F "$LOG_DIR"/*.log
    fi
    ;;
  check|doctor)
    doctor
    ;;
  *)
    usage
    exit 1
    ;;
esac
