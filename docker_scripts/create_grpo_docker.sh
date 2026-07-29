#!/usr/bin/env bash
set -euo pipefail

require_configured() {
    local name="$1"
    local value="${!name:-}"
    if [[ -z "$value" || "$value" == replace-* ]]; then
        echo "Missing or placeholder value for $name. Edit this script and replace the default parameter before running it." >&2
        exit 1
    fi
}

require_host_directory() {
    local directory="$1"
    local description="$2"
    if [[ ! -d "$directory" ]]; then
        echo "ERROR: $description does not exist: $directory" >&2
        echo "Create or mount it first to avoid Docker creating a directory with unexpected permissions." >&2
        exit 1
    fi
}

container_exists() {
    docker ps -a --format '{{.Names}}' | grep -Fxq "$1"
}

## 请在运行前修改此处的默认参数
GRPO_CONTAINER_NAME="qingnang_grpo"
GRPO_IMAGE="medflow-grpo-verl-binary:20260710"
HOST_WORKSPACE=""
##
CONTAINER_WORKSPACE="/home/workspace"
SSH_DIR="/root/.ssh"
GPU_DEVICES="all"
NETWORK_MODE="bridge"
SHM_SIZE="10g"

require_configured GRPO_CONTAINER_NAME
require_configured GRPO_IMAGE
require_configured HOST_WORKSPACE
require_configured CONTAINER_WORKSPACE
require_configured SSH_DIR

require_host_directory "$HOST_WORKSPACE" "Host workspace"
require_host_directory "$SSH_DIR" "SSH directory"

echo "GRPO_CONTAINER_NAME=$GRPO_CONTAINER_NAME"
echo "GRPO_IMAGE=$GRPO_IMAGE"
echo "HOST_WORKSPACE=$HOST_WORKSPACE"
echo "CONTAINER_WORKSPACE=$CONTAINER_WORKSPACE"
echo "NETWORK_MODE=$NETWORK_MODE"
echo "SHM_SIZE=$SHM_SIZE"

if container_exists "$GRPO_CONTAINER_NAME"; then
    echo "ERROR: container '$GRPO_CONTAINER_NAME' already exists."
    echo "Remove it first if replacement is intended: docker rm -f $GRPO_CONTAINER_NAME"
    exit 1
fi

docker run -dit \
    --name "$GRPO_CONTAINER_NAME" \
    --gpus "$GPU_DEVICES" \
    --network="$NETWORK_MODE" \
    -v "$HOST_WORKSPACE:$CONTAINER_WORKSPACE" \
    -v "$SSH_DIR:/root/.ssh" \
    --ipc=host \
    --shm-size="$SHM_SIZE" \
    --privileged \
    "$GRPO_IMAGE" \
    /bin/bash

echo "GRPO container created: $GRPO_CONTAINER_NAME"
echo "Enter it with: docker exec -it $GRPO_CONTAINER_NAME /bin/bash"
echo "Online GRPO training will fail unless Weights & Biases is logged in inside the container."
echo "Log in before starting online GRPO training:"
echo "  docker exec -it $GRPO_CONTAINER_NAME wandb login"

