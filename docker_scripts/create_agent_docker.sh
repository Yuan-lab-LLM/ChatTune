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
## 请在运行前修改此处的默认参数
AGENT_CONTAINER_NAME="agent_runtime"
AGENT_IMAGE="nvcr.io/nvidia/pytorch:25.03-py3"
HOST_WORKSPACE=""
##

CONTAINER_WORKSPACE="/home/workspace"
SSH_DIR="/root/.ssh"
SHM_SIZE="100G"
STACK_LIMIT="68719476736"

require_configured AGENT_CONTAINER_NAME
require_configured AGENT_IMAGE

if docker ps -a --format '{{.Names}}' | grep -Fxq "$AGENT_CONTAINER_NAME"; then
    echo "ERROR: container '$AGENT_CONTAINER_NAME' already exists."
    echo "Remove it first if replacement is intended: docker rm -f $AGENT_CONTAINER_NAME"
    exit 1
fi

mkdir -p "$HOST_WORKSPACE"
require_host_directory "$SSH_DIR" "SSH directory"

docker run -dit \
    --name "$AGENT_CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    --network=host \
    --privileged \
    --restart=always \
    --ulimit "stack=$STACK_LIMIT" \
    --shm-size="$SHM_SIZE" \
    --user=root \
    -v "$HOST_WORKSPACE:$CONTAINER_WORKSPACE" \
    -v "$SSH_DIR:/root/.ssh" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    -v /etc/localtime:/etc/localtime:ro \
    -w "$CONTAINER_WORKSPACE" \
    "$AGENT_IMAGE" \
    /bin/bash

echo "Agent container created: $AGENT_CONTAINER_NAME"
echo "Enter it with: docker exec -it $AGENT_CONTAINER_NAME /bin/bash"
