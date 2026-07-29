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
HOST_IP="replace-host-ip"
IMAGE_NAME="qingnang_v20260709"
IMAGE_VERSION="v1"
DOCKER_NAME="qingnang"
HOST_WORKSPACE=""
##

DOCKER_WORKSPACE="/home/workspace"
SSH_DIR="/root/.ssh"
SHM_SIZE="100G"
STACK_LIMIT="68719476736"

require_configured HOST_IP
require_configured IMAGE_NAME
require_configured DOCKER_NAME

if [[ -z "$IMAGE_VERSION" || "$IMAGE_VERSION" == "tag" ]]; then
    echo "Missing or placeholder value for IMAGE_VERSION. Edit this script and replace it before running." >&2
    exit 1
fi

echo "HOST_IP=$HOST_IP"
echo "IMAGE=$IMAGE_NAME:$IMAGE_VERSION"
echo "DOCKER_NAME=$DOCKER_NAME"
echo "HOST_WORKSPACE=$HOST_WORKSPACE"
echo "DOCKER_WORKSPACE=$DOCKER_WORKSPACE"

if docker ps -a --format '{{.Names}}' | grep -Fxq "$DOCKER_NAME"; then
    echo "ERROR: container '$DOCKER_NAME' already exists."
    echo "Remove it first if replacement is intended: docker rm -f $DOCKER_NAME"
    exit 1
fi

mkdir -p "$HOST_WORKSPACE"
require_host_directory "$SSH_DIR" "SSH directory"

docker run -dit \
    --name "$DOCKER_NAME" \
    --gpus all \
    --pid=host \
    --user=root \
    --cap-add=SYS_PTRACE \
    --privileged \
    --ipc=host \
    --network=host \
    --restart=always \
    --ulimit "stack=$STACK_LIMIT" \
    --shm-size="$SHM_SIZE" \
    -e "IEI_VISIBLE_OS_IP=$HOST_IP" \
    -v "$HOST_WORKSPACE:$DOCKER_WORKSPACE" \
    -v "$SSH_DIR:/root/.ssh" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    -v /etc/localtime:/etc/localtime:ro \
    -w "$DOCKER_WORKSPACE" \
    "$IMAGE_NAME:$IMAGE_VERSION" \
    /bin/bash

echo "Inference container created: $DOCKER_NAME"
echo "Enter it with: docker exec -it $DOCKER_NAME /bin/bash"
