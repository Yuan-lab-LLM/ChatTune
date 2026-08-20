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
TRAIN_IMAGE="qingnang_train:v20260724"
HOST_WORKSPACE=""
TRAIN_CONTAINER_NAME="qingnang_train_multi"
##

SSH_DIR="/root/.ssh"
SHM_SIZE="100G"
STACK_LIMIT="68719476736"
CONTAINER_WORKSPACE="/home/workspace"

require_configured HOST_IP
require_configured TRAIN_IMAGE
require_configured TRAIN_CONTAINER_NAME

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

install_builtin_datasets() {
    local installer="$ROOT_DIR/docker_scripts/install_builtin_datasets.sh"
    if [[ -x "$installer" ]]; then
        "$installer" "$HOST_WORKSPACE"
    elif [[ -f "$installer" ]]; then
        bash "$installer" "$HOST_WORKSPACE"
    else
        echo "Warning: built-in dataset installer not found: $installer" >&2
    fi
}

install_multinode_ssh_script() {
    local source_script="$ROOT_DIR/docker_scripts/setup_multinode_ssh.sh"
    local target_script="$HOST_WORKSPACE/setup_multinode_ssh.sh"

    if [[ ! -f "$source_script" ]]; then
        echo "ERROR: multi-node SSH setup script not found: $source_script" >&2
        exit 1
    fi

    cp "$source_script" "$target_script"
    sed -i 's/\r$//' "$target_script"
    chmod +x "$target_script"
    echo "Installed multi-node SSH setup script: $target_script"
}

container_exists() {
    docker ps -a --format '{{.Names}}' | grep -Fxq "$1"
}


create_container() {
    local container_name="$1"
    local image="$2"

    if container_exists "$container_name"; then
        echo "ERROR: container '$container_name' already exists."
        echo "Remove it first if replacement is intended: docker rm -f $container_name"
        return 1
    fi

    docker run -dit \
        --name "$container_name" \
        --gpus all \
        --ipc=host \
        --network=host \
        --pid=host \
        --user=root \
        --privileged \
        --restart=always \
        --ulimit "stack=$STACK_LIMIT" \
        --shm-size="$SHM_SIZE" \
        -e "IEI_VISIBLE_OS_IP=$HOST_IP" \
        -v "$HOST_WORKSPACE:$CONTAINER_WORKSPACE" \
        -v "$SSH_DIR:/root/.ssh" \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v /usr/bin/docker:/usr/bin/docker \
        -v /etc/localtime:/etc/localtime:ro \
        -w "$CONTAINER_WORKSPACE" \
        "$image" \
        /bin/bash

    
    echo "Container created: $container_name ($image)"
}

mkdir -p "$HOST_WORKSPACE"
require_host_directory "$SSH_DIR" "SSH directory"
install_builtin_datasets
install_multinode_ssh_script
create_container "$TRAIN_CONTAINER_NAME" "$TRAIN_IMAGE"

echo "Multi-node training container is ready: $TRAIN_CONTAINER_NAME"
echo "Multi-node SSH setup script is available in the container: $CONTAINER_WORKSPACE/setup_multinode_ssh.sh"
echo "Online multi-node training will fail unless Weights & Biases is logged in inside the container."
echo "Log in before starting online multi-node training:"
echo "  docker exec -it $TRAIN_CONTAINER_NAME wandb login"
