#! /bin/bash
set -euo pipefail

require_configured() {
    local name="$1"
    local value="${!name:-}"
    if [[ -z "$value" || "$value" == replace-* ]]; then
        echo "Missing or placeholder value for $name. Edit this script and replace the default parameter before running it." >&2
        exit 1
    fi
}

## 请在运行前修改此处的默认参数
HOST_IP="replace-host-ip"
IMAGE_NAME="replace-inference-image"
IMAGE_VERSION="tag"
DOCKER_NAME="replace-evaluate-container"
HOST_WORKSPACE="/data/llm"
DOCKER_WORKSPACE="/home/workspace"
SSH_DIR="/root/.ssh"
SHM_SIZE="100G"
STACK_LIMIT="68719476736"
##

require_configured HOST_IP
require_configured IMAGE_NAME
require_configured DOCKER_NAME

if [[ -z "$IMAGE_VERSION" || "$IMAGE_VERSION" == "tag" ]]; then
    echo "Missing or placeholder value for IMAGE_VERSION. Edit this script and replace it before running." >&2
    exit 1
fi

echo "HOST_IP=${HOST_IP}"
echo "IMAGE_NAME=${IMAGE_NAME}"
echo "IMAGE_VERSION=${IMAGE_VERSION}"
echo "DOCKER_NAME=${DOCKER_NAME}"
echo "HOST_WORKSPACE=${HOST_WORKSPACE}"
echo "DOCKER_WORKSPACE=${DOCKER_WORKSPACE}"

docker_run=`docker ps -a -q --filter "name=^/$DOCKER_NAME$"`
if [  "$docker_run" ]; then
    echo "====================================================="
    echo "ERROR: Container '$DOCKER_NAME' already exists"
    echo "Options:"
    echo "  - Change DOCKER_NAME variable to use a different name"
    echo "  - Remove existing container: docker rm -f $DOCKER_NAME"
    echo "====================================================="
    exit 1
fi

docker run -itd  \
    -v ${HOST_WORKSPACE}:${DOCKER_WORKSPACE} \
    -v ${SSH_DIR}:/root/.ssh \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    --gpus all \
    --pid=host \
    --user=root \
    --cap-add=SYS_PTRACE \
    --privileged=true \
    --ipc=host \
    --network=host \
    --restart=always \
    --ulimit stack=${STACK_LIMIT} \
    --shm-size=${SHM_SIZE} \
    --name $DOCKER_NAME \
    -w=$DOCKER_WORKSPACE \
    $IMAGE_NAME:$IMAGE_VERSION \
    /bin/bash

    #-v /etc/localtime:/etc/localtime \
    #-v /etc/timezone:/etc/timezone \
