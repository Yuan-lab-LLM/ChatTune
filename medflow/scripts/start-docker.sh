#! /bin/bash

osip=$1
host_patch=$2

# master or slave
node_tpye=$3
# master control ip
controller_ip=$4

devices=$5
model_names=$6
image_version=$7
image_name=$8

require_configured() {
    local name="$1"
    local value="${!name:-}"
    if [[ -z "$value" || "$value" == replace-* || "$value" == "tag" ]]; then
        echo "Missing or placeholder value for $name. Edit medflow/scripts/run-docker.sh before running it." >&2
        exit 1
    fi
}

require_configured image_name
require_configured image_version

docker_prefix=medflow-inference
docker_name=${docker_prefix}-${node_tpye}-${osip}
user_name=user-${osip}
docker_patch=/home/workspace
models_patch=${docker_patch}/medical_models/

echo osip=$osip
echo host_patch=$host_patch
echo docker_name=$docker_name
echo user_name=$user_name

docker_run=`docker ps -a | grep $docker_name`
if [  "$docker_run" ]; then
    echo "need docker rm -f $docker_name"
    docker rm -f $docker_name
fi

docker run -itd  \
    -v ${host_patch}:${docker_patch} \
    -v /root/.ssh:/root/.ssh \
    -v /etc/localtime:/etc/localtime \
    -v /etc/timezone:/etc/timezone \
    -e IEI_VISIBLE_MODELS_PATH=$models_patch \
    -e IEI_VISIBLE_OS_IP=$osip \
    -e IEI_VISIBLE_OS_USER_NAME=$user_name \
    -e IEI_VISIBLE_OS_DOCKER_NAME=$docker_name \
    -e IEI_VISIBLE_NODE_TYPE=$node_tpye \
    -e IEI_VISIBLE_CONTROLLER_IP=$controller_ip \
    -e IEI_VISIBLE_DEVICES=$devices    \
    -e IEI_VISIBLE_MODELS_NAME=$model_names    \
    --gpus all \
    --pid=host \
    --user=root \
    --cap-add=SYS_PTRACE \
    --privileged=true \
    --name $docker_name \
    --ipc=host \
    --network=host \
    --restart=always \
    --ulimit stack=68719476736 \
    --shm-size=100G \
    -w=$docker_patch \
    $image_name:$image_version \
    /bin/bash
