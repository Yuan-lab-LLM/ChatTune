#! /bin/bash

# $OSIP             - system ip
# $WORK_PATH        - work path
# $NODE_TYPE        - server node type, maybe {master, slave}
# $CONTROLLER_IP    - controller ip, if master  node, this ip is osip 
# $DEVICES          - gpu card index for user

# source start-docker.sh $OSIP $WORK_PATH $NODE_TYPE $CONTROLLER_IP $CARD_START $CARD_END

# config
OSIP=replace-host-ip
MASTER_IP=replace-controller-ip
WORK_PATH=/data/llm
NODE_TYPE=master
#NODE_TYPE=slave

if [ "$NODE_TYPE" == "master" ] ;then
    CONTROLLER_IP=${OSIP}
else
    CONTROLLER_IP=${MASTER_IP}
fi

DEVICE_NUM=`nvidia-smi -L | wc -l`
DEVICES=4,5,6,7

# MODEL_NAMES="model_medical_20250514"
MODEL_NAMES=$(basename "$(ls -d ${WORK_PATH}/medical_models/*/ | sort -r | head -n 1)")
IMAGE_NAME="replace-inference-image"
IMAGE_VERSION="tag"

echo "run-docker config info"
echo "OSIP=${OSIP}"
echo "WORK_PATH=${WORK_PATH}"
echo "NODE_TYPE=${NODE_TYPE}"
echo "CONTROLLER_IP=${CONTROLLER_IP}"
echo "DEVICES=${DEVICES}"
echo "DEVICE_NUM=${DEVICE_NUM}"

source ./start-docker.sh ${OSIP} ${WORK_PATH} ${NODE_TYPE} ${CONTROLLER_IP} ${DEVICES} ${MODEL_NAMES} ${IMAGE_VERSION} ${IMAGE_NAME}
