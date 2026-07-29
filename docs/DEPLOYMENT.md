# Deployment Guide

This guide only covers container, image, and workspace mount preparation for the MedFlow Runtime MP release package. After completing this page, continue with [Quickstart](QUICKSTART.md) to initialize and start Runtime.

## Components

- `studio/`: Studio web frontend/backend for login, user management, resource management, and run records.
- `agent-studio-runtime-bridge/`: bridge service between Studio and Agent API backends.
- `agent/`: Agent API, workflow orchestration, resource probing, task routing, and internal resource APIs.
- `runtime.sh`: initialization, checks, start/stop commands, status, and log scripts.
- `docker_scripts/`: scripts for creating Agent, training, evaluation/inference, and GRPO containers.
- `medflow/`: MedFlow inference services, inference operations Agent, tests, and benchmark code.

## Images and Resources

| Resource | Link | Notes |
| --- | --- | --- |
| Training image | https://pan.quark.cn/s/040388acdd9a | Extraction code: m9YH |
| Evaluation/inference image | https://pan.quark.cn/s/7ef603e26e50 | Extraction code: SH8t |
| GRPO/verl image | https://pan.quark.cn/s/c6ad41207360 | Extraction code: vJJx |
| Base model resource | https://modelscope.cn/models/MedFlow/Qingnang-32B-0630/ | Qingnang-32B-0630 base model |
| Evaluation benchmark data | https://pan.quark.cn/s/5ba6589d7933 | Extraction code: b8Pp |

## Container Types

| Container | Location | Purpose |
| --- | --- | --- |
| Agent container | single machine, center node, or compute node | Mounts the code directory and runs Runtime, Agent API, resource probing, Studio, or Bridge. |
| General training task container | GPU compute node | Runs LoRA, full-parameter, DPO, and other standard training tasks. |
| Evaluation/inference task container | GPU compute node | Runs model evaluation and inference-service tasks. |
| GRPO/verl task container | GPU compute node | Runs GRPO training and related data/model operations. |

Before running the container creation scripts, replace their default parameters. Run these scripts with root privileges, for example as `root` or with `sudo`.

| Container | Script | Fields to replace |
| --- | --- | --- |
| Agent container | `docker_scripts/create_agent_docker.sh` | `AGENT_CONTAINER_NAME`, `AGENT_IMAGE`, `HOST_WORKSPACE` |
| General training task container | `docker_scripts/create_training_dockers.sh` | `HOST_IP`, `TRAIN_IMAGE`, `TRAIN_CONTAINER_NAME`, `HOST_WORKSPACE` |
| Evaluation/inference task container | `docker_scripts/create_inference_docker.sh` | `HOST_IP`, `IMAGE_NAME`, `IMAGE_VERSION`, `DOCKER_NAME`, `HOST_WORKSPACE` |
| GRPO/verl task container | `docker_scripts/create_grpo_docker.sh` | `HOST_IP`, `GRPO_IMAGE`, `GRPO_CONTAINER_NAME`, `HOST_WORKSPACE` |

## Agent Container Mounts

When creating the container, `HOST_WORKSPACE` should point to this project code directory, that is, the MedFlow Runtime MP release package/project root.

## General Training Container Mounts

When creating the container, the host directory mounted through `HOST_WORKSPACE` must use the following structure:

```text
<training-host-workspace>/
├── models
│   ├── base                        # base weights prepared for training
│   ├── batch_train                 # batch-training output weights
│   └── dpo_train                   # enhanced-training output weights
├── dataset_daily_train             # DPO/enhanced-training datasets prepared for training
├── dataset_batch_train             # SFT/batch-training datasets prepared for training
├── eval                            # generated evaluation results
└── log                             # generated log files
    ├── batch_train
    └── dpo_train
```

### Training Data Format

The general training container uses the LLaMA-Factory training workflow. Put SFT/batch-training data under `dataset_batch_train`; Alpaca or ShareGPT format can be used. Put DPO/enhanced-training data under `dataset_daily_train`; it uses preference data format and must include chosen and rejected responses.

## GRPO/verl Container Mounts

When creating the container, the host directory mounted through `HOST_WORKSPACE` must use the following structure:

```text
<host-workspace>/
├── verl/examples/data_preprocess/data
│   ├── train                       # GRPO training set
│   └── test                        # GRPO test/evaluation set
└── models/grpo_train               # GRPO base/output model workspace
```

### Note

Training tasks use Weights & Biases to record metrics. After creating the two training containers above and before the first online training run, log in inside the training container:

```bash
wandb login
```

Notes:

- W&B login information is stored in the current user configuration inside the container. Recreate the container means logging in again.
- If login fails, check network connectivity inside the container.

## Evaluation/Inference Container Mounts

When creating the container, the host directory mounted through `HOST_WORKSPACE` must use the following structure:

```text
<inference-host-workspace>/
├── medflow                         # MedFlow directory from this project
├── medical_models                  # models used for inference
└── tests                           # evaluation benchmark data from this project
```

MedFlow inference service configuration lives under `medflow/agent/config/`:

| File | Purpose |
| --- | --- |
| `service.yaml` | Runtime parameters for the inference service stack, including ports, GPUs, models, logs, test sets, and benchmark paths. |
| `service.default.yaml` | Template used when restoring defaults. |
| `agent.yaml` | Inference Agent role, listen address, port, and LLM service configuration. |
| `nodes.yaml` | Worker node list managed by the controller. |
| `agent.both.example.yaml` | Example for the single-machine `both` role. |
| `agent.controller.example.yaml` | Example for the controller role. |
| `agent.worker.example.yaml` | Example for the worker role. |

## Single-machine Inference

For single-machine deployment, or when the center node also manages the local inference service, the Inference Agent uses the `both` role and runs controller and worker on the same machine. If `agent.yaml` does not exist, copy it from `agent.both.example.yaml`.

Before starting the Inference Agent, update:

| File | Field | Replace with |
| --- | --- | --- |
| `agent.yaml` | `HOST` / `PORT` | Inference Agent listen address and port. |
| `agent.yaml` | `LLM_URL` / `LLM_MODEL` / `LLM_API_KEY` | OpenAI-compatible LLM URL, model name, and API key used by the Inference Agent. |
| `nodes.yaml` | `NODES.<node>.HOST` / `TOOL_URL` | In single-machine mode, usually points to the local worker tool endpoint. |

`service.yaml` may be edited here in advance, or later by an administrator in Studio:

| File | Field | Replace with |
| --- | --- | --- |
| `service.yaml` | `ENV.HOST_IP` | Externally reachable IP of the current inference-service machine. |
| `service.yaml` | `ENV.CUDA_VISIBLE_DEVICES` | GPU IDs actually used by the local inference service, for example `0,1,2,3`. |
| `service.yaml` | `ENV.MODEL_NAME` / `ENV.MODEL_PATH` | Model directory name and model root that actually exist inside the container. |
| `service.yaml` | `PORTS.VLLM_OPENAI_PORT` / `PORTS.INFERENCE_PORT` / `PORTS.UI_PORT` / `PORTS.DATA_ANNOTATION_PORT` | Free local ports that can be exposed as needed. |
| `service.yaml` | `RUNTIME.TENSOR_PARALLEL_SIZE` / `RUNTIME.GPU_MEMORY_UTILIZATION` / `RUNTIME.MAX_TOKENS` | vLLM settings adjusted for GPU count, model size, and GPU memory. |

Single-machine `nodes.yaml` example:

```yaml
NODES:
  main:
    ENABLED: true
    NAME: main
    HOST: <worker-a-host>
    TOOL_URL: http://<worker-a-host>:8899/worker/tool
    ROLE: worker
```

## Multi-machine Inference

Multi-machine inference uses one controller and multiple workers. Before starting the controller Inference Agent, maintain the controller `agent.yaml` and `nodes.yaml`; before starting each worker Inference Agent, maintain that worker's `agent.yaml`. Copy controller configuration from `agent.controller.example.yaml` and worker configuration from `agent.worker.example.yaml`.

Before starting the Inference Agent, update:

| File | Field | Replace with |
| --- | --- | --- |
| Controller `nodes.yaml` | `NODES.<node>.HOST` | IP or domain of each worker reachable by the controller. |
| Controller `nodes.yaml` | `NODES.<node>.TOOL_URL` | Tool endpoint of each worker. |
| Each worker's `agent.yaml` | `ROLE` / `HOST` / `PORT` / `LLM_URL` / `LLM_MODEL` / `LLM_API_KEY` | Worker role, listen address, port, and LLM service configuration used when local LLM capability is needed. |

Each worker's `service.yaml` may be edited here in advance, or later by an administrator in Studio:

| File | Field | Replace with |
| --- | --- | --- |
| Each worker's `service.yaml` | `ENV.HOST_IP` | IP or domain of that worker reachable by the controller. |
| Each worker's `service.yaml` | `ENV.CUDA_VISIBLE_DEVICES` | GPU IDs actually used by that worker. |
| Each worker's `service.yaml` | `ENV.MODEL_NAME` / `ENV.MODEL_PATH` | Model directory name and model root that actually exist inside that worker container. |
| Each worker's `service.yaml` | `PORTS.*` | Free local ports allowed for controller/operations access; open firewalls for cross-machine access. |
| Each worker's `service.yaml` | `RUNTIME.*` | vLLM settings adjusted for that worker's GPU count, model size, and GPU memory. |

Example `nodes.yaml`:

```yaml
NODES:
  main:
    ENABLED: true
    NAME: main
    HOST: <worker-a-host>
    TOOL_URL: http://<worker-a-host>:8899/worker/tool
    ROLE: worker

  node1:
    ENABLED: true
    NAME: node1
    HOST: <worker-b-host>
    TOOL_URL: http://<worker-b-host>:8899/worker/tool
    ROLE: worker
```

Each `TOOL_URL` must point to the corresponding worker's `/worker/tool` endpoint. Disable unused workers by setting `ENABLED: false` or deleting the entry.

## Start the Inference Agent

```bash
cd medflow/agent/agent
python inference_agent.py
```

After startup, verify the controller endpoint from the Runtime center node:

```bash
curl -X POST http://<worker-a-host>:8899/inference_agent \
  -H "Content-Type: application/json" \
  -d '{"command":"查看推理状态","include_trace":true}'
```

If the response reports model-call errors, check whether `AGENT_LLM_URL` or `LLM_URL` points to a reachable OpenAI-compatible model service, and whether `AGENT_LLM_MODEL` / `LLM_MODEL` and `AGENT_LLM_API_KEY` / `LLM_API_KEY` match that service. If tool execution fails, check worker nodes, `nodes.yaml`, inference-service configuration, and port connectivity.

