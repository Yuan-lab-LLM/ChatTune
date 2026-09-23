# Deployment Guide

This guide only covers container, image, and workspace mount preparation for the MedFlow ChatTune release package. After completing this page, continue with [Quickstart](QUICKSTART.md) to initialize and start Runtime.

## Components

- `studio/`: Studio web frontend/backend for login, user management, resource management, and run records.
- `agent-studio-runtime-bridge/`: bridge service between Studio and Agent API backends.
- `agent/`: Agent API, workflow orchestration, resource probing, task routing, and internal resource APIs.
- `runtime.sh`: initialization, checks, start/stop commands, status, and log scripts.
- `docker_scripts/`: scripts for creating Agent, training, multi-node training, evaluation/inference, and GRPO containers.
- `inference-ops/`: independent vLLM operations controller/workers and benchmarks.
- `medflow/`: retained legacy application directory; it is not required by `inference-ops/`.

## Images and Resources

| Resource | Link | Notes |
| --- | --- | --- |
| Training image | https://pan.quark.cn/s/040388acdd9a | Extraction code: m9YH |
| Evaluation/inference image | https://pan.quark.cn/s/e4fe2f83f9c7 | Extraction code: QLw7 |
| GRPO/verl image | https://pan.quark.cn/s/c6ad41207360 | Extraction code: vJJx |
| Base model resource | https://modelscope.cn/models/MedFlow/Qingnang-32B-0630/ | Qingnang-32B-0630 base model |
| Evaluation benchmark data | https://pan.quark.cn/s/5ba6589d7933 | Extraction code: b8Pp |

### Training Image Patch Packages

Future training image updates will be released as patch packages. Download the corresponding `.run` package and install it with `bash xx.run`; recreating the full training image package is not required unless the release notes say otherwise.

| Resource | Link | Notes |
| --- | --- | --- |
| Training image patch package-0813 | https://pan.quark.cn/s/db69395cbde7 | Extraction code: 6K2h; LLaMA-Factory updated to 0.9.5 |
| Training image patch package-0820 | https://pan.quark.cn/s/609582f1f4c9 | Extraction code: 68uc; supports multi-node training |
| Training image patch package-0904 | https://pan.quark.cn/s/ab6dfd3a721a | Extraction code: 97TB; supports pretrain |
| Training image patch package-0904 | https://pan.quark.cn/s/f40b29e2e544 | Extraction code: dpEB;Supports general model evaluation and automatic recognition of model templates |

Current training packages expose their bundled LLaMA-Factory template registry through `/usr/local/insinfersystem/llamafactory-cli templates --json`. The Agent uses this machine-readable command to intersect the package capabilities with its platform allowlist. Legacy packages that return `Unknown command: templates` continue with the Agent compatibility allowlist and emit a warning; other command or schema failures block startup as a damaged runtime.


## Container Types

| Container | Location | Purpose |
| --- | --- | --- |
| Agent container | single machine, center node, or compute node | Mounts the code directory and runs Runtime, Agent API, resource probing, Studio, or Bridge. |
| General training task container | GPU compute node | Runs LoRA, full-parameter, DPO, and other standard training tasks. |
| Multi-node training task container | GPU compute node | Runs multi-node LoRA SFT, full-parameter SFT, and DPO enhanced training tasks. |
| Evaluation/inference task container | GPU compute node | Runs model evaluation and inference-service tasks. |
| GRPO/verl task container | GPU compute node | Runs GRPO training and related data/model operations. |

Before running the container creation scripts, replace their default parameters. Run these scripts with root privileges, for example as `root` or with `sudo`.

| Container | Script | Fields to replace |
| --- | --- | --- |
| Agent container | `docker_scripts/create_agent_docker.sh` | `AGENT_CONTAINER_NAME`, `AGENT_IMAGE`, `HOST_WORKSPACE` |
| General training task container | `docker_scripts/create_training_dockers.sh` | `HOST_IP`, `TRAIN_IMAGE`, `TRAIN_CONTAINER_NAME`, `HOST_WORKSPACE` |
| Multi-node training task container | `docker_scripts/create_multinode_training_dockers.sh` | `HOST_IP`, `TRAIN_IMAGE`, `TRAIN_CONTAINER_NAME`, `HOST_WORKSPACE` |
| Evaluation/inference task container | `docker_scripts/create_inference_docker.sh` | `HOST_IP`, `IMAGE_NAME`, `IMAGE_VERSION`, `DOCKER_NAME`, `HOST_WORKSPACE` |
| GRPO/verl task container | `docker_scripts/create_grpo_docker.sh` | `HOST_IP`, `GRPO_IMAGE`, `GRPO_CONTAINER_NAME`, `HOST_WORKSPACE` |

## Agent Container Mounts

When creating the container, `HOST_WORKSPACE` should point to this project code directory, that is, the MedFlow ChatTune release package/project root.

## General Training Container Mounts

When creating the container, the host directory mounted through `HOST_WORKSPACE` must use the following structure:

```text
<training-host-workspace>/
├── models
│   ├── base                        # base weights prepared for training
│   ├── batch_train                 # batch-training output weights
│   ├── pretrain                    # PT pretraining output weights
│   └── dpo_train                   # enhanced-training output weights
├── dataset                         # RAW/source datasets
├── dataset_daily_train             # DPO/enhanced-training datasets prepared for training
├── dataset_batch_train             # SFT/batch-training datasets prepared for training
├── dataset_pretrain                # PT/text pretraining datasets prepared for training
├── eval                            # generated evaluation results
└── log                             # generated log files
    ├── batch_train
    ├── pretrain
    └── dpo_train
```

### Training Data Format

The general training container uses the LLaMA-Factory training workflow. Put SFT/batch-training data under `dataset_batch_train`; Alpaca or ShareGPT format can be used. Put PT/text pretraining data under `dataset_pretrain`. Put DPO/enhanced-training data under `dataset_daily_train`; it uses preference data format and must include chosen and rejected responses.

### Dataset visibility and publication

The following container directories are the user-group shared dataset roots:

| Dataset type | Shared root |
| --- | --- |
| RAW/source data | `/home/workspace/dataset` |
| SFT/batch-training data | `/home/workspace/dataset_batch_train` |
| DPO/enhanced-training data | `/home/workspace/dataset_daily_train` |
| PT/text pretraining data | `/home/workspace/dataset_pretrain` |

Datasets placed under these roots and discovered by the resource node are registered and treated as shared data for the user group assigned to that container. Members of that group can list and execute them; publication does not grant access across user groups. Administrator uploads through Studio go directly to the corresponding shared root and are published immediately. Ordinary users should not bypass Studio and write to shared roots.

Datasets uploaded by an ordinary user and outputs produced by preprocessing or advanced filtering are written to that user's private directory by default:

```text
/home/workspace/medflow/users/<userId>/datasets/<type>/<datasetName>
```

A private dataset is visible, previewable, and executable only by its creator. To share it, the creator clicks **Request sharing** on the dataset card. An administrator approves it under **User Management → Publication Review**; the system then moves the dataset to its corresponding shared root and publishes it to that user group. Before approval, other group members cannot use it through the list, an explicit path, or automatic latest-dataset selection. Rejection leaves the dataset in the creator's private space.

### Resource catalog configuration

The default directories work without extra configuration. To replace a default or add a domain-specific directory, set the same `MEDFLOW_RESOURCE_CATALOG_CONFIG` JSON object for Studio and the corresponding Runtime node. You may specify only the entries that change:

```bash
MEDFLOW_RESOURCE_CATALOG_CONFIG='{
  "datasets": [
    {"id":"sft","path":"/home/workspace/dataset_batch_train"},
    {"id":"legal","path":"/home/workspace/dataset_legal","aliases":["law"],"dataTypes":["custom"]}
  ],
  "models": [
    {"id":"base","path":"/home/workspace/models/base"},
    {"id":"grpo","path":"/home/workspace/models/grpo_train"}
  ],
  "evaluation": [
    {"id":"evaluationDataset","path":"/home/workspace/tests"}
  ]
}'
```

Every directory must be an absolute path inside the container, cannot be `/`, and cannot contain `..`. New entries are returned through the existing `queryDatasets`, `queryModels`, or `queryGrpoResources` flow. Legacy types `base_train`, `batch_trained`, `pretrain`, `daily_trained`, and `inference` remain aliases. Restart Studio and Agent/Runtime after changing the value.

### Training capability manifest

The nine built-in training methods have one versioned source of truth: `agent/config/training_descriptors.v1.json`. Agent and Studio load the same file. The manifest currently requires `schemaVersion=1`; both loaders derive `requiredParams` and `resourceSelectors` from the parameter and resource definitions, so those fields do not belong in the manifest.

No additional configuration is needed when deploying the complete repository layout. If Studio and Agent are installed separately, set `MEDFLOW_TRAINING_DESCRIPTOR_PATH` to the absolute path of the same manifest in both environments:

```bash
MEDFLOW_TRAINING_DESCRIPTOR_PATH=/opt/chattune/config/training_descriptors.v1.json
```

The manifest is loaded once at process startup, so restart Studio and Agent/Runtime after changing it. A missing file, invalid JSON or schema version, conflicting descriptor ID/alias, or invalid parameter/resource field prevents startup; neither process silently falls back to an embedded copy.

The evaluation container must mount base/merged models or checkpoints and make both legacy medical paths and `/home/workspace/log/llamafactory_eval` plus `/home/workspace/eval/llamafactory` writable. Medical entrypoints retain the `legacyMedical` scope. The three `llamafactoryBuiltin` entrypoints require an executable `llamafactory-cli` and the six bundled evaluation directories under `/home/workspace/evaluation`. For a LoRA artifact selected through `CKPT_PATH`, the Agent follows `adapter_config.json.base_model_name_or_path`, resolves a template registered in the selected image, and passes it explicitly. Ambiguous cases ask the user for `TEM`. A registered `custom` resource remains management-only and cannot be executed by the general evaluator.

### Compatibility and rollback

- Capability queries add descriptors, generic classifications, and artifact fields without removing legacy fields.
- Legacy script names and natural-language aliases remain valid; `medicalTest` continues to map to `evaluationDataset`.
- Phase one removes no legacy fields. Deprecating an old alias requires at least two minor releases of compatibility observation and prior notice in release notes.
- To roll back a custom training manifest, remove `MEDFLOW_TRAINING_DESCRIPTOR_PATH` and restart Studio and Agent/Runtime to restore the repository default.
- To roll back a custom catalog, remove `MEDFLOW_RESOURCE_CATALOG_CONFIG` and restart Studio and Agent/Runtime. Existing datasets and models are not deleted automatically.

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

Training tasks use Weights & Biases to record metrics. After creating the general training container, multi-node training container, or GRPO/verl container and before the first online training run, log in inside the corresponding training container:

```bash
wandb login
```

Notes:

- W&B login information is stored in the current user configuration inside the container. Recreate the container means logging in again.
- If login fails, check network connectivity inside the container.

## Multi-node Training Container Preparation

The multi-node training container uses the same base mounts and creation flow as the general training container. After creation, besides running `wandb login`, install `openssh-server` and `pdsh` in every container that participates in multi-node training, then run `setup_multinode_ssh.sh` to configure sshd and node trust.

Before starting multi-node training, confirm that every participating machine's multi-node training Docker container has the same dataset and base model at the same container paths. For example, if the master node uses `/home/workspace/dataset_batch_train/my_sft` and `/home/workspace/models/base/qwen3`, the worker containers must also contain identical content at those exact paths; matching only the host path, or preparing the data/model on only one machine, is not sufficient.

Install SSH/pdsh dependencies and set the root password in both training containers first:

```bash
apt update
apt install openssh-server -y
apt install -y pdsh
passwd
```

Check whether pdsh passes the security check:

```bash
stat -c '%U:%G %a %n' /usr /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/pdsh
```

Normally these directories should be owned by `root:root` and must not be writable by group or other users. To fix them directly, run:

```bash
chown root:root /usr /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/pdsh
chmod go-w /usr /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/pdsh
```

Then prepare local sshd on both sides:

```bash
# 主节点容器
bash setup_multinode_ssh.sh \
  --user root \
  --self-node ds35 \
  --nodes ds35,ds36 \
  --hosts <ds35-host>,<ds36-host> \
  --ports 2222,2222 \
  --slots 4 \
  --prepare-only
```

```bash
# 从节点容器
bash setup_multinode_ssh.sh \
  --user root \
  --self-node ds36 \
  --nodes ds35,ds36 \
  --hosts <ds35-host>,<ds36-host> \
  --ports 2222,2222 \
  --slots 4 \
  --prepare-only
```

Then run the full mutual-trust setup on both sides:

```bash
bash setup_multinode_ssh.sh \
  --user root \
  --self-node ds35 \
  --nodes ds35,ds36 \
  --hosts <ds35-host>,<ds36-host> \
  --ports 2222,2222 \
  --slots 4
```

On the worker node, change `--self-node ds35` to `--self-node ds36`.

## Evaluation/Inference Container Mounts

When creating the container, the host directory mounted through `HOST_WORKSPACE` must use the following structure:

```text
<inference-host-workspace>/
├── inference-ops                   # inference-ops directory from this project
├── medical_models                  # models used for inference
└── tests                           # benchmark data supplied with the project
```

Inference Ops is a standalone vLLM operations and Benchmark service. Its configuration lives under `inference-ops/config/`. The top-level `medflow/` directory is retained legacy application code and is not a deployment dependency:

| File | Purpose |
| --- | --- |
| `service.yaml` | vLLM runtime parameters, including ports, GPUs, models, logs, and Benchmark paths. |
| `service.default.yaml` | Template used when restoring defaults. |
| `agent.yaml` | Inference Agent role, listen address, port, and LLM service configuration; empty by default. |
| `nodes.yaml` | Worker node list managed by the controller; empty by default. |
| `nodes.single.example.yaml` | Single-machine node example with node ID `local`. |
| `nodes.multi.example.yaml` | Multi-machine node example. |
| `agent.both.example.yaml` | Example for the single-machine `both` role. |
| `agent.controller.example.yaml` | Example for the controller role. |
| `agent.worker.example.yaml` | Example for the worker role. |

## Single-machine Inference

For single-machine deployment, or when the center node also manages the local inference service, the Inference Agent uses the `both` role and runs controller and worker on the same machine. Copy `agent.yaml` from `agent.both.example.yaml` and `nodes.yaml` from `nodes.single.example.yaml`.

Before starting the Inference Agent, update:

| File | Field | Replace with |
| --- | --- | --- |
| `agent.yaml` | `HOST` / `PORT` | Inference Agent listen address and port. |
| `agent.yaml` | `LLM_URL` / `LLM_MODEL` / `LLM_API_KEY` | OpenAI-compatible LLM URL, model name, and API key used by the Inference Agent. |
| `nodes.yaml` | `NODES.local.RESOURCE_NODE_ID` | Keep it as `local` so it matches the single-machine Runtime node ID. |
| `nodes.yaml` | `NODES.local.HOST` / `TOOL_URL` | Point to the local worker tool endpoint. |

`service.yaml` may be edited here in advance, or later by an administrator in Studio:

| File | Field | Replace with |
| --- | --- | --- |
| `service.yaml` | `ENV.HOST_IP` | Externally reachable IP of the current inference-service machine. |
| `service.yaml` | `ENV.CUDA_VISIBLE_DEVICES` | GPU IDs actually used by the local inference service, for example `0,1,2,3`. |
| `service.yaml` | `ENV.MODEL_NAME` / `ENV.MODEL_PATH` | Model directory name and model root that actually exist inside the container. |
| `service.yaml` | `PORTS.VLLM_OPENAI_PORT` / `ENV.MASTER_PORT` | Free local ports for the vLLM API and distributed master. |
| `service.yaml` | `RUNTIME.TENSOR_PARALLEL_SIZE` / `RUNTIME.GPU_MEMORY_UTILIZATION` / `RUNTIME.GPU_UTILIZATION_THRESHOLD` | vLLM and scheduler settings adjusted for GPU count, model size, and GPU memory. |

Single-machine `nodes.yaml` example:

```yaml
NODES:
  local:
    ENABLED: true
    NAME: local
    RESOURCE_NODE_ID: local
    HOST: 127.0.0.1
    TOOL_URL: http://127.0.0.1:8899/worker/tool
    ROLE: worker
```

## Multi-machine Inference

Multi-machine inference uses one controller and multiple workers. Before starting the controller Inference Agent, maintain the controller `agent.yaml` and `nodes.yaml`; before starting each worker Inference Agent, maintain that worker's `agent.yaml`. Copy controller configuration from `agent.controller.example.yaml` and `nodes.multi.example.yaml`, and worker configuration from `agent.worker.example.yaml`.

Before starting the Inference Agent, update:

| File | Field | Replace with |
| --- | --- | --- |
| Controller `nodes.yaml` | `NODES.<node>.HOST` | IP or domain of each worker reachable by the controller. |
| Controller `nodes.yaml` | `NODES.<node>.TOOL_URL` | Tool endpoint of each worker. |
| Controller `nodes.yaml` | `NODES.<node>.RESOURCE_NODE_ID` | Node ID of the corresponding Runtime worker. |
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
    RESOURCE_NODE_ID: node-a
    HOST: <worker-a-host>
    TOOL_URL: http://<worker-a-host>:8899/worker/tool
    ROLE: worker

  node1:
    ENABLED: true
    NAME: node1
    RESOURCE_NODE_ID: node-b
    HOST: <worker-b-host>
    TOOL_URL: http://<worker-b-host>:8899/worker/tool
    ROLE: worker
```

Each `TOOL_URL` must point to the corresponding worker's `/worker/tool` endpoint. Disable unused workers by setting `ENABLED: false` or deleting the entry.

## Inference Agent Admin Permission

The Inference Agent `/admin/*` endpoints are maintenance APIs for stale-resource cleanup and forced service, benchmark, or test stops. They should be triggered only through Studio administrator actions: the browser never sends an admin token, and Studio Server first verifies the logged-in user is an administrator before calling the Inference Agent.

Enable these endpoints with `ADMIN.ENABLED: true` in `agent.yaml`. `INFERENCE_OPS_AUTH_MODE` defaults to `trusted_network`, which disables Bearer-token checks and therefore requires network or firewall isolation so that only trusted Studio Server, main Agent, controller, and worker processes can connect; any process that reaches the port can call `/admin/*` directly. To enable service authentication, explicitly set `INFERENCE_OPS_AUTH_MODE=token` and the same `INFERENCE_OPS_API_TOKEN` in every caller and Inference Ops process. Studio still authorizes the logged-in administrator in both modes.

## Start the Inference Agent

```bash
cd inference-ops
python -m inference_ops
```

After startup, verify the controller endpoint from the Runtime center node:

```bash
curl -X POST http://<worker-a-host>:8899/inference_agent \
  -H "Content-Type: application/json" \
  -d '{"command":"查看推理状态","include_trace":true}'
```

The example above uses the default `trusted_network` mode. In `token` mode, also add `-H "Authorization: Bearer $INFERENCE_OPS_API_TOKEN"`. Resource-pool allocations travel in each request's `resource_context`; Inference Ops owns its process state and does not need a node resource API token.

If the response reports model-call errors, check whether `AGENT_LLM_URL` or `LLM_URL` points to a reachable OpenAI-compatible model service, and whether `AGENT_LLM_MODEL` / `LLM_MODEL` and `AGENT_LLM_API_KEY` / `LLM_API_KEY` match that service. If tool execution fails, check worker nodes, `nodes.yaml`, inference-service configuration, and port connectivity.
