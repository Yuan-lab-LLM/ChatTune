# Multi-node Configuration

This guide covers Runtime configuration for center nodes, compute nodes, and multi-node deployments. Prepare containers and workspace mounts first with [Deployment Guide](DEPLOYMENT.md). For the single-machine startup flow, see [Quickstart](QUICKSTART.md).

- Center node: runs Studio, Bridge, and optionally the local Agent API.
- Compute node: runs the local Agent API, resource probing service, and task containers.

The center node must be able to reach each compute node's `AGENT_API_PORT`. Each compute node must be able to reach the center node's Studio backend URL.

Every machine that participates in multi-node training must be initialized as a Runtime node with Agent API/resource probing enabled, and must have the dedicated multi-node training container configured through `MULTINODE_DOCKER_CONTAINER`. Deploying Agent only on the master node is not enough because Studio must query GPU snapshots, reserve GPU indexes, and manage the allocation for every participating node.

## Prerequisites

- Linux runtime environment for `runtime.sh`.
- Bash, `setsid`, and `curl`.
- Python 3.10 or newer.
- Node.js 20 or newer and npm 10 or newer.
- Docker and pre-created task containers for training, evaluation/inference, multi-node training, and GRPO/verl.
- OpenAI-compatible model service if Agent or inference commands need LLM calls.

## Install Dependencies

Before initialization, install dependencies in the Agent container:

```bash
python -m pip install -r agent/requirements.txt

cd studio
apt update
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs
npm ci

cd ../agent-studio-runtime-bridge
npm ci
```

## Initialize the Center Node

Run `init --profile center` on the center node and pass only the non-center compute-node Agent API URLs reachable from the center through `--nodes`. Center initialization also generates and starts the center-local Agent node `center` by default, so do not add the center node's own IP as a regular compute node in `--nodes`:

```bash
bash runtime.sh init --profile center --nodes node-b=http://<worker-b-host>:8099
```

## Fields to Edit After Generation

| Field | Value |
| --- | --- |
| `MEDFLOW_LOCAL_TRAINING_CONTAINER` | Training container name that exists on the center machine, used for center-local Agent standard training tasks. |
| `MULTINODE_DOCKER_CONTAINER` | Dedicated LLaMAFactory container name that exists on the center machine, used for multi-node LoRA SFT and DPO enhanced training. |
| `MEDFLOW_LOCAL_EVALUATE_CONTAINER` | Evaluation/inference container name that exists on the center machine, used for center-local Agent evaluation and inference-related tasks. |
| `MEDFLOW_LOCAL_GRPO_CONTAINER` | GRPO/verl container name that exists on the center machine, used for center-local Agent GRPO/verl training tasks. |
| `MODEL_NAME` / `MODEL_API_KEY` / `MODEL_BASE_URL` | OpenAI-compatible model name, API key, and service URL used by the center-local Agent. |
| `INFERENCE_AGENT_URL` | Inference Agent controller `/inference_agent` URL used by the center-local Agent. Studio administrator maintenance actions use the Studio login session; restrict Inference Agent `/admin/*` to trusted Studio Server access if enabled. |
| `STUDIO_PUBLIC_URL` | Center Studio URL opened by users in a browser. |
| `STUDIO_URL` | Center Studio backend URL used by Runtime/Agent callbacks. |

## Initialize Compute Nodes

Run `init --profile node` on each compute node. First record the shared tokens and the node-specific tokens from the center output, then run this on the corresponding compute node:

```bash
bash runtime.sh init --profile node --node-id node-a --center-url http://<center-host>:3000 --agent-url http://<worker-a-host>:8099
```

## Compute Node Fields

| Field | Value |
| --- | --- |
| `MEDFLOW_LOCAL_TRAINING_CONTAINER` | Training container name that exists on this node, used for standard training tasks. |
| `MULTINODE_DOCKER_CONTAINER` | Dedicated LLaMAFactory container name that exists on this node, used for multi-node LoRA SFT and DPO enhanced training. |
| `MEDFLOW_LOCAL_EVALUATE_CONTAINER` | Evaluation/inference container name that exists on this node, used for evaluation and inference-related tasks. |
| `MEDFLOW_LOCAL_GRPO_CONTAINER` | GRPO/verl container name that exists on this node, used for GRPO/verl training tasks. |
| `MEDFLOW_STUDIO_RUNTIME_TOKEN` | Shared Studio Runtime token copied from the center node. |
| `MEDFLOW_AGENT_API_TOKEN` | Shared Agent API token copied from the center node. |
| `MEDFLOW_RESOURCE_API_TOKEN` | Resource API token for this node from the center output; must match this node's `resourceApiToken` in center `MEDFLOW_RESOURCE_NODES`. |
| `MEDFLOW_RUNTIME_NODE_TOKEN` | Runtime node token for this node from the center output; must match this node's value in center `MEDFLOW_STUDIO_RUNTIME_NODE_TOKENS`. |
| `MODEL_NAME` / `MODEL_API_KEY` / `MODEL_BASE_URL` | OpenAI-compatible model name, API key, and service URL used by this node's Agent. |
| `INFERENCE_AGENT_URL` | Inference Agent controller `/inference_agent` URL used by tasks on this node. Studio administrator maintenance actions use the Studio login session; restrict Inference Agent `/admin/*` to trusted Studio Server access if enabled. |

## Startup Order

For two-machine or multi-machine deployments, start the center node first, then start each compute node:

```bash
bash runtime.sh check
bash runtime.sh start
bash runtime.sh status
```

`bash runtime.sh check` detects mismatches between recommended and compatibility fields, unreplaced placeholders, token reuse, current-node token mismatches against the center-node token map, center-local Agent resource token mismatches against `MEDFLOW_RESOURCE_NODES`, local container field mismatches, missing dependencies, and port conflicts.

## Common Operations

| Command | Purpose |
| --- | --- |
| `bash runtime.sh check` | Check dependencies, environment variables, ports, and Docker configuration. |
| `bash runtime.sh start` | Start services. |
| `bash runtime.sh stop` | Stop services. |
| `bash runtime.sh restart` | Restart services and reload `runtime.env`. |
| `bash runtime.sh status` | Show service status and PIDs. |
| `bash runtime.sh logs` | Show all logs. |
| `bash runtime.sh logs agent` | Show Agent logs; `studio` and `bridge` are also supported. |
| `bash runtime.sh --help` | Show all options. |

Runtime data is stored under `.runtime/` by default:

```text
.runtime/
  data/
  logs/
  agent.pid
  bridge.pid
  studio.pid
```

Key files:

- `.runtime/data/database.sqlite`: default Studio SQLite database.
- `.runtime/logs/agent.log`: Agent API log.
- `.runtime/logs/bridge.log`: Bridge log.
- `.runtime/logs/studio.log`: Studio log.
- `.runtime/*.pid`: service process IDs recorded by `runtime.sh`.

If `check` fails, fix the reported dependency, port, token, container, or placeholder issue before running `start` again.
