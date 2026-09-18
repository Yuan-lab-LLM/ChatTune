# Inference Ops

Inference Ops is a standalone vLLM operations and benchmark service. It provides controller and worker roles, vLLM instance lifecycle management, GPU and node resource management, logs, and benchmarks. The top-level `medflow/` directory contains retained legacy application code and is not a deployment or runtime dependency of Inference Ops.

## Configuration

Copy `config/agent.worker.example.yaml`, `config/agent.controller.example.yaml`, or `config/agent.both.example.yaml` to `config/agent.yaml`. `config/nodes.yaml` is empty by default; for a single-machine deployment, copy `config/nodes.single.example.yaml`, or for a multi-machine controller, copy `config/nodes.multi.example.yaml`. Then set:

```bash
export INFERENCE_OPS_AUTH_MODE=trusted_network
export INFERENCE_OPS_CONFIG_DIR=/etc/inference-ops
export INFERENCE_OPS_STATE_DIR=/var/lib/inference-ops
```

`trusted_network` is the default mode and does not require passing a token between containers. Use the container network, a firewall, or the listen address to ensure port 8899 is reachable only by trusted Studio, main Agent, controller, and worker processes. To enable Bearer authentication, explicitly set the same values for every caller and Inference Ops process:

```bash
export INFERENCE_OPS_AUTH_MODE=token
export INFERENCE_OPS_API_TOKEN='<long-random-token>'
```

Resource pool allocations are passed in each request's `resource_context`. Inference Ops maintains inference instances in its own state directory. It does not need access to a node Agent's resource API or a node resource API token.

Models and datasets are not bundled with the service. In `config/service.yaml`, point `MODEL_PATH`, `BENCHMARK_DIR`, and `GENERAL_BENCHMARK_DIR` to external directories mounted in the deployment environment. New instances use only `VLLM_OPENAI_PORT` and `MASTER_PORT`.

## Start

Prepare dependencies during deployment; service startup does not install Python or NPM dependencies. Base dependencies are declared in `requirements.txt`. Use `requirements-redis.txt` when using Redis checkpoints.

```bash
cd inference-ops
python -m inference_ops
```

Health checks do not require authentication. Example requests in the default trusted-network mode:

```bash
curl http://127.0.0.1:8899/healthz
curl http://127.0.0.1:8899/readyz
curl -X POST http://127.0.0.1:8899/inference_agent \
  -H "Content-Type: application/json" \
  -d '{"command":"查看推理状态","include_trace":true}'
```

In `token` mode, also add the `Authorization: Bearer $INFERENCE_OPS_API_TOKEN` header. Do not send a Bearer token in the default `trusted_network` mode; restrict access to the service through network isolation.

`scripts/start-service.sh` starts only `vllm serve`. Before a benchmark runs, it checks that the service is ready through the OpenAI-compatible `/v1/models` endpoint.

If `/v1/models` returns HTTP 500 and `vllm.log` reports `AttributeError: '_IncludedRouter' object has no attribute 'path'`, the endpoint is correct; the installed Prometheus FastAPI instrumentator cannot handle FastAPI's router. The start script loads `scripts/vllm_compat/sitecustomize.py` only in the vLLM process tree to support the installed FastAPI 0.141.1 and instrumentator 7.0.0 without changing container dependencies. The patch affects only route-name lookup in the metrics middleware. Stop existing instances and restart them with the updated script. If the patch fails to load, startup records the error and the workflow stops waiting.

## Legacy compatibility

An old `PROVIDER: medflow` configuration is normalized to vLLM-only configuration when read and written in the new format the next time it is saved. Schema v1 instances can still be viewed and stopped. The stop logic recognizes historical `inference`, `case2chat`, and `web/ui` PIDs so processes are not left running after an upgrade. New instances use schema v2; historical records are not rewritten automatically.

Stop any old functional-test jobs that are still running before upgrading. State-directory migration supports a preflight mode and an apply mode:

```bash
python scripts/migrate-state.py --legacy-state /old/log/root --target-state /var/lib/inference-ops
python scripts/migrate-state.py --legacy-state /old/log/root --target-state /var/lib/inference-ops --apply
```
