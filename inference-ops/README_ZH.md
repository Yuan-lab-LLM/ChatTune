# Inference Ops

Inference Ops 是独立的 vLLM 运维与 Benchmark 服务，提供 controller/worker、vLLM 实例生命周期、GPU/节点资源管理、日志和 Benchmark。顶层 `medflow/` 是暂存的旧业务目录，不是 Inference Ops 的部署或运行依赖。

## 配置

复制 `config/agent.worker.example.yaml`、`config/agent.controller.example.yaml` 或 `config/agent.both.example.yaml` 为 `config/agent.yaml`。`config/nodes.yaml` 默认为空；单机部署可复制 `config/nodes.single.example.yaml`，多机 controller 可复制 `config/nodes.multi.example.yaml`。然后设置：

```bash
export INFERENCE_OPS_AUTH_MODE=trusted_network
export INFERENCE_OPS_CONFIG_DIR=/etc/inference-ops
export INFERENCE_OPS_STATE_DIR=/var/lib/inference-ops
```

`trusted_network` 是默认模式，不需要在不同容器间传递 Token，但必须通过容器网络、防火墙或监听地址确保 8899 端口仅对可信 Studio、主 Agent、controller 和 worker 可达。需要 Bearer 鉴权时，在所有调用端和 Inference Ops 进程中显式设置相同配置：

```bash
export INFERENCE_OPS_AUTH_MODE=token
export INFERENCE_OPS_API_TOKEN='<long-random-token>'
```

资源池分配通过每次请求的 `resource_context` 传入。Inference Ops 在自己的状态目录中维护推理实例，不需要访问节点 Agent 的资源 API，也不需要节点资源 API token。

模型和数据集不随服务打包。请在 `config/service.yaml` 中把 `MODEL_PATH`、`BENCHMARK_DIR` 和 `GENERAL_BENCHMARK_DIR` 指向部署环境中的外部挂载目录。新实例只使用 `VLLM_OPENAI_PORT` 和 `MASTER_PORT`。

## 启动

依赖必须在部署阶段准备完成；服务启动过程不会安装 Python/NPM 依赖。基础依赖声明在 `requirements.txt`，使用 Redis checkpoint 时改用 `requirements-redis.txt`。

```bash
cd inference-ops
python -m inference_ops
```

健康检查无需鉴权。默认受信内网模式的调用示例：

```bash
curl http://127.0.0.1:8899/healthz
curl http://127.0.0.1:8899/readyz
curl -X POST http://127.0.0.1:8899/inference_agent \
  -H "Content-Type: application/json" \
  -d '{"command":"查看推理状态","include_trace":true}'
```

`token` 模式需额外添加 `Authorization: Bearer $INFERENCE_OPS_API_TOKEN` 请求头。默认 `trusted_network` 模式不发送 Bearer token；应通过网络隔离限制服务访问范围。

`scripts/start-service.sh` 只启动 `vllm serve`。Benchmark 在执行前通过 OpenAI-compatible `/v1/models` 接口确认服务就绪。

若 `/v1/models` 返回 500，且 `vllm.log` 出现 `AttributeError: '_IncludedRouter' object has no attribute 'path'`，是 FastAPI 路由与旧版 `prometheus-fastapi-instrumentator` 不兼容；该接口本身是正确的。启动脚本只给 vLLM 进程加载 `scripts/vllm_compat/sitecustomize.py`，兼容已安装的 FastAPI 0.141.1 与 instrumentator 7.0.0，无需修改容器依赖。补丁只处理指标中间件的路由名称查找，不改变 API 路由。已有实例需停止后用更新后的脚本重新启动；若补丁未生效，启动脚本会将实例标为失败并写明原因，工作流停止等待。

## 历史兼容

旧 `PROVIDER: medflow` 配置会在读取时归一化为 vLLM-only 配置，并在下次保存时写入新格式。schema v1 实例仍可查看和停止；停止逻辑会识别历史 `inference`、`case2chat`、`web/ui` PID，避免升级后遗留进程。新实例统一写入 schema v2，历史记录不会被自动改写。

升级前应先停止仍在运行的旧功能测试任务。状态目录迁移仍支持预检和执行模式：

```bash
python scripts/migrate-state.py --legacy-state /old/log/root --target-state /var/lib/inference-ops
python scripts/migrate-state.py --legacy-state /old/log/root --target-state /var/lib/inference-ops --apply
```
