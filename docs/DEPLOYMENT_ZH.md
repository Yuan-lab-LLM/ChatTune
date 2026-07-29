# 部署指南

本文只说明 MedFlow Runtime MP 发布包中的容器、镜像和工作区挂载准备。完成本页后，再进入 [快速开始](QUICKSTART_ZH.md) 初始化并启动 Runtime。

## 组件

- `studio/`：Studio Web 前后端，负责登录、用户管理、资源管理和运行记录。
- `agent-studio-runtime-bridge/`：Studio 与 Agent API 后端之间的桥接服务。
- `agent/`：Agent API、工作流编排、资源探测、任务路由和内部资源 API。
- `runtime.sh`：初始化、检查、启动、停止、状态和日志脚本。
- `docker_scripts/`：Agent、训练、评测/推理和 GRPO 容器创建脚本。
- `medflow/`：MedFlow 推理服务、推理运维 Agent、测试和 Benchmark 相关代码。

## 镜像和资源

| 资源 | 链接 | 备注 |
| --- | --- | --- |
| 训练镜像 | https://pan.quark.cn/s/040388acdd9a | 提取码：m9YH |
| 评测/推理镜像 | https://pan.quark.cn/s/7ef603e26e50 | 提取码：SH8t |
| GRPO/verl 镜像 | https://pan.quark.cn/s/c6ad41207360 | 提取码：vJJx |
| 基础模型资源 | https://modelscope.cn/models/MedFlow/Qingnang-32B-0630/ | Qingnang-32B-0630 基础模型 |
| 评测 Benchmark 数据 | https://pan.quark.cn/s/5ba6589d7933 | 提取码：b8Pp |

## 容器类型 

| 容器 | 部署位置 | 作用 |
| --- | --- | --- |
| Agent 容器 | 单机、中心节点或计算节点 | 挂载代码目录，运行 Runtime、Agent API、资源探测、Studio 或 Bridge。 |
| 通用训练任务容器 | GPU 计算节点 | 执行 LoRA、全参、DPO 等常规训练任务。 |
| 评测/推理任务容器 | GPU 计算节点 | 执行模型评测、推理服务相关任务。 |
| GRPO/verl 任务容器 | GPU 计算节点 | 执行 GRPO 训练及相关数据/模型操作。 |

运行生成容器的脚本之前，需要替换脚本中的默认参数。请使用 root 权限运行这些脚本。

| 容器 | 脚本 | 需要替换的字段 |
| --- | --- | --- |
| Agent 容器 | `docker_scripts/create_agent_docker.sh` | `AGENT_CONTAINER_NAME`、`AGENT_IMAGE`、`HOST_WORKSPACE` |
| 通用训练任务容器 | `docker_scripts/create_training_dockers.sh` | `HOST_IP`、`TRAIN_IMAGE`、`TRAIN_CONTAINER_NAME`、`HOST_WORKSPACE` |
| 评测/推理任务容器 | `docker_scripts/create_inference_docker.sh` | `HOST_IP`、`IMAGE_NAME`、`IMAGE_VERSION`、`DOCKER_NAME`、`HOST_WORKSPACE` |
| GRPO/verl 任务容器 | `docker_scripts/create_grpo_docker.sh` | `HOST_IP`、`GRPO_IMAGE`、`GRPO_CONTAINER_NAME`、`HOST_WORKSPACE` |

## Agent 容器挂载

容器创建时， `HOST_WORKSPACE` 应指向本项目代码目录，也就是 MedFlow Runtime MP 发布包/项目根目录。

## 通用训练容器挂载

容器创建时，HOST_WORKSPACE 所挂载的宿主机目录结构要求如下：

```text
<training-host-workspace>/
├── models
│   ├── base                        # 准备训练的基础权重文件
│   ├── batch_train                 # 存放批量训练完成的权重文件
│   └── dpo_train                   # 存放增强训练完成的权重文件
├── dataset_daily_train             # 准备训练的DPO/增强训练数据集
├── dataset_batch_train             # 准备训练的SFT/批量训练数据集
├── eval                            # 存放生成的评测结果
└── log                             # 存放生成的日志文件
    ├── batch_train
    └── dpo_train
```

### 训练数据格式

通用训练容器使用 LLaMA-Factory 训练流程。SFT/批量训练数据放在 `dataset_batch_train` 下，可使用 Alpaca 或 ShareGPT 格式。DPO/增强训练数据放在 `dataset_daily_train` 下，使用偏好数据格式，必须包含被选择回答和被拒绝回答。

## GRPO/verl 容器挂载

容器创建时，HOST_WORKSPACE 所挂载的宿主机目录结构要求如下：

```text
<host-workspace>/
├── verl/examples/data_preprocess/data
│   ├── train                       # GRPO 训练集
│   └── test                        # GRPO 测试/评测集
└── models/grpo_train               # GRPO 基础模型和输出模型工作区
```

### 注意

训练任务使用 Weights & Biases 记录训练指标。创建上述两个训练容器后、开始首次在线训练前，必须在训练容器中完成 W&B 登录：

```bash
wandb login
```

注意事项：

- W&B 登录信息保存在容器内当前用户的配置中；删除或重新创建训练容器后需要重新登录。
- 登录失败时，在容器中检查网络连通性。

## 评测/推理容器挂载

容器创建时，HOST_WORKSPACE 所挂载的宿主机目录结构要求如下：

```text
<inference-host-workspace>/
├── medflow                         # 本项目提供的 MedFlow 文件夹
├── medical_models                  # 用于推理的模型
└── tests                           # 本项目提供的评测 Benchmark 数据
```

MedFlow 推理服务配置位于 `medflow/agent/config/`：

| 文件 | 作用 |
| --- | --- |
| `service.yaml` | 推理服务栈运行参数，包括端口、GPU、模型、日志、测试集和 Benchmark 路径。 |
| `service.default.yaml` | 恢复默认配置时使用的模板。 |
| `agent.yaml` | 推理 Agent 的角色、监听地址、端口和 LLM 服务配置。 |
| `nodes.yaml` | controller 管理的 worker 节点列表。 |
| `agent.both.example.yaml` | 单机 both 角色示例。 |
| `agent.controller.example.yaml` | controller 角色示例。 |
| `agent.worker.example.yaml` | worker 角色示例。 |

## 单机推理配置

单机部署或中心节点同时管理本机推理服务时，推理 Agent 使用 `both` 角色，controller 和 worker 在同一台机器上运行。如果还没有 `agent.yaml`，可以从 `agent.both.example.yaml` 复制一份。

推理 Agent 启动前必须修改：

| 文件 | 字段 | 使用前替换为 |
| --- | --- | --- |
| `agent.yaml` | `HOST` / `PORT` | 推理 Agent 监听地址和端口。 |
| `agent.yaml` | `LLM_URL` / `LLM_MODEL` / `LLM_API_KEY` | 推理 Agent 调用的 OpenAI-compatible LLM 地址、模型名称和 API key。 |
| `nodes.yaml` | `NODES.<node>.HOST` / `TOOL_URL` | 单机模式通常指向本机 worker 工具接口。 |

`service.yaml` 可在本文件中预先修改，也可由管理员在页面修改：

| 文件 | 字段 | 使用前替换为 |
| --- | --- | --- |
| `service.yaml` | `ENV.HOST_IP` | 当前推理服务所在机器对外可访问的 IP。 |
| `service.yaml` | `ENV.CUDA_VISIBLE_DEVICES` | 本机推理服务实际使用的 GPU 编号，例如 `0,1,2,3`。 |
| `service.yaml` | `ENV.MODEL_NAME` / `ENV.MODEL_PATH` | 容器内实际存在的模型目录名和模型根目录。 |
| `service.yaml` | `PORTS.VLLM_OPENAI_PORT` / `PORTS.INFERENCE_PORT` / `PORTS.UI_PORT` / `PORTS.DATA_ANNOTATION_PORT` | 本机未占用且可按需暴露的服务端口。 |
| `service.yaml` | `RUNTIME.TENSOR_PARALLEL_SIZE` / `RUNTIME.GPU_MEMORY_UTILIZATION` / `RUNTIME.MAX_TOKENS` | 按本机 GPU 数、模型规模和显存容量调整的 vLLM 参数。 |

单机 `nodes.yaml` 示例：

```yaml
NODES:
  main:
    ENABLED: true
    NAME: main
    HOST: <worker-a-host>
    TOOL_URL: http://<worker-a-host>:8899/worker/tool
    ROLE: worker
```

## 多机推理配置

多机推理使用一个 controller 和多个 worker。controller 启动推理 Agent 前必须维护自己的 `agent.yaml` 和 `nodes.yaml`；每台 worker 启动推理 Agent 前必须维护自己的 `agent.yaml`。controller 可从 `agent.controller.example.yaml` 复制配置，worker 可从 `agent.worker.example.yaml` 复制配置。

推理 Agent 启动前必须修改：

| 文件 | 字段 | 使用前替换为 |
| --- | --- | --- |
| controller 的 `nodes.yaml` | `NODES.<node>.HOST` | 每个 worker 对 controller 可访问的 IP 或域名。 |
| controller 的 `nodes.yaml` | `NODES.<node>.TOOL_URL` | 每个 worker 的工具接口地址。 |
| 每台 worker 的 `agent.yaml` | `ROLE` / `HOST` / `PORT` / `LLM_URL` / `LLM_MODEL` / `LLM_API_KEY` | worker 角色、监听地址、端口和需要本地 LLM 能力时使用的 LLM 服务配置。 |

每台 worker 的 `service.yaml` 可在本文件中预先修改，也可由管理员在页面修改：

| 文件 | 字段 | 使用前替换为 |
| --- | --- | --- |
| 每台 worker 的 `service.yaml` | `ENV.HOST_IP` | 该 worker 对 controller 可访问的 IP 或域名。 |
| 每台 worker 的 `service.yaml` | `ENV.CUDA_VISIBLE_DEVICES` | 该 worker 实际使用的 GPU 编号。 |
| 每台 worker 的 `service.yaml` | `ENV.MODEL_NAME` / `ENV.MODEL_PATH` | 该 worker 容器内实际存在的模型目录名和模型根目录。 |
| 每台 worker 的 `service.yaml` | `PORTS.*` | 该 worker 本机未占用且允许 controller/运维访问的服务端口；跨机器访问时需放通防火墙。 |
| 每台 worker 的 `service.yaml` | `RUNTIME.*` | 按该 worker 的 GPU 数、模型规模和显存容量调整的 vLLM 参数。 |

`nodes.yaml` 示例：

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

每个 `TOOL_URL` 都要指向对应 worker 的 `/worker/tool` 接口；如果某台 worker 暂不使用，设为 `ENABLED: false` 或从文件中删除。

## 启动推理 Agent

```bash
cd medflow/agent/agent
python inference_agent.py
```

启动后可以从 Runtime 中心节点验证 controller 接口：

```bash
curl -X POST http://<worker-a-host>:8899/inference_agent \
  -H "Content-Type: application/json" \
  -d '{"command":"查看推理状态","include_trace":true}'
```

若返回模型调用错误，应检查 `AGENT_LLM_URL` 或 `LLM_URL` 指向的 OpenAI-compatible 模型服务是否可访问，并确认 `AGENT_LLM_MODEL` / `LLM_MODEL`、`AGENT_LLM_API_KEY` / `LLM_API_KEY` 与该服务匹配；若工具执行失败，应检查 worker 节点、`nodes.yaml`、推理服务配置和端口连通性。

