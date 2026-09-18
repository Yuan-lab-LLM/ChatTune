# 部署指南

本文只说明 MedFlow ChatTune 发布包中的容器、镜像和工作区挂载准备。完成本页后，再进入 [快速开始](QUICKSTART_ZH.md) 初始化并启动 Runtime。

## 组件

- `studio/`：Studio Web 前后端，负责登录、用户管理、资源管理和运行记录。
- `agent-studio-runtime-bridge/`：Studio 与 Agent API 后端之间的桥接服务。
- `agent/`：Agent API、工作流编排、资源探测、任务路由和内部资源 API。
- `runtime.sh`：初始化、检查、启动、停止、状态和日志脚本。
- `docker_scripts/`：Agent、训练、多机训练、评测/推理和 GRPO 容器创建脚本。
- `inference-ops/`：独立的 vLLM 运维 controller/worker 和 Benchmark。
- `medflow/`：暂存的旧业务目录。

## 镜像和资源

| 资源 | 链接 | 备注 |
| --- | --- | --- |
| 训练镜像 | https://pan.quark.cn/s/040388acdd9a | 提取码：m9YH |
| 评测/推理镜像 | https://pan.quark.cn/s/7ef603e26e50 | 提取码：SH8t |
| GRPO/verl 镜像 | https://pan.quark.cn/s/c6ad41207360 | 提取码：vJJx |
| 基础模型资源 | https://modelscope.cn/models/MedFlow/Qingnang-32B-0630/ | Qingnang-32B-0630 基础模型 |
| 评测 Benchmark 数据 | https://pan.quark.cn/s/5ba6589d7933 | 提取码：b8Pp |

### 训练镜像补丁包

后续训练镜像的发布将以补丁包形式进行。下载最新的 `.run` 包后，只需要执行 `bash xx.run` 完成安装；除非发布说明另有要求，不需要重新获取完整训练镜像包。

| 资源 | 链接 | 备注 |
| --- | --- | --- |
| 训练镜像补丁包-0813 | https://pan.quark.cn/s/db69395cbde7 | 提取码：6K2h llamafactory版本更新至0.9.5 |
| 训练镜像补丁包-0820 | https://pan.quark.cn/s/609582f1f4c9 | 提取码：68uc 支持多机训练 |
| 训练镜像补丁包-0904 | https://pan.quark.cn/s/ab6dfd3a721a | 提取码：97TB 支持预训练 |
| 训练镜像补丁包-0917 | https://pan.quark.cn/s/f40b29e2e544 | 提取码：dpEB 支持通用模型评估、自动识别模型tem |

## 容器类型 

| 容器 | 部署位置 | 作用 |
| --- | --- | --- |
| Agent 容器 | 单机、中心节点或计算节点 | 挂载代码目录，运行 Runtime、Agent API、资源探测、Studio 或 Bridge。 |
| 通用训练任务容器 | GPU 计算节点 | 执行 LoRA、全参、DPO 等常规训练任务。 |
| 多机训练任务容器 | GPU 计算节点 | 执行多机 LoRA SFT、多机全参 SFT 和多机 DPO 增强训练任务。 |
| 评测/推理任务容器 | GPU 计算节点 | 执行模型评测、推理服务相关任务。 |
| GRPO/verl 任务容器 | GPU 计算节点 | 执行 GRPO 训练及相关数据/模型操作。 |

运行生成容器的脚本之前，需要替换脚本中的默认参数。请使用 root 权限运行这些脚本。

| 容器 | 脚本 | 需要替换的字段 |
| --- | --- | --- |
| Agent 容器 | `docker_scripts/create_agent_docker.sh` | `AGENT_CONTAINER_NAME`、`AGENT_IMAGE`、`HOST_WORKSPACE` |
| 通用训练任务容器 | `docker_scripts/create_training_dockers.sh` | `HOST_IP`、`TRAIN_IMAGE`、`TRAIN_CONTAINER_NAME`、`HOST_WORKSPACE` |
| 多机训练任务容器 | `docker_scripts/create_multinode_training_dockers.sh` | `HOST_IP`、`TRAIN_IMAGE`、`TRAIN_CONTAINER_NAME`、`HOST_WORKSPACE` |
| 评测/推理任务容器 | `docker_scripts/create_inference_docker.sh` | `HOST_IP`、`IMAGE_NAME`、`IMAGE_VERSION`、`DOCKER_NAME`、`HOST_WORKSPACE` |
| GRPO/verl 任务容器 | `docker_scripts/create_grpo_docker.sh` | `HOST_IP`、`GRPO_IMAGE`、`GRPO_CONTAINER_NAME`、`HOST_WORKSPACE` |

## Agent 容器挂载

容器创建时， `HOST_WORKSPACE` 应指向本项目代码目录，也就是 MedFlow ChatTune 发布包/项目根目录。

## 通用训练容器挂载

容器创建时，HOST_WORKSPACE 所挂载的宿主机目录结构要求如下：

```text
<training-host-workspace>/
├── models
│   ├── base                        # 准备训练的基础权重文件
│   ├── batch_train                 # 存放批量训练完成的权重文件
│   ├── pretrain                    # 存放 PT 预训练完成的权重文件
│   └── dpo_train                   # 存放增强训练完成的权重文件
├── dataset_daily_train             # 准备训练的DPO/增强训练数据集
├── dataset_batch_train             # 准备训练的SFT/批量训练数据集
├── dataset_pretrain                # 准备训练的 PT/text 预训练数据集
├── eval                            # 存放生成的评测结果
└── log                             # 存放生成的日志文件
    ├── batch_train
    ├── pretrain
    └── dpo_train
```

### 训练数据格式

通用训练容器使用 LLaMA-Factory 训练流程。SFT/批量训练数据放在 `dataset_batch_train` 下，可使用 Alpaca 或 ShareGPT 格式。PT/text 预训练数据放在 `dataset_pretrain` 下。DPO/增强训练数据放在 `dataset_daily_train` 下，使用偏好数据格式，必须包含被选择回答和被拒绝回答。

### 资源目录配置

默认目录可直接使用。需要替换默认目录或增加领域数据目录时，在 Studio 和对应 Runtime 节点的环境中设置相同的 `MEDFLOW_RESOURCE_CATALOG_CONFIG`。它是 JSON 对象，可只覆盖需要变更的条目：

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

目录必须是容器内绝对路径，不能是 `/` 且不能包含 `..`。新增目录会通过现有 `queryDatasets`、`queryModels` 或 `queryGrpoResources` 查询返回；旧类型 `base_train`、`batch_trained`、`pretrain`、`daily_trained`、`inference` 仍作为 alias 保留。配置完成后重启 Studio 和 Agent/Runtime 进程使配置生效。

### 训练能力清单

9 个内置训练方法由唯一的版本化清单 `agent/config/training_descriptors.v1.json` 定义，Agent 和 Studio 会读取同一文件。清单当前要求 `schemaVersion=1`；`requiredParams` 和 `resourceSelectors` 由两端加载器根据参数与资源配置生成，不需要写入清单。

使用完整仓库结构部署时不需要额外配置。若 Studio 与 Agent 分开安装，必须在两者的环境中将 `MEDFLOW_TRAINING_DESCRIPTOR_PATH` 设置为同一份清单的绝对路径：

```bash
MEDFLOW_TRAINING_DESCRIPTOR_PATH=/opt/chattune/config/training_descriptors.v1.json
```

清单仅在进程启动时加载，修改后需要重启 Studio 和 Agent/Runtime。文件缺失、JSON 或版本非法、descriptor ID/alias 冲突、参数或资源字段非法时进程会直接启动失败，不会静默回退到内置副本。

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

训练任务使用 Weights & Biases 记录训练指标。创建通用训练容器、多机训练容器或 GRPO/verl 容器后、开始首次在线训练前，必须在对应训练容器中完成 W&B 登录：

```bash
wandb login
```

注意事项：

- W&B 登录信息保存在容器内当前用户的配置中；删除或重新创建训练容器后需要重新登录。
- 登录失败时，在容器中检查网络连通性。

## 多机训练容器准备

多机训练容器的基础挂载和创建方式与通用训练容器一致，创建后除了完成 `wandb login`，还需要在每个参与多机训练的容器中安装 `openssh-server` 和 `pdsh`，然后运行 `setup_multinode_ssh.sh` 配置 sshd 与节点互信。

启动多机训练前，必须确认每台参与训练机器的多机训练 Docker 容器内，在相同容器路径下存在相同的数据集和基础模型。例如主节点使用 `/home/workspace/dataset_batch_train/my_sft` 和 `/home/workspace/models/base/qwen3` 时，从节点容器内也必须存在内容一致的这两个路径；只在宿主机路径相同或只在其中一台机器准备数据/模型都不够。

两个训练容器里都先设置 root 密码：

```bash
apt update
apt install openssh-server -y
apt install -y pdsh
passwd
```

检查pdsh是否通过安全检查：

```bash
stat -c '%U:%G %a %n' /usr /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/pdsh
```

正常这些目录应该是 root:root，并且不能被 group/other 写。直接修复可以执行：

```bash
chown root:root /usr /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/pdsh
chmod go-w /usr /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/pdsh
```

然后两边先各自准备本地 sshd：

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

然后两边再跑完整互信：

```bash
bash setup_multinode_ssh.sh \
  --user root \
  --self-node ds35 \
  --nodes ds35,ds36 \
  --hosts <ds35-host>,<ds36-host> \
  --ports 2222,2222 \
  --slots 4
```

从节点把 `--self-node ds35` 改成 `--self-node ds36`。

## 评测/推理容器挂载

容器创建时，HOST_WORKSPACE 所挂载的宿主机目录结构要求如下：

```text
<inference-host-workspace>/
├── inference-ops                         # 本项目提供的 inference-ops 文件夹
├── medical_models                  # 用于推理的模型
└── tests                           # 本项目提供的评测 Benchmark 数据
```

Inference Ops 是独立的 vLLM 运维与 Benchmark 服务，配置位于 `inference-ops/config/`。顶层 `medflow/` 是暂存的旧业务目录，不是部署依赖：

| 文件 | 作用 |
| --- | --- |
| `service.yaml` | vLLM 运行参数，包括端口、GPU、模型、日志和 Benchmark 路径。 |
| `service.default.yaml` | 恢复默认配置时使用的模板。 |
| `agent.yaml` | 推理 Agent 的角色、监听地址、端口和 LLM 服务配置，默认为空。 |
| `nodes.yaml` | controller 管理的 worker 节点列表，默认为空。 |
| `nodes.single.example.yaml` | 单机节点示例，节点 ID 为 `local`。 |
| `nodes.multi.example.yaml` | 多机节点示例。 |
| `agent.both.example.yaml` | 单机 both 角色示例。 |
| `agent.controller.example.yaml` | controller 角色示例。 |
| `agent.worker.example.yaml` | worker 角色示例。 |

## 单机推理配置

单机部署或中心节点同时管理本机推理服务时，推理 Agent 使用 `both` 角色，controller 和 worker 在同一台机器上运行。可以从 `agent.both.example.yaml` 复制 `agent.yaml`，并从 `nodes.single.example.yaml` 复制 `nodes.yaml`。

推理 Agent 启动前必须修改：

| 文件 | 字段 | 使用前替换为 |
| --- | --- | --- |
| `agent.yaml` | `HOST` / `PORT` | 推理 Agent 监听地址和端口。 |
| `agent.yaml` | `LLM_URL` / `LLM_MODEL` / `LLM_API_KEY` | 推理 Agent 调用的 OpenAI-compatible LLM 地址、模型名称和 API key。 |
| `nodes.yaml` | `NODES.local.RESOURCE_NODE_ID` | 保持为 `local`，与单机 Runtime 的节点 ID 一致。 |
| `nodes.yaml` | `NODES.local.HOST` / `TOOL_URL` | 指向本机 worker 工具接口。 |

`service.yaml` 可在本文件中预先修改，也可由管理员在页面修改：

| 文件 | 字段 | 使用前替换为 |
| --- | --- | --- |
| `service.yaml` | `ENV.HOST_IP` | 当前推理服务所在机器对外可访问的 IP。 |
| `service.yaml` | `ENV.CUDA_VISIBLE_DEVICES` | 本机推理服务实际使用的 GPU 编号，例如 `0,1,2,3`。 |
| `service.yaml` | `ENV.MODEL_NAME` / `ENV.MODEL_PATH` | 容器内实际存在的模型目录名和模型根目录。 |
| `service.yaml` | `PORTS.VLLM_OPENAI_PORT` / `ENV.MASTER_PORT` | 本机未占用的 vLLM API 端口和分布式 master 端口。 |
| `service.yaml` | `RUNTIME.TENSOR_PARALLEL_SIZE` / `RUNTIME.GPU_MEMORY_UTILIZATION` / `RUNTIME.GPU_UTILIZATION_THRESHOLD` | 按本机 GPU 数、模型规模和显存容量调整的 vLLM 与调度参数。 |

单机 `nodes.yaml` 示例：

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

## 多机推理配置

多机推理使用一个 controller 和多个 worker。controller 启动推理 Agent 前必须维护自己的 `agent.yaml` 和 `nodes.yaml`；每台 worker 启动推理 Agent 前必须维护自己的 `agent.yaml`。controller 可从 `agent.controller.example.yaml` 和 `nodes.multi.example.yaml` 复制配置，worker 可从 `agent.worker.example.yaml` 复制配置。

推理 Agent 启动前必须修改：

| 文件 | 字段 | 使用前替换为 |
| --- | --- | --- |
| controller 的 `nodes.yaml` | `NODES.<node>.HOST` | 每个 worker 对 controller 可访问的 IP 或域名。 |
| controller 的 `nodes.yaml` | `NODES.<node>.TOOL_URL` | 每个 worker 的工具接口地址。 |
| controller 的 `nodes.yaml` | `NODES.<node>.RESOURCE_NODE_ID` | 与对应 Runtime worker 的节点 ID 一致。 |
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

每个 `TOOL_URL` 都要指向对应 worker 的 `/worker/tool` 接口；如果某台 worker 暂不使用，设为 `ENABLED: false` 或从文件中删除。

## 推理 Agent Admin 权限

推理 Agent 的 `/admin/*` 接口用于清理 stale 资源、查看和强制停止服务实例、Benchmark 与测试任务。该能力应只通过 Studio 管理员入口触发：前端不传 admin token，Studio Server 先校验当前登录用户是否为管理员，再调用推理 Agent。

在 `agent.yaml` 中设置 `ADMIN.ENABLED: true` 可开启这些接口。`INFERENCE_OPS_AUTH_MODE` 默认为 `trusted_network`，此时不校验 Bearer Token，必须通过内部网络或防火墙确保只有可信 Studio Server、主 Agent、controller 和 worker 能访问；任何能访问该端口的进程都能直接调用 `/admin/*`。需要服务间鉴权时，在所有调用端和 Inference Ops 进程中显式设置 `INFERENCE_OPS_AUTH_MODE=token` 及相同的 `INFERENCE_OPS_API_TOKEN`。Studio 在两种模式下都仍需先校验登录用户的管理员身份。

## 启动推理 Agent

```bash
cd inference-ops
python -m inference_ops
```

启动后可以从 Runtime 中心节点验证 controller 接口：

```bash
curl -X POST http://<worker-a-host>:8899/inference_agent \
  -H "Content-Type: application/json" \
  -d '{"command":"查看推理状态","include_trace":true}'
```

以上为默认 `trusted_network` 模式。`token` 模式需额外添加 `-H "Authorization: Bearer $INFERENCE_OPS_API_TOKEN"`。资源池分配通过请求中的 `resource_context` 传递；Inference Ops 自主管理推理实例状态，不需要节点资源 API token。

若返回模型调用错误，应检查 `AGENT_LLM_URL` 或 `LLM_URL` 指向的 OpenAI-compatible 模型服务是否可访问，并确认 `AGENT_LLM_MODEL` / `LLM_MODEL`、`AGENT_LLM_API_KEY` / `LLM_API_KEY` 与该服务匹配；若工具执行失败，应检查 worker 节点、`nodes.yaml`、推理服务配置和端口连通性。
