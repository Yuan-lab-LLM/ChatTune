# 多机部署配置

本文说明 Runtime 的中心节点、计算节点和多节点配置。创建容器和工作区挂载请先看 [部署指南](DEPLOYMENT_ZH.md)，单机启动流程见 [快速开始](QUICKSTART_ZH.md)。

- 中心节点：运行 Studio、Bridge，并可选运行本机 Agent API。
- 计算节点：运行本机 Agent API、资源探测服务和任务容器。

中心节点必须能访问每台计算节点的 `AGENT_API_PORT`，每台计算节点必须能访问中心节点的 Studio 后端地址。

每台参与多机训练的机器都必须初始化为 Runtime 节点，并启用 Agent API/资源探测服务，同时通过 `MULTINODE_DOCKER_CONTAINER` 配置独立的多机训练容器。只在 master 节点部署 Agent 不够；Studio 需要向每个参与节点查询 GPU 快照、预约 GPU 卡号并管理 allocation。

## 前置条件

- Linux 运行环境，`runtime.sh` 依赖 Bash。
- Bash、`setsid` 和 `curl`。
- Python 3.10 或更高版本。
- Node.js 20 或更高版本、npm 10 或更高版本。
- Docker，以及预先创建好的训练、评测/推理、多机训练和 GRPO/verl 任务容器。
- 如需执行 Agent 或推理命令，需要可访问的 OpenAI-compatible 模型服务。

## 安装依赖

在Agent容器中安装依赖：

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

## 中心节点初始化

中心节点先执行 `init --profile center`，用 `--nodes` 只填写中心之外、中心可访问的计算节点 Agent API 地址。中心初始化会默认生成并启动本机 Agent 节点 `center`，不要把中心节点自己的 IP 再作为普通计算节点写进 `--nodes`：

```bash
bash runtime.sh init --profile center --nodes node-b=http://<worker-b-host>:8099
```

## 生成后重点修改的字段

| 字段 | 填写内容 |
| --- | --- |
| `MEDFLOW_LOCAL_TRAINING_CONTAINER` | 中心本机实际存在的训练容器名，用于中心本机 Agent 的常规训练任务。 |
| `MULTINODE_DOCKER_CONTAINER` | 中心本机实际存在的多机训练 LLaMAFactory 容器名，用于多机 LoRA SFT 和 DPO 增强训练。 |
| `MEDFLOW_LOCAL_EVALUATE_CONTAINER` | 中心本机实际存在的评测/推理容器名，用于中心本机 Agent 的模型评测和推理相关任务。 |
| `MEDFLOW_LOCAL_GRPO_CONTAINER` | 中心本机实际存在的 GRPO/verl 容器名，用于中心本机 Agent 的 GRPO/verl 训练任务。 |
| `MODEL_NAME` / `MODEL_API_KEY` / `MODEL_BASE_URL` | 中心本机 Agent 调用的 OpenAI-compatible 模型名、API key 和服务地址。 |
| `INFERENCE_AGENT_URL` | 中心本机 Agent 使用的推理 Agent controller `/inference_agent` 地址；页面管理员运维动作使用 Studio 登录态判权，如启用推理 Agent `/admin/*`，应限制为仅可信 Studio Server 可访问。 |
| `STUDIO_PUBLIC_URL` | 用户浏览器访问中心 Studio 的地址。 |
| `STUDIO_URL` | Runtime/Agent 回调中心 Studio 后端的地址。 |

## 计算节点初始化

每台计算节点执行 `init --profile node`。先从中心节点输出中记录共享 token 和该节点专属 token，然后在对应计算节点执行：

```bash
bash runtime.sh init --profile node --node-id node-b --center-url http://<center-host>:3000 --agent-url http://<worker-b-host>:8099
```

## 计算节点重点字段

| 字段 | 填写内容 |
| --- | --- |
| `MEDFLOW_LOCAL_TRAINING_CONTAINER` | 该节点实际存在的训练容器名，用于常规训练任务。 |
| `MULTINODE_DOCKER_CONTAINER` | 该节点实际存在的多机训练 LLaMAFactory 容器名，用于多机 LoRA SFT 和 DPO 增强训练。 |
| `MEDFLOW_LOCAL_EVALUATE_CONTAINER` | 该节点实际存在的评测/推理容器名，用于模型评测和推理相关任务。 |
| `MEDFLOW_LOCAL_GRPO_CONTAINER` | 该节点实际存在的 GRPO/verl 容器名，用于 GRPO/verl 训练任务。 |
| `MEDFLOW_STUDIO_RUNTIME_TOKEN` | 从中心节点复制的共享 Studio Runtime token。 |
| `MEDFLOW_AGENT_API_TOKEN` | 从中心节点复制的共享 Agent API token。 |
| `MEDFLOW_RESOURCE_API_TOKEN` | 中心输出中该节点的资源 API token，必须匹配中心 `MEDFLOW_RESOURCE_NODES` 中对应节点的 `resourceApiToken`。 |
| `MEDFLOW_RUNTIME_NODE_TOKEN` | 中心输出中该节点的 Runtime 节点 token，必须匹配中心 `MEDFLOW_STUDIO_RUNTIME_NODE_TOKENS` 中对应节点的值。 |
| `MODEL_NAME` / `MODEL_API_KEY` / `MODEL_BASE_URL` | 该节点 Agent 调用的 OpenAI-compatible 模型名、API key 和服务地址。 |
| `INFERENCE_AGENT_URL` | 该节点任务使用的推理 Agent controller `/inference_agent` 地址；页面管理员运维动作使用 Studio 登录态判权，如启用推理 Agent `/admin/*`，应限制为仅可信 Studio Server 可访问。 |

## 启动顺序

双机或多机部署时，先启动中心节点，再启动各计算节点：

```bash
bash runtime.sh check
bash runtime.sh start
bash runtime.sh status
```

`bash runtime.sh check` 会提前发现以下问题：推荐字段和兼容字段不一致、placeholder 未替换、token 复用、当前节点 token 与中心节点 token 映射不一致、中心本机 Agent 的资源 token 与 `MEDFLOW_RESOURCE_NODES` 不一致、本地容器字段错配、依赖缺失和端口冲突。

## 常用运维命令

| 命令 | 用途 |
| --- | --- |
| `bash runtime.sh check` | 检查依赖、环境变量、端口和 Docker 配置。 |
| `bash runtime.sh start` | 启动服务。 |
| `bash runtime.sh stop` | 停止服务。 |
| `bash runtime.sh restart` | 重启并重新读取 `runtime.env`。 |
| `bash runtime.sh status` | 查看服务状态和 PID。 |
| `bash runtime.sh logs` | 查看全部日志。 |
| `bash runtime.sh logs agent` | 查看 Agent 日志；也可使用 `studio` 或 `bridge`。 |
| `bash runtime.sh --help` | 查看全部参数。 |

默认运行数据位于 `.runtime/`：

```text
.runtime/
  data/
  logs/
  agent.pid
  bridge.pid
  studio.pid
```

其中：

- `.runtime/data/database.sqlite`：默认 Studio SQLite 数据库。
- `.runtime/logs/agent.log`：Agent API 日志。
- `.runtime/logs/bridge.log`：Bridge 日志。
- `.runtime/logs/studio.log`：Studio 日志。
- `.runtime/*.pid`：`runtime.sh` 记录的服务进程 PID。

如果 `check` 失败，先按提示修复依赖、端口、token、容器或 placeholder 问题，再重新启动。
