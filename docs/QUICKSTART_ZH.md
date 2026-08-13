# 快速开始

本文说明如何在单机上完成最小启动验证。启动前请先按 [部署指南](DEPLOYMENT_ZH.md) 创建好 Agent、训练、评测/推理和 GRPO/verl 容器。

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

## 初始化 Runtime

在仓库根目录执行：

```bash
bash runtime.sh init --profile single
```

如果 Studio 需要被其他机器访问，初始化时传入本机可达 IP：

```bash
bash runtime.sh init --profile single --iphost <本机IP>
```

该命令会生成 `runtime.env`，其中包含本地运行时 token 和初始管理员密码。需要重新生成配置时，使用 `--force` 显式覆盖。

## 生成后重点修改的字段

| 字段 | 填写内容 |
| --- | --- |
| `MEDFLOW_LOCAL_TRAINING_CONTAINER` | 本机实际存在的训练容器名，用于常规训练任务。 |
| `MEDFLOW_LOCAL_EVALUATE_CONTAINER` | 本机实际存在的评测/推理容器名，用于模型评测和推理相关任务。 |
| `MEDFLOW_LOCAL_GRPO_CONTAINER` | 本机实际存在的 GRPO/verl 容器名，用于 GRPO/verl 训练任务。 |
| `MODEL_NAME` / `MODEL_API_KEY` / `MODEL_BASE_URL` | Agent 调用的 OpenAI-compatible 模型名、API key 和服务地址。 |
| `INFERENCE_AGENT_URL` | 在部署手册中启动的推理 Agent controller `/inference_agent` 地址；页面管理员运维动作使用 Studio 登录态判权，如启用推理 Agent `/admin/*`，应限制为仅可信 Studio Server 可访问。 |
| `STUDIO_PUBLIC_URL` | 用户浏览器访问 Studio 的地址；跨机器访问时不要保留 `127.0.0.1`。 |
| `STUDIO_URL` | Runtime/Agent 回调 Studio 后端的地址，默认后端端口为 `3000`；跨机器访问时改成宿主机可达 IP 或域名。 |
| `MEDFLOW_RESOURCE_NODES` | Studio 资源池节点列表，单机通常包含本机节点的 `id`、`name`、`baseUrl`；`baseUrl` 要指向本机 Agent API。 |
| `AGENT_API_BACKENDS` | Bridge 可访问的 Agent API 地址列表，单机通常指向本机 `AGENT_API_PORT`。 |

## 默认值

`runtime.env` 下半部分的超时、GPU 默认数、workflow 路径和可执行文件覆盖项属于默认值。通常保持 `init` 生成值即可；只有在端口、GPU 策略、工作流目录或 Python/npm 路径确实不同的时候再调整。

## 检查并启动

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
