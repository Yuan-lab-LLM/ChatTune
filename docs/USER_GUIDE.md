# User Guide

This guide covers common Studio operations for ordinary users.

## Login

Open `STUDIO_PUBLIC_URL` in a browser and log in with the account created by an administrator.

Login sessions are valid for 7 days by default. After 5 consecutive failed login attempts, the account is locked for about 15 minutes by default. The lock state is currently stored in Studio process memory and is cleared after the process restarts.

Users can change their password from the account menu. After a successful password change, existing sessions become invalid and the user must log in again.

## Available Natural-language Commands

After logging in to Studio, users can enter natural-language tasks directly in the chat box.

Note: The “Clear context” action in the upper-right area of the run page clears the current page conversation context and starts a new context. The system updates the current session username with the new context, so concepts such as “current user” and “my” remain consistent only within the same context. After clearing context, if you need to continue operating on tasks or instances from the previous context, explicitly include the task ID, instance ID, workflow ID, or container name in the command.

## One-click Workflow

A one-click workflow connects one complete model iteration. After the user selects the training dataset and evaluation set, the system proceeds through these stages:

1. Model training: automatically runs LoRA batch training or enhanced training based on data location.
2. Medical evaluation: runs single-model evaluation on the trained model.
3. Model publication: copies training outputs to the configured release directory and creates a deployable version.
4. Inference deployment: updates inference configuration, restarts the inference service, and confirms service status.
5. Benchmark evaluation: runs inference Benchmark with the selected evaluation set.

After the workflow starts, the page status bar continuously shows the current stage and progress for training, evaluation, publication, deployment, and benchmark evaluation. Workflow state is persisted; if a task fails, handle the displayed reason and then continue.

### Page Steps

1. Open the project's run page and select the "One-click workflow" button in the upper-right area.
2. In the right-side workflow panel, select a training dataset. The training dataset is required; if the list is stale, click "Refresh datasets".
3. Select an evaluation set. The evaluation set is optional; if omitted, `2024.json` is used by default.
4. Review the command preview in the panel, then click "Generate command". This places the command into the chat input.
5. Confirm that the generated command and datasets are correct, then click the send button in the lower-right corner of the chat box to start the workflow.
6. After the workflow starts, track each stage in the page status bar.

### Workflow Control Commands

| Purpose | Natural-language example |
| --- | --- |
| View status | `查看一键工作流状态` |
| Stop workflow | `停止当前一键工作流` |
| Continue failed task | `继续上次失败的一键工作流` |
| View a specific workflow | `查看一键工作流状态 <工作流ID>` |
| Stop a specific workflow | `停止一键工作流 <工作流ID>` |

Workflow IDs look like `wf-YYYYMMDDHHMMSS-xxxxxxxx` and are displayed in status messages after startup. When multiple workflows exist, include the workflow ID in control commands.

## Data Processing

| Purpose | Natural-language example |
| --- | --- |
| Data preprocessing | `执行数据预处理` |
| Advanced data filtering | `执行数据高级筛选处理` |
| Advanced data filtering(Specified threshold) | `执行数据高级筛选，数据清洗阈值=60` |

When required parameters are missing, Agent will ask follow-up questions.

Data preprocessing supports `sft` and `dpo` data formats. For medical text data, it supports selecting the data purpose as `inspection`, `diagnosis`, or `prescription`.

Data preprocessing uses the latest dated data by default. To use an older dataset, select it in the left data-management tab and click `Preprocess with this`.

## Single-machine Training

| Purpose | Natural-language example |
| --- | --- |
| LoRA batch training | `执行lora批量训练` |
| Custom LoRA batch-training parameters | `执行lora批量训练，MBS=1，ACC=8，LR=1e-7` |
| Full-parameter batch training | `执行全参批量训练` |
| DPO enhanced training | `执行增强训练` |
| GRPO training | `执行grpo训练` |

When required parameters are missing, Agent will ask follow-up questions.

The current version of batch training and enhanced training supports the Qwen3 series for both training and evaluation by default. If you need to use other models, please explicitly specify the model template via the TEM or template parameter when executing training, for example: 执行lora批量训练，TEM=llama3 or 执行增强训练，template=deepseekr1.

LoRA batch training and full-parameter batch training use the latest model under `/home/workspace/models/base` by default. To use another model, specify it explicitly, for example: `执行lora批量训练，模型在xxxx`.

LoRA batch training and full-parameter batch training use the latest dated dataset by default. To use an older dataset, select it in the left data-management tab and click `Train with this`.

## Multi-node Training

Multi-node LoRA SFT and DPO enhanced training are available after an administrator configures a training resource pool and sets `MULTINODE_DOCKER_CONTAINER` on the participating Runtime nodes. The Agent supports requests such as `执行多机lora批量训练` and `执行多机增强训练`.

Every machine that participates in the training job must be a Runtime node with Agent API/resource probing available. Ask an administrator to add each participating machine to the training resource pool before starting dual-node or multi-node training.

If the request does not specify whether the task is SFT or DPO, Agent will ask you to choose `多机lora批量训练` or `多机增强训练`. The resource pool assigns nodes, GPU indexes, `MASTER_ADDR`, `MASTER_PORT`, and the allocation file automatically.

When the user does not specify the resource shape, multi-node LoRA SFT and multi-node enhanced training both request `2` nodes and `1` GPU per node by default. Administrators can change this environment default with `MEDFLOW_MULTINODE_NODE_COUNT` and `MEDFLOW_MULTINODE_GPUS_PER_NODE`. The resource pool still decides the concrete GPU indexes according to node `allowedGpuIndexes`, current GPU availability, active reservations, and user-group quota.

If you need to specify the number of GPUs per node, add `gpus-per-node` to the command:

| Purpose | Natural-language example |
| --- | --- |
| Multi-node LoRA with default resources | `执行多机lora批量训练` |
| Multi-node LoRA with 2 GPUs per node | `执行多机lora批量训练，gpus-per-node=2` |
| Multi-node enhanced training with default resources | `执行多机增强训练` |

Multi-node LoRA defaults to the latest model in /home/workspace/models/base and the most recent dataset. Explicit specification is required to use any other model or data.

## Medical Evaluation

| Purpose | Natural-language example |
| --- | --- |
| Single-model evaluation | `执行单模型评估` |
| Two-model comparison evaluation | `执行双模型评估` |
| Checkpoint evaluation | `执行checkpoint评估` |

When required parameters are missing, Agent will ask follow-up questions.

Checkpoint evaluation defaults to supporting the Qwen3 series. If you are evaluating checkpoints from other models, please specify the template via the TEM or template parameter in the evaluation command.

## Status and Stop Tasks

| Purpose | Natural-language example |
| --- | --- |
| Query training status | `查询当前训练状态` |
| Query evaluation status | `查询当前评估状态` |
| Stop training | `停止当前训练任务` |
| Stop evaluation | `停止当前模型评估任务` |
| Stop data processing | `停止当前数据处理任务` |

To avoid stopping the wrong task when multiple tasks are running, include the task type, PID, workflow ID, or container name. After starting a task, keep the PID, run ID, and container information returned by the system so you can query or stop it later.

## Inference Services, Functional Tests, and Benchmarks

Inference commands are forwarded to the Inference Agent. Before running them, make sure `INFERENCE_AGENT_URL` points to the Inference Agent controller `/inference_agent` endpoint, controller and workers are started, and the target node has created and configured the inference Docker container. Inference-service YAML and node configuration are documented in [Deployment Guide](DEPLOYMENT.md).

### View and Modify Inference Configuration

| Purpose | Natural-language example |
| --- | --- |
| View all configuration; returns all nodes in multi-machine deployment | `查看推理配置文件` |
| View a specific node configuration | `查看节点main/node1的推理配置` |

Modify inference configuration by changing parameters on the configuration page and saving them. This is administrator-only.

### Inference Service and Node Operations

Ordinary users can only view and operate inference service instances they are authorized to access. Viewing all instances or stopping another user's instance requires an administrator account. For stop operations, run the preview command first and confirm only after verifying the target.

| Purpose | Natural-language example | Notes |
| --- | --- | --- |
| View service status; returns all nodes in multi-machine deployment | `查看推理状态` | Returns overall inference service status. |
| View startup status | `查看推理服务启动状态` | Returns current startup task status. |
| Start service | `启动推理服务` | Starts the inference service. |
| Stop service | `关闭推理服务` | Stops the inference service. |
| View logs | `查看推理服务日志` | Shows service logs. |
| Operate on a specific node | `查看节点 <节点名称> 的推理服务状态`, `重启节点 <节点名称> 的推理服务` | Targets a specific node in multi-machine deployment. |
| View current user's inference service instances | `查看当前推理服务实例列表` | Returns inference service instances for the current user. |
| View a specific instance status | `查看推理服务实例 instance_id=20260731_161456_26261a30 的状态` | Ordinary users can only view their own instances; admins can view specified instances. |
| Preview stopping an instance | `预览停止推理服务实例 instance_id=20260731_161456_26261a30` | Only checks the instance to be stopped and blocking tasks; it does not stop anything. |
| Confirm stopping an instance | `确认停止推理服务实例 instance_id=20260731_161456_26261a30` | Ordinary users can only stop their own instances; admins can stop specified instances. |
| Admin view all instances | `查看所有推理服务实例` | Admin command for troubleshooting cross-user instances. |

For multi-machine deployment, include the node name in the command to avoid operating on the wrong node.

### Inference Functional Tests

Ordinary users can only view and operate their own functional test tasks. Viewing all tasks or stopping another user's task requires an administrator account. For stop operations, run the preview command first.

| Purpose | Natural-language example | Notes |
| --- | --- | --- |
| View test scripts | `查看可用的推理功能测试脚本` | Returns currently available scripts. |
| Run a test | `运行推理功能测试脚本 basicmedicalrecord.sh` | Runs a test with an actual script name. |
| View test status | `查看推理功能测试状态` | Shows current test status. |
| Preview stopping a functional test | `预览停止功能测试 test_run_id=test_20260803_153000` | Only checks the target test status; it does not terminate the process. |
| Confirm stopping a functional test | `确认停止功能测试 test_run_id=test_20260803_153000` | Ordinary users can only stop their own tasks; admins can stop specified tasks. |
| Admin view all functional test tasks | `查看所有功能测试任务` | Returns functional test run records for all users. |

View available scripts first, then run tests with the actual script name returned by the system.

### Inference Benchmarks

Ordinary users can only view and operate their own Benchmark jobs. Viewing all jobs or stopping another user's job requires an administrator account. For stop operations, run the preview command first.

| Purpose | Natural-language example | Notes |
| --- | --- | --- |
| View available Benchmarks | `查看可用的推理基准测试` | Returns currently available Benchmarks. |
| Start Benchmark | `运行推理基准测试2024.json` | Starts the specified Benchmark. |
| View status | `查看推理基准测试2024.json状态` | Shows job status. |
| View progress | `查看推理基准测试2024.json进度` | Shows execution progress. |
| View results | `查看推理基准测试2024.json结果` | Shows result information. |
| Stop Benchmark | `停止推理基准测试2024.json` | Stops the current user's Benchmark job. |
| Run on a specific node | `在节点 <节点名称> 运行推理基准测试2024.json` | Runs on a specific node in multi-machine deployment. |
| Preview stopping a Benchmark | `预览停止 benchmark job_id=bench_20260803_153000` | Only checks the target task status; it does not terminate the process. |
| Confirm stopping a Benchmark | `确认停止 benchmark job_id=bench_20260803_153000` | Ordinary users can only stop their own tasks; admins can stop specified tasks. |
| Admin view all Benchmark jobs | `查看所有 benchmark 任务` | Returns Benchmark jobs for all users. |

The system supports medical multiple-choice datasets such as `2021.json`, `2024.json`, `step1.json`, `step2.json`, and `step3.json`, plus MedBench and general Benchmarks. Inference benchmark evaluation depends on the developer-provided `tests` directory, which contains currently supported benchmarks. Before deployment, confirm that this directory is mounted with `medflow/` into the inference container, and that Benchmark data directories in `service.yaml` point to actual paths inside the container. Actual available items are determined by the response from "查看可用的推理基准测试".

### Admin Maintenance Commands

The following commands clean up local stale resources and require an administrator account. For cleanup operations, run the preview command first and confirm only after verifying the target.

| Purpose | Natural-language example | Notes |
| --- | --- | --- |
| Preview local stale-resource cleanup | `预览清理本地残留资源` | Shows stale instances, residual processes, or records eligible for cleanup without applying changes. |
| Apply local stale-resource cleanup | `确认执行本地残留资源清理` | Applies cleanup according to the preview result. |
