# User Guide

This guide covers common Studio operations for ordinary users.

## Login

Open `STUDIO_PUBLIC_URL` in a browser and log in with the account created by an administrator.

Login sessions are valid for 7 days by default. After 5 consecutive failed login attempts, the account is locked for about 15 minutes by default. The lock state is currently stored in Studio process memory and is cleared after the process restarts.

Users can change their password from the account menu. After a successful password change, existing sessions become invalid and the user must log in again.

## Available Natural-language Commands

After logging in to Studio, users can enter natural-language tasks directly in the chat box.

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

Data preprocessing supports `sft` and `dpo` data formats. It also supports selecting data purpose as `inspection`, `diagnosis`, or `prescription`.

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

LoRA batch training and full-parameter batch training use the latest dated dataset by default. To use an older dataset, select it in the left data-management tab and click `Train with this`.

## Multi-node Training

Multi-node training is planned for the next phase.

## Medical Evaluation

| Purpose | Natural-language example |
| --- | --- |
| Single-model evaluation | `执行单模型评估` |
| Two-model comparison evaluation | `执行双模型评估` |
| Checkpoint evaluation | `执行checkpoint评估` |

When required parameters are missing, Agent will ask follow-up questions.

Checkpoint evaluation defaults to supporting the Qwen3 series. If you are evaluating checkpoints from other models, please specify the template via the TEM or template parameter in the evaluation command.

## Inference Services, Functional Tests, and Benchmarks

Inference commands are forwarded to the Inference Agent. Before running them, make sure `INFERENCE_AGENT_URL` points to the Inference Agent controller `/inference_agent` endpoint, controller and workers are started, and the target node has created and configured the inference Docker container. Inference-service YAML and node configuration are documented in [Deployment Guide](DEPLOYMENT.md).

### View and Modify Inference Configuration

| Purpose | Natural-language example |
| --- | --- |
| View all configuration; returns all nodes in multi-machine deployment | `查看推理配置文件` |
| View a specific node configuration | `查看节点main/node1的推理配置` |

Modify inference configuration by changing parameters on the configuration page and saving them. This is administrator-only.

### Inference Service and Node Operations

| Purpose | Natural-language example |
| --- | --- |
| View service status; returns all nodes in multi-machine deployment | `查看推理状态` |
| View startup status | `查看推理服务启动状态` |
| Start service | `启动推理服务` |
| Stop service | `关闭推理服务` |
| View logs | `查看推理服务日志` |
| Operate on a specific node | `查看节点 <节点名称> 的推理服务状态`, `重启节点 <节点名称> 的推理服务` |

For multi-machine deployment, include the node name in the command to avoid operating on the wrong node.

### Inference Functional Tests

| Purpose | Natural-language example |
| --- | --- |
| View test scripts | `查看可用的推理功能测试脚本` |
| Run a test | `运行推理功能测试脚本 basicmedicalrecord.sh` |
| View test status | `查看推理功能测试状态` |

View available scripts first, then run tests with the actual script name returned by the system.

### Inference Benchmarks

| Purpose | Natural-language example |
| --- | --- |
| View available Benchmarks | `查看可用的推理基准测试` |
| Start Benchmark | `运行推理基准测试2024.json` |
| View status | `查看推理基准测试2024.json状态` |
| View progress | `查看推理基准测试2024.json进度` |
| View results | `查看推理基准测试2024.json结果` |
| Stop Benchmark | `停止推理基准测试2024.json` |
| Run on a specific node | `在节点 <节点名称> 运行推理基准测试2024.json` |

The system supports medical multiple-choice datasets such as `2021.json`, `2024.json`, `step1.json`, `step2.json`, and `step3.json`, plus MedBench and general Benchmarks. Inference benchmark evaluation depends on the developer-provided `tests` directory, which contains currently supported benchmarks. Before deployment, confirm that this directory is mounted with `medflow/` into the inference container, and that Benchmark data directories in `service.yaml` point to actual paths inside the container. Actual available items are determined by the response from "查看可用的推理基准测试".

## Status and Stop Tasks

| Purpose | Natural-language example |
| --- | --- |
| Query training status | `查询当前训练状态` |
| Query evaluation status | `查询当前评估状态` |
| Stop training | `停止当前训练任务` |
| Stop evaluation | `停止当前模型评估任务` |
| Stop data processing | `停止当前数据处理任务` |

To avoid stopping the wrong task when multiple tasks are running, include the task type, PID, workflow ID, or container name. After starting a task, keep the returned PID, run ID, and container information for later status queries or stop commands.

