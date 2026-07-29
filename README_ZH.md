# MedFlow Runtime MP

MedFlow Runtime MP 是一个完整的项目，用于通过 Studio、Agent API、资源管理、任务容器和 MedFlow 推理服务运行 MedFlow Agent 工作流。

本仓库包含：

- `runtime.sh`：运行时初始化、配置检查、启动停止、状态和日志管理。
- `studio/`：Studio Web 前后端，负责登录、用户、用户组、资源、运行记录和审计视图。
- `agent/`：Agent API、工作流编排、资源探测、内部资源 API 和 MedFlow 自研 Agent 工具。
- `agent-studio-runtime-bridge/`：Studio 与 Agent API 后端之间的桥接服务。
- `docker_scripts/`：Agent、训练、评测/推理和 GRPO 容器创建脚本。
- `medflow/`：MedFlow 推理服务、推理运维 Agent、功能测试和 Benchmark 相关代码。

## 文档

- [文档索引](docs/README_ZH.md) 
- [部署指南](docs/DEPLOYMENT_ZH.md) 
- [快速开始](docs/QUICKSTART_ZH.md)
- [管理员指南](docs/ADMIN_GUIDE_ZH.md) 
- [用户指南](docs/USER_GUIDE_ZH.md) 
- [多机部署配置](docs/MULTI_NODE_ZH.md) 

## 项目来源

本项目的前身为医疗 MedFlow 项目 [MedFlow2025/medflow](https://github.com/MedFlow2025/medflow)，并基于 qingnang 系列文本生成模型（[ModelScope: MedFlow](https://modelscope.cn/models/MedFlow/Qingnang-32B-0630)）构建。

## 致谢

MedFlow Runtime MP 使用并借鉴了以下开源项目。我们真诚感谢这些项目的作者、维护者和贡献者社区。

- [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)：用于 LLaMA-Factory 训练流程。
- [verl](https://github.com/verl-project/verl)：用于 GRPO 训练流程。
- [AgentScope](https://github.com/agentscope-ai/agentscope)：通过外部 Python 依赖 `agentscope-runtime` 使用，并由该依赖解析兼容的 `agentscope` 包版本。

## 许可证和声明

本仓库使用 Apache License 2.0 发布，详见 [LICENSE](LICENSE)。

子项目归属、第三方声明和外部资产边界见 [NOTICE](NOTICE)。子目录中已有的许可证文件和版权头应继续保留。

模型权重、外部数据集、容器镜像和云服务可能有独立条款和访问要求，不会因为本仓库开源而自动获得授权。

安全问题请按 [SECURITY.md](SECURITY.md) 私下报告。

## 医疗和数据免责声明

本项目仅用于研发、测试和部署集成，不构成医疗建议，也不能作为诊断或治疗的唯一依据。

## 贡献

欢迎在 Apache-2.0 下贡献代码和文档。开发规范、测试建议和敏感数据规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。

