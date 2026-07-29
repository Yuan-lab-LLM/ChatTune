# MedFlow Runtime MP

MedFlow Runtime MP is a complete runtime project for running MedFlow Agent workflows through Studio, Agent API, resource management, task containers, and MedFlow inference services.

This repository includes:

- `runtime.sh`: runtime initialization, configuration checks, start/stop commands, status inspection, and log management.
- `studio/`: Studio web frontend and backend for login, users, user groups, resources, run records, and audit views.
- `agent/`: Agent API, workflow orchestration, resource probing, internal resource APIs, and MedFlow-specific Agent tools. 
- `agent-studio-runtime-bridge/`: bridge service between Studio and the Agent API backend.
- `docker_scripts/`: scripts for creating Agent, training, evaluation/inference, and GRPO containers.
- `medflow/`: MedFlow inference services, inference operation agents, functional tests, and benchmark code.

## Documentation

- [Documentation index](docs/README.md) 
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Quickstart](docs/QUICKSTART.md) 
- [Admin Guide](docs/ADMIN_GUIDE.md) 
- [User Guide](docs/USER_GUIDE.md) 
- [Multi-node Configuration](docs/MULTI_NODE.md) 

## Project Origins

This project originated from the medical MedFlow project [MedFlow2025/medflow](https://github.com/MedFlow2025/medflow) and is built on the qingnang series text-generation models ([ModelScope: MedFlow](https://modelscope.cn/models/MedFlow/Qingnang-32B-0630)).

## Acknowledgements

MedFlow Runtime MP uses and builds on the following open-source projects. We are grateful to their authors, maintainers, and contributor communities.

- [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory), used for the LLaMA-Factory training workflow. 
- [verl](https://github.com/verl-project/verl), used for GRPO training workflows. 
- [AgentScope](https://github.com/agentscope-ai/agentscope), used through the external `agentscope-runtime` Python dependency, which resolves the compatible `agentscope` package version.

## License and Notices

This repository is released under the Apache License 2.0. See [LICENSE](LICENSE).

Subproject ownership, third-party notices, and external asset boundaries are documented in [NOTICE](NOTICE). Existing license files and copyright headers in subdirectories should be preserved.

Model weights, external datasets, container images, and cloud services may have separate terms and access requirements. Open-sourcing this repository does not automatically grant permission to use those external assets.

Please report security issues privately according to [SECURITY.md](SECURITY.md).

## Medical and Data Disclaimer

This project is intended only for research, testing, and deployment integration. It is not medical advice and must not be used as the sole basis for diagnosis or treatment.

## Contributing

Code and documentation contributions are welcome under Apache-2.0. Development conventions, testing recommendations, and sensitive-data rules are described in [CONTRIBUTING.md](CONTRIBUTING.md).

