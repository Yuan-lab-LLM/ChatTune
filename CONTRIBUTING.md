# Contributing to MedFlow ChatTune

Thank you for helping improve MedFlow ChatTune. This project is released under Apache-2.0, and contributions are accepted under the same license unless explicitly stated otherwise.

## Development Guidelines

- Keep changes focused and avoid unrelated refactors in the same pull request.
- Do not commit secrets, API keys, private model paths, internal hostnames, runtime logs, SQLite databases, or generated cache files.
- Do not add real patient data or other personal information. Test data must be synthetic or properly de-identified and documented.
- Preserve third-party license headers and notices when modifying vendored or derived code.
- Prefer backward-compatible changes for public module paths and runtime configuration fields.

## Before Submitting

Run the relevant checks for the area you changed:

```bash
bash -n runtime.sh
bash -n docker_scripts/*.sh
cd agent && python -m pytest tests/runlocal_monitor_test.py
cd ../studio && npm run build
cd ../agent-studio-runtime-bridge && npm run build
```

Some checks require local Python/Node dependencies, Docker, GPUs, or model services. If you cannot run a check, mention that in the pull request and explain why.

## Pull Request Notes

A useful pull request includes:

- What changed and why.
- How it was tested.
- Any migration or compatibility impact.
- Any data, model, image, or third-party dependency license impact.

