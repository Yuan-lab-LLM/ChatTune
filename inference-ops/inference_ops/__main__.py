import argparse

import uvicorn

from .inference_agent import (
    INFERENCE_AGENT_HOST,
    INFERENCE_AGENT_PORT,
    app,
    main as run_cli,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference Ops service")
    parser.add_argument("--mode", choices=["api", "cli"], default="api")
    args = parser.parse_args()
    if args.mode == "cli":
        run_cli()
        return
    uvicorn.run(app, host=INFERENCE_AGENT_HOST, port=INFERENCE_AGENT_PORT)


if __name__ == "__main__":
    main()
