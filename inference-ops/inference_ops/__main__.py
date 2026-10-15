import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference Ops service")
    parser.add_argument("--mode", choices=["api", "cli"], default="api")
    parser.add_argument("--env-file", help="Operator environment file; existing process variables take precedence")
    args = parser.parse_args()
    if args.env_file:
        os.environ["INFERENCE_OPS_ENV_FILE"] = args.env_file

    # Load configuration before importing the app and its module-level settings.
    from .inference_agent import INFERENCE_AGENT_HOST, INFERENCE_AGENT_PORT, app, main as run_cli
    import uvicorn

    if args.mode == "cli":
        run_cli()
        return
    uvicorn.run(app, host=INFERENCE_AGENT_HOST, port=INFERENCE_AGENT_PORT)


if __name__ == "__main__":
    main()
