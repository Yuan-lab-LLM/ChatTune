import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
START_SCRIPT = REPO / "inference-ops" / "scripts" / "start-service.sh"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux process groups")
class VllmStartServiceTests(unittest.TestCase):
    @unittest.skipUnless(
        all(shutil.which(command) for command in ("bash", "curl", "setsid", "python3")),
        "requires bash, curl, setsid, and python3",
    )
    def test_fake_vllm_start_ready_status_log_and_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake_bin = root / "bin"
            pid_dir = root / "pids"
            log_dir = root / "logs"
            fake_bin.mkdir()
            pid_dir.mkdir()
            log_dir.mkdir()
            config = root / "service.runtime.yaml"
            config.write_text("PROVIDER: vllm\n", encoding="utf-8")

            yq = fake_bin / "yq"
            yq.write_text(
                """#!/usr/bin/env bash
case "$1" in
  *PROVIDER*) echo vllm ;;
  *VLLM_OPENAI_PORT*) echo "$FAKE_VLLM_PORT" ;;
  *HOST_IP*) echo 127.0.0.1 ;;
  *MODEL_NAME*) echo fake-model ;;
  *MODEL_PATH*) echo /models/ ;;
  *CUDA_VISIBLE_DEVICES*) echo 0 ;;
  *MASTER_PORT*) echo "$FAKE_MASTER_PORT" ;;
  *TENSOR_PARALLEL_SIZE*) echo 1 ;;
  *GPU_MEMORY_UTILIZATION*) echo 0.5 ;;
  *) exit 2 ;;
esac
""",
                encoding="utf-8",
            )

            lsof = fake_bin / "lsof"
            lsof.write_text(
                """#!/usr/bin/env bash
port=""
for value in "$@"; do
  case "$value" in -iTCP:*) port="${value#-iTCP:}" ;; esac
done
python3 - "$port" <<'PY'
import socket, sys
try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.2):
        pass
except OSError:
    raise SystemExit(1)
PY
""",
                encoding="utf-8",
            )

            vllm = fake_bin / "vllm"
            vllm.write_text(
                """#!/usr/bin/env bash
if [[ "${FAKE_VLLM_EXIT_IMMEDIATELY:-}" == "1" ]]; then
  echo "simulated vLLM startup failure" >&2
  exit 42
fi
host=127.0.0.1
port=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    *) shift ;;
  esac
done
exec python3 - "$host" "$port" <<'PY'
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import sys
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/v1/models":
            self.send_response(404); self.end_headers(); return
        if os.environ.get("FAKE_VLLM_BAD_ROUTER") == "1":
            print("AttributeError: '_IncludedRouter' object has no attribute 'path'", flush=True)
            self.send_response(500); self.end_headers(); return
        body = b'{"object":"list","data":[]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *_args):
        pass
ThreadingHTTPServer((sys.argv[1], int(sys.argv[2])), Handler).serve_forever()
PY
""",
                encoding="utf-8",
            )

            for executable in (yq, lsof, vllm):
                executable.chmod(0o755)

            vllm_port = free_port()
            master_port = free_port()
            while master_port == vllm_port:
                master_port = free_port()
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}{os.pathsep}{env.get('PATH', '')}",
                    "FAKE_VLLM_PORT": str(vllm_port),
                    "FAKE_MASTER_PORT": str(master_port),
                    "SERVICE_RUN_ID": "fake-vllm-test",
                    "SERVICE_RUN_LOG_DIR": str(log_dir),
                    "SERVICE_PID_DIR": str(pid_dir),
                    "INFERENCE_OPS_START_TIMEOUT_SECONDS": "10",
                }
            )

            vllm_pid = 0
            legacy_process = None
            try:
                subprocess.run(
                    ["bash", str(START_SCRIPT), "start", str(config)],
                    env=env,
                    check=True,
                    timeout=20,
                    capture_output=True,
                    text=True,
                )
                status = json.loads((log_dir / "status.json").read_text(encoding="utf-8"))
                self.assertEqual(2, status["schema_version"])
                self.assertEqual("vllm", status["provider"])
                self.assertEqual({"vllm": vllm_port, "master": master_port}, status["ports"])
                self.assertEqual("finished", status["status"])
                self.assertTrue((log_dir / "vllm.log").is_file())
                self.assertEqual({"start-service.pid", "vllm.pid"}, {p.name for p in pid_dir.glob("*.pid")})
                vllm_pid = int((pid_dir / "vllm.pid").read_text(encoding="utf-8"))
                os.kill(vllm_pid, 0)

                # A schema-v1 deployment may have left an old medical-service PID.
                # The v2 stop path must still terminate that recorded process.
                legacy_process = subprocess.Popen(
                    [
                        "setsid",
                        "env",
                        "SERVICE_RUN_ID=fake-vllm-test",
                        "python3",
                        "-c",
                        "import time; time.sleep(60)",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                (pid_dir / "inference.pid").write_text(
                    str(legacy_process.pid), encoding="utf-8"
                )

                subprocess.run(
                    ["bash", str(START_SCRIPT), "stop", str(config)],
                    env=env,
                    check=True,
                    timeout=20,
                    capture_output=True,
                    text=True,
                )
                stopped = json.loads((log_dir / "status.json").read_text(encoding="utf-8"))
                self.assertEqual("stopped", stopped["status"])
                legacy_process.wait(timeout=5)

                failed_pid_dir = root / "failed-pids"
                failed_log_dir = root / "failed-logs"
                failed_pid_dir.mkdir()
                failed_log_dir.mkdir()
                failed_env = dict(env)
                failed_env.update(
                    {
                        "SERVICE_RUN_ID": "fake-vllm-failed-test",
                        "SERVICE_RUN_LOG_DIR": str(failed_log_dir),
                        "SERVICE_PID_DIR": str(failed_pid_dir),
                        "FAKE_VLLM_EXIT_IMMEDIATELY": "1",
                        "INFERENCE_OPS_START_TIMEOUT_SECONDS": "30",
                    }
                )
                started = time.monotonic()
                failed = subprocess.run(
                    ["bash", str(START_SCRIPT), "start", str(config)],
                    env=failed_env,
                    check=False,
                    timeout=10,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(0, failed.returncode)
                self.assertLess(time.monotonic() - started, 10)
                failed_status = json.loads(
                    (failed_log_dir / "status.json").read_text(encoding="utf-8")
                )
                self.assertEqual("failed", failed_status["status"])
                self.assertIn("vLLM exited", failed_status["error"])
                self.assertIsNotNone(failed_status["finished_at"])

                router_pid_dir = root / "router-pids"
                router_log_dir = root / "router-logs"
                router_pid_dir.mkdir()
                router_log_dir.mkdir()
                router_env = dict(env)
                router_env.update(
                    {
                        "SERVICE_RUN_ID": "fake-vllm-router-test",
                        "SERVICE_RUN_LOG_DIR": str(router_log_dir),
                        "SERVICE_PID_DIR": str(router_pid_dir),
                        "FAKE_VLLM_BAD_ROUTER": "1",
                        "INFERENCE_OPS_START_TIMEOUT_SECONDS": "30",
                    }
                )
                started = time.monotonic()
                router_failed = subprocess.run(
                    ["bash", str(START_SCRIPT), "start", str(config)],
                    env=router_env,
                    check=False,
                    timeout=10,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(0, router_failed.returncode)
                self.assertLess(time.monotonic() - started, 10)
                router_status = json.loads(
                    (router_log_dir / "status.json").read_text(encoding="utf-8")
                )
                self.assertEqual("failed", router_status["status"])
                self.assertIn("prometheus-fastapi-instrumentator", router_status["error"])
            finally:
                if vllm_pid:
                    try:
                        os.killpg(vllm_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if legacy_process and legacy_process.poll() is None:
                    try:
                        os.killpg(legacy_process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
