"""Load operator-owned environment files without executing shell commands."""
import os
from pathlib import Path
import re
import shlex


def load_runtime_environment(environ=None):
    """Existing process variables win; the package-local .env is optional."""
    environ = os.environ if environ is None else environ
    configured = str(environ.get("INFERENCE_OPS_ENV_FILE") or "").strip()
    path = Path(configured).expanduser() if configured else Path(__file__).resolve().parents[1] / ".env"
    if not path.is_file():
        if configured:
            raise ValueError(f"INFERENCE_OPS_ENV_FILE does not exist: {path}")
        return

    values = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
        if not match:
            raise ValueError(f"Invalid environment assignment in {path} at line {number}")
        key, value = match.groups()
        if value.startswith(("'", '"')):
            lexer = shlex.shlex(value, posix=True)
            lexer.whitespace_split = True
            try:
                parts = list(lexer)
            except ValueError:
                raise ValueError(f"Invalid environment value in {path} at line {number}") from None
            if len(parts) != 1:
                raise ValueError(f"Invalid environment value in {path} at line {number}")
            value = parts[0]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        values[key] = value
    for key, value in values.items():
        environ.setdefault(key, value)
