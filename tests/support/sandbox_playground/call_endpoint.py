"""Call a deployed endpoint and display its result."""

import base64
import json
import math
import os
import re
import sys
import time
from pathlib import Path


def redact(value):
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str) and len(value) >= 1000:
        try:
            image = base64.b64decode(value, validate=True)
        except ValueError:
            return value
        return f"<base64 jpeg, {math.ceil(len(image) / 1024)} KB>"
    return value


def main(name: str, *, output_dir: Path, endpoints=None) -> int:
    """Fire an explicitly selected endpoint; tests inject a fake registry."""
    if not os.environ.get("MOBILERUN_API_KEY"):
        print(
            "MOBILERUN_API_KEY missing: run `source ~/.agent-env` first",
            file=sys.stderr,
        )
        return 1
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", name):
        print("Invalid endpoint name", file=sys.stderr)
        return 1
    path = output_dir / f"{name}.last.jpg"
    if path.is_symlink():
        print("Invalid screenshot path", file=sys.stderr)
        return 1
    try:
        if endpoints is None:
            from mobilerun.sandbox import cloud

            endpoints = cloud
        started = time.monotonic()
        run = endpoints[name].fire()
        print(f"fired invocation {run.invocation_id}")
        print(f"execution {run.id} running")
        result = run.result()
        if hasattr(result, "model_dump"):
            result = result.model_dump(mode="json")
        print(f"done in {time.monotonic() - started:.1f} s")
        display = redact(result)
        screenshot = result.get("screenshot")
        if screenshot:
            image = base64.b64decode(screenshot, validate=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image)
            display["screenshot"] = f"<base64 jpeg, {math.ceil(len(image) / 1024)} KB>"
            print(json.dumps(display, indent=2, ensure_ascii=False))
            print(f"screenshot: {path}")
        else:
            print(json.dumps(display, indent=2, ensure_ascii=False))
        return 0
    except Exception:
        print("Endpoint call failed", file=sys.stderr)
        return 1
