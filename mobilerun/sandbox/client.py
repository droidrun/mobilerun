"""Client and public models for the Mobilerun sandbox scripts API."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Literal, TypedDict, Unpack

import httpx
from pydantic import BaseModel

from mobilerun.clients.local.login_store import resolve_cloud_api_key
from mobilerun.config.settings import get_settings
from mobilerun.sandbox.errors import SandboxNotImplemented

_NOT_IMPLEMENTED = (
    "mobilerun sandbox: not implemented yet (waiting for the sandbox-api release)"
)


class RequestOptions(TypedDict, total=False):
    json: dict[str, Any]
    headers: dict[str, str]
    params: dict[str, str | bool]


class RunError(BaseModel):
    code: str
    message: str
    retryable: bool = False


class RunResult(BaseModel):
    outcome: Literal["succeeded", "partial", "failed"]
    result: Any = None
    error: RunError | None = None
    partial_reason: str | None = None
    state: dict[str, Any] = {}
    run_id: str | None = None
    logs: list[Any] = []


class DeployedScript(BaseModel):
    script_id: str
    version: int | None = None
    status: str
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None


def idempotency_key(name: str, source: str, manifest: dict[str, Any]) -> str:
    return hashlib.sha256(
        name.encode() + source.encode() + repr(sorted(manifest.items())).encode()
    ).hexdigest()


class SandboxClient:
    def __init__(self, *, transport: httpx.BaseTransport | None = None) -> None:
        """Prepare the configured client; requests await backend release."""
        settings = get_settings()
        self.base_url = settings.MOBILERUN_CLOUD_BASE_URL.rstrip("/")
        key = resolve_cloud_api_key()
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.timeout = settings.MOBILERUN_CLOUD_API_TIMEOUT
        self._transport = transport

    def _request(self, method: str, path: str, **kwargs: Unpack[RequestOptions]) -> Any:
        """Perform a backend request (currently unavailable)."""
        del method, path, kwargs
        raise SandboxNotImplemented(_NOT_IMPLEMENTED)

    def create_script(
        self,
        *,
        name: str,
        source: str,
        description: str | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a script with a stable idempotency key."""
        manifest = manifest or {}
        body = {"name": name, "source": source, "manifest": manifest}
        if description is not None:
            body["description"] = description
        return self._request(
            "POST",
            "/sandbox/scripts",
            json=body,
            headers={
                **self.headers,
                "Idempotency-Key": idempotency_key(name, source, manifest),
            },
        )

    def get_script(self, script_id: str) -> dict[str, Any]:
        """Retrieve a script's inspection status."""
        return self._request("GET", f"/sandbox/scripts/{script_id}")

    def list_scripts(self, *, name: str | None = None, latest: bool = False) -> Any:
        """List scripts, optionally by name and latest version."""
        return self._request(
            "GET",
            "/sandbox/scripts",
            params={
                k: v
                for k, v in {"name": name, "latest": latest or None}.items()
                if v is not None
            },
        )

    def get_source(self, script_id: str) -> Any:
        """Retrieve the source for a script."""
        return self._request("GET", f"/sandbox/scripts/{script_id}/source")

    def delete_script(self, script_id: str) -> Any:
        """Delete a script."""
        return self._request("DELETE", f"/sandbox/scripts/{script_id}")

    def wait_until_inspected(
        self, script_id: str, *, timeout: float, interval: float
    ) -> dict[str, Any]:
        """Wait until backend inspection completes."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            data = self.get_script(script_id)
            if data.get("status") in {"ready", "rejected"}:
                return data
            time.sleep(interval)
        raise TimeoutError(f"Sandbox script inspection timed out: {script_id}")

    def start_run(self, *_: object, **__: object) -> None:
        """Start a remote script run."""
        raise SandboxNotImplemented(_NOT_IMPLEMENTED)


def create_script(
    *,
    name: str,
    source: str,
    description: str | None = None,
    manifest: dict[str, Any] | None = None,
) -> None:
    """Create a remote script."""
    raise SandboxNotImplemented(_NOT_IMPLEMENTED)


def get_script(script_id: str) -> None:
    """Retrieve a remote script."""
    raise SandboxNotImplemented(_NOT_IMPLEMENTED)


def list_scripts(*, name: str | None = None, latest: bool = False) -> None:
    """List remote scripts."""
    raise SandboxNotImplemented(_NOT_IMPLEMENTED)


def delete_script(script_id: str) -> None:
    """Delete a remote script."""
    raise SandboxNotImplemented(_NOT_IMPLEMENTED)


def wait_until_inspected(script_id: str, *, timeout: float, interval: float) -> None:
    """Wait for script inspection to complete."""
    raise SandboxNotImplemented(_NOT_IMPLEMENTED)
