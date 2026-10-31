"""OAuth 2.0 Device Authorization Grant (RFC 8628) client for Mobilerun.

Python port of the mobilerun-ios ``pkg/deviceauth`` flow: request a device
code, poll the better-auth token endpoint until the user approves, and return
the session access token. Transport + storage only — callers handle UI.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

GRANT_TYPE_DEVICE_CODE = "urn:ietf:params:oauth:grant-type:device_code"

_REQUEST_TIMEOUT = 10.0
_MIN_POLL_INTERVAL = 5.0
_SLOW_DOWN_BUMP = 5.0


class DeviceAuthError(Exception):
    """An OAuth 2.0 error response from the token endpoint."""

    def __init__(self, code: str, description: str = "") -> None:
        self.code = code
        self.description = description
        super().__init__(f"{code}: {description}" if description else code)


@dataclass
class CodeResponse:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int


def _endpoint(auth_url: str, path: str) -> str:
    return auth_url.rstrip("/") + path


def request_code(auth_url: str, client_id: str) -> CodeResponse:
    """POST /device/code — returns the device + user codes."""
    resp = httpx.post(
        _endpoint(auth_url, "/device/code"),
        json={"client_id": client_id, "scope": "openid profile email"},
        headers={"Accept": "application/json"},
        timeout=_REQUEST_TIMEOUT,
    )
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(
            f"device-code endpoint returned {resp.status_code}: {resp.text.strip()}"
        )
    d = resp.json()
    return CodeResponse(
        device_code=d["device_code"],
        user_code=d["user_code"],
        verification_uri=d.get("verification_uri", ""),
        verification_uri_complete=d.get("verification_uri_complete", ""),
        expires_in=int(d.get("expires_in", 0)),
        interval=int(d.get("interval", 5)),
    )


def _exchange(auth_url: str, client_id: str, device_code: str) -> str:
    resp = httpx.post(
        _endpoint(auth_url, "/device/token"),
        json={
            "grant_type": GRANT_TYPE_DEVICE_CODE,
            "device_code": device_code,
            "client_id": client_id,
        },
        headers={"Accept": "application/json"},
        timeout=_REQUEST_TIMEOUT,
    )
    if 200 <= resp.status_code < 300:
        token = resp.json().get("access_token")
        if not token:
            raise RuntimeError("token response missing access_token")
        return token
    try:
        err = resp.json()
        code = err.get("error")
    except Exception:
        code = None
    if not code:
        raise RuntimeError(f"token endpoint returned {resp.status_code}")
    raise DeviceAuthError(code, err.get("error_description", ""))


def poll_token(auth_url: str, client_id: str, code: CodeResponse) -> str:
    """Poll /device/token until approval; return the access token.

    Handles ``authorization_pending`` and ``slow_down`` internally. Raises
    ``DeviceAuthError`` on terminal errors (access_denied, expired_token, ...).
    """
    interval = max(float(code.interval), _MIN_POLL_INTERVAL)
    deadline = time.monotonic() + code.expires_in

    while True:
        time.sleep(interval)
        try:
            return _exchange(auth_url, client_id, code.device_code)
        except DeviceAuthError as e:
            if e.code == "authorization_pending":
                pass
            elif e.code == "slow_down":
                interval += _SLOW_DOWN_BUMP
            else:
                raise
        if time.monotonic() > deadline:
            raise DeviceAuthError("expired_token", "device code expired")


def fetch_session(auth_url: str, token: str) -> Optional[dict]:
    """GET /get-session with the token as Bearer. None if no active session."""
    resp = httpx.get(
        _endpoint(auth_url, "/get-session"),
        headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
        timeout=_REQUEST_TIMEOUT,
    )
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"get-session endpoint returned {resp.status_code}")
    return resp.json()  # better-auth returns null when there's no session


# -- credential storage ------------------------------------------------------


def credential_path() -> Path:
    from mobilerun.config_manager.credential_paths import OAUTH_CREDENTIAL_DIR

    return OAUTH_CREDENTIAL_DIR / "mobilerun-cloud.json"


def _chmod_private(path: Path, mode: int) -> None:
    if os.name != "nt":
        os.chmod(path, mode)


def save_token(token: str, auth_url: str) -> Path:
    """Persist the session token to the credentials dir."""
    path = credential_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _chmod_private(path.parent, 0o700)

    payload = json.dumps({"access_token": token, "auth_url": auth_url})
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
        _chmod_private(tmp_path, 0o600)
        os.replace(tmp_path, path)
        _chmod_private(path, 0o600)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


def load_credentials() -> Optional[dict]:
    path = credential_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def clear_credentials() -> bool:
    """Delete the saved login. Returns True if a file was removed."""
    path = credential_path()
    if path.exists():
        path.unlink()
        return True
    return False
