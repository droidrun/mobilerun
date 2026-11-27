"""OAuth 2.0 Device Authorization Grant (RFC 8628) client for Mobilerun.

Python port of the mobilerun-ios ``pkg/deviceauth`` flow: request a device
code, poll the better-auth token endpoint until the user approves, and return
the session access token. Transport + storage only — callers handle UI.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, TypedDict, Unpack

import httpx

from mobilerun.clients.cloud.billing import requires
from mobilerun.clients.cloud.errors import cloud_error
from mobilerun.config.settings import get_settings
from mobilerun.errors import CloudAPIError, DeviceAuthError
from mobilerun.models.billing import Billing


@dataclass
class CodeResponse:
    """Values returned by the OAuth device authorization flow."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int


_RequestOptions = TypedDict(
    "_RequestOptions",
    {"json": dict[str, str], "headers": dict[str, str], "timeout": float},
    total=False,
)


_GRANT_TYPE_DEVICE_CODE = "urn:ietf:params:oauth:grant-type:device_code"


async def _request(
    method: str, url: str, **kwargs: Unpack[_RequestOptions]
) -> httpx.Response:
    """Make an authentication request without exposing its URL."""
    try:
        async with httpx.AsyncClient() as client:
            return await client.request(method, url, **kwargs)
    except httpx.HTTPError as exc:
        raise cloud_error(exc, "Contact authentication service") from exc


@requires(Billing.NO_COMMERCE)
async def request_code(
    auth_url: str, client_id: str, *, billing: Billing
) -> CodeResponse:
    """Request OAuth device and user codes."""
    resp = await _request(
        "POST",
        auth_url.rstrip("/") + "/device/code",
        json={"client_id": client_id, "scope": "openid profile email"},
        headers={"Accept": "application/json"},
        timeout=get_settings().MOBILERUN_DEVICE_AUTH_REQUEST_TIMEOUT,
    )
    if not (200 <= resp.status_code < 300):
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise cloud_error(exc, "Request device code") from exc
    d = resp.json()
    return CodeResponse(
        device_code=d["device_code"],
        user_code=d["user_code"],
        verification_uri=d.get("verification_uri", ""),
        verification_uri_complete=d.get("verification_uri_complete", ""),
        expires_in=int(d.get("expires_in", 0)),
        interval=int(d.get("interval", 5)),
    )


async def _exchange(auth_url: str, client_id: str, device_code: str) -> str:
    """Exchange an approved device code for a token."""
    resp = await _request(
        "POST",
        auth_url.rstrip("/") + "/device/token",
        json={
            "grant_type": _GRANT_TYPE_DEVICE_CODE,
            "device_code": device_code,
            "client_id": client_id,
        },
        headers={"Accept": "application/json"},
        timeout=get_settings().MOBILERUN_DEVICE_AUTH_REQUEST_TIMEOUT,
    )
    if 200 <= resp.status_code < 300:
        token = resp.json().get("access_token")
        if not token:
            raise CloudAPIError(operation="Exchange device code", code="missing_token")
        return token
    try:
        err = resp.json()
        code = err.get("error")
    except Exception:
        code = None
    allowed = {"authorization_pending", "slow_down", "access_denied", "expired_token"}
    if code in allowed and resp.status_code == 400:
        raise DeviceAuthError(code)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise cloud_error(exc, "Exchange device code") from exc
    raise CloudAPIError(operation="Exchange device code", code="auth_failed")


@requires(Billing.NO_COMMERCE)
async def poll_token(
    auth_url: str, client_id: str, code: CodeResponse, *, billing: Billing
) -> str:
    """Poll for approval and return the access token.

    Handles ``authorization_pending`` and ``slow_down`` internally. Raises
    ``DeviceAuthError`` on terminal errors (access_denied, expired_token, ...).
    """
    interval = max(
        float(code.interval), get_settings().MOBILERUN_DEVICE_AUTH_MIN_POLL_INTERVAL
    )
    deadline = time.monotonic() + code.expires_in

    while True:
        await asyncio.sleep(interval)
        try:
            return await _exchange(auth_url, client_id, code.device_code)
        except DeviceAuthError as e:
            if e.code == "authorization_pending":
                pass
            elif e.code == "slow_down":
                interval += get_settings().MOBILERUN_DEVICE_AUTH_SLOW_DOWN_BUMP
            else:
                raise
        if time.monotonic() > deadline:
            raise DeviceAuthError("expired_token", "device code expired")


@requires(Billing.NO_COMMERCE)
async def fetch_session(
    auth_url: str, token: str, *, billing: Billing
) -> Optional[dict]:
    """Retrieve the session; None means no active session."""
    resp = await _request(
        "GET",
        auth_url.rstrip("/") + "/get-session",
        headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
        timeout=get_settings().MOBILERUN_DEVICE_AUTH_REQUEST_TIMEOUT,
    )
    if not (200 <= resp.status_code < 300):
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise cloud_error(exc, "Retrieve authentication session") from exc
    return resp.json()  # better-auth returns null when there's no session
