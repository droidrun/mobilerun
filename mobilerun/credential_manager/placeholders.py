"""Credential placeholder resolution helpers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mobilerun.credential_manager.credential_manager import CredentialManager


CREDENTIAL_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.:\-]*)\s*\}\}")


async def resolve_credential_placeholders(
    value: Any,
    credential_manager: "CredentialManager | None",
) -> Any:
    """Resolve credential placeholders in a value without mutating the input.

    Strings may contain placeholders such as ``{{PASSWORD}}``. Known credential
    IDs are replaced with their secret values at execution time. Unknown
    placeholders are left untouched so non-secret templating keeps working.
    """
    if credential_manager is None:
        return value

    if isinstance(value, str):
        return await _resolve_string(value, credential_manager)

    if isinstance(value, list):
        return [
            await resolve_credential_placeholders(item, credential_manager)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            [
                await resolve_credential_placeholders(item, credential_manager)
                for item in value
            ]
        )

    if isinstance(value, dict):
        return {
            key: item
            if key == "secret_id"
            else await resolve_credential_placeholders(item, credential_manager)
            for key, item in value.items()
        }

    return value


async def _resolve_string(
    text: str,
    credential_manager: "CredentialManager",
) -> str:
    if "{{" not in text:
        return text

    matches = list(CREDENTIAL_PLACEHOLDER_RE.finditer(text))
    if not matches:
        return text

    try:
        available_keys = set(await credential_manager.get_keys())
    except Exception:
        return text

    parts: list[str] = []
    cursor = 0
    for match in matches:
        parts.append(text[cursor : match.start()])
        secret_id = match.group(1)
        if secret_id in available_keys:
            try:
                parts.append(await credential_manager.resolve_key(secret_id))
            except Exception:
                parts.append(match.group(0))
        else:
            parts.append(match.group(0))
        cursor = match.end()

    parts.append(text[cursor:])
    return "".join(parts)
