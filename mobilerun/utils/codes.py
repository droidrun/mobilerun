"""Verification-code extraction and polling."""

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping

_CODE = re.compile(r"(?<!\d)\d{4,8}(?!\d)")
_SPLIT_CODE = re.compile(r"(?<!\d)\d{2,4}(?:\s*-\s*|[ \t]+)\d{2,4}(?!\d)")


def extract_code(text: str | None) -> str | None:
    match = _CODE.search(text or "")
    return match.group() if match else None


def verification_read_codes(role: str, action: str, result: object) -> Iterator[str]:
    """Extract candidates from declared verification reads, never from sends."""
    if action == "wait_for_code" and isinstance(result, str):
        if result:
            yield result
        return
    if role not in {"phone_number", "mailbox"} or action not in {
        "messages",
        "emails",
        "read",
        "read_message",
        "list_messages",
        "inbox",
    }:
        return
    for item in result if isinstance(result, (list, tuple)) else (result,):
        if isinstance(item, str):
            yield from (
                m.group()
                for pattern in (_CODE, _SPLIT_CODE)
                for m in pattern.finditer(item)
            )
            continue
        get = (
            item.get
            if isinstance(item, Mapping)
            else lambda name, item=item: getattr(item, name, None)
        )
        code = get("code")
        if isinstance(code, str) and code:
            yield code
        for field in ("body", "text", "subject", "html"):
            text = get(field)
            if isinstance(text, str):
                yield from (
                    m.group()
                    for pattern in (_CODE, _SPLIT_CODE)
                    for m in pattern.finditer(text)
                )


async def wait_for_code(
    fetch: Callable[[str | None], Awaitable[str | None]],
    *,
    timeout: float = 60,
    sender: str | None = None,
) -> str:
    """Poll an async fetch(sender) returning a code or None."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for verification code")
        try:
            code = await asyncio.wait_for(fetch(sender), remaining)
        except TimeoutError:
            raise TimeoutError("Timed out waiting for verification code") from None
        if code:
            return code
        await asyncio.sleep(min(1, max(0, deadline - time.monotonic())))
