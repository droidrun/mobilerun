"""One timeout-bounded polling loop for SMS and email codes."""

import asyncio
import time
from collections.abc import Awaitable, Callable


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
