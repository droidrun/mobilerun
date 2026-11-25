"""Keep transport diagnostics for signed transfers out of application logs."""

import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar

import httpx

_signed_transfer = ContextVar("mobilerun_signed_transfer", default=False)


def _allow_transport_log(record: logging.LogRecord) -> bool:
    return not _signed_transfer.get()


@contextmanager
def _signed_transfer_logs() -> Iterator[None]:
    """Silence only this task, including redirect URLs and httpcore header traces."""
    for logger in (
        "httpx",
        "mobilerun_sdk._base_client",
        "httpcore.http11",
        "httpcore.http2",
        "httpcore.connection",
        "httpcore.proxy",
        "httpcore.socks",
    ):
        # addFilter is idempotent. Leave it installed for concurrent transfers.
        logging.getLogger(logger).addFilter(_allow_transport_log)
    token = _signed_transfer.set(True)
    try:
        yield
    finally:
        _signed_transfer.reset(token)


@asynccontextmanager
async def _signed_client(client: httpx.AsyncClient) -> AsyncIterator[httpx.AsyncClient]:
    """Open and close an owned transfer client without logging signed URLs."""
    with _signed_transfer_logs():
        async with client as opened:
            yield opened
