"""Errors raised by the Jev agent. Messages are user-facing and secret-free."""

from __future__ import annotations


class JevError(Exception):
    """Base class for Jev failures."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.request_id = request_id


class JevConfigError(JevError):
    """Local configuration is missing or invalid, e.g. no API key."""


class JevAuthError(JevError):
    """TypeSafe rejected the API key (HTTP 401/403)."""


class JevRequestError(JevError):
    """TypeSafe rejected the request (permanent 4xx)."""


class JevRateLimitError(JevError):
    """TypeSafe kept rate limiting the request (HTTP 429)."""


class JevServiceError(JevError):
    """TypeSafe kept failing with a server error (HTTP 5xx/529)."""


class JevTimeoutError(JevError):
    """TypeSafe did not answer in time."""


class JevTransportError(JevError):
    """TypeSafe could not be reached."""


class JevResponseError(JevError):
    """TypeSafe answered with a body that cannot be used."""
