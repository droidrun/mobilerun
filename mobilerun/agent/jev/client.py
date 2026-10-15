"""Async client for the TypeSafe inference API."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import math
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from mobilerun.agent.jev.errors import (
    JevAuthError,
    JevConfigError,
    JevError,
    JevRateLimitError,
    JevRequestError,
    JevResponseError,
    JevServiceError,
    JevTimeoutError,
    JevTransportError,
)
from mobilerun.config_manager import env_keys

logger = logging.getLogger("mobilerun")

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
REQUEST_ID_HEADER = "x-typesafe-request-id"
MISSING_KEY_MESSAGE = (
    "No TypeSafe API key found. Set TYPESAFE_API_KEY or run 'mobilerun configure jev'."
)

_RETRY_STATUSES = frozenset({408, 429, *range(500, 600)})
_MAX_RETRY_AFTER = 10.0
_MAX_BACKOFF = 5.0
_MAX_DETAIL_CHARS = 300
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_KEY_SOURCES = ("auto", "env", "file")
_AUTH_HINT = "Check TYPESAFE_API_KEY or the key saved with 'mobilerun configure jev'."


@dataclass(frozen=True)
class JevResponse:
    """A parsed ``POST /v1/systemone`` response."""

    answers: dict[str, dict[str, Any]]
    model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    request_id: str | None = None
    latency_ms: int = 0


def resolve_typesafe_api_key(source: str = "auto") -> str:
    """Return the TypeSafe API key from the saved key file and/or the environment.

    Args:
        source: "auto" (saved file first, then TYPESAFE_API_KEY), "env" or "file".

    Raises:
        JevConfigError: No key is available from the requested source.
    """
    if source not in _KEY_SOURCES:
        raise JevConfigError(
            "agent.jev.api_key_source must be 'auto', 'env' or 'file'."
        )
    key = (env_keys.resolve_env_key("typesafe", source) or "").strip()
    if key:
        return key
    if source == "env":
        raise JevConfigError(
            "No TypeSafe API key found. Set TYPESAFE_API_KEY "
            "(agent.jev.api_key_source is 'env')."
        )
    if source == "file":
        raise JevConfigError(
            "No saved TypeSafe API key found. Run 'mobilerun configure jev' "
            "(agent.jev.api_key_source is 'file')."
        )
    raise JevConfigError(MISSING_KEY_MESSAGE)


def _user_agent() -> str:
    try:
        return f"mobilerun/{importlib.metadata.version('mobilerun')}"
    except importlib.metadata.PackageNotFoundError:
        return "mobilerun/unknown"


def _validate_key(api_key: Any) -> str:
    if not isinstance(api_key, str) or not api_key.strip():
        raise JevConfigError(MISSING_KEY_MESSAGE)
    key = api_key.strip()
    if not key.isascii() or not key.isprintable() or " " in key:
        raise JevConfigError(
            "The TypeSafe API key is malformed: it must be printable ASCII without "
            "whitespace. Check TYPESAFE_API_KEY or run 'mobilerun configure jev'."
        )
    return key


def _display_url(base_url: str) -> str:
    try:
        url = httpx.URL(base_url)
    except Exception as e:  # httpx.InvalidURL and friends
        raise JevConfigError(
            f"Invalid TypeSafe base_url ({type(e).__name__})."
        ) from None
    if url.scheme not in ("http", "https") or not url.host:
        raise JevConfigError("TypeSafe base_url must be an http(s) URL.")
    if url.scheme == "http" and url.host not in _LOOPBACK_HOSTS:
        raise JevConfigError(
            "TypeSafe base_url must use https (http only for localhost)."
        )
    return str(url.copy_with(username=None, password=None, query=None, fragment=None))


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _parse_retry_after(headers: httpx.Headers) -> float | None:
    """Return the server-requested delay in seconds, if any."""
    raw = headers.get("retry-after-ms")
    if raw is not None:
        try:
            value = float(raw.strip() or "0")
        except ValueError:
            value = math.nan
        if math.isfinite(value) and value >= 0:
            return value / 1000
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw.strip() or "0")
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(raw).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return None
    return value if math.isfinite(value) and value >= 0 else None


def _extract_detail(body: Any) -> str | None:
    """Pull a human-readable message out of an error body (TypeSafe SDK order)."""
    if isinstance(body, str):
        return body or None
    if not isinstance(body, dict):
        return None
    error, message, detail = body.get("error"), body.get("message"), body.get("detail")
    if isinstance(error, str):
        return error
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(message, str):
        return message
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict) and isinstance(detail.get("message"), str):
        return detail["message"]
    if isinstance(detail, list):
        # Validation entries also carry "input"/"ctx" (echoed request data); skip them.
        parts = []
        for entry in detail:
            if not isinstance(entry, dict) or not isinstance(entry.get("msg"), str):
                continue
            loc = entry.get("loc")
            path = (
                ".".join(str(item) for item in loc if item != "body")
                if isinstance(loc, list)
                else ""
            )
            parts.append(f"{path}: {entry['msg']}" if path else entry["msg"])
        return "; ".join(parts) or None
    return None


def _parse_usage(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {}
    return {
        key: int(value)
        for key, value in usage.items()
        if isinstance(key, str)
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    }


def _response_detail(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        text = response.text.strip()
        # Skip HTML error pages; they are noise in a one-line message.
        if not text or text.startswith("<"):
            return None
        # Not truncated here: the key must be scrubbed before any cut.
        return text
    return _extract_detail(body)


class JevClient:
    """Async client for TypeSafe's ``/v1/systemone`` and ``/v1/models`` endpoints.

    Inference has no side effects, so timeouts, connection errors, 408, 429 and
    5xx responses are retried. The API key is held privately and never appears
    in messages, logs or ``repr``.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_delay: float = 0.5,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = _validate_key(api_key)
        if not isinstance(model, str) or not model.strip():
            raise JevConfigError("The Jev model name is empty.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise JevConfigError(
                "The TypeSafe timeout must be a positive number of seconds."
            )
        if max_retries < 0:
            raise JevConfigError("The TypeSafe max_retries must not be negative.")
        self.model = model.strip()
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._display_base_url = _display_url(self.base_url).rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.retry_delay = max(0.0, float(retry_delay))
        self._owns_http = http_client is None
        self._http = http_client if http_client is not None else httpx.AsyncClient()
        self._user_agent = _user_agent()

    def __repr__(self) -> str:
        return f"JevClient(model={self.model!r}, base_url={self._display_base_url!r})"

    async def __aenter__(self) -> JevClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def systemone(self, state: Any, questions: dict[str, dict]) -> JevResponse:
        """Ask Jev one question set about ``state``.

        Raises:
            JevError: A subclass describing the failure; no action should follow.
        """
        body = {"model": self.model, "state": state, "questions": questions}
        try:
            # Lone surrogates from screen text must not abort the step.
            content = json.dumps(body, ensure_ascii=False, allow_nan=False).encode(
                "utf-8", errors="replace"
            )
        except (TypeError, ValueError) as e:
            raise JevRequestError(
                f"The Jev request could not be encoded as JSON ({type(e).__name__})."
            ) from None
        started = time.monotonic()
        response = await self._send("POST", "/v1/systemone", content)
        latency_ms = int((time.monotonic() - started) * 1000)
        request_id = response.headers.get(REQUEST_ID_HEADER)
        data = self._json_object(response)
        answers = data.get("answers")
        if not isinstance(answers, dict) or not all(
            isinstance(answer, dict) for answer in answers.values()
        ):
            raise self._response_error(
                response, "'answers' is missing or not an object of answers"
            )
        model = data.get("model")
        parsed = JevResponse(
            answers=answers,
            model=model if isinstance(model, str) else None,
            usage=_parse_usage(data.get("usage")),
            request_id=request_id,
            latency_ms=latency_ms,
        )
        suffix = f" (request {request_id})" if request_id else ""
        logger.debug(
            f"Jev response: model {parsed.model}, {latency_ms} ms, "
            f"{len(content)} request bytes{suffix}"
        )
        return parsed

    async def list_models(self) -> list[str]:
        """Return the model names TypeSafe lists (aliases only); doubles as a key check."""
        response = await self._send("GET", "/v1/models", None)
        models = self._json_object(response).get("models")
        if not isinstance(models, list):
            raise self._response_error(response, "'models' is missing or not a list")
        return [
            entry["name"]
            for entry in models
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        ]

    async def aclose(self) -> None:
        """Close the HTTP client if this instance created it."""
        if self._owns_http:
            await self._http.aclose()

    def _headers(self, attempt: int) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "User-Agent": self._user_agent,
        }
        if attempt:
            headers["X-TypeSafe-Retry-Count"] = str(attempt)
        return headers

    async def _send(
        self, method: str, path: str, content: bytes | None
    ) -> httpx.Response:
        url = f"{self.base_url}{path}"
        attempts = self.max_retries + 1
        attempt = 0
        while True:
            headers = self._headers(attempt)
            if content is not None:
                headers["Content-Type"] = "application/json"
            try:
                response = await self._http.request(
                    method,
                    url,
                    content=content,
                    headers=headers,
                    timeout=httpx.Timeout(self.timeout),
                )
            except httpx.TimeoutException as e:
                if attempt + 1 >= attempts:
                    raise JevTimeoutError(
                        self._scrub(
                            f"TypeSafe did not respond within {self.timeout:g}s "
                            f"({_plural(attempts, 'attempt')}). No action was executed."
                        )
                    ) from e
                reason, delay = type(e).__name__, self._backoff(attempt)
            except httpx.TransportError as e:
                if attempt + 1 >= attempts:
                    raise self._transport_error(e) from e
                reason, delay = type(e).__name__, self._backoff(attempt)
            except httpx.HTTPError as e:
                raise self._transport_error(e) from e
            else:
                status = response.status_code
                if 200 <= status < 300:
                    return response
                if status not in _RETRY_STATUSES or attempt + 1 >= attempts:
                    raise self._status_error(response, attempt + 1)
                request_id = response.headers.get(REQUEST_ID_HEADER)
                reason = f"HTTP {status}" + (
                    f" (request {request_id})" if request_id else ""
                )
                hinted = _parse_retry_after(response.headers)
                delay = (
                    min(hinted, _MAX_RETRY_AFTER)
                    if hinted is not None
                    else self._backoff(attempt)
                )
            logger.debug(
                self._scrub(
                    f"Jev request attempt {attempt + 1}/{attempts} failed: {reason}; "
                    f"retrying in {delay:.2f}s"
                )
            )
            await asyncio.sleep(delay)
            attempt += 1

    def _backoff(self, attempt: int) -> float:
        return min(self.retry_delay * 2**attempt, _MAX_BACKOFF)

    def _scrub(self, text: str) -> str:
        return text.replace(self._api_key, "[redacted]")

    def _clean_detail(self, detail: str | None) -> str | None:
        if not detail:
            return None
        text = " ".join(self._scrub(detail).split())
        if len(text) > _MAX_DETAIL_CHARS:
            text = text[: _MAX_DETAIL_CHARS - 1] + "…"
        return text or None

    def _transport_error(self, error: Exception) -> JevTransportError:
        return JevTransportError(
            f"Could not reach TypeSafe at {self._display_base_url} "
            f"({type(error).__name__}). No action was executed."
        )

    def _status_error(self, response: httpx.Response, attempts: int) -> JevError:
        status = response.status_code
        request_id = response.headers.get(REQUEST_ID_HEADER)
        tag = f"HTTP {status}" + (f", request {request_id}" if request_id else "")
        detail = self._clean_detail(_response_detail(response))
        tried = _plural(attempts, "attempt")
        error_type: type[JevError]
        if status == 401 or (
            status == 403 and (detail is None or "key" in detail.lower())
        ):
            error_type = JevAuthError
            message = f"TypeSafe rejected the API key ({tag}). {_AUTH_HINT}"
        elif status == 403:
            error_type = JevAuthError
            message = f"TypeSafe denied access ({tag}): {detail}. {_AUTH_HINT}"
        elif status == 429:
            error_type = JevRateLimitError
            message = f"TypeSafe rate limit reached ({tag}) after {tried}."
        elif status == 408:
            error_type = JevTimeoutError
            message = f"TypeSafe timed out the request ({tag}) after {tried}. No action was executed."
        elif status == 529:
            error_type = JevServiceError
            message = f"TypeSafe is overloaded ({tag}) after {tried}."
        elif status >= 500:
            error_type = JevServiceError
            message = f"TypeSafe failed with a server error ({tag}) after {tried}."
        elif status >= 400:
            error_type = JevRequestError
            message = f"TypeSafe rejected the request ({tag})" + (
                f": {detail}" if detail else "."
            )
        else:
            error_type = JevResponseError
            message = f"TypeSafe returned an unexpected response: {tag}."
        return error_type(self._scrub(message), status=status, request_id=request_id)

    def _response_error(self, response: httpx.Response, what: str) -> JevResponseError:
        request_id = response.headers.get(REQUEST_ID_HEADER)
        suffix = f" (request {request_id})" if request_id else ""
        return JevResponseError(
            self._scrub(f"TypeSafe returned an unexpected response: {what}{suffix}."),
            status=response.status_code,
            request_id=request_id,
        )

    def _json_object(self, response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError:
            raise self._response_error(response, "the body is not valid JSON") from None
        if not isinstance(data, dict):
            raise self._response_error(response, "the body is not a JSON object")
        return data
