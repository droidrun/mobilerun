"""Sanitized cloud transport errors; never include server text or URLs."""

import httpx

from mobilerun.errors import (
    CloudAPIError,
    CloudAuthenticationError,
    CloudConflict,
    CloudConnectionError,
    CloudDeviceNotFound,
    CloudPermissionDenied,
    CloudRateLimited,
    CloudResourceNotFound,
    CloudServiceUnavailable,
    CloudTimeout,
    CloudValidationError,
    InvalidCloudDeviceId,
)


def cloud_error(
    exc: BaseException, operation: str, *, device: bool = False
) -> CloudAPIError:
    """Build a safe public error from transport metadata, not response content."""
    # SDK and httpx exception classes expose different optional metadata fields.
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = response.status_code
    request_id = getattr(exc, "request_id", None)
    if not request_id and response is not None:
        request_id = response.headers.get("x-request-id")
    kind = CloudAPIError
    code = "request_failed"
    if status == 401:
        kind = CloudAuthenticationError
    elif status == 403:
        kind = CloudPermissionDenied
    elif status == 404:
        kind = CloudDeviceNotFound if device else CloudResourceNotFound
    elif status == 409:
        kind = CloudConflict
    elif status in {400, 422}:
        kind = CloudValidationError
        body = getattr(exc, "body", None)
        detail = body.get("detail", []) if isinstance(body, dict) else []
        if isinstance(detail, list) and any(
            isinstance(item, dict) and "deviceId" in item.get("loc", [])
            for item in detail
        ):
            kind = InvalidCloudDeviceId
        if status == 422 and isinstance(body, dict):
            if (
                operation == "Create number"
                and body.get("code") == "unsupported_country"
            ):
                code = "unsupported_number_country"
            elif (
                operation == "Create cloud phone"
                and body.get("code") == "DEVICE_TYPE_UNAVAILABLE"
            ):
                code = "device_type_unavailable"
    elif status == 429:
        kind = CloudRateLimited
    elif type(status) is int and status >= 500:
        kind = CloudServiceUnavailable
    elif (
        isinstance(exc, (TimeoutError, httpx.TimeoutException))
        or type(exc).__name__ == "APITimeoutError"
    ):
        kind = CloudTimeout
    elif (
        isinstance(exc, httpx.RequestError)
        or type(exc).__name__ == "APIConnectionError"
    ):
        kind = CloudConnectionError
    return kind(operation=operation, status=status, request_id=request_id, code=code)
