from datetime import datetime, timezone


def to_api_timestamp(value: datetime) -> str:
    """UTC ISO-8601 with a trailing Z, as the Mobilerun API requires."""
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
