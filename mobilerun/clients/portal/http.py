"""Portal HTTP URL helpers."""

from mobilerun_core_local.driver.android import AndroidPortalHttpDriver


def create_android_portal_http_driver(
    url: str, token: str | None
) -> AndroidPortalHttpDriver:
    """Return an unconnected Portal HTTP driver, without ADB."""
    if not token:
        raise ValueError(
            "Android Portal HTTP needs the Portal auth token. Copy it from the "
            "Portal app and set MOBILERUN_DEVICE_TOKEN or device.auth_token."
        )
    return AndroidPortalHttpDriver(url, token)


def is_portal_http_url(device: str | None) -> bool:
    """True when ``device`` is a Portal HTTP base URL rather than an ADB serial."""
    return bool(device) and device.lower().startswith(("http://", "https://"))
