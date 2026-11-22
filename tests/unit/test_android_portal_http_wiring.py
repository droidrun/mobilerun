import pytest
from mobilerun_core_local.driver.android import AndroidPortalHttpDriver

from mobilerun.clients.portal.http import (
    create_android_portal_http_driver,
    is_portal_http_url,
)

PORTAL_URL = "http://192.168.1.20:8080"


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("http://192.168.1.20:8080", True),
        ("HTTPS://phone.local:8080", True),
        ("emulator-5554", False),
        ("192.168.1.20:5555", False),
        ("", False),
        (None, False),
    ],
)
def test_is_portal_http_url(device, expected):
    assert is_portal_http_url(device) is expected


def test_portal_http_driver_requires_token():
    with pytest.raises(ValueError, match="MOBILERUN_DEVICE_TOKEN"):
        create_android_portal_http_driver(PORTAL_URL, None)


def test_portal_http_driver_normalizes_url():
    driver = create_android_portal_http_driver(PORTAL_URL + "/", "sekret")

    assert isinstance(driver, AndroidPortalHttpDriver)
    assert driver.url == PORTAL_URL
    assert driver.token == "sekret"
