import pytest

from mobilerun.utils.redaction import mask_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.invalid:8443/path/", "https://example.invalid:8443/path/"),
        (
            "https://receiver.example/in?opaque-sentinel",
            "https://receiver.example/in?•••",
        ),
        (
            "http://user:secret@example.org/in?one=secret&bare&one=two#secret",
            "http://example.org/in?one=•••&•••&one=•••",
        ),
        ("file://host/private", "[redacted]"),
        ("https:///missing-host", "[redacted]"),
        ("https://[broken", "[redacted]"),
    ],
)
def test_mask_url(url, expected):
    assert mask_url(url) == expected
