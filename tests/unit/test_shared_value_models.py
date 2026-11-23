import importlib

import pytest


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("ProxyStatus", "proxy"),
        ("RecordedAction", "recording"),
        ("CredentialMetadata", "credentials"),
        ("UIElement", "ui"),
        ("PaymentSnapshot", "checkout"),
    ],
)
def test_public_value_model_identity(name: str, module: str) -> None:
    models = importlib.import_module("mobilerun.models")
    home = importlib.import_module(f"mobilerun.models.{module}")
    assert getattr(models, name) is getattr(home, name)


def test_existing_core_exports_keep_model_identity() -> None:
    from mobilerun.core.proxy import ProxyStatus
    from mobilerun.core.state import UIElement
    from mobilerun.models import ProxyStatus as ModelProxyStatus
    from mobilerun.models import UIElement as ModelUIElement

    assert ProxyStatus is ModelProxyStatus
    assert UIElement is ModelUIElement
