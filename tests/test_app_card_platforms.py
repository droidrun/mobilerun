import asyncio
import json
from types import SimpleNamespace

import pytest

from mobilerun.app_cards.providers.composite_provider import CompositeAppCardProvider
from mobilerun.app_cards.providers.local_provider import LocalAppCardProvider
from mobilerun.app_cards.providers.server_provider import ServerAppCardProvider


@pytest.fixture
def cards_dir(tmp_path):
    (tmp_path / "android").mkdir()
    (tmp_path / "ios").mkdir()
    (tmp_path / "android" / "tiktok.md").write_text("ANDROID TIKTOK")
    (tmp_path / "ios" / "tiktok.md").write_text("IOS TIKTOK")
    (tmp_path / "shared.md").write_text("SHARED")
    (tmp_path / "app_cards.json").write_text(
        json.dumps(
            {
                "com.zhiliaoapp.musically": {
                    "android": "android/tiktok.md",
                    "ios": "ios/tiktok.md",
                },
                "com.example.notes": {"ios": "ios/tiktok.md", "default": "shared.md"},
                "com.example.both": "shared.md",
                "com.example.broken": ["shared.md"],
            }
        )
    )
    return tmp_path


def _load(provider, package, platform):
    return asyncio.run(provider.load_app_card(package, "goal", platform))


def test_local_provider_picks_the_card_for_each_platform(cards_dir) -> None:
    provider = LocalAppCardProvider(str(cards_dir))

    assert _load(provider, "com.zhiliaoapp.musically", "android") == "ANDROID TIKTOK"
    assert _load(provider, "com.zhiliaoapp.musically", "ios") == "IOS TIKTOK"
    assert _load(provider, "com.zhiliaoapp.musically", None) == ""


def test_local_provider_falls_back_to_default_and_plain_paths(cards_dir) -> None:
    provider = LocalAppCardProvider(str(cards_dir))

    assert _load(provider, "com.example.notes", "android") == "SHARED"
    assert _load(provider, "com.example.notes", "ios") == "IOS TIKTOK"
    assert _load(provider, "com.example.both", "ios") == "SHARED"
    assert _load(provider, "com.example.both", "android") == "SHARED"


def test_local_provider_ignores_an_invalid_mapping_value(cards_dir) -> None:
    provider = LocalAppCardProvider(str(cards_dir))

    assert _load(provider, "com.example.broken", "android") == ""


def test_server_provider_sends_the_platform(monkeypatch) -> None:
    sent = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json):
            sent.append(json)
            return SimpleNamespace(status_code=200, json=lambda: {"app_card": "CARD"})

    monkeypatch.setattr(
        "mobilerun.app_cards.providers.server_provider.httpx.AsyncClient", FakeClient
    )
    provider = ServerAppCardProvider(server_url="https://cards.example")

    assert _load(provider, "com.zhiliaoapp.musically", "ios") == "CARD"
    assert _load(provider, "com.zhiliaoapp.musically", "android") == "CARD"
    assert [payload["platform"] for payload in sent] == ["ios", "android"]


def test_composite_provider_passes_the_platform_to_the_local_fallback(
    cards_dir, monkeypatch
) -> None:
    provider = CompositeAppCardProvider(
        server_url="https://cards.example", app_cards_dir=str(cards_dir)
    )

    async def no_server_card(*args, **kwargs):
        return ""

    monkeypatch.setattr(provider.server_provider, "load_app_card", no_server_card)

    assert _load(provider, "com.zhiliaoapp.musically", "ios") == "IOS TIKTOK"


def test_manager_loads_the_card_for_the_device_platform(cards_dir) -> None:
    from mobilerun.agent.manager.manager_agent import ManagerAgent

    calls = []

    class RecordingProvider:
        async def load_app_card(self, package_name, instruction="", platform=None):
            calls.append((package_name, platform))
            return "CARD"

    manager = ManagerAgent.__new__(ManagerAgent)
    manager.app_card_config = SimpleNamespace(enabled=True)
    manager.app_card_provider = RecordingProvider()
    manager.shared_state = SimpleNamespace(
        current_package_name="com.zhiliaoapp.musically",
        instruction="goal",
        platform="iOS",
        app_card="",
    )

    asyncio.run(manager._load_app_card())

    assert calls == [("com.zhiliaoapp.musically", "ios")]
    assert manager.shared_state.app_card == "CARD"


@pytest.mark.parametrize(
    ("platform", "expected"),
    [("Android", "android"), ("iOS", "ios"), ("VisualRemote", None), (None, None)],
)
def test_manager_sends_only_android_or_ios(platform, expected) -> None:
    from mobilerun.agent.manager.manager_agent import _card_platform

    assert _card_platform(platform) == expected
