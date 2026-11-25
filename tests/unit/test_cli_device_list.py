import asyncio
import json
from types import SimpleNamespace

from click.testing import CliRunner


def test_cloud_listing_uses_settings_base_url(monkeypatch):
    from mobilerun.cli.devices.base import discovery
    from mobilerun.clients.cloud import client

    monkeypatch.setenv("MOBILERUN_CLOUD_BASE_URL", "https://configured.example")
    monkeypatch.setattr(discovery, "resolve_cloud_api_key", lambda: "fake-key")
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.devices = SimpleNamespace(list=self._list)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def _list(self, **kwargs):
            return SimpleNamespace(items=[], pagination=SimpleNamespace(has_next=False))

    monkeypatch.setattr(client, "AsyncMobilerun", FakeClient)
    asyncio.run(discovery.discover_devices(local=False))
    assert calls[0]["base_url"] == "https://configured.example"


def test_list_human_and_json(monkeypatch):
    from mobilerun.cli.devices import list as listing
    from mobilerun.cli.devices.base.discovery import DiscoveredDevice

    async def discover(**kwargs):
        return [
            DiscoveredDevice("usb", "local", None, "device", None),
            DiscoveredDevice("cloud", "cloud", "Phone", "ready", "android"),
        ]

    monkeypatch.setattr(listing, "discover_devices", discover)
    runner = CliRunner()
    result = runner.invoke(listing.devices, ["--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == [
        {"id": "usb", "source": "local", "default": False, "state": "device"},
        {
            "id": "cloud",
            "source": "cloud",
            "default": False,
            "name": "Phone",
            "state": "ready",
            "type": "android",
        },
    ]
    result = runner.invoke(listing.devices, [])
    assert result.exit_code == 0, result.output
    assert "Local devices · 1" in result.stdout
    assert "usb" in result.stdout
    assert "Cloud devices · 1" in result.stdout
    assert "Phone    cloud    ● ready    android" in result.stdout


def test_cloud_list_errors_are_fatal(monkeypatch):
    from mobilerun.cli.devices import list as listing
    from mobilerun.cli.devices.base import discovery

    async def fail(**kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(discovery, "resolve_cloud_api_key", lambda: "key")
    monkeypatch.setattr(discovery, "list_devices", fail)
    result = CliRunner().invoke(listing.devices, ["--cloud"])
    assert result.exit_code == 1
    assert "Error listing cloud devices: offline" in result.stderr


def test_cloud_list_uses_public_discovery(monkeypatch):
    from mobilerun.cli.devices import list as listing
    from mobilerun.cli.devices.base.discovery import DiscoveredDevice

    async def discover(*, local, cloud, base_url, billing, strict):
        assert local is False and cloud is True and strict is True
        return [DiscoveredDevice("cloud-1", "cloud", "Phone", "ready", "android")]

    monkeypatch.setattr(listing, "discover_devices", discover)
    result = CliRunner().invoke(listing.devices, ["--cloud", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)[0]["id"] == "cloud-1"
