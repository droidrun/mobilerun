import asyncio
from types import SimpleNamespace

from click.testing import CliRunner

from mobilerun.cli import device_commands
from mobilerun.cli.main import cli


def _saved_credential(monkeypatch, tmp_path):
    from mobilerun.config_manager import credential_paths

    monkeypatch.setattr(credential_paths, "OAUTH_CREDENTIAL_DIR", tmp_path)
    (tmp_path / "mobilerun-cloud.json").write_text(
        '{"access_token": "saved-token", "auth_url": "https://example.test"}'
    )


def test_resolve_cloud_api_key_prefers_new_env(monkeypatch, tmp_path):
    _saved_credential(monkeypatch, tmp_path)
    monkeypatch.setenv("MOBILERUN_API_KEY", "new-token")
    monkeypatch.setenv("MOBILERUN_CLOUD_API_KEY", "old-token")

    assert device_commands.resolve_cloud_api_key() == "new-token"


def test_resolve_cloud_api_key_accepts_old_env(monkeypatch, tmp_path):
    _saved_credential(monkeypatch, tmp_path)
    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    monkeypatch.setenv("MOBILERUN_CLOUD_API_KEY", "old-token")

    assert device_commands.resolve_cloud_api_key() == "old-token"


def test_resolve_cloud_api_key_empty_new_env_falls_back(monkeypatch, tmp_path):
    _saved_credential(monkeypatch, tmp_path)
    monkeypatch.setenv("MOBILERUN_API_KEY", "")
    monkeypatch.setenv("MOBILERUN_CLOUD_API_KEY", "old-token")

    assert device_commands.resolve_cloud_api_key() == "old-token"


def test_resolve_cloud_api_key_reads_saved_credential(monkeypatch, tmp_path):
    from mobilerun.config_manager import credential_paths

    monkeypatch.setattr(credential_paths, "OAUTH_CREDENTIAL_DIR", tmp_path)
    monkeypatch.delenv("MOBILERUN_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    (tmp_path / "mobilerun-cloud.json").write_text(
        '{"access_token": "saved-token", "auth_url": "https://example.test"}'
    )

    assert device_commands.resolve_cloud_api_key() == "saved-token"


def test_cloud_uuid_auto_routes_only_with_cloud_credential(monkeypatch):
    uuid = "123e4567-e89b-12d3-a456-426614174000"
    cloud_calls = []
    android_calls = []

    class FakeCloudDriver:
        def __init__(self, **kwargs):
            cloud_calls.append(kwargs)

        async def connect(self):
            pass

    class FakeAndroidDriver:
        def __init__(self, serial, use_tcp, portal_mode="auto"):
            android_calls.append((serial, use_tcp, portal_mode))

        async def connect(self):
            pass

    monkeypatch.setattr(device_commands, "resolve_cloud_api_key", lambda: "cloud-token")
    monkeypatch.setattr(
        "mobilerun_core_local.driver.cloud.CloudDriver",
        FakeCloudDriver,
    )
    driver, is_ios = asyncio.run(
        device_commands._create_driver(uuid, None, None, False)
    )

    assert isinstance(driver, FakeCloudDriver)
    assert is_ios is False
    assert cloud_calls[0]["device_id"] == uuid
    assert cloud_calls[0]["api_key"] == "cloud-token"

    cloud_calls.clear()
    monkeypatch.setattr(device_commands, "resolve_cloud_api_key", lambda: None)
    monkeypatch.setattr(
        device_commands.ConfigLoader,
        "load",
        lambda _path: SimpleNamespace(
            device=SimpleNamespace(
                serial="local-serial",
                use_tcp=False,
                platform="android",
                auto_setup=False,
                portal_mode="auto",
            )
        ),
    )
    monkeypatch.setattr(device_commands, "AndroidDriver", FakeAndroidDriver)
    driver, is_ios = asyncio.run(
        device_commands._create_driver(uuid, None, None, False)
    )

    assert isinstance(driver, FakeAndroidDriver)
    assert is_ios is False
    assert cloud_calls == []
    assert android_calls[-1] == (uuid, False, "auto")


def test_devices_cloud_missing_key_exits_nonzero(monkeypatch):
    monkeypatch.setattr("mobilerun.cli.main.resolve_cloud_api_key", lambda: None)

    result = CliRunner().invoke(cli, ["devices", "--cloud"])

    assert result.exit_code != 0
    assert "MOBILERUN_API_KEY" in result.output


def test_devices_cloud_sdk_error_exits_nonzero(monkeypatch):
    class FakeDevices:
        async def list(self):
            raise RuntimeError("boom")

    class FakeClient:
        def __init__(self, **_kwargs):
            self.devices = FakeDevices()

    monkeypatch.setattr("mobilerun.cli.main.resolve_cloud_api_key", lambda: "token")
    monkeypatch.setattr("mobilerun_sdk.AsyncMobilerun", FakeClient)

    result = CliRunner().invoke(cli, ["devices", "--cloud"])

    assert result.exit_code != 0
    assert "Error listing cloud devices" in result.output
    assert "boom" in result.output


def test_devices_cloud_empty_list_is_success(monkeypatch):
    class FakeResponse:
        items = []

    class FakeDevices:
        async def list(self):
            return FakeResponse()

    class FakeClient:
        def __init__(self, **_kwargs):
            self.devices = FakeDevices()

    monkeypatch.setattr("mobilerun.cli.main.resolve_cloud_api_key", lambda: "token")
    monkeypatch.setattr("mobilerun_sdk.AsyncMobilerun", FakeClient)

    result = CliRunner().invoke(cli, ["devices", "--cloud"])

    assert result.exit_code == 0
    assert "No cloud devices found" in result.output


def test_devices_plain_keeps_local_listing_when_cloud_errors(monkeypatch):
    class FakeAdbDevice:
        serial = "emulator-5554"

    class FakeDevices:
        async def list(self):
            raise RuntimeError("cloud broke")

    class FakeClient:
        def __init__(self, **_kwargs):
            self.devices = FakeDevices()

    async def fake_adb_list():
        return [FakeAdbDevice()]

    monkeypatch.setattr("mobilerun.cli.main.resolve_cloud_api_key", lambda: "token")
    monkeypatch.setattr("mobilerun.cli.main.adb.list", fake_adb_list)
    monkeypatch.setattr("mobilerun_sdk.AsyncMobilerun", FakeClient)

    result = CliRunner().invoke(cli, ["devices"])

    assert result.exit_code == 0, result.output
    assert "emulator-5554" in result.output
    assert "cloud broke" not in result.output
