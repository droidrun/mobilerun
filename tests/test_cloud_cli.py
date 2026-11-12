
from types import SimpleNamespace
from unittest.mock import Mock

import mobilerun_core
import pytest
from click.testing import CliRunner

import mobilerun.config_manager
from mobilerun.cli import device_commands, selection
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

    assert selection.resolve_cloud_api_key() == "new-token"


def test_resolve_cloud_api_key_accepts_old_env(monkeypatch, tmp_path):
    _saved_credential(monkeypatch, tmp_path)
    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    monkeypatch.setenv("MOBILERUN_CLOUD_API_KEY", "old-token")

    assert selection.resolve_cloud_api_key() == "old-token"


def test_resolve_cloud_api_key_empty_new_env_falls_back(monkeypatch, tmp_path):
    _saved_credential(monkeypatch, tmp_path)
    monkeypatch.setenv("MOBILERUN_API_KEY", "")
    monkeypatch.setenv("MOBILERUN_CLOUD_API_KEY", "old-token")

    assert selection.resolve_cloud_api_key() == "old-token"


def test_resolve_cloud_api_key_reads_primary_env(monkeypatch, tmp_path):
    from mobilerun.config_manager import credential_paths

    monkeypatch.setattr(credential_paths, "OAUTH_CREDENTIAL_DIR", tmp_path)
    monkeypatch.delenv("MOBILERUN_CLOUD_API_KEY", raising=False)
    monkeypatch.setenv("MOBILERUN_API_KEY", "primary-token")

    assert selection.resolve_cloud_api_key() == "primary-token"


def test_resolve_cloud_api_key_reads_saved_credential(monkeypatch, tmp_path):
    from mobilerun.config_manager import credential_paths

    monkeypatch.setattr(credential_paths, "OAUTH_CREDENTIAL_DIR", tmp_path)
    monkeypatch.delenv("MOBILERUN_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    (tmp_path / "mobilerun-cloud.json").write_text(
        '{"access_token": "saved-token", "auth_url": "https://example.test"}'
    )

    assert selection.resolve_cloud_api_key() == "saved-token"


def test_cloud_uuid_requires_credential_without_falling_back_to_adb(monkeypatch):
    import click
    import pytest

    from mobilerun.cli import selection
    from mobilerun.config_manager import MobileConfig

    monkeypatch.setattr(mobilerun.config_manager.ConfigLoader, "load", lambda _: MobileConfig())
    monkeypatch.setattr(selection, "resolve_cloud_api_key", lambda: None)
    with pytest.raises(click.ClickException, match="No cloud API key"):
        with device_commands.device_driver({"device": "123e4567-e89b-12d3-a456-426614174000", "api_url": "https://example.test/v1"}):
            pass


def test_devices_cloud_missing_key_exits_nonzero(monkeypatch):
    monkeypatch.setattr(selection, "resolve_cloud_api_key", lambda: None)

    result = CliRunner().invoke(cli, ["device", "list", "-t", "cloud"])

    assert result.exit_code != 0
    assert "MOBILERUN_API_KEY" in result.output


def test_cloud_facade_receives_url_key_and_id(monkeypatch):
    from mobilerun.cli import selection
    from mobilerun.config_manager import MobileConfig

    calls = []
    class FakeCore:
        def close(self):
            pass

        def __init__(self, **kwargs):
            calls.append(kwargs)
        def connect(self, device_id, **kwargs):
            calls.append({"device_id": device_id, **kwargs})
            return SimpleNamespace(close=Mock())

    monkeypatch.setattr(mobilerun.config_manager.ConfigLoader, "load", lambda _: MobileConfig())
    monkeypatch.setattr(mobilerun_core, "Mobilerun", FakeCore)
    monkeypatch.setattr(selection, "resolve_cloud_api_key", lambda: "test-key")
    device_id = "123e4567-e89b-12d3-a456-426614174000"
    with device_commands.device_driver({"device": device_id, "api_url": "https://example.test/v1"}):
        pass
    assert calls[0] == {"bearer": "test-key", "base_url": "https://example.test/v1"}
    assert calls[1]["device_id"] == device_id
    assert calls[1]["backend"] == "cloud"


def test_cloud_listing_errors_are_failures(monkeypatch):
    from mobilerun.cli import device_lifecycle

    class FakeCore:
        def close(self):
            pass

        def list_devices(self, **kwargs):
            assert kwargs["scope"] in {"cloud", "all"}
            raise RuntimeError("cloud broke")

    monkeypatch.setattr(device_lifecycle, "core_client", lambda *_, **kw: FakeCore())
    for flags in (["-t", "cloud"], ["--all"]):
        result = CliRunner().invoke(cli, ["device", "list", *flags])
        assert result.exit_code == 1
        assert "cloud broke" in result.output


def test_local_listing_uses_local_scope(monkeypatch):
    from mobilerun.cli import device_lifecycle

    class FakeCore:
        def close(self):
            pass

        def list_devices(self, **kwargs):
            assert kwargs["scope"] == "local"
            return [SimpleNamespace(device_id="serial", capabilities={"backend": "local-android-adb"}, name=None, state=None, device_type=None, close=Mock())]

    monkeypatch.setattr(device_lifecycle, "core_client", lambda *_, **kw: FakeCore())
    result = CliRunner().invoke(cli, ["device", "list", "-j"])
    assert result.exit_code == 0
    assert '"device_id": "serial"' in result.stdout


@pytest.mark.parametrize("primary,legacy,expected", [
    ("primary-token", "legacy-token", "primary-token"),
    ("", "legacy-token", "legacy-token"),
    (None, "legacy-token", "legacy-token"),
    ("", "", "saved-token"),
    (None, None, "saved-token"),
])
def test_cloud_key_env_order_and_empty_fallback(monkeypatch, primary, legacy, expected):
    from mobilerun.cli import deviceauth

    for name, value in (("MOBILERUN_API_KEY", primary), ("MOBILERUN_CLOUD_API_KEY", legacy)):
        monkeypatch.delenv(name, raising=False)
        if value is not None:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(deviceauth, "load_credentials", lambda: {"api_key": "saved-token"})
    assert selection.resolve_cloud_api_key() == expected
