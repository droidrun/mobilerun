"""CloudDevice resolves credentials at construction time."""

import json

import pytest

cloud = pytest.importorskip("mobilerun_core_local.driver.cloud")


def _device(monkeypatch, *, api_key=None, base_url=None):
    from mobilerun.core.devices.cloud import CloudDevice

    calls = []

    def init(self, device_id, api_key=None, base_url=None, **kwargs):
        calls.append((device_id, api_key, base_url, kwargs))
        self.device_id = device_id
        self._client = None
        self._write_client = None

    monkeypatch.setattr(cloud.CloudDriver, "__init__", init)
    device = CloudDevice("device-1", api_key=api_key, base_url=base_url)
    return device, calls[0]


def test_explicit_key_precedes_environment_and_saved_login(monkeypatch, tmp_path):
    monkeypatch.setenv("MOBILERUN_API_KEY", "environment-secret")
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path))
    path = tmp_path / "credentials" / "mobilerun-cloud.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"api_key": "saved-secret"}))

    _, call = _device(monkeypatch, api_key="explicit-secret")

    assert call[1] == "explicit-secret"


def test_environment_key_precedes_saved_login(monkeypatch, tmp_path):
    monkeypatch.setenv("MOBILERUN_API_KEY", "environment-secret")
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path))
    path = tmp_path / "credentials" / "mobilerun-cloud.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"api_key": "saved-secret"}))

    _, call = _device(monkeypatch)

    assert call[1] == "environment-secret"


@pytest.mark.parametrize("field", ["api_key", "access_token"])
def test_saved_login_key_variants(monkeypatch, tmp_path, field):
    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path))
    path = tmp_path / "credentials" / "mobilerun-cloud.json"
    path.parent.mkdir()
    path.write_text(json.dumps({field: "saved-secret"}))

    _, call = _device(monkeypatch)

    assert call[1] == "saved-secret"


def test_missing_credentials_raises_safe_auth_error(monkeypatch, tmp_path):
    from mobilerun.core.devices.cloud import CloudDevice
    from mobilerun.errors import CloudAuthRequired

    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path))

    with pytest.raises(CloudAuthRequired) as error:
        CloudDevice("device-1")
    assert "mobilerun account login" in str(error.value)
    assert "secret" not in str(error.value)


def test_default_base_url_comes_from_settings(monkeypatch):
    from mobilerun.config.settings import get_settings

    monkeypatch.setenv("MOBILERUN_API_KEY", "test-secret")
    _, call = _device(monkeypatch)

    assert call[2] == get_settings().MOBILERUN_CLOUD_BASE_URL


def test_credentials_are_not_exposed_in_device_repr_or_string(monkeypatch):
    monkeypatch.setenv("MOBILERUN_API_KEY", "never-print-this")
    device, _ = _device(monkeypatch)

    assert "never-print-this" not in repr(device)
    assert "never-print-this" not in str(device)
