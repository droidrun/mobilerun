"""Keep tests independent of the developer's configuration and credentials."""

import os
import tempfile
from pathlib import Path

import pytest


def _isolation_environment(root):
    home = root / "home"
    home.mkdir(exist_ok=True)
    config = root / "config"
    return {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "XDG_CONFIG_HOME": str(root / "xdg-config"),
        "XDG_DATA_HOME": str(root / "xdg-data"),
        "XDG_CACHE_HOME": str(root / "xdg-cache"),
        "APPDATA": str(root / "appdata"),
        "LOCALAPPDATA": str(root / "localappdata"),
        "MOBILERUN_USER_CONFIG_DIR": str(config),
        "MOBILERUN_CONFIG": str(config / "config.yaml"),
    }


def _secret_names():
    return [
        key
        for key in os.environ
        if key.startswith("MOBILERUN_")
        or key.endswith(("_API_KEY", "_TOKEN", "_SECRET_KEY", "_OAUTH_CREDENTIAL_PATH"))
        or key
        in {
            "LANGFUSE_PUBLIC_KEY",
            "OTEL_EXPORTER_OTLP_HEADERS",
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
            "ANDROID_HOME",
            "ANDROID_SDK_ROOT",
            "PHOENIX_URL",
            "PHOENIX_PROJECT_NAME",
            "phoenix_url",
            "phoenix_project_name",
        }
    ]


class RealEnv:
    """Private environment snapshot whose public representation contains names only."""

    def __init__(self, values):
        self.__values = values.copy()

    def __repr__(self):
        return f"RealEnv(keys={sorted(self.__values)!r})"

    __str__ = __repr__

    def apply(self, monkeypatch):
        __tracebackhide__ = True
        for name in self.__values:
            monkeypatch.setenv(name, self.__values[name])
        if not os.environ.get("MOBILERUN_API_KEY") and os.environ.get(
            "MOBILERUN_API_KEY"
        ):
            monkeypatch.setenv("MOBILERUN_API_KEY", os.environ["MOBILERUN_API_KEY"])


# Capture only in memory before collection-time isolation. Never write these to disk.
_real_secrets = RealEnv(
    {
        key: os.environ[key]
        for key in _secret_names()
        if key.endswith(("_API_KEY", "_TOKEN", "_SECRET_KEY"))
        or key == "LANGFUSE_PUBLIC_KEY"
    }
)
_real_config_dir = Path(
    os.environ.get("MOBILERUN_USER_CONFIG_DIR")
    or (Path.home() / "Library/Application Support/droidrun")
)
_real_android_sdk = (
    os.environ.get("ANDROID_HOME")
    or os.environ.get("ANDROID_SDK_ROOT")
    or str(Path.home() / "Library/Android/sdk")
)


def pytest_addoption(parser):
    parser.addoption("--smoke", action="store_true", help="Enable real smoke tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--smoke"):
        skip = pytest.mark.skip(reason="smoke tests: run with --smoke")
        for item in items:
            if "smoke" in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def real_env(monkeypatch, isolate_user_configuration, request):
    smoke_dir = Path(__file__).parent / "unit" / "smoke"
    is_smoke = request.node.get_closest_marker(
        "smoke"
    ) or request.node.path.is_relative_to(smoke_dir)
    if not request.config.getoption("--smoke") or not is_smoke:
        raise RuntimeError("real_env requires a smoke test and --smoke")
    _real_secrets.apply(monkeypatch)
    return _real_secrets


@pytest.fixture
def real_android_sdk():
    return _real_android_sdk


@pytest.fixture
def copied_auth_profiles(tmp_path, isolate_user_configuration):
    import shutil

    source = _real_config_dir / "credentials" / "auth-profiles.json"
    destination = tmp_path / "config" / "credentials" / "auth-profiles.json"
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination.chmod(0o600)
    return destination


# Test modules can import application code during collection, before fixtures run.
_collection_home = tempfile.TemporaryDirectory(prefix="mobilerun-pytest-")
for _key in _secret_names():
    if not _key.startswith("MOBILERUN_SMOKE_"):
        os.environ.pop(_key, None)
os.environ.update(_isolation_environment(Path(_collection_home.name)))


@pytest.fixture(autouse=True)
def isolate_user_configuration(tmp_path, monkeypatch, request):
    for key in _secret_names():
        if not (
            key.startswith("MOBILERUN_SMOKE_")
            and request.node.get_closest_marker("smoke")
        ):
            monkeypatch.delenv(key, raising=False)
    for key, value in _isolation_environment(tmp_path).items():
        monkeypatch.setenv(key, value)


@pytest.fixture(autouse=True)
def isolate_device_discovery(monkeypatch, request):
    """Never enumerate real devices during unit tests."""
    if request.node.get_closest_marker("smoke"):
        return
    from unittest.mock import AsyncMock

    from mobilerun.clients.portal import discovery

    monkeypatch.setattr(discovery.adb, "list", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        discovery,
        "discover_ios_device",
        AsyncMock(side_effect=ValueError("No iOS device found")),
    )
