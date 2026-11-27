"""Isolation and offline safety contracts for the pytest harness."""

import asyncio
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import platformdirs
import pytest

from mobilerun.clients.cloud import devices
from mobilerun.config.settings import get_settings
from mobilerun.core.devices import CloudDevice
from mobilerun.models import Billing, Plan
from tests.support import smoke_phone
from tests.support.paths import REPO_ROOT

TESTS = REPO_ROOT / "tests"
ROOT_CONFTEST = TESTS / "conftest.py"
NAME = "smoke run-123"


def test_user_paths_are_isolated(tmp_path):
    from mobilerun.clients.local.login_store import credential_path
    from mobilerun.config.loader import ConfigLoader

    settings = get_settings()
    assert settings.MOBILERUN_CONFIG is None
    for path in (
        settings.MOBILERUN_USER_CONFIG_DIR,
        ConfigLoader.get_user_config_path(),
        credential_path(),
    ):
        assert path.is_relative_to(tmp_path)


def test_real_env_snapshot_repr_and_failure_hide_values(request):
    conftest = request.config.pluginmanager.get_plugin(str(ROOT_CONFTEST))
    snapshot = conftest.RealEnv({"SAMPLE_API_KEY": "fake-credential-xyz"})
    assert "SAMPLE_API_KEY" in repr(snapshot)
    assert "fake-credential-xyz" not in repr(snapshot)
    assert "fake-credential-xyz" not in str(snapshot)
    with pytest.raises(AssertionError) as error:
        assert snapshot == {}
    assert "fake-credential-xyz" not in str(error.value)


def test_real_env_requires_smoke_marker_and_option(monkeypatch, tmp_path, request):
    conftest = request.config.pluginmanager.get_plugin(str(ROOT_CONFTEST))
    monkeypatch.setattr(
        conftest, "_real_secrets", conftest.RealEnv({"FAKE_API_KEY": "fake"})
    )
    for marked, enabled in [(False, False), (True, False), (False, True)]:
        request = NS(
            node=NS(
                get_closest_marker=lambda _, marked=marked: marked,
                path=tmp_path / "test_unit.py",
            ),
            config=NS(getoption=lambda _, enabled=enabled: enabled),
        )
        with pytest.raises((RuntimeError, pytest.skip.Exception)):
            conftest.real_env._fixture_function(monkeypatch, None, request)
        assert "FAKE_API_KEY" not in os.environ

    request.node.get_closest_marker = lambda _: True
    conftest.real_env._fixture_function(monkeypatch, None, request)
    assert os.environ["FAKE_API_KEY"] == "fake"
    monkeypatch.delenv("FAKE_API_KEY")
    request.node.get_closest_marker = lambda _: None
    request.node.path = Path(conftest.__file__).parent / "unit/smoke/test_example.py"
    conftest.real_env._fixture_function(monkeypatch, None, request)
    assert os.environ["FAKE_API_KEY"] == "fake"


def test_collection_and_fixture_remove_environment_overrides(tmp_path):
    names = [
        "ANDROID_SERIAL",
        "CODEX_HOME",
        "MOBILERUN_CONFIG",
        "MOBILERUN_SECRET_PASSWORD",
        "MOBILERUN_DEFAULT_DEVICE",
        "MOBILERUN_CLOUD_DEVICE_TYPE",
        "MOBILERUN_ANDROID_DEVICE_TYPE",
        "OPENAI_OAUTH_CREDENTIAL_PATH",
        "ANTHROPIC_OAUTH_CREDENTIAL_PATH",
        "GEMINI_OAUTH_CREDENTIAL_PATH",
        "GROK_OAUTH_CREDENTIAL_PATH",
        "SAMPLE_API_KEY",
        "SAMPLE_TOKEN",
        "SAMPLE_SECRET_KEY",
    ]
    env = os.environ.copy()
    env.update(dict.fromkeys(names, "fake"))
    script = """
import importlib.util, os, sys
from pathlib import Path
from types import SimpleNamespace
import pytest
spec = importlib.util.spec_from_file_location("isolator", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
names = sys.argv[3:]
assert all(name not in os.environ for name in names)
os.environ.update(dict.fromkeys(names, "fake"))
with pytest.MonkeyPatch.context() as patch:
    request = SimpleNamespace(node=SimpleNamespace(get_closest_marker=lambda _: None))
    module.isolate_user_configuration._fixture_function(Path(sys.argv[2]), patch, request)
    assert all(name not in os.environ for name in names)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT_CONFTEST),
            str(tmp_path),
            *names,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_mixed_collection_keeps_offline_environment_isolated(tmp_path):
    """Collect real nodes together, but never set up or run the smoke node."""
    ambient = tmp_path / "ambient"
    config = ambient / "config"
    config.mkdir(parents=True)
    script = """
import os, sys
from pathlib import Path
import pytest

ambient = Path(os.environ['HOME'])
reads = []

def record_reads(event, args):
    if event in {'open', 'os.listdir', 'os.scandir'} and isinstance(args[0], str):
        if Path(args[0]).absolute().is_relative_to(ambient):
            reads.append(args[0])

class Probe:
    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, items):
        offline, smoke = items
        assert offline.nodeid == sys.argv[1]
        assert smoke.nodeid == sys.argv[2]
        assert offline.get_closest_marker('smoke') is None
        assert {m.name for m in smoke.iter_markers() if m.name.startswith('smoke')} == {
            'smoke', 'smoke_cloud'
        }
        assert 'SAMPLE_API_KEY' not in os.environ
        assert not Path.home().is_relative_to(ambient)
        assert not Path(os.environ['MOBILERUN_USER_CONFIG_DIR']).is_relative_to(ambient)
        items[:] = [offline]

    def pytest_runtest_call(self, item):
        assert item.nodeid == sys.argv[1]
        with pytest.raises(RuntimeError, match='requires a smoke test'):
            item._request.getfixturevalue('real_env')
        assert 'SAMPLE_API_KEY' not in os.environ

sys.addaudithook(record_reads)
result = pytest.main(['-q', '-p', 'no:cacheprovider', *sys.argv[1:]], plugins=[Probe()])
assert not reads, reads
sys.exit(result)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "tests/unit/harness/test_isolation.py::test_user_paths_are_isolated",
            "tests/unit/smoke/test_sandbox_functions.py::test_device_free_sandbox_endpoint",
        ],
        cwd=REPO_ROOT,
        env={
            "HOME": str(ambient),
            "USERPROFILE": str(ambient),
            "MOBILERUN_USER_CONFIG_DIR": str(config),
            "SAMPLE_API_KEY": "fake-sentinel",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_smoke_restores_original_android_sdk_path(tmp_path):
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "real-home")
    env.pop("ANDROID_HOME", None)
    env.pop("ANDROID_SDK_ROOT", None)
    script = """
import importlib.util, os, sys
from pathlib import Path
import pytest

def load(path):
    spec = importlib.util.spec_from_file_location("isolator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
root = load(sys.argv[1])
smoke = load(sys.argv[2])
assert Path.home() != Path(sys.argv[3])
with pytest.MonkeyPatch.context() as patch:
    smoke.android_sdk_for_smoke._fixture_function(patch, root._real_android_sdk, None)
    assert os.environ["ANDROID_HOME"] == str(Path(sys.argv[3]) / "Library/Android/sdk")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT_CONFTEST),
            str(TESTS / "unit/smoke/conftest.py"),
            env["HOME"],
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_smoke_apk_defaults_to_user_cache_and_respects_override(monkeypatch, tmp_path):
    monkeypatch.delenv("MOBILERUN_SMOKE_APK", raising=False)
    expected = Path(platformdirs.user_cache_dir("droidrun")) / "smoke" / "counter.apk"
    assert expected.is_relative_to(tmp_path), "Expected the isolated pytest cache"
    assert get_settings().MOBILERUN_SMOKE_APK == str(expected)

    override = str(tmp_path / "custom.apk")
    monkeypatch.setenv("MOBILERUN_SMOKE_APK", override)
    assert get_settings().MOBILERUN_SMOKE_APK == override


def phone(id, name=NAME, billing="minute"):
    return NS(id=id, name=name, billing_strategy=billing)


@pytest.fixture
def inventory(monkeypatch):
    listed, terminate = AsyncMock(), AsyncMock()
    monkeypatch.setattr(devices, "list_devices", listed)
    monkeypatch.setattr(devices, "terminate", terminate)
    monkeypatch.setattr(smoke_phone, "cloud_client", lambda **kw: nullcontext(None))
    monkeypatch.setattr(smoke_phone.asyncio, "sleep", AsyncMock())
    return listed, terminate


def lost_phone():
    return NS(cloud_id=None, name=None, close=AsyncMock())


def test_single_exact_minute_match_terminated_and_verified(inventory):
    listed, terminate = inventory
    other = phone("other", NAME + "-other")
    listed.side_effect = [[other], [other, phone("owned")], [other]]
    asyncio.run(smoke_phone.close_smoke_phone(lost_phone(), NAME))
    terminate.assert_awaited_once_with("owned", billing=Billing.INCLUDED, client=None)
    assert listed.await_count == 3


@pytest.mark.parametrize(
    "matches",
    [
        [phone("one"), phone("two")],
        [phone("one"), phone("two", billing="subscription")],
        [phone("subscription-id", billing="subscription")],
    ],
)
def test_unsafe_matches_never_deleted(inventory, matches):
    listed, terminate = inventory
    listed.return_value = matches
    with pytest.raises(AssertionError) as caught:
        asyncio.run(smoke_phone.close_smoke_phone(lost_phone(), NAME))
    assert NAME in str(caught.value)
    assert all(item.id in str(caught.value) for item in matches)
    terminate.assert_not_awaited()


def test_readiness_failure_rolled_back_without_scanning(monkeypatch, inventory):
    listed, terminate = inventory
    failure = TimeoutError("readiness failed")
    monkeypatch.setattr(devices, "create", AsyncMock(return_value=phone("owned")))
    monkeypatch.setattr(devices, "wait_ready", AsyncMock(side_effect=failure))

    async def exercise():
        device = CloudDevice(
            api_key="fake", billing=Billing.PAID, plan=Plan.MINUTE, name=NAME
        )
        try:
            await device
        finally:
            await smoke_phone.close_smoke_phone(device, NAME)

    with pytest.raises(TimeoutError) as caught:
        asyncio.run(exercise())
    assert caught.value is failure and not getattr(failure, "__notes__", [])
    listed.assert_not_awaited()
    terminate.assert_awaited_once()


def test_missing_phone_preserves_create_failure_with_unresolved_note(inventory):
    listed, terminate = inventory
    listed.return_value = []
    failure = TimeoutError("create response lost")

    async def exercise():
        try:
            raise failure
        finally:
            await smoke_phone.close_smoke_phone(lost_phone(), NAME)

    with pytest.raises(TimeoutError) as caught:
        asyncio.run(exercise())
    assert caught.value is failure
    assert any(
        f"unresolved: phone named {NAME!r} may exist" in note
        for note in failure.__notes__
    )
    assert 1 < listed.await_count <= 30
    terminate.assert_not_awaited()
