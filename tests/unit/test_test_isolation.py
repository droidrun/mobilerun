from pathlib import Path


def test_user_paths_are_isolated(tmp_path):
    from mobilerun.clients.local.login_store import credential_path
    from mobilerun.config.loader import ConfigLoader
    from mobilerun.config.settings import get_settings

    settings = get_settings()
    assert settings.MOBILERUN_CONFIG is None
    for path in (
        settings.MOBILERUN_USER_CONFIG_DIR,
        ConfigLoader.get_user_config_path(),
        credential_path(),
    ):
        assert path.is_relative_to(tmp_path)


def test_real_env_snapshot_repr_and_failure_hide_values(request):
    import pytest

    conftest = request.config.pluginmanager.get_plugin(
        str(Path(__file__).parents[1] / "conftest.py")
    )
    snapshot = conftest.RealEnv({"SAMPLE_API_KEY": "fake-credential-xyz"})
    assert "SAMPLE_API_KEY" in repr(snapshot)
    assert "fake-credential-xyz" not in repr(snapshot)
    assert "fake-credential-xyz" not in str(snapshot)
    with pytest.raises(AssertionError) as error:
        assert snapshot == {}
    assert "fake-credential-xyz" not in str(error.value)


def test_real_env_requires_smoke_marker_and_option(monkeypatch, tmp_path, request):
    from types import SimpleNamespace

    import pytest

    conftest = request.config.pluginmanager.get_plugin(
        str(Path(__file__).parents[1] / "conftest.py")
    )
    monkeypatch.setattr(
        conftest, "_real_secrets", conftest.RealEnv({"FAKE_API_KEY": "fake"})
    )
    for marked, enabled in [(False, False), (True, False), (False, True)]:
        request = SimpleNamespace(
            node=SimpleNamespace(
                get_closest_marker=lambda _, marked=marked: marked,
                path=tmp_path / "test_unit.py",
            ),
            config=SimpleNamespace(getoption=lambda _, enabled=enabled: enabled),
        )
        with pytest.raises((RuntimeError, pytest.skip.Exception)):
            conftest.real_env._fixture_function(monkeypatch, None, request)
        assert "FAKE_API_KEY" not in __import__("os").environ

    request.node.get_closest_marker = lambda _: True
    conftest.real_env._fixture_function(monkeypatch, None, request)
    assert __import__("os").environ["FAKE_API_KEY"] == "fake"
    monkeypatch.delenv("FAKE_API_KEY")
    request.node.get_closest_marker = lambda _: None
    request.node.path = Path(conftest.__file__).parent / "unit/smoke/test_example.py"
    conftest.real_env._fixture_function(monkeypatch, None, request)
    assert __import__("os").environ["FAKE_API_KEY"] == "fake"


def test_collection_and_fixture_remove_environment_overrides(tmp_path):
    import os
    import subprocess
    import sys

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
for name in names: os.environ[name] = "fake"
with pytest.MonkeyPatch.context() as patch:
    module.isolate_user_configuration._fixture_function(Path(sys.argv[2]), patch, SimpleNamespace(node=SimpleNamespace(get_closest_marker=lambda _: None)))
    assert all(name not in os.environ for name in names)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).parents[1] / "conftest.py"),
            str(tmp_path),
            *names,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_smoke_restores_original_android_sdk_path(tmp_path):
    import os
    import subprocess
    import sys

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
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(root / "conftest.py"),
            str(root / "unit/smoke/conftest.py"),
            env["HOME"],
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
