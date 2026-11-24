"""Safety checks for the introduction demo's command-line entrypoint."""

import asyncio
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from tests.demos import intro_demo


def test_import_does_not_load_agent_addon_or_deploy():
    code = """import builtins
import mobilerun.sandbox as sandbox

def blocked(*args, **kwargs):
    raise AssertionError('demo attempted deployment during import')

sandbox.deploy = blocked
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'mobilerun.agents' or name.startswith('mobilerun.agents.'):
        raise AssertionError('demo imported agents eagerly')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import tests.demos.intro_demo
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_sandbox_example_runs_locally():
    greeting = intro_demo.hello(intro_demo.GreetingInput(name="Ada"))
    assert greeting.message == "Hello, Ada!"


def test_main_without_agent_uses_base_device(monkeypatch):
    device = Mock(_ensure_connected=AsyncMock())
    cloud = Mock(_ensure_connected=AsyncMock())
    android = Mock(return_value=device)
    monkeypatch.setattr(intro_demo, "AGENT", False)
    monkeypatch.setattr(intro_demo, "AndroidDevice", android)
    monkeypatch.setattr(intro_demo, "CloudDevice", Mock(return_value=cloud))
    monkeypatch.setattr(intro_demo, "sandbox_and_skills", Mock())
    for name in (
        "local_demos",
        "cloud_demos",
        "pool_demo",
        "new_cloud_device_demo",
        "emulator_demo",
    ):
        monkeypatch.setattr(intro_demo, name, AsyncMock())
    asyncio.run(intro_demo.main())
    android.assert_called_once_with(intro_demo.LOCAL_SERIAL)
    intro_demo.local_demos.assert_awaited_once_with(device)
    intro_demo.cloud_demos.assert_awaited_once_with(cloud)


def test_help_exits_without_running_demo(monkeypatch):
    main = AsyncMock()
    run = Mock()
    monkeypatch.setattr(intro_demo, "main", main)
    monkeypatch.setattr(intro_demo.asyncio, "run", run)
    with pytest.raises(SystemExit) as exc:
        intro_demo.cli(["--help"])
    assert exc.value.code == 0
    main.assert_not_called()
    run.assert_not_called()


@pytest.mark.parametrize("new_phone", [False, True])
def test_declining_does_not_run_demo(monkeypatch, capsys, new_phone):
    main = AsyncMock()
    monkeypatch.setattr(intro_demo, "main", main)
    monkeypatch.setattr(intro_demo, "NEW_CLOUD_DEVICE", new_phone)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "n")
    intro_demo.cli([])
    main.assert_not_called()
    assert "cloud task" in prompts[0].lower()
    assert "llm" in prompts[0].lower()
    assert ("new cloud phone" in prompts[0].lower()) is new_phone
    assert "Cancelled" in capsys.readouterr().out


def test_yes_runs_main(monkeypatch):
    main = AsyncMock()
    monkeypatch.setattr(intro_demo, "main", main)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("unexpected prompt"))
    intro_demo.cli(["--yes"])
    main.assert_awaited_once_with()


def test_script_help_in_isolated_mode():
    root = Path(__file__).resolve().parents[2]
    script = root / "tests/demos/intro_demo.py"
    # Fail closed: if --help ever reaches asyncio.run, abort before the demo runs.
    launcher = (
        "import asyncio, runpy, sys\n"
        "def _blocked(coro, *a, **kw):\n"
        "    coro.close()\n"
        "    raise SystemExit('demo started during --help')\n"
        "asyncio.run = _blocked\n"
        f"sys.path[:] = [p for p in sys.path if p != {str(root)!r}]\n"
        f"sys.argv = [{str(script)!r}, '--help']\n"
        f"runpy.run_path({str(script)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", launcher],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--yes" in result.stdout
