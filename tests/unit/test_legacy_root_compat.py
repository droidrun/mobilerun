import importlib
import logging
import subprocess
import sys
import warnings

import pytest
from click.testing import CliRunner

from mobilerun.addons.agents import agents_available
from mobilerun.cli.app import cli


@pytest.mark.parametrize(
    "legacy,module,attribute,args",
    [
        ("run", "mobilerun.agents.cli.agent.run", "run", ["hello"]),
        ("devices", "mobilerun.cli.devices.list", "devices", ["--cloud"]),
        ("login", "mobilerun.cli.auth.login", "login", ["--client-id", "test"]),
        ("logout", "mobilerun.cli.auth.logout", "logout", []),
        ("whoami", "mobilerun.cli.auth.status", "whoami", []),
        ("connect", "mobilerun.cli.devices.connect", "connect", ["serial"]),
        ("disconnect", "mobilerun.cli.devices.disconnect", "disconnect", ["serial"]),
        ("setup", "mobilerun.cli.devices.setup", "setup", ["--latest"]),
        ("ping", "mobilerun.cli.devices.ping", "ping", []),
        (
            "configure",
            "mobilerun.agents.cli.agent.llm.configure",
            "configure",
            ["--model", "test"],
        ),
    ],
)
def test_legacy_command_delegates(monkeypatch, legacy, module, attribute, args):
    if module.startswith("mobilerun.agents."):
        pytest.importorskip(
            "mobilerun.agents", reason="agent command belongs to add-on"
        )
    command = getattr(importlib.import_module(module), attribute)
    received = []
    monkeypatch.setattr(command, "callback", lambda **kwargs: received.append(kwargs))
    result = CliRunner().invoke(cli, [legacy, *args])
    assert result.exit_code == 0, result.output
    assert received
    assert "deprecated" in result.stderr.lower()


def test_package_import_is_lazy_and_does_not_configure_logging():
    code = """import logging, sys, mobilerun
assert "mobilerun.agents" not in sys.modules
logger = logging.getLogger("mobilerun")
assert not logger.handlers
assert logger.level == logging.NOTSET and logger.propagate
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "name",
    [
        "MobileAgent",
        "MobileAgentState",
        "load_llm",
        "ResultEvent",
        "HtmlErrorPageFilter",
        "AgentConfig",
        "FastAgentConfig",
        "ManagerConfig",
        "ExecutorConfig",
        "AppCardConfig",
        "CredentialsConfig",
        "ToolsConfig",
        "LLMProfile",
        "AndroidDriver",
        "DeviceDriver",
        "RecordingDriver",
        "MacroPlayer",
        "replay_macro_file",
        "replay_macro_folder",
        "DroidAgent",
        "DroidAgentState",
        "DroidConfig",
    ],
)
def test_legacy_exports_warn(name):
    import mobilerun
    from mobilerun._legacy.aliases import _LEGACY_EXPORTS

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if (
            _LEGACY_EXPORTS[name][0].startswith("mobilerun.agents")
            and not agents_available()
        ):
            with pytest.raises(
                ModuleNotFoundError, match="pip install mobilerun-agents"
            ):
                getattr(mobilerun, name)
        else:
            assert getattr(mobilerun, name) is not None
    assert any(
        issubclass(w.category, DeprecationWarning) and name in str(w.message)
        for w in caught
    )


def test_module_entrypoint():
    result = subprocess.run(
        [sys.executable, "-m", "mobilerun", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "Commands:" in result.stdout


def test_configure_provider_subcommand(monkeypatch):
    pytest.importorskip("mobilerun.agents", reason="agent command belongs to add-on")
    from mobilerun.agents.cli.agent.llm.login.openai import configure_openai

    received = []
    monkeypatch.setattr(
        configure_openai, "callback", lambda **kwargs: received.append(kwargs)
    )
    result = CliRunner().invoke(cli, ["configure", "openai", "--model", "example"])
    assert result.exit_code == 0, result.output
    assert received[0]["model"] == "example"
    assert "agent llm login" in result.stderr


def test_default_logging_is_idempotent_and_replaceable():
    from mobilerun.logs.handlers import configure_logging, ensure_agent_logging

    logger = logging.getLogger("mobilerun")
    root = logging.getLogger()
    previous = logger.handlers[:], logger.level, logger.propagate
    root_handlers = root.handlers[:]
    try:
        root.handlers.clear()
        logger.handlers.clear()
        ensure_agent_logging()
        first = logger.handlers[:]
        assert len(first) == 1
        assert type(first[0]).__name__ == "CLILogHandler"
        ensure_agent_logging()
        assert logger.handlers == first
        replacement = logging.NullHandler()
        configure_logging(False, replacement)
        ensure_agent_logging()
        assert logger.handlers == [replacement]
    finally:
        logger.handlers, logger.level, logger.propagate = previous
        root.handlers = root_handlers


def test_run_legacy_short_options_and_device(monkeypatch):
    pytest.importorskip("mobilerun.agents", reason="agent command belongs to add-on")
    from mobilerun.agents.cli.agent.run import run

    received = []
    monkeypatch.setattr(run, "callback", lambda **kwargs: received.append(kwargs))
    result = CliRunner().invoke(
        cli, ["run", "hello", "-c", "config.yaml", "-a", "external", "-d", "serial"]
    )
    assert result.exit_code == 0, result.output
    assert received[0]["config"] == "config.yaml"
    assert received[0]["agent"] == "external"
    # `-d` is now the shared --device/-d option; @device_command merges it into DeviceTarget.
    assert received[0]["device_target"] == "serial"
    assert received[0]["target"] is None
