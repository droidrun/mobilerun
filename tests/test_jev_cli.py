import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from click.testing import CliRunner

import mobilerun.agent.jev.client as jev_client_module
from mobilerun.agent.droid.events import FinalizeEvent
from mobilerun.agent.jev.errors import JevAuthError, JevTransportError
from mobilerun.agent.jev.events import JevDecisionEvent, JevEndEvent, JevStepEvent
from mobilerun.cli.event_handler import EventHandler
from mobilerun.cli.main import cli, run_command
from mobilerun.config_manager import ConfigLoader, env_keys
from mobilerun.config_manager.config_manager import JevConfig, MobileConfig
from mobilerun.config_manager.migrations import CURRENT_VERSION

SECRET = "ts-test-secret-4242"


@pytest.fixture(autouse=True)
def isolate_api_keys(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(env_keys, "AUTH_PROFILES_PATH", tmp_path / "auth-profiles.json")
    # setenv first so teardown also removes a key that save_env_keys exported.
    monkeypatch.setenv("TYPESAFE_API_KEY", "placeholder")
    monkeypatch.delenv("TYPESAFE_API_KEY")


@pytest.fixture
def config_path(monkeypatch, tmp_path):
    path = tmp_path / "config" / "config.yaml"
    monkeypatch.setattr(
        ConfigLoader, "get_user_config_path", classmethod(lambda cls: path)
    )
    monkeypatch.delenv("MOBILERUN_CONFIG", raising=False)
    monkeypatch.delenv("DROIDRUN_CONFIG", raising=False)
    return path


@pytest.fixture
def mobilerun_caplog(caplog):
    logger = logging.getLogger("mobilerun")
    previous = (logger.propagate, logger.level)
    logger.propagate = True
    logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="mobilerun")
    yield caplog
    logger.propagate, level = previous
    logger.setLevel(level)


def _saved_keys() -> dict:
    path = env_keys.AUTH_PROFILES_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("apiKeys", {})


def _install_fake_client(monkeypatch, *, models=None, error=None) -> list:
    instances = []

    class FakeJevClient:
        def __init__(self, api_key, **kwargs):
            self.api_key = api_key
            self.kwargs = kwargs
            self.closed = False
            instances.append(self)

        async def list_models(self):
            if error is not None:
                raise error
            return list(models or ["jev-latest", "jev-preview"])

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr(jev_client_module, "JevClient", FakeJevClient, raising=False)
    return instances


# ── run options ──────────────────────────────────────────────────


def test_run_forwards_jev_options() -> None:
    run = AsyncMock(return_value=True)
    with patch("mobilerun.cli.main.run_command", run):
        result = CliRunner().invoke(
            cli,
            [
                "run",
                "Search for the bridge",
                "--agent",
                "jev",
                "--model",
                "jev-1.14.0",
                "--min-confidence",
                "0.5",
                "--var",
                "query=Golden Gate Bridge",
                "--var",
                "note=a=b",
            ],
        )

    assert result.exit_code == 0, result.output
    kwargs = run.await_args.kwargs
    assert kwargs["agent"] == "jev"
    assert kwargs["model"] == "jev-1.14.0"
    assert kwargs["min_confidence"] == 0.5
    assert kwargs["variables"] == {"query": "Golden Gate Bridge", "note": "a=b"}


def test_run_agent_choice_is_case_insensitive() -> None:
    run = AsyncMock(return_value=True)
    with patch("mobilerun.cli.main.run_command", run):
        result = CliRunner().invoke(cli, ["run", "Open Settings", "--agent", "JEV"])

    assert result.exit_code == 0, result.output
    assert run.await_args.kwargs["agent"] == "jev"


def test_run_without_jev_options_passes_none() -> None:
    run = AsyncMock(return_value=True)
    with patch("mobilerun.cli.main.run_command", run):
        result = CliRunner().invoke(cli, ["run", "Open Settings"])

    assert result.exit_code == 0, result.output
    kwargs = run.await_args.kwargs
    assert kwargs["agent"] is None
    assert kwargs["min_confidence"] is None
    assert kwargs["variables"] is None


@pytest.mark.parametrize("name", ["droidrun", "DroidRun"])
def test_run_accepts_legacy_droidrun_agent(name) -> None:
    run = AsyncMock(return_value=True)
    with patch("mobilerun.cli.main.run_command", run):
        result = CliRunner().invoke(cli, ["run", "Open Settings", "--agent", name])

    assert result.exit_code == 0, result.output
    assert run.await_args.kwargs["agent"] == "droidrun"


def test_run_rejects_unknown_agent() -> None:
    run = AsyncMock(return_value=True)
    with patch("mobilerun.cli.main.run_command", run):
        result = CliRunner().invoke(cli, ["run", "Open Settings", "--agent", "nope"])

    assert result.exit_code == 2
    run.assert_not_awaited()


@pytest.mark.parametrize("item", ["query", "=s3cret-value", " =x"])
def test_run_rejects_malformed_var_without_echoing_it(item: str) -> None:
    run = AsyncMock(return_value=True)
    with patch("mobilerun.cli.main.run_command", run):
        result = CliRunner().invoke(cli, ["run", "Open Settings", "--var", item])

    assert result.exit_code == 2
    assert "NAME=VALUE" in result.output
    assert "s3cret-value" not in result.output
    run.assert_not_awaited()


@pytest.mark.parametrize("value", ["1.5", "-0.1", "nan", "inf"])
def test_run_rejects_min_confidence_out_of_range(value: str) -> None:
    run = AsyncMock(return_value=True)
    with patch("mobilerun.cli.main.run_command", run):
        result = CliRunner().invoke(
            cli, ["run", "Open Settings", "--min-confidence", value]
        )

    assert result.exit_code == 2
    assert "between 0 and 1" in result.output
    run.assert_not_awaited()


# ── run_command routing ──────────────────────────────────────────


class _FakeHandler:
    async def stream_events(self):
        if False:
            yield None

    def __await__(self):
        async def done():
            return SimpleNamespace(success=True)

        return done().__await__()


def _run_with_fake_agent(config: MobileConfig, **kwargs) -> tuple[bool, list, Mock]:
    created_agents: list[dict] = []

    class FakeAgent:
        def __init__(self, **agent_kwargs):
            created_agents.append(agent_kwargs)

        def run(self):
            return _FakeHandler()

    load_llm = Mock(return_value=object())
    with (
        patch("mobilerun.cli.main.ConfigLoader.load", return_value=config),
        patch("mobilerun.cli.main.MobileAgent", FakeAgent),
        patch("mobilerun.cli.main.load_llm", load_llm),
        patch("mobilerun.cli.main._setup_cli_logging"),
    ):
        success = asyncio.run(run_command("Search", debug=False, **kwargs))
    return success, created_agents, load_llm


def test_run_command_routes_jev_options(mobilerun_caplog) -> None:
    success, agents, load_llm = _run_with_fake_agent(
        MobileConfig(),
        agent="JEV",
        provider="GoogleGenAI",
        model="jev-1.14.0",
        min_confidence=0.5,
        variables={"query": "Golden Gate Bridge"},
    )

    assert success
    load_llm.assert_not_called()
    config = agents[0]["config"]
    assert config.agent.name == "jev"
    assert config.agent.jev.model == "jev-1.14.0"
    assert config.agent.jev.min_confidence == 0.5
    assert agents[0]["llms"] is None
    assert agents[0]["variables"] == {"query": "Golden Gate Bridge"}
    assert "--provider is ignored with the Jev agent" in mobilerun_caplog.text
    assert "🤖 Agent: Jev (jev-1.14.0)" in mobilerun_caplog.text
    assert "Agent mode" not in mobilerun_caplog.text
    assert "Vision settings" not in mobilerun_caplog.text


def test_run_command_uses_jev_from_config() -> None:
    config = MobileConfig.from_dict(
        {"agent": {"name": "Jev", "jev": {"model": "jev-preview"}}}
    )

    success, agents, load_llm = _run_with_fake_agent(config)

    assert success
    load_llm.assert_not_called()
    routed = agents[0]["config"]
    assert routed.agent.name == "jev"
    assert routed.agent.jev.model == "jev-preview"
    assert routed.agent.jev.min_confidence == JevConfig().min_confidence


def test_run_command_keeps_llm_path_for_mobilerun(mobilerun_caplog) -> None:
    success, agents, load_llm = _run_with_fake_agent(
        MobileConfig(),
        provider="GoogleGenAI",
        model="gemini-3.8-flash",
        min_confidence=0.5,
        variables={"city": "Paris"},
    )

    assert success
    load_llm.assert_called_once_with("GoogleGenAI", model="gemini-3.8-flash")
    config = agents[0]["config"]
    assert config.agent.name == "mobilerun"
    assert config.agent.jev == JevConfig()
    assert agents[0]["llms"] is load_llm.return_value
    assert agents[0]["variables"] == {"city": "Paris"}
    assert "🤖 Agent mode: direct execution" in mobilerun_caplog.text
    assert "--min-confidence only applies to the Jev agent" in mobilerun_caplog.text


# ── configure jev ────────────────────────────────────────────────


def test_configure_jev_saves_key_and_selects_jev(monkeypatch, config_path) -> None:
    clients = _install_fake_client(monkeypatch)

    result = CliRunner().invoke(
        cli,
        [
            "configure",
            "jev",
            "--api-key",
            f"  {SECRET} ",
            "--model",
            "jev-1.14.0",
            "--min-confidence",
            "0.4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert _saved_keys() == {"typesafe": SECRET}
    config = ConfigLoader.load()
    assert config.agent.name == "jev"
    assert config.agent.jev.model == "jev-1.14.0"
    assert config.agent.jev.min_confidence == 0.4
    assert len(clients) == 1
    assert clients[0].api_key == SECRET
    assert clients[0].kwargs["model"] == "jev-1.14.0"
    assert clients[0].closed
    assert SECRET not in result.output
    assert SECRET not in config_path.read_text()
    assert str(env_keys.AUTH_PROFILES_PATH) in result.output.replace("\n", "")
    assert "TypeSafe accepted the key" in result.output


def _write_jev_settings(config_path, settings: str) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(f"_version: {CURRENT_VERSION}\nagent:\n  jev:\n{settings}")


def test_configure_jev_accepts_quoted_min_confidence(monkeypatch, config_path) -> None:
    _install_fake_client(monkeypatch)
    _write_jev_settings(config_path, '    min_confidence: "0.5"\n')

    result = CliRunner().invoke(cli, ["configure", "jev", "--api-key", SECRET])

    assert result.exit_code == 0, result.output
    assert "Min confidence: 0.5" in result.output
    assert ConfigLoader.load().agent.jev.min_confidence == 0.5


def test_configure_jev_rejects_invalid_config_before_saving(
    monkeypatch, config_path
) -> None:
    clients = _install_fake_client(monkeypatch)
    _write_jev_settings(config_path, "    min_confidence: 30\n")
    before = config_path.read_text()

    result = CliRunner().invoke(cli, ["configure", "jev", "--api-key", SECRET])

    assert result.exit_code == 1
    assert "agent.jev.min_confidence must be between 0 and 1" in result.output
    assert "Nothing was saved." in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert _saved_keys() == {}
    assert config_path.read_text() == before
    assert clients == []


def test_configure_jev_uses_env_key_without_prompt(monkeypatch, config_path) -> None:
    _install_fake_client(monkeypatch)
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)

    result = CliRunner().invoke(cli, ["configure", "jev"])

    assert result.exit_code == 0, result.output
    assert "Using TYPESAFE_API_KEY from the environment" in result.output
    assert _saved_keys() == {"typesafe": SECRET}
    assert SECRET not in result.output


def test_configure_jev_prompts_for_hidden_key(monkeypatch, config_path) -> None:
    _install_fake_client(monkeypatch)

    result = CliRunner().invoke(cli, ["configure", "jev"], input=f"{SECRET}\n")

    assert result.exit_code == 0, result.output
    assert "TypeSafe API key" in result.output
    assert _saved_keys() == {"typesafe": SECRET}
    assert SECRET not in result.output


def test_configure_jev_enter_keeps_saved_key(monkeypatch, config_path) -> None:
    clients = _install_fake_client(monkeypatch)
    env_keys.save_env_keys({"typesafe": SECRET})
    monkeypatch.delenv("TYPESAFE_API_KEY")

    result = CliRunner().invoke(cli, ["configure", "jev"], input="\n")

    assert result.exit_code == 0, result.output
    assert clients[0].api_key == SECRET
    assert _saved_keys() == {"typesafe": SECRET}
    assert SECRET not in result.output


def test_configure_jev_refuses_rejected_key(monkeypatch, config_path) -> None:
    _install_fake_client(
        monkeypatch,
        error=JevAuthError(
            "TypeSafe rejected the API key (HTTP 401). Check TYPESAFE_API_KEY "
            "or the key saved with 'mobilerun configure jev'.",
            status=401,
        ),
    )

    result = CliRunner().invoke(cli, ["configure", "jev", "--api-key", SECRET])

    assert result.exit_code == 1
    assert "rejected the API key" in result.output
    assert "Nothing was saved" in result.output
    assert _saved_keys() == {}
    assert ConfigLoader.load().agent.name == "mobilerun"
    assert SECRET not in result.output


def test_configure_jev_saves_when_typesafe_is_unreachable(
    monkeypatch, config_path
) -> None:
    _install_fake_client(
        monkeypatch,
        error=JevTransportError(
            "Could not reach TypeSafe at https://api.typesafe.ai (ConnectError)."
        ),
    )

    result = CliRunner().invoke(cli, ["configure", "jev", "--api-key", SECRET])

    assert result.exit_code == 0, result.output
    assert "Could not verify the key" in result.output
    assert _saved_keys() == {"typesafe": SECRET}
    assert ConfigLoader.load().agent.name == "jev"


def test_configure_jev_redacts_key_from_unexpected_errors(
    monkeypatch, config_path
) -> None:
    _install_fake_client(monkeypatch, error=RuntimeError(f"boom {SECRET}"))

    result = CliRunner().invoke(cli, ["configure", "jev", "--api-key", SECRET])

    assert result.exit_code == 0, result.output
    assert "[redacted]" in result.output
    assert SECRET not in result.output


def test_configure_jev_no_verify_skips_typesafe(monkeypatch, config_path) -> None:
    clients = _install_fake_client(monkeypatch)

    result = CliRunner().invoke(
        cli, ["configure", "jev", "--api-key", SECRET, "--no-verify"]
    )

    assert result.exit_code == 0, result.output
    assert clients == []
    assert _saved_keys() == {"typesafe": SECRET}


def test_configure_jev_no_set_default_keeps_agent(monkeypatch, config_path) -> None:
    _install_fake_client(monkeypatch)

    result = CliRunner().invoke(
        cli, ["configure", "jev", "--api-key", SECRET, "--no-set-default"]
    )

    assert result.exit_code == 0, result.output
    assert ConfigLoader.load().agent.name == "mobilerun"
    assert _saved_keys() == {"typesafe": SECRET}


@pytest.mark.parametrize("key", ["has space", "tab\tkey", "ключ"])
def test_configure_jev_rejects_malformed_key(monkeypatch, config_path, key) -> None:
    clients = _install_fake_client(monkeypatch)

    result = CliRunner().invoke(cli, ["configure", "jev", "--api-key", key])

    assert result.exit_code == 1
    assert "malformed" in result.output
    assert clients == []
    assert _saved_keys() == {}


@pytest.mark.parametrize(
    ("model", "warned"),
    [("jev-latest", False), ("jev-1.14.0", False), ("jev-foo", True)],
)
def test_configure_jev_warns_about_unlisted_model(
    monkeypatch, config_path, model: str, warned: bool
) -> None:
    _install_fake_client(monkeypatch, models=["jev-latest", "jev-preview"])

    result = CliRunner().invoke(
        cli, ["configure", "jev", "--api-key", SECRET, "--model", model]
    )

    assert result.exit_code == 0, result.output
    assert ("is not listed" in result.output) is warned


# ── event rendering ──────────────────────────────────────────────


def _info_messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]


def test_jev_decision_events_render_at_info(mobilerun_caplog) -> None:
    handler = EventHandler()
    handler.handle(
        JevDecisionEvent(
            step=2,
            status="action",
            operation="TAP",
            label="[12] Search",
            confidence=0.91,
            target_confidence=0.8,
            model="jev-1.14.0",
            latency_ms=420,
        )
    )
    handler.handle(
        JevDecisionEvent(step=3, status="done", operation="DONE", confidence=0.95)
    )
    handler.handle(
        JevDecisionEvent(
            step=4,
            status="uncertain",
            operation="TAP",
            label="[3] OK",
            confidence=0.12,
        )
    )

    assert _info_messages(mobilerun_caplog) == [
        "🧭 Step 2: TAP [12] Search (confidence 0.91, target 0.80)",
        "🧭 Step 3: done: DONE (confidence 0.95)",
        "🧭 Step 4: uncertain: TAP [3] OK (confidence 0.12)",
    ]


def test_jev_step_failures_are_visible(mobilerun_caplog) -> None:
    handler = EventHandler()
    handler.handle(
        JevStepEvent(step=1, operation="TAP", success=True, summary="Tapped Search")
    )
    handler.handle(
        JevStepEvent(
            step=2, operation="TYPE_TEXT", success=False, summary="No focused field"
        )
    )
    handler.handle(JevEndEvent(success=True, reason="done", steps=2))

    assert _info_messages(mobilerun_caplog) == ["❌ TYPE_TEXT failed: No focused field"]


def test_finalize_event_rendering_is_unchanged(mobilerun_caplog) -> None:
    EventHandler().handle(FinalizeEvent(success=True, reason="Settings opened"))

    assert _info_messages(mobilerun_caplog) == ["🎉 Goal achieved: Settings opened"]
