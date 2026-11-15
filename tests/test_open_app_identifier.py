import asyncio
from types import SimpleNamespace

import pytest

from mobilerun.agent.fast_agent.xml_parser import parse_tool_calls_detailed
from mobilerun.agent.utils import actions
from mobilerun.agent.utils.actions import open_bundle_id
from mobilerun.agent.utils.signatures import build_tool_registry


class _Driver:
    def __init__(self) -> None:
        self.started = []

    async def start_app(self, package, activity=None):
        self.started.append(package)
        return f"App started: {package}"


class _Recorder:
    def __init__(self) -> None:
        self.actions = []

    def record_action(self, action, **kwargs) -> None:
        self.actions.append(action)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def instant(_seconds):
        return None

    monkeypatch.setattr(actions.asyncio, "sleep", instant)


@pytest.mark.parametrize("arg", ["bundle_id", "app_id"])
def test_open_app_strips_whitespace_around_the_identifier(arg) -> None:
    driver = _Driver()
    recorder = _Recorder()
    ctx = SimpleNamespace(driver=driver, macro_recorder=recorder)

    result = asyncio.run(open_bundle_id(**{arg: " com.burbn.instagram\n"}, ctx=ctx))

    assert result.success
    assert driver.started == ["com.burbn.instagram"]
    assert recorder.actions[0]["package"] == "com.burbn.instagram"


def test_open_app_rejects_a_blank_identifier() -> None:
    driver = _Driver()

    result = asyncio.run(
        open_bundle_id(bundle_id=" \n", ctx=SimpleNamespace(driver=driver))
    )

    assert result.success is False
    assert "exact app identifier is required" in result.summary
    assert driver.started == []


def test_ios_open_app_tool_call_with_a_trailing_newline_launches_the_app() -> None:
    async def run():
        registry, _ = await build_tool_registry(platform="ios")
        parsed = parse_tool_calls_detailed(
            "<function_calls>\n"
            '<invoke name="open_app">\n'
            '<parameter name="bundle_id">com.burbn.instagram\n</parameter>\n'
            "</invoke>\n"
            "</function_calls>",
            registry.get_param_types(),
        )
        call = parsed.calls[0]
        driver = _Driver()
        result = await registry.execute(
            call.name, call.parameters, SimpleNamespace(driver=driver)
        )
        return call, result, driver

    call, result, driver = asyncio.run(run())

    assert call.parameters == {"bundle_id": "com.burbn.instagram\n"}
    assert result.success
    assert driver.started == ["com.burbn.instagram"]


def test_name_based_open_app_starts_an_offered_app_id_without_the_llm(monkeypatch):
    from mobilerun.agent.action_context import ActionCandidateApp

    def no_llm(**kwargs):
        raise AssertionError("An offered app id must not ask the app opener LLM")

    monkeypatch.setattr(actions, "AppStarter", no_llm)
    driver = _Driver()
    ctx = SimpleNamespace(
        driver=driver,
        app_opener_llm=object(),
        action_candidate_apps=(
            ActionCandidateApp("com.google.android.deskclock", "Clock"),
        ),
    )

    result = asyncio.run(actions.open_app("com.google.android.deskclock", ctx=ctx))

    assert result.success
    assert driver.started == ["com.google.android.deskclock"]


def test_name_based_open_app_still_asks_the_llm_for_a_name(monkeypatch):
    from mobilerun.agent.action_context import ActionCandidateApp

    asked = []

    class FakeAppStarter:
        def __init__(self, **kwargs):
            pass

        async def run(self, app_description):
            asked.append(app_description)
            return "App started: com.google.android.deskclock"

    monkeypatch.setattr(actions, "AppStarter", FakeAppStarter)
    ctx = SimpleNamespace(
        driver=_Driver(),
        app_opener_llm=object(),
        streaming=False,
        action_candidate_apps=(
            ActionCandidateApp("com.google.android.deskclock", "Clock"),
        ),
    )

    result = asyncio.run(actions.open_app("Clock", ctx=ctx))

    assert result.success
    assert asked == ["Clock"]
