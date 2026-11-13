import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun import DecisionConfig, MobileConfig
from mobilerun.agent.fast_agent.xml_parser import ToolCall
from mobilerun.agent.selectors import (
    ActionCandidate,
    Decision,
    DecisionSelector,
    SelectionRequest,
)
from mobilerun.agent.utils.prompt_resolver import PromptResolver
from mobilerun.config_manager.prompt_loader import PromptLoader


def _request(*names, **kwargs):
    return SelectionRequest(
        goal="Check Battery status",
        options=tuple(ActionCandidate(ToolCall(name), name) for name in names),
        tools={name: {} for name in (*names, "type", "complete")},
        **kwargs,
    )


def _instructions(request):
    model = SimpleNamespace(decide=AsyncMock(return_value=Decision(0)))
    asyncio.run(DecisionSelector(model).select(request))
    return model.decide.call_args.args[0].instructions


def test_default_prompt_explains_missing_typing_and_available_scrolling():
    text = _instructions(_request("click", "swipe", "open_app"))

    assert "You cannot write arbitrary text" in text
    assert "Avoid opening search bars" in text
    assert "Swipe up to scroll down" in text
    assert "not reset it to its main page" in text
    assert "No screenshot is supplied" in text
    assert "complete_success" in text
    assert "not output XML tool calls or a reasoning narrative" in text


@pytest.mark.parametrize("typing_tool", ["type", "type_text"])
def test_typing_capability_requires_an_enabled_candidate(typing_tool):
    request = _request("click", typing_tool)
    text = _instructions(request)
    assert "Text entry is limited to the exact text and target" in text
    assert "You cannot write arbitrary text" not in text

    del request.tools[typing_tool]
    text = _instructions(request)
    assert "You cannot write arbitrary text" in text


def test_secret_candidate_does_not_imply_arbitrary_text_entry():
    text = _instructions(_request("click", "type_secret"))
    assert "You cannot write arbitrary text" in text


def test_prompt_omits_unavailable_capabilities():
    request = _request("click", protocol="executor", screenshot=b"image")
    text = _instructions(request)
    assert "Swipe up to scroll down" not in text
    assert "complete_success" not in text
    assert "complete_failure" not in text
    assert "Use the supplied screenshot" in text
    assert "No screenshot is supplied" not in text


def test_decision_config_round_trip_and_custom_path(tmp_path, monkeypatch):
    prompt = tmp_path / "decision.jinja2"
    prompt.write_text("Goal: {{ goal }}; scroll={{ can_scroll }}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config = MobileConfig.from_dict(
        {
            "agent": {"decision": {"system_prompt": prompt.name}},
        }
    )
    restored = MobileConfig.from_dict(config.to_dict())

    assert isinstance(restored.agent.decision, DecisionConfig)
    assert restored.agent.get_decision_system_prompt_path() == str(prompt)
    assert (
        asyncio.run(
            PromptLoader.load_prompt(
                restored.agent.get_decision_system_prompt_path(),
                _request("swipe").get_decision_prompt_variables(),
            )
        )
        == "Goal: Check Battery status; scroll=True"
    )


def test_decision_prompt_variables_support_custom_templates():
    resolver = PromptResolver(
        custom_prompts={
            "decision_system": (
                "{{ goal }} {{ protocol }} {{ can_type }} {{ can_scroll }} "
                "{{ can_complete }} {{ vision }} {{ variables.name }} "
                "{{ available_tools | sort | join(',') }} "
                "registered_type={{ 'type' in registered_tools }}"
            )
        }
    )
    text = PromptLoader.render_template(
        resolver.get_prompt("decision_system"),
        _request(
            "click", "swipe", variables={"name": "Battery"}
        ).get_decision_prompt_variables(),
    )

    assert text == (
        "Check Battery status xml False True True False Battery "
        "click,complete,swipe registered_type=True"
    )
    assert "decision_system" in PromptResolver.get_valid_prompt_keys()


@pytest.mark.parametrize("decision", [None, {}])
def test_missing_or_null_decision_config_uses_bundled_default(decision):
    config = MobileConfig.from_dict({"agent": {"decision": decision}})
    assert config.agent.decision == DecisionConfig()
    assert config.agent.get_decision_system_prompt_path().endswith(
        "config/prompts/decision/system.jinja2"
    )


def test_standalone_selector_renders_after_filtering_without_mutating_request():
    model = SimpleNamespace(decide=AsyncMock(return_value=Decision(0)))
    request = _request("click", "type")
    del request.tools["type"]

    asyncio.run(DecisionSelector(model).select(request))

    supplied = model.decide.call_args.args[0]
    assert "You cannot write arbitrary text" in supplied.instructions
    assert len(supplied.options) == 1
    assert request.instructions == ""
    assert len(request.options) == 2


def test_standalone_selector_preserves_explicit_instructions():
    model = SimpleNamespace(decide=AsyncMock(return_value=Decision(0)))
    request = _request("click", instructions="Custom decision instructions")

    asyncio.run(DecisionSelector(model).select(request))

    assert model.decide.call_args.args[0].instructions == request.instructions


def test_delegate_that_can_type_makes_search_fields_usable():
    from llama_index.core.llms.mock import MockLLM

    from mobilerun.agent.selectors import DelegatingSelector, LlamaIndexLLMSelector

    model = SimpleNamespace(decide=AsyncMock(return_value=Decision(0)))
    selector = DelegatingSelector(
        DecisionSelector(model), LlamaIndexLLMSelector(MockLLM())
    )
    asyncio.run(selector.select(_request("click")))
    text = model.decide.call_args.args[0].instructions

    assert "choose delegate to type the text" in text
    assert "Avoid opening search bars" not in text


def test_typing_candidates_take_precedence_over_the_delegate():
    variables = _request("click", "type").get_decision_prompt_variables(True)
    text = asyncio.run(
        PromptLoader.load_prompt(
            MobileConfig().agent.get_decision_system_prompt_path(), variables
        )
    )

    assert "Text entry is limited to the exact text and target" in text
    assert "choose delegate to type the text" not in text


def test_delegating_fallback_renders_its_own_typing_guidance():
    from llama_index.core.llms.mock import MockLLM

    from mobilerun.agent.selectors import (
        DelegatingSelector,
        FallbackSelector,
        LlamaIndexLLMSelector,
    )

    path = MobileConfig().agent.get_decision_system_prompt_path()

    async def render(variables):
        return await PromptLoader.load_prompt(path, variables)

    request = _request("click")
    request.render_instructions = render
    request.instructions = asyncio.run(render(request.get_decision_prompt_variables()))
    plain = SimpleNamespace(decide=AsyncMock(return_value=None))
    model = SimpleNamespace(decide=AsyncMock(return_value=Decision(0)))
    selector = FallbackSelector(
        DecisionSelector(plain),
        [DelegatingSelector(DecisionSelector(model), LlamaIndexLLMSelector(MockLLM()))],
    )

    asyncio.run(selector.select(request))

    assert "Avoid opening search bars" in plain.decide.call_args.args[0].instructions
    text = model.decide.call_args.args[0].instructions
    assert "choose delegate to type the text" in text
    assert "Avoid opening search bars" not in text
