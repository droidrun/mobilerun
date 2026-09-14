import asyncio
from types import SimpleNamespace

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai import OpenAI
from pydantic import ValidationError

from mobilerun.agent.manager.manager_agent import ManagerAgent
from mobilerun.agent.manager.structured_output import (
    ManagerDecision,
    parse_structured_response,
    structured_chat_options,
)
from mobilerun.agent.utils.inference import acall_with_retries
from mobilerun.agent.utils.prompt_resolver import PromptResolver
from mobilerun.config_manager.config_manager import AgentConfig, ManagerConfig
from mobilerun.config_manager.prompt_loader import PromptLoader


def _decision_json(**overrides):
    values = {
        "thought": "continue",
        "memory_update": "",
        "progress_summary": "opened settings",
        "plan": ["Open Display", "Change brightness"],
        "current_subgoal": None,
        "success": None,
        "answer": None,
    }
    values.update(overrides)
    return ManagerDecision(**values).model_dump_json()


def test_manager_decision_normalizes_plan_and_derives_subgoal():
    decision = ManagerDecision.model_validate_json(_decision_json())

    assert decision.plan == ["Open Display", "Change brightness"]
    assert decision.current_subgoal == "Open Display"
    assert decision.as_manager_fields()["plan"] == "Open Display\nChange brightness"


def test_manager_decision_keeps_first_plan_item_canonical():
    decision = ManagerDecision.model_validate_json(_decision_json(current_subgoal="Different step"))

    assert decision.current_subgoal == "Open Display"


@pytest.mark.parametrize(
    "overrides",
    [
        {"success": True, "answer": "done"},
        {"plan": None, "current_subgoal": None, "success": None, "answer": None},
        {"plan": None, "current_subgoal": "orphan", "success": True, "answer": "done"},
    ],
)
def test_manager_decision_rejects_mixed_or_incomplete_states(overrides):
    values = {
        "thought": "",
        "memory_update": "",
        "progress_summary": "",
        "plan": ["Next"],
        "current_subgoal": "Next",
        "success": None,
        "answer": None,
    }
    values.update(overrides)

    with pytest.raises(ValidationError):
        ManagerDecision(**values)


def test_parse_structured_response_accepts_fenced_prompt_fallback():
    response = ChatResponse(message=ChatMessage(content=f"```json\n{_decision_json()}\n```"))

    decision = parse_structured_response(response, "prompt")

    assert decision.current_subgoal == "Open Display"


def test_parse_structured_response_reads_anthropic_tool_payload():
    payload = ManagerDecision.model_validate_json(_decision_json()).model_dump()
    response = ChatResponse(
        message=ChatMessage(
            content="",
            additional_kwargs={
                "tool_calls": [{"name": "manager_decision", "input": payload, "type": "tool_use"}]
            },
        )
    )

    decision = parse_structured_response(response, "tool")

    assert decision.plan == ["Open Display", "Change brightness"]


@pytest.mark.parametrize(
    ("class_name", "model", "expected_mode", "expected_key"),
    [
        (GoogleGenAI.class_name(), "gemini-3.8-flash", "native", "generation_config"),
        ("openai_responses_llm", "gpt-5.5", "native", "text"),
        (OpenAI.class_name(), "gpt-4o-mini", "native", "response_format"),
        ("openai_responses_llm", "grok-4.6", "prompt", None),
        ("Ollama", "qwen3", "native", "format"),
        ("MobilerunAnthropic", "claude-sonnet-5", "tool", "tools"),
        ("MobilerunAnthropic", "claude-fable-5-1", "prompt", None),
        ("OpenAILike", "provider/model", "prompt", None),
    ],
)
def test_structured_chat_options_are_explicit_by_adapter(
    class_name, model, expected_mode, expected_key
):
    llm = SimpleNamespace(class_name=lambda: class_name, model=model)

    mode, kwargs = structured_chat_options(llm)

    assert mode == expected_mode
    if expected_key is None:
        assert kwargs == {}
    else:
        assert expected_key in kwargs


def test_structured_output_is_limited_to_auto_selected_stateful_prompt():
    agent = object.__new__(ManagerAgent)
    agent.config = ManagerConfig(system_prompt=None, stateless=False)
    agent.prompt_resolver = PromptResolver()
    assert agent._uses_structured_output()

    agent.config = ManagerConfig(system_prompt="config/prompts/manager/system.jinja2")
    assert not agent._uses_structured_output()

    agent.config = ManagerConfig(system_prompt=None, stateless=False)
    agent.prompt_resolver = PromptResolver({"manager_system": "custom"})
    assert not agent._uses_structured_output()


def test_bundled_prompt_switches_contract_without_changing_legacy_prompt():
    prompt_path = AgentConfig().get_manager_system_prompt_path()
    common = {
        "platform": "android",
        "structured_manager_output": True,
        "manager_decision_schema": '{"type": "object"}',
        "output_schema": {
            "properties": {"title": {"description": "Collected title", "type": "string"}},
            "required": ["title"],
        },
    }

    structured = asyncio.run(PromptLoader.load_prompt(prompt_path, common))
    legacy = asyncio.run(
        PromptLoader.load_prompt(prompt_path, {**common, "structured_manager_output": False})
    )

    assert "Return exactly one JSON object" in structured
    assert '<request_accomplished success="true">' not in structured
    assert "in the `answer` field" in structured
    assert "do NOT output JSON" not in structured
    assert "Use `<plan>` for unfinished work" in legacy
    assert "Return exactly one JSON object" not in legacy


def test_acall_forwards_schema_kwargs_and_accepts_tool_only_response():
    captured = {}

    class FakeLLM:
        async def achat(self, messages, **kwargs):
            captured.update(kwargs)
            return ChatResponse(
                message=ChatMessage(
                    content="",
                    additional_kwargs={"tool_calls": [{"name": "manager_decision"}]},
                )
            )

    response = asyncio.run(
        acall_with_retries(
            FakeLLM(),
            [ChatMessage(content="decide")],
            llm_kwargs={"format": {"type": "object"}},
        )
    )

    assert captured == {"format": {"type": "object"}}
    assert response.message.additional_kwargs["tool_calls"]


def test_structured_validation_performs_only_one_semantic_retry(monkeypatch):
    invalid = ChatResponse(message=ChatMessage(content="{}"))
    valid = ChatResponse(message=ChatMessage(content=_decision_json()))
    calls = []

    async def fake_call(*args, **kwargs):
        calls.append((args, kwargs))
        return valid

    monkeypatch.setattr("mobilerun.agent.manager.manager_agent.acall_with_retries", fake_call)
    agent = object.__new__(ManagerAgent)
    agent.llm = object()

    decision, repair_response, retries = asyncio.run(
        agent._validate_structured_and_retry([ChatMessage(content="decide")], invalid, "prompt", {})
    )

    assert decision.current_subgoal == "Open Display"
    assert repair_response is valid
    assert retries == 1
    assert len(calls) == 1
