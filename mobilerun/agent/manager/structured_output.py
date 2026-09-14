"""Structured response contract for the stateful manager."""

from __future__ import annotations

import json
import re
from typing import Any

from llama_index.core.base.llms.types import ChatResponse
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator


class ManagerDecision(BaseModel):
    """Provider-friendly, flat manager response schema."""

    model_config = ConfigDict(extra="forbid")

    thought: str
    memory_update: str
    progress_summary: str
    plan: list[str] | None
    current_subgoal: str | None
    success: bool | None
    answer: str | None

    @model_validator(mode="after")
    def validate_control_result(self) -> "ManagerDecision":
        self.thought = self.thought.strip()
        self.memory_update = self.memory_update.strip()
        self.progress_summary = self.progress_summary.strip()
        self.answer = self.answer.strip() if self.answer else None
        self.current_subgoal = self.current_subgoal.strip() if self.current_subgoal else None
        self.plan = [item.strip() for item in (self.plan or []) if item.strip()] or None

        if self.plan is not None:
            if self.success is not None or self.answer is not None:
                raise ValueError("a plan cannot be combined with a final result")
            self.current_subgoal = self.plan[0]
            return self

        if self.current_subgoal is not None:
            raise ValueError("current_subgoal requires a plan")
        if self.success is None or self.answer is None:
            raise ValueError("a final result requires success and answer")
        return self

    def as_manager_fields(self) -> dict[str, Any]:
        """Map the structured response to the existing manager state contract."""

        return {
            "thought": self.thought,
            "memory": self.memory_update,
            "plan": "\n".join(self.plan or []),
            "current_subgoal": self.current_subgoal or "",
            "answer": self.answer or "",
            "success": self.success,
            "progress_summary": self.progress_summary,
        }


def manager_output_schema_json() -> str:
    """Return the schema as JSON for inclusion in the bundled prompt."""

    return json.dumps(ManagerDecision.model_json_schema(), indent=2)


def structured_chat_options(llm: Any) -> tuple[str, dict[str, Any]]:
    """Return the strongest safe schema constraint supported by this adapter.

    The fallback mode still asks for JSON in the bundled prompt and validates it
    locally, but does not claim that the provider enforced the schema.
    """

    class_name = llm.class_name()
    schema = ManagerDecision.model_json_schema()

    if class_name in {"GenAI", "GoogleGenAI", "MobilerunGoogleGenAI"}:
        return (
            "native",
            {
                "generation_config": {
                    "response_mime_type": "application/json",
                    "response_schema": ManagerDecision,
                }
            },
        )

    if class_name in {
        "OpenAIResponses",
        "MobilerunOpenAIResponses",
        "openai_responses_llm",
    } and not _is_grok(llm):
        return (
            "native",
            {
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "manager_decision",
                        "schema": schema,
                        "strict": True,
                    }
                }
            },
        )

    if class_name in {"OpenAI", "openai_llm"}:
        return (
            "native",
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "manager_decision",
                        "schema": schema,
                        "strict": True,
                    },
                }
            },
        )

    if class_name in {"Ollama", "Ollama_llm"}:
        return "native", {"format": schema}

    if class_name in {"Anthropic", "MobilerunAnthropic", "Anthropic_LLM"} and not _is_fable(llm):
        return (
            "tool",
            {
                "tools": [
                    {
                        "name": "manager_decision",
                        "description": "Return the next manager decision.",
                        "input_schema": schema,
                    }
                ],
                "tool_choice": {"type": "tool", "name": "manager_decision"},
            },
        )

    return "prompt", {}


def parse_structured_response(response: ChatResponse, enforcement_mode: str) -> ManagerDecision:
    """Validate a structured manager response from text or a forced tool call."""

    if enforcement_mode == "tool":
        for tool_call in response.message.additional_kwargs.get("tool_calls", []):
            payload = _tool_call_payload(tool_call)
            if payload is not None:
                return ManagerDecision.model_validate(payload)
        raise ValueError("structured manager response did not contain a tool call")

    content = response.message.content or ""
    try:
        return ManagerDecision.model_validate_json(content)
    except ValidationError as direct_error:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match is None:
            raise direct_error
        return ManagerDecision.model_validate_json(match.group(0))


def _is_fable(llm: Any) -> bool:
    model = str(getattr(llm, "model", "")).lower()
    return "fable" in model


def _is_grok(llm: Any) -> bool:
    model = str(getattr(llm, "model", "")).lower()
    return model.startswith("grok")


def _tool_call_payload(tool_call: Any) -> dict[str, Any] | None:
    if hasattr(tool_call, "model_dump"):
        tool_call = tool_call.model_dump()
    if not isinstance(tool_call, dict):
        return None

    name = tool_call.get("name")
    payload = tool_call.get("input")
    if name is None and isinstance(tool_call.get("function"), dict):
        function = tool_call["function"]
        name = function.get("name")
        payload = function.get("arguments")
    if name != "manager_decision":
        return None
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload if isinstance(payload, dict) else None
