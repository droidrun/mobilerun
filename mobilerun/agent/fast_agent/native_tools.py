"""Native function-calling support for FastAgent.

Converts provider tool-call blocks into the same ToolCall objects the XML
protocol produces, and builds the tool-result messages that answer them.
"""

import json
from typing import Any, Dict, List, Optional

from llama_index.core.base.llms.types import ChatMessage, ToolCallBlock

from mobilerun.agent.fast_agent.xml_parser import (
    ToolCall,
    ToolCallParseResult,
    ToolCallParseStatus,
    ToolResult,
    _coerce_param,
)


def native_tool_kwargs(
    llm: Any, specs: List[Dict[str, Any]], parallel: bool
) -> Dict[str, Any]:
    """Build the request arguments that send tools to the LLM."""
    if getattr(llm, "native_tool_api", "responses") == "chat_completions":
        return {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": spec["name"],
                        "description": spec["description"],
                        "parameters": spec["parameters"],
                    },
                }
                for spec in specs
            ],
            # DeepSeek thinking mode rejects tool_choice="required".
            "tool_choice": "auto",
            "parallel_tool_calls": parallel,
        }
    return {"tools": specs, "tool_choice": "required", "parallel_tool_calls": parallel}


def parse_native_tool_calls(
    message: ChatMessage, param_types: Optional[Dict[str, str]] = None
) -> ToolCallParseResult:
    """Split a native tool-calling response into its text and tool calls."""
    calls = [
        _to_tool_call(block, param_types)
        for block in message.blocks
        if isinstance(block, ToolCallBlock)
    ]
    return ToolCallParseResult(
        thought=(message.content or "").strip(),
        calls=calls,
        status=ToolCallParseStatus.VALID if calls else ToolCallParseStatus.NO_MARKUP,
    )


def tool_result_messages(
    calls: List[ToolCall], results: List[ToolResult]
) -> List[ChatMessage]:
    """Answer every native tool call; calls without a result were not run."""
    messages = []
    for index, call in enumerate(calls):
        if index < len(results):
            result = results[index]
            output = f"Error: {result.output}" if result.is_error else result.output
        else:
            output = "Error: Not executed."
        messages.append(
            ChatMessage(
                role="tool",
                content=output or "Done",
                additional_kwargs={"tool_call_id": call.call_id},
            )
        )
    return messages


def _to_tool_call(
    block: ToolCallBlock, param_types: Optional[Dict[str, str]]
) -> ToolCall:
    name = block.tool_name
    call_id = block.tool_call_id
    arguments: Any = block.tool_kwargs
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return ToolCall(
                name=name, error="arguments are not valid JSON", call_id=call_id
            )
    if not isinstance(arguments, dict):
        return ToolCall(
            name=name, error="arguments must be a JSON object", call_id=call_id
        )

    parameters: Dict[str, Any] = {}
    for param_name, value in arguments.items():
        # Models sometimes send numbers, booleans, or lists as strings.
        if isinstance(value, str) and param_types:
            if param_types.get(param_name, "string") != "string":
                try:
                    value = _coerce_param(param_name, value, param_types)
                except ValueError as e:
                    return ToolCall(
                        name=name, parameters=parameters, error=str(e), call_id=call_id
                    )
        parameters[param_name] = value
    return ToolCall(name=name, parameters=parameters, call_id=call_id)
