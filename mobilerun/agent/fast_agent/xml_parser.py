"""XML tool-call parsing and result formatting.

Parses LLM responses containing <function_calls> blocks into structured
ToolCall objects, and formats tool results as <function_results> XML
for injection back into the conversation.
"""

import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html import escape
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mobilerun")

OPEN_TAG = "<function_calls>"
CLOSE_TAG = "</function_calls>"

_PARAM_RE = re.compile(
    r'(<parameter\s+name="[^"]*">)(.*?)(</parameter>)',
    re.DOTALL,
)


@dataclass
class ToolCall:
    """A parsed tool invocation from the LLM response."""

    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ToolResult:
    """Result from executing a single tool."""

    name: str
    output: str
    is_error: bool = False


def parse_tool_calls(
    text: str, param_types: Optional[Dict[str, str]] = None
) -> Tuple[str, List[ToolCall]]:
    """Parse tool calls from LLM response text.

    Args:
        text: Raw LLM response text.
        param_types: Optional {param_name: type_string} map for coercion.
                     If None, all values are kept as strings.

    Returns:
        Tuple of (text_before_tool_calls, list_of_tool_calls).
        If no tool calls found, returns (full_text, []).
    """
    if OPEN_TAG not in text:
        return text.strip(), []

    parts = text.split(OPEN_TAG)
    text_before = parts[0].strip()

    call_blocks: List[List[ToolCall]] = []
    for part in parts[1:]:
        close_idx = part.find(CLOSE_TAG)
        if close_idx == -1:
            continue  # Malformed — no closing tag, skip

        block = part[:close_idx].strip()
        if not block:
            continue

        calls = _parse_tool_call_block(block, param_types)
        if calls:
            call_blocks.append(calls)

    deduped_blocks = _drop_adjacent_duplicate_blocks(call_blocks)
    return text_before, [call for block in deduped_blocks for call in block]


def format_tool_results(results: List[ToolResult]) -> str:
    """Format tool results as XML for injection into conversation.

    Args:
        results: List of tool results to format.

    Returns:
        XML string with <function_results> wrapper.
    """
    lines = ["<function_results>"]

    for result in results:
        if result.is_error:
            lines.append(
                f"<result>\n<name>{result.name}</name>\n"
                f"<error>{result.output}</error>\n</result>"
            )
        else:
            lines.append(
                f"<result>\n<name>{result.name}</name>\n"
                f"<output>{result.output}</output>\n</result>"
            )

    lines.append("</function_results>")
    return "\n".join(lines)


def format_tool_calls(calls: List[ToolCall]) -> str:
    """Format parsed tool calls as XML for logging/trajectory output."""
    lines = [OPEN_TAG]
    for call in calls:
        lines.append(f'<invoke name="{call.name}">')
        for name, value in call.parameters.items():
            lines.append(
                f'<parameter name="{name}">{_format_param_value(value)}</parameter>'
            )
        lines.append("</invoke>")
    lines.append(CLOSE_TAG)
    return "\n".join(lines)


def _parse_tool_call_block(
    block: str, param_types: Optional[Dict[str, str]]
) -> List[ToolCall]:
    block = _sanitize_param_content(block)

    try:
        root = ET.fromstring(f"<root>{block}</root>")
    except ET.ParseError:
        logger.warning("Failed to parse tool call XML block, skipping")
        return []

    calls: List[ToolCall] = []
    for invoke in root.findall("invoke"):
        name = invoke.get("name", "")
        if not name:
            continue

        params: Dict[str, Any] = {}
        error: Optional[str] = None
        for param in invoke.findall("parameter"):
            param_name = param.get("name", "")
            param_value = param.text or ""
            if param_name:
                try:
                    params[param_name] = _coerce_param(
                        param_name, param_value, param_types
                    )
                except ValueError as e:
                    error = str(e)
                    break

        calls.append(ToolCall(name=name, parameters=params, error=error))
    return calls


def _drop_adjacent_duplicate_blocks(
    blocks: List[List[ToolCall]],
) -> List[List[ToolCall]]:
    """Drop exact adjacent duplicate <function_calls> blocks."""
    if not blocks:
        return blocks

    deduped = [blocks[0]]
    for block in blocks[1:]:
        previous = deduped[-1]
        if block == previous:
            logger.debug("Dropping duplicate adjacent tool-call block")
            continue
        deduped.append(block)
    return deduped


def _format_param_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return escape(json.dumps(value, separators=(",", ":")))
    if value is None:
        return ""
    return escape(str(value))


def _sanitize_param_content(block: str) -> str:
    """Escape XML-unsafe characters inside parameter values.

    Parameter values often contain raw code or text with <, >, &
    which would break XML parsing. This escapes content inside
    <parameter> tags only, leaving the XML structure intact.
    """

    def _escape(m: re.Match) -> str:
        pre, content, post = m.group(1), m.group(2), m.group(3)
        clean = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return pre + clean + post

    return _PARAM_RE.sub(_escape, block)


def _coerce_param(
    name: str, value: str, param_types: Optional[Dict[str, str]] = None
) -> Any:
    """Coerce string parameter value to expected type.

    Args:
        name: Parameter name.
        value: Raw string value from XML.
        param_types: Optional type map. If None, returns value as-is.
    """
    if param_types is None:
        return value

    expected = param_types.get(name, "string")

    if expected == "boolean":
        return value.strip().lower() == "true"

    if expected == "number":
        value = value.strip()
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                raise ValueError(
                    f"parameter '{name}' expected number, got '{value}'"
                ) from None

    if expected == "list":
        value = value.strip()
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
            return [parsed]  # Single element — wrap in list
        except (json.JSONDecodeError, ValueError):
            raise ValueError(
                f"parameter '{name}' expected list, got '{value}'"
            ) from None

    return value
