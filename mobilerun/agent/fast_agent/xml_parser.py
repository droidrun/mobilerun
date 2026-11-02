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
from enum import Enum
from html import escape
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mobilerun")

OPEN_TAG = "<function_calls>"
CLOSE_TAG = "</function_calls>"

_PARAM_RE = re.compile(
    r"""(<parameter\s+name=(?:"[^"]*"|'[^']*')\s*>)(.*?)(</parameter>)""",
    re.DOTALL,
)
_PARAM_OPEN_RE = re.compile(r"<parameter(?=[\s>/])")

_DSML_MARKUP_PATTERN = r"(?:<|＜)\s*[/／]?\s*｜+DSML｜+"
_DSML_MARKUP_RE = re.compile(_DSML_MARKUP_PATTERN, re.IGNORECASE)
_TOOL_CALL_MARKUP_RE = re.compile(
    r"(?:<|＜)\s*[/／]?\s*(?:function_calls?|tool_calls?|invoke|parameter)\b"
    r"|" + _DSML_MARKUP_PATTERN,
    re.IGNORECASE,
)
_CANONICAL_BLOCK_TAGS = ("<invoke", "</invoke", "<parameter", "</parameter")
# Characters that can follow a <function_calls> mentioned in prose.
_PROSE_PUNCTUATION = "`.,;:!?)"
# Tool-call markup that marks a non-canonical block between canonical openers.
_BLOCK_CONTENT_RE = re.compile(
    r"(?:<|＜)\s*[/／]?\s*(?:function_call|tool_calls?|invoke|parameter)\b|"
    + _DSML_MARKUP_PATTERN,
    re.IGNORECASE,
)


@dataclass
class ToolCall:
    """A parsed tool invocation from the LLM response."""

    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    # Native tool calls carry the provider's id for matching results.
    call_id: Optional[str] = None


class ToolCallParseStatus(str, Enum):
    """Classification of tool-call markup in an LLM response."""

    VALID = "valid"
    NO_MARKUP = "no_markup"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class ToolCallParseResult:
    """Detailed result used internally to distinguish missing and invalid markup.

    ``end`` is the text offset where the first tool-call block ends;
    ``ignored_blocks`` counts later blocks that contain tool-call markup.
    """

    thought: str
    calls: List[ToolCall]
    status: ToolCallParseStatus
    end: Optional[int] = None
    ignored_blocks: int = 0


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

    result = parse_tool_calls_detailed(text, param_types)
    return result.thought, result.calls


def parse_tool_calls_detailed(
    text: str, param_types: Optional[Dict[str, str]] = None
) -> ToolCallParseResult:
    """Parse tool calls and classify whether markup was absent, valid, or malformed.

    Only the first tool-call block is parsed: the first <function_calls>
    always starts it. If it is malformed (including any tool-call markup
    before that opener), the result is malformed even when a later block is
    valid.
    """
    if OPEN_TAG not in text:
        markup_match = _TOOL_CALL_MARKUP_RE.search(text)
        if markup_match:
            return ToolCallParseResult(
                thought=text[: markup_match.start()].strip(),
                calls=[],
                status=ToolCallParseStatus.MALFORMED,
            )
        return ToolCallParseResult(
            thought=text.strip(),
            calls=[],
            status=ToolCallParseStatus.NO_MARKUP,
        )

    parts = text.split(OPEN_TAG)
    leading = _TOOL_CALL_MARKUP_RE.search(parts[0])
    if leading:
        # Markup before the first opener is the first block.
        end = len(parts[0])
        return ToolCallParseResult(
            thought=parts[0][: leading.start()].strip(),
            calls=[],
            status=ToolCallParseStatus.MALFORMED,
            end=end,
            ignored_blocks=_count_tool_call_blocks(text, end),
        )

    text_before = parts[0].strip()
    part = parts[1]
    part_start = len(parts[0]) + len(OPEN_TAG)

    close_idx = part.find(CLOSE_TAG)
    if close_idx == -1:
        # Unclosed block ends where the next one starts, if any.
        end = part_start + len(part) if len(parts) > 2 else None
        return ToolCallParseResult(
            thought=text_before,
            calls=[],
            status=ToolCallParseStatus.MALFORMED,
            end=end,
            ignored_blocks=(
                _count_tool_call_blocks(text, end) if end is not None else 0
            ),
        )

    block = part[:close_idx].strip()
    calls: List[ToolCall] = []
    # Parameter-value sanitization intentionally preserves raw text such as
    # code and literal DSML. Provider-specific markup is malformed only
    # when it occurs outside a trustworthy canonical parameter payload.
    if block and not _has_noncanonical_markup(block):
        calls = _parse_tool_call_block(block, param_types)

    end = part_start + close_idx + len(CLOSE_TAG)
    return ToolCallParseResult(
        thought=text_before,
        calls=calls,
        status=ToolCallParseStatus.VALID if calls else ToolCallParseStatus.MALFORMED,
        end=end,
        ignored_blocks=_count_tool_call_blocks(text, end),
    )


def _count_tool_call_blocks(text: str, end: int) -> int:
    """Count blocks after offset ``end``; prose mentioning the tag is not a block."""
    parts = text[end:].split(OPEN_TAG)
    count = 1 if _BLOCK_CONTENT_RE.search(parts[0]) else 0
    parts[0] = text[:end] + parts[0]
    return count + sum(
        1 for i in range(1, len(parts)) if not _is_prose_mention(parts[i - 1], parts[i])
    )


def _is_prose_mention(before: str, after: str) -> bool:
    """Whether an unclosed opener sits mid-sentence before a word or punctuation.

    An opener closed before the next one, or followed by anything else (a
    tag, JSON, brackets, quotes), is a block.
    """
    if CLOSE_TAG in after:
        return False
    line_before = before[before.rfind("\n") + 1 :].strip()
    line_after = after.split("\n", 1)[0].strip()
    return bool(line_before and line_after) and (
        line_after[0].isalpha() or line_after[0] in _PROSE_PUNCTUATION
    )


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


_ADD_MEMORY_RE = re.compile(
    r"<add_memory(?:\s+[^>]*)?>(.+?)</add_memory>",
    re.DOTALL,
)


def extract_add_memory(text: str) -> str:
    """Extract and combine content from all ``<add_memory>`` tags in LLM response text."""
    matches = _ADD_MEMORY_RE.findall(text)
    if not matches:
        return ""
    return "\n".join(m.strip() for m in matches if m.strip())


def _parse_tool_call_block(
    block: str, param_types: Optional[Dict[str, str]]
) -> List[ToolCall]:
    block = _sanitize_param_content(block)
    # Comments and processing instructions are dropped by the XML parser,
    # which would hide them from the checks below.
    if "<!" in block or "<?" in block:
        logger.warning("Comment or declaration in tool call XML block, skipping block")
        return []

    try:
        root = ET.fromstring(f"<root>{block}</root>")
    except (ET.ParseError, UnicodeError):
        logger.warning("Failed to parse tool call XML block, skipping")
        return []

    if _has_text_outside_parameters(root):
        logger.warning("Text outside tool calls in XML block, skipping block")
        return []

    for child in root:
        if child.tag != "invoke":
            logger.warning(
                "Unexpected <%s> element in tool call XML block, skipping block",
                child.tag,
            )
            return []

    calls: List[ToolCall] = []
    for invoke in root.findall("invoke"):
        name = invoke.get("name", "")
        if not name:
            logger.warning("Tool call without a name in XML block, skipping block")
            return []

        params: Dict[str, Any] = {}
        error: Optional[str] = None
        for param in invoke:
            param_name = param.get("name", "")
            if param.tag != "parameter" or not param_name or len(param):
                logger.warning(
                    "Invalid tool call parameter in XML block, skipping block"
                )
                return []
            try:
                params[param_name] = _coerce_param(
                    param_name, param.text or "", param_types
                )
            except ValueError as e:
                error = str(e)
                break

        calls.append(ToolCall(name=name, parameters=params, error=error))
    return calls


def _has_text_outside_parameters(root: ET.Element) -> bool:
    """Return whether the block has text anywhere except inside parameter values."""
    texts = [root.text]
    for invoke in root:
        texts += [invoke.tail, invoke.text]
        texts += [param.tail for param in invoke]
    return any(text and text.strip() for text in texts)


def _has_noncanonical_markup(block: str) -> bool:
    """Return whether tool-call markup other than plain <invoke>/<parameter>
    tags (DSML, full-width or alternate tags) occurs outside a trustworthy
    parameter payload."""
    payload_spans: List[Tuple[int, int]] = []
    for parameter in _PARAM_RE.finditer(block):
        payload = parameter.group(2)
        # A nested parameter opener means the regex may have crossed
        # from a missing close tag into a later parameter. Do not hide DSML in
        # that span from structural validation.
        if _PARAM_OPEN_RE.search(payload):
            continue
        payload_spans.append(parameter.span(2))

    for marker in _TOOL_CALL_MARKUP_RE.finditer(block):
        if marker.group(0) in _CANONICAL_BLOCK_TAGS:
            continue
        if not any(
            start <= marker.start() and marker.end() <= end
            for start, end in payload_spans
        ):
            return True
    return False


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
