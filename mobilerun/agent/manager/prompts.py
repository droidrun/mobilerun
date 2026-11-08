"""
Prompts for the ManagerAgent.
"""

import re
from dataclasses import dataclass

FINAL_RESPONSE_TAGS = ("request_accomplished", "answer")

_MANAGER_RESULT_TAGS = ("plan", *FINAL_RESPONSE_TAGS)
_MANAGER_METADATA_TAGS = ("thought", "add_memory", "progress_summary")
_MANAGER_RESPONSE_ENVELOPE_TAGS = (*_MANAGER_RESULT_TAGS, *_MANAGER_METADATA_TAGS)
# Possessive quantifiers avoid pathological backtracking on truncated quoted attrs.
_TAG_TOKEN_RE = re.compile(
    r"""<(?P<closing>/)?(?P<tag>[A-Za-z][\w:.-]*)
    (?P<attrs>(?:[^<>"']++|"[^"]*"|'[^']*')*+)>""",
    re.DOTALL | re.VERBOSE,
)
_SPACED_MANAGER_RESULT_TAG_RE = re.compile(
    r"""<(?:\s+/?\s*|/\s+)(?P<tag>plan|request_accomplished|answer)\b""",
    re.IGNORECASE,
)
_TRAILING_UNTERMINATED_MANAGER_RESULT_TAG_RE = re.compile(
    r"""<(?P<closing>/)?(?P<tag>plan|request_accomplished|answer)\b[^<>]*\Z""",
    re.IGNORECASE,
)
_ATTRIBUTE_RE = re.compile(
    r"""\s+(?P<name>[^\s=/>]+)
    (?:\s*=\s*(?:(?P<quote>["'])(?P<quoted_value>.*?)(?P=quote)
    |(?P<bare_value>[^\s>]+)))?""",
    re.DOTALL | re.VERBOSE,
)
_NEXT_STEP_TAG_RE = re.compile(r"<\s*/?\s*next_step(?=[\s/>]|\Z)", re.IGNORECASE)
_NEXT_STEP_BLOCK_RE = re.compile(
    r"\A<next_step>(.*?)</next_step>", re.DOTALL | re.IGNORECASE
)


@dataclass(frozen=True)
class ManagerResponseValidation:
    is_valid: bool
    error_message: str | None = None


class ManagerResponseValidationError(RuntimeError):
    """Raised when the manager cannot produce a valid response after retries."""


def _find_tag_matches(response: str, tag: str) -> list[re.Match[str]]:
    pattern = re.compile(
        rf"<{tag}\b(?P<attrs>[^>]*)>(?P<body>.*?)</{tag}>",
        re.IGNORECASE | re.DOTALL,
    )
    return list(pattern.finditer(response))


def _tag_content(match: re.Match[str]) -> str:
    return match.group("body").strip()


def _success_from_attrs(attrs: str) -> bool | None:
    """Return a final tag's one valid, explicit success attribute.

    Only an attribute whose *name* is exactly ``success`` is accepted.  This
    deliberately rejects look-alikes such as ``data-success`` and strings in
    another attribute's value, as well as duplicate or unquoted attributes.
    """
    success_attributes: list[tuple[str | None, str]] = []
    position = 0

    while position < len(attrs):
        if not attrs[position:].strip():
            break

        match = _ATTRIBUTE_RE.match(attrs, position)
        if not match:
            return None

        if match.group("name").lower() == "success":
            value = match.group("quoted_value") or match.group("bare_value")
            success_attributes.append((match.group("quote"), value or ""))
        position = match.end()

    if len(success_attributes) != 1:
        return None

    quote, value = success_attributes[0]
    if quote is None or value.lower() not in {"true", "false"}:
        return None
    return value.lower() == "true"


def _find_top_level_manager_result_tags(
    response: str,
) -> tuple[list[dict[str, str]], list[str]]:
    """Find well-formed top-level result tags and report unsafe control tags.

    Manager output is XML-like rather than a complete XML document, so a small
    stack parser is more robust than globally matching ``<plan>`` and terminal
    tags.  In particular, manager response tags inside metadata or a result
    must not turn into a top-level plan or final result.  Ordinary markup in
    a top-level result remains body text.
    """
    stack: list[dict[str, str | int]] = []
    results: list[dict[str, str]] = []
    errors = [
        f"malformed <{match.group('tag').lower()}> tag"
        for match in _SPACED_MANAGER_RESULT_TAG_RE.finditer(response)
    ]
    errors.extend(
        "unterminated "
        f"{'closing ' if match.group('closing') else ''}"
        f"<{match.group('tag').lower()}> tag"
        for match in _TRAILING_UNTERMINATED_MANAGER_RESULT_TAG_RE.finditer(response)
    )

    for match in _TAG_TOKEN_RE.finditer(response):
        tag = match.group("tag").lower()
        attrs = match.group("attrs")
        is_closing = bool(match.group("closing"))
        is_self_closing = not is_closing and attrs.rstrip().endswith("/")
        if is_self_closing:
            attrs = attrs.rstrip()[:-1]

        if is_closing:
            if attrs.strip():
                if tag in _MANAGER_RESULT_TAGS:
                    errors.append(f"malformed closing <{tag}> tag")
                continue

            if not stack or stack[-1]["tag"] != tag:
                if tag in _MANAGER_RESULT_TAGS:
                    errors.append(f"unmatched closing <{tag}> tag")
                continue

            opening = stack.pop()
            if opening["tag"] in _MANAGER_RESULT_TAGS and opening["depth"] == 0:
                results.append(
                    {
                        "tag": str(opening["tag"]),
                        "attrs": str(opening["attrs"]),
                        "body": response[int(opening["body_start"]) : match.start()],
                    }
                )
            continue

        if is_self_closing:
            if tag in _MANAGER_RESULT_TAGS:
                errors.append(f"self-closing <{tag}> tag")
            continue

        has_open_top_level_result = any(
            opening["tag"] in _MANAGER_RESULT_TAGS and opening["depth"] == 0
            for opening in stack
        )
        if has_open_top_level_result and tag not in _MANAGER_RESPONSE_ENVELOPE_TAGS:
            continue

        depth = len(stack)
        if tag in _MANAGER_RESULT_TAGS and depth:
            errors.append(f"nested <{tag}> tag")
        stack.append(
            {
                "tag": tag,
                "attrs": attrs,
                "depth": depth,
                "start": match.start(),
                "body_start": match.end(),
            }
        )

    for opening in stack:
        if opening["tag"] in _MANAGER_RESULT_TAGS:
            errors.append(f"unclosed <{opening['tag']}> tag")

    return results, errors


def _parse_current_subgoal(
    plan: str, require_next_step: bool
) -> tuple[str, str | None]:
    """Extract the next instruction and any format error from a plan."""
    script = re.match(r"<script>(.*?)</script>", plan, re.DOTALL)
    if script:
        return f"<script>{script[1].strip()}</script>", None

    markers = _NEXT_STEP_TAG_RE.findall(plan)
    if markers:
        block = _NEXT_STEP_BLOCK_RE.match(plan)
        if not block or len(markers) != 2 or not block[1].strip():
            return "", (
                "Begin <plan> with a single nonempty <next_step>...</next_step> block. "
                "Escape literal tag text inside the instruction."
            )

        # Remove framing newlines without trimming the instruction's whitespace.
        step = re.sub(r"\A\r?\n|\r?\n\Z", "", block[1])
        step = step.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        return step, None

    lines = [line.strip() for line in plan.splitlines() if line.strip()]
    if require_next_step and len(lines) > 1:
        return "", (
            "Begin the multiline <plan> with <next_step>...</next_step> containing "
            "the complete next action and its text. Place later actions after the block."
        )

    first_line = lines[0] if lines else ""
    first_line = re.sub(r"^\d+\.\s*", "", first_line)
    first_line = re.sub(r"^[-*]\s*", "", first_line)
    first_line = re.sub(r"^•\s*", "", first_line)
    return first_line.strip(), None


def parse_manager_response(response: str, *, require_next_step: bool = False) -> dict:
    """
    Parse manager LLM response into structured dict.

    Extracts XML-style tags from the response:
    - <thought>...</thought>
    - <add_memory>...</add_memory>
    - <plan>...</plan>
    - <request_accomplished success="true|false">...</request_accomplished> (answer)

    Also derives:
    - current_subgoal: the explicit next step, a leading script, or the legacy first line
    - success: bool | None parsed from request_accomplished success attribute

    Args:
        response: Raw LLM response text
        require_next_step: Require an explicit step for multiline text plans.

    Returns:
        Dict with keys:
            - thought: str
            - memory: str
            - plan: str
            - current_subgoal: str (the next instruction or script)
            - subgoal_error: str | None (invalid next-step format)
            - answer: str (from request_accomplished tag)
            - success: bool | None (True/False if task complete, None if still in progress)
    """

    def extract(tag: str) -> str:
        """Extract content between XML-style tags (handles attributes)."""
        matches = _find_tag_matches(response, tag)
        if matches:
            return _tag_content(matches[0])
        return ""

    def extract_all(tag: str) -> str:
        """Extract and combine content from all occurrences of a tag."""
        matches = [_tag_content(match) for match in _find_tag_matches(response, tag)]
        if not matches:
            return ""
        return "\n".join(match for match in matches if match)

    thought = extract("thought")
    memory_section = extract_all("add_memory")
    progress_summary = extract("progress_summary")
    result_tags, result_tag_errors = _find_top_level_manager_result_tags(response)
    plan_matches = [tag for tag in result_tags if tag["tag"] == "plan"]
    final_matches = [tag for tag in result_tags if tag["tag"] in FINAL_RESPONSE_TAGS]
    plan = plan_matches[0]["body"].strip() if plan_matches else ""

    final_match = None
    for match in final_matches:
        if match["body"].strip():
            final_match = match
            break
    if final_match is None and final_matches:
        final_match = final_matches[0]

    answer = final_match["body"].strip() if final_match else ""
    success = None
    final_tag = None
    if final_match:
        final_tag = final_match["tag"]
        success = _success_from_attrs(final_match["attrs"])

    final_counts = {
        tag: sum(1 for match in final_matches if match["tag"] == tag)
        for tag in FINAL_RESPONSE_TAGS
    }

    current_subgoal, subgoal_error = _parse_current_subgoal(plan, require_next_step)

    return {
        "thought": thought,
        "plan": plan,
        "memory": memory_section,
        "current_subgoal": current_subgoal,
        "subgoal_error": subgoal_error,
        "answer": answer,
        "success": success,
        "progress_summary": progress_summary,
        "final_tag": final_tag,
        "result_tag_errors": result_tag_errors,
        "tag_counts": {
            "plan": len(plan_matches),
            "request_accomplished": final_counts["request_accomplished"],
            "answer": final_counts["answer"],
            "final": len(final_matches),
        },
    }


def validate_manager_response(parsed: dict) -> ManagerResponseValidation:
    """Validate the manager output contract parsed by parse_manager_response."""
    plan = (parsed.get("plan") or "").strip()
    answer = (parsed.get("answer") or "").strip()
    success = parsed.get("success")
    result_tag_errors = parsed.get("result_tag_errors") or []
    tag_counts = parsed.get("tag_counts") or {}
    plan_count = tag_counts.get("plan", 1 if plan else 0)
    final_count = tag_counts.get("final", 1 if answer else 0)

    if result_tag_errors:
        return ManagerResponseValidation(
            is_valid=False,
            error_message="Manager response contains malformed or nested control tags. "
            "Provide one complete, top-level <plan> or final answer tag.",
        )

    if plan_count > 1:
        return ManagerResponseValidation(
            is_valid=False,
            error_message="Manager response contains multiple <plan> tags. "
            "Provide exactly one <plan>, or exactly one final answer tag.",
        )

    if final_count > 1:
        return ManagerResponseValidation(
            is_valid=False,
            error_message="Manager response contains multiple final answer tags. "
            "Provide exactly one <request_accomplished> or <answer> tag.",
        )

    if plan_count == 1 and final_count == 1:
        return ManagerResponseValidation(
            is_valid=False,
            error_message="Manager response must provide exactly one of <plan> "
            "or a final answer tag, not both.",
        )

    if plan_count == 1 and not plan:
        return ManagerResponseValidation(
            is_valid=False,
            error_message="Manager <plan> tag must not be empty.",
        )

    if final_count == 1 and not answer:
        return ManagerResponseValidation(
            is_valid=False,
            error_message="Manager final answer tag must not be empty.",
        )

    if answer:
        if success is None:
            return ManagerResponseValidation(
                is_valid=False,
                error_message='Final answer tag must include success="true" or success="false".',
            )
        return ManagerResponseValidation(is_valid=True)

    if plan:
        if subgoal_error := parsed.get("subgoal_error"):
            return ManagerResponseValidation(
                is_valid=False, error_message=subgoal_error
            )
        return ManagerResponseValidation(is_valid=True)

    return ManagerResponseValidation(
        is_valid=False,
        error_message="Manager response must provide exactly one of <plan> or a final answer tag.",
    )
