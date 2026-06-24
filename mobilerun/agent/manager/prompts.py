"""
Prompts for the ManagerAgent.
"""

import re
from dataclasses import dataclass

FINAL_RESPONSE_TAGS = ("request_accomplished", "answer")

_SUCCESS_ATTR_RE = re.compile(
    r"""\bsuccess\s*=\s*(?P<quote>["'])(?P<value>true|false)(?P=quote)""",
    re.IGNORECASE | re.DOTALL,
)
_SUCCESS_ATTR_PRESENT_RE = re.compile(r"\bsuccess\s*=", re.IGNORECASE | re.DOTALL)
_FINAL_TAG_RE = re.compile(
    r"<(?P<tag>request_accomplished|answer)\b(?P<attrs>[^>]*)>"
    r"(?P<body>.*?)"
    r"</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_OPEN_TAG_RE = re.compile(
    r"<(?P<tag>request_accomplished|answer)\b(?P<attrs>[^>]*)>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ManagerResponseValidation:
    is_valid: bool
    error_message: str | None = None
    can_continue_with_plan: bool = False
    can_accept_final_without_success: bool = False


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
    match = _SUCCESS_ATTR_RE.search(attrs)
    if not match:
        return None
    return match.group("value").lower() == "true"


def strip_manager_final_tags(response: str) -> str:
    """Remove final answer tags while leaving reasoning and plan tags intact."""
    return _FINAL_TAG_RE.sub("", response).strip()


def add_default_success_to_final_tag(response: str) -> str:
    """Add success=true to a final tag that omitted the success attribute."""

    def replace(match: re.Match[str]) -> str:
        attrs = match.group("attrs")
        if _SUCCESS_ATTR_PRESENT_RE.search(attrs):
            return match.group(0)
        return f'<{match.group("tag")}{attrs} success="true">'

    return _FINAL_OPEN_TAG_RE.sub(replace, response, count=1)


def parse_manager_response(response: str) -> dict:
    """
    Parse manager LLM response into structured dict.

    Extracts XML-style tags from the response:
    - <thought>...</thought>
    - <add_memory>...</add_memory>
    - <plan>...</plan>
    - <request_accomplished success="true|false">...</request_accomplished> (answer)

    Also derives:
    - current_subgoal: first line of plan (with list markers removed)
    - If first item is <script> tag, extract script content as current_subgoal
    - success: bool | None parsed from request_accomplished success attribute

    Args:
        response: Raw LLM response text

    Returns:
        Dict with keys:
            - thought: str
            - memory: str
            - plan: str
            - current_subgoal: str (first line of plan, cleaned, or script content)
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
    plan = extract("plan")
    progress_summary = extract("progress_summary")
    plan_matches = _find_tag_matches(response, "plan")
    final_matches = list(_FINAL_TAG_RE.finditer(response))
    final_matches.sort(key=lambda match: match.start())

    final_match = None
    for match in final_matches:
        if _tag_content(match):
            final_match = match
            break
    if final_match is None and final_matches:
        final_match = final_matches[0]

    answer = _tag_content(final_match) if final_match else ""
    success = None
    final_tag = None
    success_attr_present = False
    if final_match:
        final_tag = final_match.group("tag").lower()
        final_attrs = final_match.group("attrs")
        success_attr_present = bool(_SUCCESS_ATTR_PRESENT_RE.search(final_attrs))
        success = _success_from_attrs(final_attrs)

    final_counts = {
        tag: sum(1 for match in final_matches if match.group("tag").lower() == tag)
        for tag in FINAL_RESPONSE_TAGS
    }

    # Parse current subgoal from first line of plan
    current_goal_text = plan

    # Check if first item is a <script> tag
    script_match = re.search(
        r"^\s*<script>(.*?)</script>", current_goal_text, re.DOTALL
    )

    if script_match:
        # Script is first task - extract script content with tag
        current_subgoal = f"<script>{script_match.group(1).strip()}</script>"
    else:
        # Regular subgoal - use existing logic
        plan_lines = [
            line.strip() for line in current_goal_text.splitlines() if line.strip()
        ]
        if plan_lines:
            first_line = plan_lines[0]
        else:
            first_line = current_goal_text.strip()

        # Remove common list markers like "1.", "-", "*", or bullet characters
        first_line = re.sub(
            r"^\s*\d+\.\s*", "", first_line
        )  # Remove "1. ", "2. ", etc.
        first_line = re.sub(r"^\s*[-*]\s*", "", first_line)  # Remove "- " or "* "
        first_line = re.sub(r"^\s*•\s*", "", first_line)  # Remove bullet "• "

        current_subgoal = first_line.strip()

    return {
        "thought": thought,
        "plan": plan,
        "memory": memory_section,
        "current_subgoal": current_subgoal,
        "answer": answer,
        "success": success,
        "success_attr_present": success_attr_present,
        "progress_summary": progress_summary,
        "final_tag": final_tag,
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
    success_attr_present = bool(parsed.get("success_attr_present"))
    tag_counts = parsed.get("tag_counts") or {}
    plan_count = tag_counts.get("plan", 1 if plan else 0)
    final_count = tag_counts.get("final", 1 if answer else 0)

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
            can_continue_with_plan=plan_count == 1 and bool(plan),
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
            can_continue_with_plan=plan_count == 1 and bool(plan),
        )

    if plan and answer:
        return ManagerResponseValidation(
            is_valid=False,
            error_message="Manager response must provide exactly one of <plan> "
            "or a final answer tag, not both.",
            can_continue_with_plan=plan_count == 1 and final_count == 1,
        )

    if answer:
        if success is None:
            return ManagerResponseValidation(
                is_valid=False,
                error_message='Final answer tag must include success="true" '
                'or success="false".',
                can_accept_final_without_success=not success_attr_present,
            )
        return ManagerResponseValidation(is_valid=True)

    if plan:
        return ManagerResponseValidation(is_valid=True)

    return ManagerResponseValidation(
        is_valid=False,
        error_message="Manager response must provide exactly one of <plan> "
        "or a final answer tag.",
    )
