"""Prompt template rendering and prompt text lookup contracts."""

from importlib.util import find_spec
from pathlib import Path

import pytest
from jinja2 import Environment

from mobilerun.utils.prompts import PromptLoader, load_prompt_text

AGENTS_SPEC = find_spec("mobilerun_agents")
PROMPTS_DIR = (
    Path(next(iter(AGENTS_SPEC.submodule_search_locations))) / "assets" / "prompts"
    if AGENTS_SPEC and AGENTS_SPEC.submodule_search_locations
    else None
)


def _actions(n):
    return [
        {
            "action": f"click({i})",
            "summary": f"s{i}",
            "outcome": i % 2 == 0,
            "error": "" if i % 2 == 0 else f"err{i}",
        }
        for i in range(n)
    ]


_SIGNATURES = {
    "click": {
        "description": "Click an element",
        "parameters": {
            "index": {"type": "int", "required": True},
            "long": {"type": "bool", "required": False, "default": False},
        },
    },
    "wait": {"description": "Wait", "parameters": {}},
}
_SCHEMA = {
    "description": "Result",
    "properties": {
        "title": {"type": "string", "description": "The title"},
        "count": {"type": "integer"},
    },
    "required": ["title"],
}

_POPULATED = {
    "instruction": "Open settings",
    "goal": "Open settings",
    "platform": "android",
    "device_date": "2026-01-01",
    "device_state": "state",
    "current_state": "cur",
    "previous_state": "prev",
    "previous_plan": "plan0",
    "plan": "1. a 2. b",
    "subgoal": "do a",
    "memory": "mem",
    "last_thought": "thought",
    "progress_summary": "progress",
    "progress_status": "status",
    "extra_context": "card",
    "important_notes": "notes",
    "custom_tools_descriptions": "- tool: desc",
    "tool_descriptions": "<tool>x</tool>",
    "available_secrets": ["pw", "token"],
    "available_tools": {"click", "type"},
    "variables": {"user": "bob"},
    "output_schema": _SCHEMA,
    "parallel_tools": True,
    "vision": True,
    "screenshot_only": True,
    "atomic_actions": _SIGNATURES,
    "action_history": _actions(7),
    "error_history": _actions(3),
}
_EMPTY = {
    "instruction": "",
    "goal": "",
    "platform": "ios",
    "variables": {},
    "available_secrets": [],
    "available_tools": set(),
    "output_schema": None,
    "parallel_tools": False,
    "vision": False,
    "screenshot_only": False,
    "atomic_actions": {},
    "action_history": [],
    "error_history": None,
}
_STRING_HISTORY = {**_POPULATED, "action_history": "a1\na2"}

TEMPLATES = sorted(PROMPTS_DIR.rglob("*.jinja2")) if PROMPTS_DIR else []


def test_base_owned_template_renders():
    source = "Hello {{ name }}!"
    assert PromptLoader.render_template(source, {"name": "world"}) == "Hello world!"


def test_bundled_templates_found():
    if AGENTS_SPEC is None:
        pytest.skip("agent prompt templates belong to the optional agents add-on")
    assert PROMPTS_DIR is not None and PROMPTS_DIR.is_dir()
    assert len(TEMPLATES) >= 8


@pytest.mark.parametrize(
    "path", TEMPLATES, ids=lambda p: str(p.relative_to(PROMPTS_DIR))
)
@pytest.mark.parametrize(
    "variables",
    [_POPULATED, _EMPTY, _STRING_HISTORY],
    ids=["populated", "empty", "string-history"],
)
def test_bundled_templates_render_identically(path, variables):
    source = path.read_text(encoding="utf-8")
    plain = Environment(
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False
    )
    expected = plain.from_string(source).render(**variables)
    assert PromptLoader.render_template(source, variables) == expected


def test_load_prompt_text_reads_bundled_file_and_matches_previous_guidance():
    assert load_prompt_text("state/screenshot_only_guidance.md") == (
        "Prefer click_at on the center of visible text or controls, especially in "
        "dense lists, adjacent rows, and compact menus. Use click_area only for "
        "large, unambiguous targets. If a tap does not change the screen, do not "
        "repeat the same coordinate; choose a better point on the intended target "
        "or use navigation. For text entry, focus a field with a coordinate action "
        "first, then use direct text typing."
    )


def test_load_prompt_text_prefers_working_directory_override(tmp_path, monkeypatch):
    relative_path = "assets/prompts/state/screenshot_only_guidance.md"
    override = tmp_path / relative_path
    override.parent.mkdir(parents=True)
    override.write_text("working directory override", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    load_prompt_text.cache_clear()
    try:
        assert load_prompt_text("state/screenshot_only_guidance.md") == (
            "working directory override"
        )
    finally:
        load_prompt_text.cache_clear()
