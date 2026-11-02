"""Tests for sandboxed prompt rendering in PromptLoader."""

from pathlib import Path

import pytest
from jinja2 import Environment
from jinja2.exceptions import SecurityError
from jinja2.sandbox import ImmutableSandboxedEnvironment

import mobilerun
from mobilerun.config_manager.prompt_loader import PromptLoader

PROMPTS_DIR = Path(mobilerun.__file__).parent / "config" / "prompts"

ESCAPE_PAYLOADS = [
    "{{ cycler.__init__.__globals__ }}",
    "{{ ''.__class__.__mro__[1].__subclasses__() }}",
    "{{ '{0.__class__}'.format(cycler) }}",
    "{{ cycler|attr('__init__')|attr('__globals__') }}",
    "{{ lipsum.__globals__ }}",
    "{{ range.__self__ }}",
]


@pytest.mark.parametrize("payload", ESCAPE_PAYLOADS)
def test_escape_attempts_do_not_expose_internals(payload):
    try:
        out = PromptLoader.render_template(payload)
    except SecurityError:
        return
    for marker in ("<module", "<class", "<function"):
        assert marker not in out


def test_template_cannot_mutate_dict():
    d = {"a": 1}
    with pytest.raises(SecurityError):
        PromptLoader.render_template("{% set _ = d.update({'x': 1}) %}", {"d": d})
    assert d == {"a": 1}


def test_template_cannot_mutate_list():
    items = [1, 2]
    with pytest.raises(SecurityError):
        PromptLoader.render_template("{% set _ = items.append(3) %}", {"items": items})
    assert items == [1, 2]


def test_cached_environment_is_sandboxed_with_expected_options():
    PromptLoader._env = None
    env = PromptLoader._get_environment()
    assert isinstance(env, ImmutableSandboxedEnvironment)
    assert env.trim_blocks is True
    assert env.lstrip_blocks is True
    assert env.keep_trailing_newline is False
    assert PromptLoader._get_environment() is env


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
    "app_card": "card",
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

TEMPLATES = sorted(PROMPTS_DIR.rglob("*.jinja2"))


def test_bundled_templates_found():
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
