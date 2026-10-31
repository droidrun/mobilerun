import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from mobilerun.agent.manager.stateless_manager_agent import StatelessManagerAgent
from mobilerun.agent.utils.prompt_resolver import PromptResolver
from mobilerun.config_manager.config_manager import (
    DEFAULT_MANAGER_SYSTEM_PROMPT,
    DEFAULT_STATELESS_MANAGER_SYSTEM_PROMPT,
    AgentConfig,
    ManagerConfig,
)
from mobilerun.config_manager.migrations import CURRENT_VERSION, migrate

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(coro):
    return asyncio.run(coro)


def _stateless_agent(
    manager_config: ManagerConfig,
    prompt_resolver: PromptResolver | None = None,
) -> StatelessManagerAgent:
    agent = object.__new__(StatelessManagerAgent)
    agent.agent_config = AgentConfig(manager=manager_config)
    agent.prompt_resolver = prompt_resolver or PromptResolver()
    agent.shared_state = SimpleNamespace(
        instruction="Read Android version",
        platform="android",
        device_date="2026-07-22",
        previous_plan="Open Settings",
        previous_formatted_device_state="Previous UI",
        agent_memory="Memory",
        last_thought="Last thought",
        progress_summary="Opened Settings",
        formatted_device_state="Current UI",
    )
    agent._build_action_history = lambda: []
    return agent


def test_manager_auto_prompt_uses_the_mode_appropriate_bundled_template():
    stateful_path = AgentConfig(
        manager=ManagerConfig(stateless=False)
    ).get_manager_system_prompt_path()
    stateless_path = AgentConfig(
        manager=ManagerConfig(stateless=True)
    ).get_manager_system_prompt_path()

    assert Path(stateful_path).as_posix().endswith(DEFAULT_MANAGER_SYSTEM_PROMPT)
    assert Path(stateless_path).as_posix().endswith(DEFAULT_STATELESS_MANAGER_SYSTEM_PROMPT)


def test_runtime_custom_prompt_precedes_an_explicit_manager_prompt(monkeypatch):
    configured_prompt = REPO_ROOT / "mobilerun/config/prompts/manager/trained.jinja2"
    agent = _stateless_agent(
        ManagerConfig(stateless=True, system_prompt=str(configured_prompt)),
        PromptResolver({"manager_system": "custom {{ current_state }}"}),
    )

    async def unexpected_file_load(*args, **kwargs):
        raise AssertionError("A runtime custom prompt must take precedence.")

    monkeypatch.setattr(
        "mobilerun.agent.manager.stateless_manager_agent.PromptLoader.load_prompt",
        unexpected_file_load,
    )

    assert _run(agent._build_prompt()) == "custom Current UI"


@pytest.mark.parametrize("platform", ("android", "ios"))
def test_default_stateless_prompt_renders_platform_and_progress_context(platform):
    agent = _stateless_agent(ManagerConfig(stateless=True))
    agent.shared_state.platform = platform

    rendered = _run(agent._build_prompt())

    for expected_text in (
        f"operate a {platform} device",
        "<previous_plan>\nOpen Settings",
        "<memory>\nMemory",
        "<progress_summary>\nOpened Settings",
        "<current_state>\nCurrent UI",
    ):
        assert expected_text in rendered


@pytest.mark.parametrize(
    ("template_path", "contract_start", "expected_examples"),
    [
        (
            "mobilerun/config/prompts/manager/system.jinja2",
            "Use `<plan>` for unfinished work.",
            (
                """**When work remains:**
<plan>
Update or copy the existing plan according to the current page and progress. The first item must be the next subgoal.
</plan>""",
                """<request_accomplished success="true">
Confirmation of the completed request.
</request_accomplished>""",
                """<request_accomplished success="false">
Explanation of why the request cannot be completed.
</request_accomplished>""",
            ),
        ),
        (
            "mobilerun/config/prompts/manager/stateless.jinja2",
            "Use `<plan>` for unfinished work.",
            (
                """**When work remains:**
<plan>
Updated plan with numbered steps. The first item is what the executor will do next.
</plan>""",
                """<answer success="true">
Confirmation of the completed task.
</answer>""",
                """<answer success="false">
Explanation of why the task cannot be completed.
</answer>""",
            ),
        ),
    ],
)
def test_default_manager_prompts_keep_examples_without_cardinality_rules(
    template_path, contract_start, expected_examples
):
    prompt = (REPO_ROOT / template_path).read_text(encoding="utf-8")
    contract = prompt.split(contract_start, maxsplit=1)[1].lower()

    for example in expected_examples:
        assert example in prompt
    for removed_rule in (
        "exactly one",
        "never include both",
        "not both",
        "never repeat",
        "return only",
        "one of",
    ):
        assert removed_rule not in contract


@pytest.mark.parametrize(
    ("system_prompt", "expected_prompt"),
    [
        (DEFAULT_MANAGER_SYSTEM_PROMPT, None),
        ("config/prompts/manager/rev1.jinja2", "config/prompts/manager/rev1.jinja2"),
    ],
)
def test_v7_migration_only_converts_the_old_canonical_manager_prompt_path(
    system_prompt, expected_prompt
):
    config = {
        "_version": 6,
        "agent": {"manager": {"system_prompt": system_prompt, "stateless": True}},
    }

    migrated = migrate(config)

    assert migrated["_version"] == CURRENT_VERSION == 7
    assert migrated["agent"]["manager"]["system_prompt"] == expected_prompt
