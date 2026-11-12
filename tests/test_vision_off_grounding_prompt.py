import subprocess
from pathlib import Path

import pytest

from mobilerun.config_manager.prompt_loader import PromptLoader

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    "fast_agent/system.jinja2",
    "manager/system.jinja2",
    "manager/stateless.jinja2",
    "manager/rev1.jinja2",
    "manager/trained.jinja2",
    "executor/system.jinja2",
    "executor/rev1.jinja2",
]
SCREENSHOT_INSTRUCTIONS = [
    "Analyze the current device state and screenshot",
    "assess the current status and the provided screenshot",
    "**Assess** the current screenshot",
    "inspect the screenshot",
]


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("screenshot_only", [False, True])
def test_vision_off_grounded(prompt, screenshot_only):
    template = (ROOT / "mobilerun/config/prompts" / prompt).read_text()
    rendered = PromptLoader.render_template(
        template,
        {
            "vision": False,
            "screenshot_only": screenshot_only,
            "atomic_actions": {},
        },
    )
    block = (
        (ROOT / "mobilerun/config/prompts/shared/vision_off_grounding.jinja2")
        .read_text()
        .strip()
    )
    assert rendered.count(block) == 1
    for instruction in SCREENSHOT_INSTRUCTIONS:
        assert instruction not in rendered


@pytest.mark.parametrize("prompt", PROMPTS)
def test_vision_on_matches_develop(prompt):
    path = f"mobilerun/config/prompts/{prompt}"
    original = subprocess.check_output(
        ["git", "show", f"origin/develop:{path}"], cwd=ROOT, text=True
    )
    current = (ROOT / path).read_text()
    context = {"vision": True, "screenshot_only": True, "atomic_actions": {}}
    rendered = PromptLoader.render_template(current, context)
    block = (
        (ROOT / "mobilerun/config/prompts/shared/vision_off_grounding.jinja2")
        .read_text()
        .strip()
    )
    assert block not in rendered
    assert rendered == PromptLoader.render_template(original, context)
