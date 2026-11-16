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
def test_vision_on_instructions(prompt):
    path = f"mobilerun/config/prompts/{prompt}"
    current = (ROOT / path).read_text()
    context = {"vision": True, "screenshot_only": True, "atomic_actions": {}}
    rendered = PromptLoader.render_template(current, context)
    block = (
        (ROOT / "mobilerun/config/prompts/shared/vision_off_grounding.jinja2")
        .read_text()
        .strip()
    )
    assert block not in rendered
    if not prompt.startswith("executor/"):
        assert rendered.count("## Reading Values and Reporting Results") == 1
        assert "supplied screenshot" in rendered
        assert "host-side HTTP" in rendered
        assert "could not be read" in rendered
        assert "previous state or memory" in rendered
    assert "Vision is off" not in rendered
