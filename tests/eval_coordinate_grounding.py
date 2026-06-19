"""Manual eval: per-model coordinate grounding through the real a11y+vision pipeline.

Not collected by pytest (no ``test_`` prefix) — it needs a connected Android
device and configured LLM credentials. Run it whenever adding a model/provider
or touching the coordinate contract:

    .venv/bin/python tests/eval_coordinate_grounding.py [serial]

For each provider it builds the real ``AndroidStateProvider(vision_enabled=True)``
state, attaches the exact resized+grid screenshot the agents send, asks the
model for click_at coordinates of known on-screen labels, converts them through
``_convert_action_point``, and scores against native a11y bounds.

Background (measured 2026-06-11, issue #350): model output spaces are
model-dependent when the contract is absent — gpt-5.4-mini answered in a
~1.18x-downscaled space, claude-opus-4-7 in near-native space, so no fixed
multiplier can be correct. With the resize+grid+declared-dims contract both
answered in the declared space (mean y-error 16px / 28px vs 213px broken).
"""

import asyncio
import re
import sys
from types import SimpleNamespace

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, TextBlock

from mobilerun.agent.utils.actions import _convert_action_point
from mobilerun.agent.utils.llm_picker import load_llm
from mobilerun.agent.utils.vision_sizing import VisionResizePolicy
from mobilerun.tools.filters import ConciseFilter
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.ui.provider import (
    AndroidStateProvider,
    resize_model_screenshot_with_grid,
)

PROVIDERS = [
    ("openai", "OpenAIResponses", "gpt-5.4-mini"),
    ("anthropic", "Anthropic", "claude-opus-4-8"),
    # claude-sonnet-4-6 uses Anthropic's standard 1568 visual-token budget, so it
    # downsizes a 2048-declared screenshot and would undershoot ~23% without the
    # per-model resize policy (issue #365). Keep it here so the eval covers an
    # AFFECTED model, not just the high-res opus that masked the bug.
    ("anthropic", "Anthropic", "claude-sonnet-4-6"),
    # GoogleGenAI Gemini vision quota is tight for repeated calls; add it back
    # when evaluating Gemini support.
]
MAX_TARGETS = 4


def _parse_xy(text: str):
    m = re.search(r'"x"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*"y"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    if m:
        return float(m.group(1)), float(m.group(2))
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return (float(nums[0]), float(nums[1])) if len(nums) >= 2 else None


def _pick_targets(state):
    flat = []

    def walk(elements):
        for element in elements:
            flat.append(element)
            walk(element.get("children") or [])

    walk(state.elements)
    targets, seen = [], set()
    for element in flat:
        text = (element.get("text") or "").strip()
        if not text or len(text) <= 6 or not element.get("bounds"):
            continue
        if text in seen:
            continue
        left, top, right, bottom = [int(p) for p in element["bounds"].split(",")]
        if 150 <= right - left <= 1050 and 30 <= bottom - top <= 260 and top >= 150:
            seen.add(text)
            targets.append((text, (left, top, right, bottom)))
    step = max(1, len(targets) // MAX_TARGETS)
    return targets[::step][:MAX_TARGETS]


async def main(serial: str) -> None:
    from mobilerun_core_cli.driver.android import AndroidDriver

    driver = AndroidDriver(serial=serial)
    await driver.connect()
    provider = AndroidStateProvider(
        driver,
        tree_filter=ConciseFilter(),
        tree_formatter=IndexedFormatter(),
        vision_enabled=True,
    )
    screenshot = await driver.screenshot()
    # Device-pixel target bounds are model-independent; pick them once.
    targets = _pick_targets(await provider.get_state())
    if not targets:
        print("No suitable targets on screen — open a list-style screen (e.g. Settings).")
        sys.exit(1)
    print(f"targets: {[t for t, _ in targets]}")

    for name, provider_key, model in PROVIDERS:
        try:
            llm = load_llm(provider_key, model=model)
        except Exception as e:
            print(f"--- {name}: SKIP ({type(e).__name__}: {e})")
            continue
        # Declare + send this model's EFFECTIVE size, exactly like production, so
        # the eval validates the fixed path (not the pre-fix fixed-2048 space).
        provider.vision_resize_policy = VisionResizePolicy([model])
        state = await provider.get_state()
        sent_image = resize_model_screenshot_with_grid(provider, screenshot)
        ctx = SimpleNamespace(ui=state, state_provider=provider)
        print(
            f"--- {name} ({model}) declared="
            f"{provider.model_screenshot_width}x{provider.model_screenshot_height} ---"
        )
        for text, (left, top, right, bottom) in targets:
            message = ChatMessage(
                role="user",
                blocks=[
                    TextBlock(
                        text=(
                            "You control an Android phone. Here is the current "
                            "device state:\n\n" + state.formatted_text + "\n\n"
                            f"You want to click_at the list item with text '{text}'. "
                            'Respond with ONLY JSON {"x": <int>, "y": <int>} '
                            "for the click_at action."
                        )
                    ),
                    ImageBlock(image=sent_image),
                ],
            )
            response = await llm.achat([message])
            xy = _parse_xy(str(response.message.content))
            if xy is None:
                print(f"  {text[:32]:34} UNPARSEABLE: {str(response.message.content)[:60]!r}")
                continue
            try:
                device_x, device_y = _convert_action_point(xy[0], xy[1], ctx=ctx)
            except ValueError as e:
                print(f"  {text[:32]:34} REJECTED: {str(e)[:70]}")
                continue
            center_x, center_y = (left + right) / 2, (top + bottom) / 2
            hit = left <= device_x <= right and top <= device_y <= bottom
            dist = ((device_x - center_x) ** 2 + (device_y - center_y) ** 2) ** 0.5
            print(
                f"  {text[:32]:34} model={tuple(int(v) for v in xy)} -> "
                f"native=({device_x},{device_y}) center=({center_x:.0f},{center_y:.0f}) "
                f"hit={hit} dist={dist:.0f}px"
            )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "emulator-5554"))
