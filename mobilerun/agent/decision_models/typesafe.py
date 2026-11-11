"""Native TypeSafe Jev decisions, without a chat/LLM compatibility layer."""

import asyncio
import logging
import math
import time
from contextlib import suppress
from io import BytesIO
from typing import Any

from llama_index.core.base.llms.types import ChatMessage
from PIL import Image as PILImage
from PIL import ImageOps, UnidentifiedImageError

from mobilerun.agent.selectors.base import SelectionRequest
from mobilerun.agent.selectors.decision import Decision
from mobilerun.config_manager.config_manager import AgentConfig
from mobilerun.config_manager.prompt_loader import PromptLoader

logger = logging.getLogger("mobilerun")


async def _log_request_progress(started: float) -> None:
    """Keep slow uploads visibly distinct from a stalled agent loop."""
    while True:
        await asyncio.sleep(5)
        logger.info(
            "Jev request still in flight (%.0fs); waiting for TypeSafe.",
            time.monotonic() - started,
        )


def _prepare_screenshot(screenshot: bytes) -> bytes:
    return _prepare_image(screenshot)[0]


def _prepare_image(screenshot: bytes) -> tuple[bytes, tuple[float, float]]:
    """Fit TypeSafe's 280–1568px per-side limits without cropping the UI.

    https://docs.typesafe.ai/images#limits
    Jev selects element-index actions, so this provider-local resize must not
    change the UI state's native bounds or the driver's tap coordinates.
    """
    try:
        with PILImage.open(BytesIO(screenshot)) as original:
            if getattr(original, "is_animated", False):
                raise ValueError("TypeSafe screenshots must be still images")
            image = ImageOps.exif_transpose(original).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("TypeSafe requires a valid screenshot image") from exc

    width, height = image.size
    scale = min(1.0, 1568 / max(width, height))
    if min(width, height) * scale < 280:
        scale = min(280 / min(width, height), 1568 / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if size != image.size:
        image = image.resize(size, PILImage.Resampling.LANCZOS)
    # Very narrow/tall images cannot satisfy both limits by scaling alone.
    padded = (max(280, size[0]), max(280, size[1]))
    image = ImageOps.pad(image, padded, color="white", centering=(0, 0))
    output = BytesIO()
    # Full-screen PNGs with wallpaper and coordinate grids can exceed 700 KB.
    # Baseline JPEG is supported by TypeSafe and keeps the same dimensions and
    # full UI while substantially reducing upload time on slow connections.
    image.save(output, format="JPEG", quality=85, optimize=True, progressive=False)
    # Share of the sent image the screenshot covers; padding is right/bottom.
    return output.getvalue(), (size[0] / padded[0], size[1] / padded[1])


MAX_CHOICES = 255


_LOCATE_INSTRUCTIONS = (
    "If the chosen action taps an on-screen element, the element to tap."
)


class TypeSafeDecisionModel:
    """Choose only among supplied actions; abstain on low-confidence answers.

    SDK constructor options (including api_key, base_url and timeout) are
    forwarded unchanged. Authentication defaults to TYPESAFE_API_KEY.
    """

    screenshot_max_side = 1568
    # A Choice accepts 255 options, three of which are abstain and completion.
    max_options = MAX_CHOICES - 3
    # Box locates on-screen taps, so screenshot-only screens need no grid.
    locates_points = True

    def __init__(
        self,
        model: str | None = None,
        min_confidence: float = 0.0,
        **kwargs: Any,
    ):
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        try:
            from typesafe_sdk import AsyncTypeSafeClient
        except ImportError as exc:
            raise ImportError(
                "TypeSafe decision models require the TypeSafe SDK. "
                'Install it with: pip install "mobilerun[typesafe]"'
            ) from exc
        self._client_type = AsyncTypeSafeClient
        self._client_kwargs = dict(kwargs)
        if model is not None:
            self._client_kwargs["model"] = model
        self.min_confidence = min_confidence

    async def decide(self, request: SelectionRequest) -> Decision | None:
        from typesafe_sdk import Choice

        if not request.options:
            raise ValueError("TypeSafe decisions require non-empty action options")

        # Opaque, unique labels preserve option identity even when descriptions
        # collide. The returned label is never interpreted as a tool call.
        indices = {f"action_{i}": i for i in range(len(request.options))}
        criteria: dict[str, Any] = {
            label: {
                "description": request.options[i].description,
                "tool": request.options[i].action.name,
                "parameters": request.options[i].action.parameters,
            }
            for label, i in indices.items()
        }
        criteria["abstain"] = "None of the available actions safely advances the goal."
        if "complete" in request.tools:
            criteria["complete_success"] = (
                "The goal is already achieved, as verified by the current state."
            )
            criteria["complete_failure"] = (
                "The goal cannot be achieved with the available capabilities."
            )
        if len(criteria) > MAX_CHOICES:
            raise ValueError(
                f"TypeSafe accepts at most {MAX_CHOICES} choices, got {len(criteria)}"
            )

        state: dict[str, Any] = {
            "goal": request.goal,
            "memory": request.memory,
            "history": [
                (
                    {"role": item.role.value, "content": item.content or ""}
                    if isinstance(item, ChatMessage)
                    else item
                )
                for item in request.history
            ],
            "variables": request.variables,
        }
        if request.state is not None:
            state.update(
                ui=request.state.formatted_text,
                focused_text=request.state.focused_text,
                phone_state=request.state.phone_state,
            )
        request_kwargs: dict[str, Any] = {}
        questions: dict[str, Any] = {}
        if request.screenshot is not None:
            from typesafe_sdk import Image

            prepared_screenshot, coverage = _prepare_image(request.screenshot)
            state["screenshot"] = Image(prepared_screenshot)
            if any(option.point_space for option in request.options):
                from typesafe_sdk import Box

                questions["target"] = Box(instructions=_LOCATE_INSTRUCTIONS)
            # Vision is already explicitly enabled by the caller. TypeSafe also
            # requires this per-request opt-in for its experimental image API.
            request_kwargs["extra_headers"] = {"TypeSafe-Danger-Zone-Images": "true"}

        instructions = request.instructions
        if not instructions:
            instructions = await PromptLoader.load_prompt(
                AgentConfig().get_decision_system_prompt_path(),
                request.get_decision_prompt_variables(),
            )
        started = time.monotonic()
        logger.info(
            "Jev selecting from %d actions%s...",
            len(request.options),
            (
                f" (image {len(prepared_screenshot) / 1024:.0f} KiB)"
                if request.screenshot is not None
                else ""
            ),
        )
        progress = asyncio.create_task(_log_request_progress(started))
        # A scoped client avoids leaking HTTP resources, including on cancellation.
        try:
            async with self._client_type(**self._client_kwargs) as client:
                result = await client.system_one(
                    state=state,
                    questions={
                        "action": Choice(instructions=instructions, criteria=criteria),
                        **questions,
                    },
                    **request_kwargs,
                )
        finally:
            progress.cancel()
            with suppress(asyncio.CancelledError):
                await progress
        answer = result.choices.get("action")
        target = result.boxes.get("target") if questions else None
        logger.info(
            "Jev returned %s (confidence=%s)%s in %.2fs.",
            getattr(answer, "choice", "<missing>"),
            getattr(answer, "confidence", None),
            (
                f", target {target.box} (confidence={target.confidence})"
                if target is not None
                else ""
            ),
            time.monotonic() - started,
        )
        if answer is None or answer.choice not in criteria:
            raise ValueError("TypeSafe returned an unknown or missing action choice")
        confidence = answer.confidence
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("TypeSafe returned an invalid action confidence")
        if confidence < self.min_confidence or answer.choice == "abstain":
            return None
        if answer.choice in {"complete_success", "complete_failure"}:
            success = answer.choice == "complete_success"
            return Decision(
                success=success,
                message=(
                    "Goal achieved."
                    if success
                    else "Goal cannot be achieved with the available actions."
                ),
            )
        index = indices[answer.choice]
        if request.options[index].point_space is None:
            return Decision(option_index=index)
        if target is None:
            raise ValueError("TypeSafe did not locate the on-screen tap")
        box = target.box
        return Decision(
            option_index=index,
            point=(
                min(1.0, (box.left + box.right) / 2 / coverage[0]),
                min(1.0, (box.top + box.bottom) / 2 / coverage[1]),
            ),
        )
