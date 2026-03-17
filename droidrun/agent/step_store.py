"""StepStore — in-memory history of UI states and screenshots per step.

Stores snapshots from each FastAgent step and exposes them via virtual
file paths (``/steps/{n}/ui_state.txt``, ``/steps/{n}/screenshot.png``).
Two action functions (``read_step_file``, ``grep_steps``) are registered
as agent tools for on-demand lookup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from droidrun.agent.action_result import ActionResult

if TYPE_CHECKING:
    from droidrun.agent.action_context import ActionContext

_PATH_RE = re.compile(r"^/steps/(\d+)/(ui_state\.txt|screenshot\.png)$")


@dataclass
class StepSnapshot:
    ui_state_text: str
    screenshot: Optional[bytes]


class StepStore:
    """In-memory store keyed by step number."""

    def __init__(self) -> None:
        self._steps: dict[int, StepSnapshot] = {}

    def save(self, step: int, ui_state_text: str, screenshot: Optional[bytes]) -> None:
        self._steps[step] = StepSnapshot(
            ui_state_text=ui_state_text, screenshot=screenshot
        )

    def read_file(
        self, path: str, offset: int = 0, limit: int = 0
    ) -> tuple[Optional[str], Optional[bytes], Optional[str]]:
        """Read a virtual file.

        Returns ``(text, image_bytes, error)``.  Exactly one of
        *text* / *image_bytes* is set on success, or *error* on failure.
        """
        m = _PATH_RE.match(path)
        if not m:
            return (
                None,
                None,
                f"Invalid path: {path}. "
                "Expected /steps/{n}/ui_state.txt or /steps/{n}/screenshot.png",
            )
        step = int(m.group(1))
        file_type = m.group(2)
        snap = self._steps.get(step)
        if snap is None:
            return (
                None,
                None,
                f"No data for step {step}. "
                f"Available steps: {sorted(self._steps.keys())}",
            )
        if file_type == "screenshot.png":
            if snap.screenshot is None:
                return None, None, f"No screenshot saved for step {step}"
            return None, snap.screenshot, None
        # ui_state.txt
        lines = snap.ui_state_text.splitlines()
        if offset > 0:
            lines = lines[offset:]
        if limit > 0:
            lines = lines[:limit]
        return "\n".join(lines), None, None

    def search(self, pattern: str, steps: list) -> tuple[str, Optional[str]]:
        """Search UI state text across steps.

        Returns ``(matches_text, error)``.  *error* is set for invalid regex.
        """
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return "", f"Invalid regex pattern: {e}"
        # Normalize step values to int — XML parser may pass strings
        int_steps: list[int] = []
        for s in steps:
            try:
                int_steps.append(int(s))
            except (ValueError, TypeError):
                continue
        matches: list[str] = []
        for step_num in int_steps:
            snap = self._steps.get(step_num)
            if snap is None:
                continue
            for line_num, line in enumerate(snap.ui_state_text.splitlines(), 1):
                if regex.search(line):
                    matches.append(f"/steps/{step_num}/ui_state.txt:{line_num}: {line}")
        if not matches:
            return "No matches found.", None
        return "\n".join(matches), None


# -- Action functions (registered as tools) --------------------------------


async def read_step_file(
    path: str, offset: int = 0, limit: int = 0, *, ctx: "ActionContext"
) -> ActionResult:
    """Read a file from step history."""
    text, image, error = ctx.step_store.read_file(path, offset, limit)
    if error:
        return ActionResult(success=False, summary=error)
    if image is not None:
        return ActionResult(
            success=True, summary=f"Screenshot from {path}", image=image
        )
    return ActionResult(success=True, summary=text or "")


async def grep_steps(
    pattern: str, steps: list, *, ctx: "ActionContext"
) -> ActionResult:
    """Search UI state history with regex."""
    matches, error = ctx.step_store.search(pattern, steps)
    if error:
        return ActionResult(success=False, summary=error)
    return ActionResult(success=True, summary=matches)
