"""RecordingDriver — transparent proxy that logs user-visible actions.

Wraps any ``DeviceDriver`` and appends dicts to ``self.log`` for every
mutating action.  Read-only methods (screenshot, get_ui_tree, …) pass
through automatically via ``__getattr__``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from droidrun.tools.driver.base import DeviceDriver


class RecordingDriver:
    """Proxy that records device-level actions for macro replay.

    Dict key is ``"action_type"`` — consistent with ``replay.py`` and
    ``trajectory.py`` expectations.
    """

    def __init__(self, inner: DeviceDriver) -> None:
        self.inner = inner
        self.log: List[Dict[str, Any]] = []

    @property
    def platform(self) -> str:
        return self.inner.platform

    @property
    def supported(self) -> set[str]:
        return self.inner.supported

    @property
    def supported_buttons(self) -> set[str]:
        return self.inner.supported_buttons

    def __getattr__(self, name: str):
        """Delegate all non-overridden attribute lookups to the inner driver."""
        return getattr(self.inner, name)

    # -- recorded actions ----------------------------------------------------

    async def tap(self, x: int, y: int) -> None:
        await self.inner.tap(x, y)
        self.log.append({"action_type": "tap", "x": x, "y": y})

    async def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: float = 1000,
    ) -> None:
        await self.inner.swipe(x1, y1, x2, y2, duration_ms)
        self.log.append(
            {
                "action_type": "swipe",
                "start_x": x1,
                "start_y": y1,
                "end_x": x2,
                "end_y": y2,
                "duration_ms": duration_ms,
            }
        )

    async def input_text(self, text: str, clear: bool = False) -> bool:
        result = await self.inner.input_text(text, clear)
        self.log.append({"action_type": "input_text", "text": text, "clear": clear})
        return result

    async def press_button(self, button: str) -> None:
        await self.inner.press_button(button)
        self.log.append({"action_type": "button_press", "button": button})

    async def start_app(self, package: str, activity: Optional[str] = None) -> str:
        result = await self.inner.start_app(package, activity)
        self.log.append(
            {
                "action_type": "start_app",
                "package": package,
                "activity": activity,
            }
        )
        return result

    async def drag(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: float = 3.0,
    ) -> None:
        await self.inner.drag(x1, y1, x2, y2, duration)
        self.log.append(
            {
                "action_type": "drag",
                "start_x": x1,
                "start_y": y1,
                "end_x": x2,
                "end_y": y2,
                "duration": duration,
            }
        )
