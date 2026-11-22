"""Action-level macro recorder for schema v2."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from mobilerun.macro.models import StateInput, build_screen_snapshot


class MacroRecorder:
    def __init__(self) -> None:
        self.actions: List[Dict[str, Any]] = []
        self._last_recorded_at_ms: Optional[int] = None

    def record_action(
        self,
        action: Mapping[str, Any],
        *,
        pre_ui: StateInput = None,
        post_ui: StateInput = None,
    ) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)
        elapsed_ms = (
            0
            if self._last_recorded_at_ms is None
            else max(0, now_ms - self._last_recorded_at_ms)
        )

        snapshot = build_screen_snapshot(pre_ui) if pre_ui is not None else None
        macro_action = {
            **action,
            "pre_state": snapshot,
            "post_state": (
                build_screen_snapshot(post_ui) if post_ui is not None else None
            ),
            "screen": snapshot.get("screen", {}) if snapshot else {},
            "recorded_at_ms": now_ms,
            "elapsed_since_previous_ms": elapsed_ms,
        }

        self.actions.append(macro_action)
        self._last_recorded_at_ms = now_ms
        return macro_action

    def record_wait(
        self, duration: float, *, pre_ui: StateInput = None, post_ui: StateInput = None
    ) -> Dict[str, Any]:
        return self.record_action(
            {"action_type": "wait", "duration": duration},
            pre_ui=pre_ui,
            post_ui=post_ui,
        )
