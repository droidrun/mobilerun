"""Humanized input over the explicit framework device contract."""

import asyncio
import random
from typing import TypeVar

from mobilerun.core.devices.base import Device
from mobilerun.core.devices.components.wrapper import DeviceWrapper

D = TypeVar("D", bound=Device)


class StealthDevice(DeviceWrapper[D]):
    async def input_text(
        self, text: str, clear: bool = False, stealth: bool = False, wpm: int = 0
    ) -> bool:
        if "{{" in text or "}}" in text:
            return await self._inner.input_text(text, clear, stealth, wpm)
        words = text.split(" ")
        for i, word in enumerate(words):
            if not await self._inner.input_text(word, clear and i == 0, stealth, wpm):
                return False
            if i < len(words) - 1:
                if not await self._inner.input_text(" ", False, stealth, wpm):
                    return False
                await asyncio.sleep(random.uniform(0.1, 0.3))
        return True

    async def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
    ) -> None:
        await self._inner._swipe_humanized(x1, y1, x2, y2, duration_ms)
