"""StealthDriver — human-like interaction wrapper for device drivers.

Wraps any ``DeviceDriver`` to provide:
- Curved Bezier swipe paths with easing and micro-jitter
- Word-by-word typing with random inter-word delays

Non-overridden methods delegate to the inner driver via ``__getattr__``.
"""

from __future__ import annotations

import asyncio
import math
import random
from typing import Any, List, Tuple

from droidrun.tools.driver.base import DeviceDriver

# ---------------------------------------------------------------------------
# Path generation helpers
# ---------------------------------------------------------------------------


def _ease_in_out_cubic(t: float) -> float:
    """Cubic easing for natural acceleration/deceleration."""
    if t < 0.5:
        return 4 * t**3
    return 1 - pow(-2 * t + 2, 3) / 2


def _perlin_noise_1d(x: float, seed: int = 0) -> float:
    """Simple 1-D Perlin-like noise for micro-jitter simulation."""
    random.seed(seed + int(x * 1000))
    freq1 = random.uniform(0.5, 1.5)
    freq2 = random.uniform(2.0, 3.0)
    freq3 = random.uniform(4.0, 6.0)
    return (
        math.sin(x * freq1) * 0.5
        + math.sin(x * freq2) * 0.3
        + math.sin(x * freq3) * 0.2
    )


def generate_curved_path(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    num_points: int = 15,
) -> List[Tuple[int, int]]:
    """Generate a curved path using a quadratic Bezier curve.

    Includes velocity profiling (ease-in/ease-out), micro-jitter via
    Perlin noise, and randomized control points for organic curves.
    """
    distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5

    if distance <= 100:
        num_points = max(5, int(num_points / 3))

    mid_x = (start_x + end_x) / 2
    mid_y = (start_y + end_y) / 2

    curve_intensity = random.uniform(0.1, 0.25)
    max_offset = distance * curve_intensity
    offset = random.uniform(-max_offset, max_offset)

    dx = end_x - start_x
    dy = end_y - start_y

    if distance > 0:
        perp_x = -dy / distance
        perp_y = dx / distance
        control_x = mid_x + perp_x * offset
        control_y = mid_y + perp_y * offset
    else:
        control_x = mid_x
        control_y = mid_y

    noise_seed = random.randint(0, 10000)
    jitter_intensity = min(2.0, distance * 0.01)

    points: List[Tuple[int, int]] = []
    for i in range(num_points):
        linear_t = i / (num_points - 1)
        eased_t = _ease_in_out_cubic(linear_t)

        x = (
            (1 - eased_t) ** 2 * start_x
            + 2 * (1 - eased_t) * eased_t * control_x
            + eased_t**2 * end_x
        )
        y = (
            (1 - eased_t) ** 2 * start_y
            + 2 * (1 - eased_t) * eased_t * control_y
            + eased_t**2 * end_y
        )

        jitter_x = _perlin_noise_1d(linear_t * 10, noise_seed) * jitter_intensity
        jitter_y = _perlin_noise_1d(linear_t * 10, noise_seed + 1000) * jitter_intensity

        points.append((int(x + jitter_x), int(y + jitter_y)))

    return points


# ---------------------------------------------------------------------------
# StealthDriver
# ---------------------------------------------------------------------------


class StealthDriver:
    """Transparent proxy that adds human-like randomness to device I/O.

    Overrides:
    - ``swipe()`` → curved Bezier path via ``input motionevent``
    - ``input_text()`` → word-by-word typing with random delays

    Everything else delegates to the inner driver.
    """

    def __init__(self, inner: DeviceDriver) -> None:
        self.inner = inner

    @property
    def platform(self) -> str:
        return self.inner.platform

    @property
    def supported(self) -> set[str]:
        return self.inner.supported

    @property
    def supported_buttons(self) -> set[str]:
        return self.inner.supported_buttons

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    # -- overridden actions --------------------------------------------------

    async def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: float = 1000,
    ) -> None:
        """Perform a curved swipe using motionevent commands."""
        await self.inner.ensure_connected()
        path_points = generate_curved_path(x1, y1, x2, y2)

        x0, y0 = path_points[0]
        await self.inner.device.shell(f"input motionevent DOWN {x0} {y0}")

        delay = duration_ms / 1000 / len(path_points)
        for x, y in path_points[1:]:
            await asyncio.sleep(delay)
            await self.inner.device.shell(f"input motionevent MOVE {x} {y}")

        x_end, y_end = path_points[-1]
        await self.inner.device.shell(f"input motionevent UP {x_end} {y_end}")

    async def input_text(self, text: str, clear: bool = False) -> bool:
        """Type text word-by-word with random delays between words."""
        words = text.split(" ")

        for i, word in enumerate(words):
            ok = await self.inner.input_text(word, clear=(clear and i == 0))
            if not ok:
                return False

            if i < len(words) - 1:
                space_ok = await self.inner.input_text(" ", clear=False)
                if not space_ok:
                    return False
                await asyncio.sleep(random.uniform(0.1, 0.3))

        return True
