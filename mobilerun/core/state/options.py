"""Typed construction options shared by device state-provider factories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mobilerun.core.state.filters import TreeFilter

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device
    from mobilerun.core.state.base import StateProvider


class VisionResizePolicy(Protocol):
    def effective_dims(self, width: int, height: int) -> tuple[int, int]: ...


@dataclass(frozen=True)
class StateProviderOptions:
    vision: bool = False
    normalized: bool = False
    stealth: bool = False
    vision_resize_policy: VisionResizePolicy | None = None
    tree_filter: TreeFilter | None = None


StateProviderFactory = Callable[["Device", StateProviderOptions], "StateProvider"]


def android_state_provider(
    device: Device, options: StateProviderOptions
) -> StateProvider:
    from mobilerun.core.state.android import AndroidStateProvider
    from mobilerun.core.state.filters import ConciseFilter, DetailedFilter
    from mobilerun.core.state.formatters import IndexedFormatter

    return AndroidStateProvider(
        device,
        tree_filter=(
            options.tree_filter
            if options.tree_filter is not None
            else (ConciseFilter() if options.vision else DetailedFilter())
        ),
        tree_formatter=IndexedFormatter(),
        vision_enabled=options.vision,
        use_normalized=options.normalized,
        stealth=options.stealth,
        vision_resize_policy=options.vision_resize_policy,
    )


def ios_state_provider(device: Device, options: StateProviderOptions) -> StateProvider:
    from mobilerun.core.state.ios import IOSStateProvider

    return IOSStateProvider(
        device,
        vision_enabled=options.vision,
        use_normalized=options.normalized,
        vision_resize_policy=options.vision_resize_policy,
    )


def screenshot_state_provider(
    device: Device, options: StateProviderOptions
) -> StateProvider:
    from mobilerun.core.state.screenshot import ScreenshotOnlyStateProvider

    return ScreenshotOnlyStateProvider(
        device,
        vision_resize_policy=options.vision_resize_policy,
        use_normalized=options.normalized,
    )
