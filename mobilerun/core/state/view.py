"""A small controller for a cached observation, not a second snapshot model."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.ui_state import UIState
from mobilerun.errors import DeviceNotReady, UnsupportedStateCapability

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device


class UIStateView:
    def __init__(
        self,
        device: Device,
        default_provider: Callable[[], StateProvider] | None = None,
    ) -> None:
        self._device = device
        self._default_provider = default_provider
        self._factory: Callable[[Device], StateProvider] | None = None
        self._provider: StateProvider | None = None
        self._current: UIState | None = None

    def _attach(self, provider: StateProvider | None) -> None:
        if callable(provider):
            raise TypeError(
                "Pass a StateProvider instance; use state_provider_factory= for factories"
            )
        self._factory = None
        self._provider = provider
        self._current = None

    def _attach_factory(self, factory: Callable[[Device], StateProvider]) -> None:
        self._factory = factory
        self._provider = None
        self._current = None

    @property
    def _resolved_provider(self) -> StateProvider:
        if self._provider is None:
            if self._factory is not None:
                self._provider = self._factory(self._device)
            elif self._default_provider is not None:
                self._provider = self._default_provider()
            else:
                raise DeviceNotReady("No state provider is configured")
        return self._provider

    def _reset_default(self) -> None:
        self._attach(None)

    @property
    def capabilities(self) -> set[str]:
        return set(self._resolved_provider.capabilities)

    @property
    def current(self) -> UIState:
        if self._current is None:
            raise DeviceNotReady(
                "UI state has not been refreshed; call await device.state.refresh()"
            )
        return self._current

    def _require(self, capability: str) -> None:
        if capability not in self.capabilities:
            raise UnsupportedStateCapability(self._resolved_provider, capability)

    async def refresh(self) -> UIState:
        self._current = await self._resolved_provider.get_state()
        return self._current

    async def _final(self) -> UIState:
        return await self._resolved_provider.get_final_state()
