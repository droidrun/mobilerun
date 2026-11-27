"""Optional structural observations around a recorded device action."""

from __future__ import annotations

import json
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from mobilerun.core.state.screenshot import ScreenshotOnlyStateProvider
from mobilerun.core.state.snapshot import DeviceSnapshot
from mobilerun.models import Fingerprint
from mobilerun.utils.redaction import _redact_secrets

from .writer import recording_scope

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device
    from mobilerun.core.state.ui_state import UIState


class FingerprintCapture:
    """Reuse the latest observation until an action or explicit refresh invalidates it."""

    def __init__(
        self, device: Device, inner: Device, full: bool, settle_seconds: float
    ) -> None:
        self._device = device
        self._inner = inner
        self._full = full
        self._settle_seconds = settle_seconds
        self._invalidated = False
        self._last_current: UIState | None = None
        self._snapshot: DeviceSnapshot | None = None
        self._saved: (
            tuple[DeviceSnapshot, bool, dict[Literal["json", "png"], str]] | None
        ) = None

    def invalidate(self) -> None:
        self._invalidated = True

    async def capture(self, *, fresh: bool = False) -> Fingerprint | None:
        return self.fingerprint(await self._sample(fresh=fresh))

    async def sample(self, *, fresh: bool = False) -> DeviceSnapshot | None:
        return await self._sample(fresh=fresh) if self._full else None

    async def _sample(self, *, fresh: bool) -> DeviceSnapshot | None:
        with recording_scope(enabled=False):
            return await self._sample_unrecorded(fresh=fresh)

    async def _sample_unrecorded(self, *, fresh: bool) -> DeviceSnapshot | None:
        current = self._device.state._current or self._inner.state._current
        if (
            not fresh
            and not self._invalidated
            and current is self._last_current
            and self._snapshot is not None
        ):
            return self._snapshot
        try:
            self._snapshot = None
            if fresh or self._invalidated or current is None:
                provider = self._device.state._resolved_provider
                if isinstance(provider, ScreenshotOnlyStateProvider) and not self._full:
                    return None
                state = await provider.get_final_state()
            else:
                state = current
            self._invalidated = False
            self._last_current = current
            self._snapshot = (
                await DeviceSnapshot._from_ui(
                    self._device, state, screenshot=True, redact=False
                )
                if self._full
                else DeviceSnapshot(datetime.now(UTC), state)
            )
            return self._snapshot
        except Exception:
            self.invalidate()
            return None

    async def after(self, *, cancelled: bool) -> DeviceSnapshot | None:
        self.invalidate()
        if cancelled or not self._full:
            return None
        if self._settle_seconds:
            with suppress(Exception), recording_scope(enabled=False):
                await self._inner.wait(self._settle_seconds)
        return await self.sample(fresh=True)

    def fingerprint(self, snapshot: DeviceSnapshot | None) -> Fingerprint | None:
        if snapshot is None or DeviceSnapshot.is_sensitive(self._device):
            return None
        fingerprint = snapshot.fingerprint
        contains_code = self._device._extension_state.contains_sensitive_text
        if any(
            _redact_secrets(value) != value or contains_code(value)
            for value in (fingerprint.app, fingerprint.activity, *fingerprint.elements)
            if value is not None
        ):
            return None
        return (
            fingerprint
            if fingerprint.app or fingerprint.activity or fingerprint.elements
            else None
        )

    def save(
        self,
        run_dir: Path,
        index: int,
        phase: Literal["before", "after"],
        snapshot: DeviceSnapshot | None,
    ) -> dict[Literal["json", "png"], str] | None:
        """Persist redacted evidence, referring to the previous after files when safe."""
        if snapshot is None:
            return None
        sensitive = DeviceSnapshot.is_sensitive(self._device)
        if (
            self._saved is not None
            and self._saved[0] is snapshot
            and self._saved[1] == sensitive
        ):
            return self._saved[2]
        files: dict[Literal["json", "png"], str] = {}
        stem = f"screens/{index}.{phase}"
        with suppress(OSError, ValueError, TypeError):
            (run_dir / "screens").mkdir(mode=0o700, exist_ok=True)
            (run_dir / f"{stem}.json").write_text(
                json.dumps(snapshot.to_json_data(device=self._device), allow_nan=False),
                encoding="utf-8",
            )
            files["json"] = f"{stem}.json"
            if snapshot.screenshot is not None:
                (run_dir / f"{stem}.png").write_bytes(snapshot.screenshot)
                files["png"] = f"{stem}.png"
        self._saved = (snapshot, sensitive, files) if files else None
        return files or None
