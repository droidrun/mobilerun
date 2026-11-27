"""One observed device state, with value-safe serialization for evidence sinks."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import JsonValue

from mobilerun.core.credentials.masking import current_masks
from mobilerun.core.state.formatters.indexed_formatter import IndexedFormatter
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.models import Fingerprint
from mobilerun.models.context import ContextItem
from mobilerun.utils.redaction import _active_secrets, mask_secure_ui, mask_step_args
from mobilerun.utils.screen_fingerprint import screen_fingerprint

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device
    from mobilerun.core.state.ui_state import UIState

_UI_FIELDS = (
    "elements",
    "formatted_text",
    "focused_text",
    "phone_state",
    "screen_width",
    "screen_height",
    "use_normalized",
    "coordinate_scale_x",
    "coordinate_scale_y",
    "coordinate_contract_active",
    "model_screenshot_width",
    "model_screenshot_height",
)


@dataclass
class DeviceSnapshot:
    """A UI observation and public extension context; PNG bytes stay out of JSON."""

    at: datetime
    ui: UIState = field(repr=False)
    screenshot: bytes | None = field(default=None, repr=False)
    context: list[ContextItem] = field(default_factory=list, repr=False)

    @classmethod
    async def observe(
        cls, device: Device, *, screenshot: bool = False, redact: bool = True
    ) -> DeviceSnapshot:
        """Refresh live UI and context, optionally taking an unredacted screenshot."""
        from mobilerun.core.runs import recording_scope

        with recording_scope(enabled=False):
            return await cls._from_ui(
                device,
                await device.state.refresh(),
                screenshot=screenshot,
                redact=redact,
            )

    @classmethod
    async def _from_ui(
        cls, device: Device, ui: UIState, *, screenshot: bool, redact: bool
    ) -> DeviceSnapshot:
        at = datetime.now(UTC)
        context = list(await device.context())
        result = cls(at, ui, context=context)
        if screenshot:
            try:
                image = await device.screenshot()
            except DeviceOperationNotSupported:
                pass
            else:
                result.screenshot = image or None
        if not redact:
            return result
        safe_ui = copy(ui)
        masked = result._ui_payload(device, live=True)
        for name, value in masked.items():
            setattr(safe_ui, name, value)
        return cls(at, safe_ui, result.screenshot, result._context_payload(device))

    @staticmethod
    def is_sensitive(device: Device) -> bool:
        masks = current_masks.get()
        return bool(
            device._extension_state.sensitive_ui
            or _active_secrets.get()
            or (masks and masks.active)
        )

    @property
    def fingerprint(self) -> Fingerprint:
        payload: dict[str, JsonValue] = {
            "phone_state": self.ui.phone_state,
            "elements": [*self.ui.elements],
        }
        safe = mask_secure_ui(payload)
        return screen_fingerprint(safe if safe is payload else {})

    def _ui_payload(
        self, device: Device | None, *, live: bool = False
    ) -> dict[str, JsonValue]:
        ui = {name: getattr(self.ui, name) for name in _UI_FIELDS}
        rerender = False
        sensitive = device is not None and self.is_sensitive(device)
        if not live and sensitive:
            # A secret may have passed through a scope that already exited.
            # IDs and app/activity can be supplied dynamically, so omit them too.
            ui.update(
                elements=[], phone_state={}, formatted_text="***", focused_text="***"
            )
        else:
            secure = mask_secure_ui(ui, retain_targeting=live)
            rerender = live and (sensitive or secure is not ui)
            ui = secure
        safe = mask_step_args(
            ui,
            contains_code=(
                device._extension_state.contains_sensitive_text
                if device is not None
                else None
            ),
        )
        if live and (rerender or safe != ui):
            formatter = IndexedFormatter()
            formatter.screen_width = self.ui.screen_width
            formatter.screen_height = self.ui.screen_height
            formatter.use_normalized = self.ui.use_normalized
            elements = safe.get("elements")
            if isinstance(elements, list):
                safe["formatted_text"] = formatter._format_ui_elements_text(
                    [node for node in elements if isinstance(node, dict)]
                )
        return safe

    def _context_payload(self, device: Device | None) -> list[ContextItem]:
        if device is not None and self.is_sensitive(device):
            return []
        contains_code = (
            device._extension_state.contains_sensitive_text
            if device is not None
            else None
        )
        items = []
        for item in self.context:
            safe = mask_step_args(
                {"source": item.source, "kind": item.kind, "data": dict(item.data)},
                contains_code=contains_code,
            )
            if isinstance(safe["data"], dict):
                items.append(
                    ContextItem(str(safe["source"]), str(safe["kind"]), safe["data"])
                )
        return items

    def to_json_data(self, *, device: Device | None = None) -> dict[str, JsonValue]:
        """Serialize redacted UI fields and context; supply the device for taint checks."""
        return {
            "at": self.at.astimezone(UTC).isoformat(),
            "ui": self._ui_payload(device),
            "device_context": [
                {"source": item.source, "kind": item.kind, "data": dict(item.data)}
                for item in self._context_payload(device)
            ],
        }
