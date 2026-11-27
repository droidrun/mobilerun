"""Fingerprint matching, replay argument checks, and CLI logging."""

import logging
from collections.abc import Mapping

from mobilerun.core.credentials.masking import current_masks
from mobilerun.core.state.ui_state import UIState
from mobilerun.macro.settings import get_macro_settings
from mobilerun.models import Fingerprint
from mobilerun.utils.redaction import _redact_secrets
from mobilerun.utils.screen_fingerprint import fingerprint_matches, screen_fingerprint


def capture_fingerprint(
    state: UIState | Mapping[str, object] | None,
) -> Fingerprint | None:
    """Normalize either a state-provider view or a raw device dump."""
    if state is None:
        return None
    if not isinstance(state, Mapping):
        state = {"phone_state": state.phone_state, "elements": state.elements}
    observed = screen_fingerprint(state)
    masks = current_masks.get()
    if masks is not None and masks.active:
        masked = masks.snapshot(
            {
                "phone_state": {"app": observed.app, "activity": observed.activity},
                "nodes": [{"id": key} for key in observed.elements],
            }
        )
        observed = Fingerprint(
            app=masked["phone_state"]["app"],
            activity=masked["phone_state"]["activity"],
            elements=tuple(node["id"] for node in masked["nodes"]),
        )
    if any(
        _redact_secrets(value) != value
        for value in (observed.app, observed.activity, *observed.elements)
        if value is not None
    ):
        return None
    return observed if observed.app or observed.activity or observed.elements else None


def compare_states(
    expected: Fingerprint, observed: Fingerprint | None, threshold: float | None = None
) -> bool:
    """Check a macro guard against normalized, nonempty screen evidence."""
    if threshold is None:
        threshold = get_macro_settings().MOBILERUN_MACRO_STATE_THRESHOLD
    return fingerprint_matches(expected, observed, threshold)


def has_masked_arguments(value: object) -> bool:
    """Detect reserved redaction markers inside nested action arguments."""
    if isinstance(value, str):
        return any(
            marker in value
            for marker in ("***", "<redacted>", "[redacted]", "[secret:")
        )
    if isinstance(value, dict):
        return any(
            has_masked_arguments(key) or has_masked_arguments(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(has_masked_arguments(item) for item in value)
    return False


def configure_macro_logging(debug: bool = False) -> logging.Logger:
    """Configure logging for the macro CLI."""
    logger = logging.getLogger("mobilerun-macro")
    logger.handlers = []
    level = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(levelname)s %(name)s %(message)s" if debug else "%(message)s"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
