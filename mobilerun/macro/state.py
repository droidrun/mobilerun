"""State snapshots for guarded macro replay."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

MACRO_SCHEMA_VERSION = "2.0"
UNSUPPORTED_SCHEMA_MESSAGE = (
    "Unsupported macro schema. Re-record this macro with the current Mobilerun version."
)


def normalize_ui_state(ui_state: Any) -> Dict[str, Any]:
    """Normalize a UI state into the stable fields used by macro replay."""
    elements = _extract_elements(ui_state)
    phone_state = _extract_phone_state(ui_state)
    screen = _extract_screen(ui_state)

    nodes = []
    for element in _walk_elements(elements):
        node = _normalize_element(element)
        if node:
            nodes.append(node)

    return {
        "phone_state": {
            "package": _first_present(
                phone_state, "package", "appPackage", "bundle_id"
            ),
            "activity": _first_present(phone_state, "activity", "appActivity"),
        },
        "screen": screen,
        "nodes": nodes,
    }


def build_screen_snapshot(ui_state: Any) -> Dict[str, Any]:
    """Return the serializable screen/action state snapshot for a macro action."""
    return normalize_ui_state(ui_state)


def node_semantic_key(node: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        node.get("resource_id"),
        node.get("class"),
        node.get("text"),
        node.get("content_description"),
        node.get("clickable"),
        node.get("enabled"),
        node.get("focused"),
    )


def _extract_elements(ui_state: Any) -> List[Dict[str, Any]]:
    if ui_state is None:
        return []
    if isinstance(ui_state, list):
        return ui_state
    if isinstance(ui_state, dict):
        value = ui_state.get("elements")
        if value is None:
            value = ui_state.get("a11y_tree")
        if value is None:
            value = ui_state.get("nodes")
        return value or []
    return getattr(ui_state, "elements", []) or []


def _extract_phone_state(ui_state: Any) -> Dict[str, Any]:
    if ui_state is None:
        return {}
    if isinstance(ui_state, dict):
        return ui_state.get("phone_state") or ui_state.get("device_context") or {}
    return getattr(ui_state, "phone_state", {}) or {}


def _extract_screen(ui_state: Any) -> Dict[str, Any]:
    if ui_state is None:
        return {}
    if isinstance(ui_state, dict):
        width = ui_state.get("screen_width") or ui_state.get("width")
        height = ui_state.get("screen_height") or ui_state.get("height")
        device_context = ui_state.get("device_context") or {}
        screen_bounds = device_context.get("screen_bounds") or {}
        width = width or screen_bounds.get("width")
        height = height or screen_bounds.get("height")
    else:
        # Taps are recorded in device units, so record the device size.
        width = getattr(ui_state, "device_width", None) or getattr(
            ui_state, "screen_width", None
        )
        height = getattr(ui_state, "device_height", None) or getattr(
            ui_state, "screen_height", None
        )

    return {"width": width, "height": height}


def _walk_elements(elements: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for element in elements:
        if not isinstance(element, dict):
            continue
        yield element
        children = element.get("children") or []
        yield from _walk_elements(children)


def _normalize_element(element: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    bounds = _parse_bounds(
        _first_present(element, "bounds", "boundsInScreen", "bounds_in_screen")
    )
    node = {
        "resource_id": _first_present(
            element,
            "resourceId",
            "resource_id",
            "viewIdResourceName",
            "id",
        ),
        "class": _first_present(element, "className", "class_name", "class", "type"),
        "text": _normalize_text(_first_present(element, "text", "label", "value")),
        "content_description": _normalize_text(
            _first_present(
                element,
                "contentDescription",
                "content_desc",
                "content_description",
                "description",
            )
        ),
        "clickable": _normalize_bool(
            _first_present(element, "clickable", "isClickable")
        ),
        "enabled": _normalize_bool(_first_present(element, "enabled", "isEnabled")),
        "focused": _normalize_bool(_first_present(element, "focused", "isFocused")),
        "bounds": bounds,
    }

    if not any(value is not None and value != "" for value in node.values()):
        return None
    return node


def _first_present(data: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    return " ".join(str(value).split())


def _normalize_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def _parse_bounds(value: Any) -> Optional[Dict[str, int]]:
    raw = _bounds_tuple(value)
    if raw is None:
        return None

    left, top, right, bottom = raw
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "center_x": (left + right) // 2,
        "center_y": (top + bottom) // 2,
    }


def _bounds_tuple(value: Any) -> Optional[Tuple[int, int, int, int]]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.replace("[", "").replace("]", "").replace(" ", ",")
        parts = [part for part in cleaned.split(",") if part]
        if len(parts) == 4:
            try:
                return tuple(int(float(part)) for part in parts)  # type: ignore[return-value]
            except ValueError:
                return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return tuple(int(float(part)) for part in value)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        if all(key in value for key in ("left", "top", "right", "bottom")):
            try:
                return (
                    int(float(value["left"])),
                    int(float(value["top"])),
                    int(float(value["right"])),
                    int(float(value["bottom"])),
                )
            except (TypeError, ValueError):
                return None
    return None
