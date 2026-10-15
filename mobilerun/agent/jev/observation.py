"""Screen model Jev decides on, built from Mobilerun UI state."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from mobilerun.tools.formatters.indexed_formatter import IndexedFormatter
from mobilerun.tools.ui.state import UIState

LABEL_MAX_CHARS = 160
VISIBLE_TEXT_MAX_ITEMS = 200
VISIBLE_TEXT_MAX_CHARS = 300
PASSWORD_LABEL = "[password field]"

_ANDROID_EDITABLE_SUFFIXES = ("EditText", "AutoCompleteTextView")
_ANDROID_KEYBOARD_PACKAGES = frozenset(
    {"com.samsung.android.honeyboard", "com.touchtype.swiftkey"}
)
# iOS reports XCUI type names (no package dots); the WDA fork prefixes them.
_XCUI_PREFIX = "XCUIElementType"
_IOS_EDITABLE = frozenset({"TextField", "SearchField", "SecureTextField", "TextView"})
_IOS_KEYBOARD = frozenset({"Key", "Keyboard"})
# Keyboard keys that submit the focused field.
RETURN_KEY_LABELS = frozenset(
    {"return", "search", "go", "done", "send", "next", "join", "route", "enter"}
)
_IOS_CLICKABLE = frozenset(
    {
        "Button",
        "Cell",
        "Link",
        "Switch",
        "Icon",
        "Tab",
        "MenuItem",
        "SegmentedControl",
    }
)

Bounds = Tuple[int, int, int, int]


@dataclass
class JevElement:
    """One on-screen element; ``index`` is the UIState index used by ``click``."""

    index: int
    label: str
    clickable: bool
    editable: bool
    scrollable: bool
    enabled: bool
    focused: bool
    password: bool
    checkable: bool
    checked: bool
    selected: bool
    value: str
    bounds: Bounds
    keyboard: bool
    hint: str = ""
    resource_id: str = ""
    class_name: str = ""
    # Pre-order descendants follow at index+1..index+descendants (0 if unknown).
    descendants: int = 0
    # App element whose tap point lies under the on-screen keyboard.
    covered: bool = False


@dataclass
class JevScreen:
    platform: str
    package: str
    app_name: str
    width: int
    height: int
    elements: List[JevElement]
    focused: Optional[JevElement]
    keyboard_visible: bool
    visible_text: List[str]
    fingerprint: str
    # On-screen keyboard key that submits input, for backends without an Enter key.
    return_key: Optional[JevElement] = None


class JevIndexedFormatter(IndexedFormatter):
    """``IndexedFormatter`` that also stores Jev's flags under ``element["jev"]``.

    Only the namespaced key is added, so macro snapshots normalize exactly like
    plain ``IndexedFormatter`` output and guarded replay keeps matching.
    """

    def __init__(self) -> None:
        super().__init__()
        self._keyboard_until = 0

    def format(
        self, filtered_tree: Optional[Dict[str, Any]], phone_state: Dict[str, Any]
    ) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, Any]]:
        self._keyboard_until = 0
        return super().format(filtered_tree, phone_state)

    def _format_node(self, node: Dict[str, Any], index: int) -> Dict[str, Any]:
        formatted = super()._format_node(node, index)
        descendants, label = _subtree_summary(node)
        keyboard = index <= self._keyboard_until or _is_keyboard_node(node)
        if keyboard:
            # Pre-order indexing: the subtree occupies the next `descendants` indices.
            self._keyboard_until = max(self._keyboard_until, index + descendants)

        class_name = node.get("className") or ""
        ios_class = _ios_class(class_name)
        password = _flag(node, "isPassword", "password") or (
            ios_class == "SecureTextField"
        )
        raw_text = _clean(node.get("text"))
        # iOS --local (WDA) reads an empty field back as its placeholder, which
        # is the field's own label: that text is a hint, not a value.
        showing_hint = _flag(node, "isShowingHintText") or (
            ios_class in _IOS_EDITABLE
            and bool(raw_text)
            and raw_text == _clean(node.get("contentDescription"))
        )
        hint = _clean(node.get("hint")) or (raw_text if showing_hint else "")
        formatted["jev"] = {
            "clickable": _flag(node, "isClickable", "clickable")
            or _flag(node, "isLongClickable", "longClickable"),
            "editable": _flag(node, "isEditable", "editable")
            or class_name.endswith(_ANDROID_EDITABLE_SUFFIXES)
            or ios_class in _IOS_EDITABLE,
            "scrollable": _flag(node, "isScrollable", "scrollable"),
            "enabled": _first(node, "isEnabled", "enabled") not in (False, "false"),
            "focused": _flag(node, "isFocused", "focused"),
            "password": password,
            "checkable": _flag(node, "isCheckable", "checkable"),
            "checked": _flag(node, "isChecked", "checked"),
            "selected": _flag(node, "isSelected", "selected"),
            "text": "" if password or showing_hint else raw_text,
            "description": _clean(node.get("contentDescription")),
            "hint": hint,
            "label": label,
            "keyboard": keyboard,
            "descendants": descendants,
        }
        return formatted


def build_screen(
    ui_state: UIState,
    *,
    platform: str,
    tapped_input: Optional[JevElement] = None,
) -> JevScreen:
    """Build Jev's view of a UI state from Android, iOS ``--local`` or legacy iOS.

    ``tapped_input`` is the text field Jev tapped last; with the keyboard up it
    identifies the focused field on backends that report no input focus.
    """
    phone = ui_state.phone_state if isinstance(ui_state.phone_state, dict) else {}
    width = _int(ui_state.screen_width)
    height = _int(ui_state.screen_height)

    parsed_elements: List[Tuple[JevElement, List[str]]] = []
    tap_points: Dict[int, Tuple[int, int]] = {}
    jev_formatted = False
    for raw in _walk(ui_state.elements or []):
        jev_formatted = jev_formatted or isinstance(raw.get("jev"), dict)
        parsed = _parse_element(raw, width, height)
        if parsed is not None:
            parsed_elements.append(parsed)
            point = _tap_point(raw.get("bounds"))
            if point is not None:
                tap_points[parsed[0].index] = point
    elements = [element for element, _ in parsed_elements]
    if not jev_formatted:
        _mark_field_icons(elements)
    ios = platform == "ios"
    hidden = _mark_covered(elements, tap_points) if ios else set()

    texts: Dict[str, None] = {}
    for element, own_texts in parsed_elements:
        if element.keyboard or element.index in hidden:
            continue
        for text in own_texts:
            if len(texts) >= VISIBLE_TEXT_MAX_ITEMS:
                break
            texts.setdefault(text[:VISIBLE_TEXT_MAX_CHARS], None)

    package = str(phone.get("packageName") or "")
    # iOS isEditable can be a guess; legacy iOS focus arrives via focusedElement.
    input_reported = bool(phone.get("keyboardVisible")) or (
        not ios and bool(phone.get("isEditable"))
    )
    keyboard_visible = input_reported or any(e.keyboard for e in elements)
    # iOS --local names a guessed field there, not the focused one.
    focus_hint = None if ios and jev_formatted else phone.get("focusedElement")
    focused = _focused_element(
        elements,
        input_reported=input_reported,
        focus_hint=focus_hint,
        tapped_input=tapped_input if keyboard_visible else None,
        strict_hint=ios,
    )
    return JevScreen(
        platform=platform,
        package=package,
        app_name=str(phone.get("currentApp") or ""),
        width=width,
        height=height,
        elements=elements,
        focused=focused,
        keyboard_visible=keyboard_visible,
        visible_text=list(texts),
        fingerprint=_fingerprint(package, phone, elements, focused),
        return_key=_return_key(elements) if keyboard_visible else None,
    )


def _mark_field_icons(elements: List[JevElement]) -> None:
    # Legacy iOS: a search magnifier, or an icon repeating the field's own name,
    # is decoration inside the field, not a control.
    fields = [e for e in elements if e.editable]
    for element in elements:
        if element.class_name == "Image" and any(
            _contains(field.bounds, element.bounds)
            and (
                field.class_name == "SearchField"
                or element.label in (field.label, field.hint)
            )
            for field in fields
        ):
            element.clickable = False


def _mark_covered(
    elements: List[JevElement], tap_points: Dict[int, Tuple[int, int]]
) -> Set[int]:
    """Mark app elements the keyboard covers; return those fully inside it."""
    keyboards = [e.bounds for e in elements if _ios_class(e.class_name) == "Keyboard"]
    hidden: Set[int] = set()
    if not keyboards:
        return hidden
    for element in elements:
        if element.keyboard:
            continue
        inside = any(_contains(outer, element.bounds) for outer in keyboards)
        point = tap_points.get(element.index)
        element.covered = inside or (
            point is not None and any(_holds(outer, point) for outer in keyboards)
        )
        if inside:
            hidden.add(element.index)
    return hidden


def _return_key(elements: List[JevElement]) -> Optional[JevElement]:
    keys = [
        e
        for e in elements
        if e.keyboard
        and not e.covered
        and e.enabled
        and _class_key(e.class_name) != "Keyboard"
        and e.label.strip().lower() in RETURN_KEY_LABELS
    ]
    if len(keys) > 1:
        # iOS names its own keys in lowercase ("search", "return").
        keys = [
            e
            for e in keys
            if e.label == e.label.lower() or _class_key(e.class_name) == "Key"
        ]
    return keys[0] if len(keys) == 1 else None


def _contains(outer: Bounds, inner: Bounds) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _holds(outer: Bounds, point: Tuple[int, int]) -> bool:
    return outer[0] <= point[0] < outer[2] and outer[1] <= point[1] < outer[3]


def _tap_point(value: Any) -> Optional[Tuple[int, int]]:
    # The bounds centre, as UIState.get_element_coords taps it.
    bounds = _parse_bounds(value, 0, 0)
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    return (left + right) // 2, (top + bottom) // 2


def _focused_element(
    elements: List[JevElement],
    *,
    input_reported: bool,
    focus_hint: Any = None,
    tapped_input: Optional[JevElement] = None,
    strict_hint: bool = False,
) -> Optional[JevElement]:
    # Never guess between several inputs; a field the keyboard covers can be focused.
    inputs = [e for e in elements if e.editable and e.enabled and not e.keyboard]
    focused = [e for e in inputs if e.focused]
    if focused:
        return focused[0] if len(focused) == 1 else None
    hinted = _match_focus_hint(inputs, focus_hint) or _focused_text_view(
        elements, inputs, focus_hint
    )
    if hinted is not None:
        return hinted
    if tapped_input is not None:
        tapped = [
            e
            for e in inputs
            if e.class_name == tapped_input.class_name
            and e.bounds == tapped_input.bounds
        ]
        if len(tapped) == 1:
            return tapped[0]
    if input_reported and len(inputs) == 1:
        # On iOS, a hint naming another kind of field rules the single input out.
        if (
            not strict_hint
            or not isinstance(focus_hint, dict)
            or _class_key(str(focus_hint.get("className") or ""))
            == _class_key(inputs[0].class_name)
        ):
            return inputs[0]
    return None


def _focused_text_view(
    elements: List[JevElement], inputs: List[JevElement], hint: Any
) -> Optional[JevElement]:
    """Promote the text view legacy iOS reports focused although it looked static."""
    if not isinstance(hint, dict):
        return None
    if _ios_class(str(hint.get("className") or "")) != "TextView":
        return None
    resource_id = str(hint.get("resourceId") or "")

    def matches(e: JevElement) -> bool:
        return _ios_class(e.class_name) == "TextView" and (
            not resource_id or e.resource_id == resource_id
        )

    if any(matches(e) for e in inputs):
        return None
    views = [
        e
        for e in elements
        if matches(e) and not e.editable and e.enabled and not e.keyboard
    ]
    if len(views) != 1:
        return None
    views[0].editable = True
    return views[0]


def _match_focus_hint(inputs: List[JevElement], hint: Any) -> Optional[JevElement]:
    """The one input matching phone_state.focusedElement, if exactly one does."""
    if not isinstance(hint, dict):
        return None
    class_key = _class_key(str(hint.get("className") or ""))
    resource_id = str(hint.get("resourceId") or "")
    text = _clean(hint.get("text"))
    if class_key:
        inputs = [e for e in inputs if _class_key(e.class_name) == class_key]
    if resource_id:
        matches = [e for e in inputs if e.resource_id == resource_id]
    elif text:
        matches = [e for e in inputs if e.value == text]
    else:
        matches = inputs if class_key else []
    return matches[0] if len(matches) == 1 else None


def _fingerprint(
    package: str,
    phone: Dict[str, Any],
    elements: List[JevElement],
    focused: Optional[JevElement] = None,
) -> str:
    rows = sorted(
        (e.label, e.value, e.checked, e.selected, e.focused, e.enabled, list(e.bounds))
        for e in elements
    )
    # Raw IME flags: a BACK that only hides the keyboard still changes the screen.
    flags = [bool(phone.get("keyboardVisible")), bool(phone.get("isEditable"))]
    # Resolved focus: moving to another field changes the screen even when no
    # element reports focus.
    focus = [list(focused.bounds), focused.class_name] if focused else None
    payload = json.dumps(
        [package, flags, focus, rows], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()


def _parse_element(
    raw: Dict[str, Any], width: int, height: int
) -> Optional[Tuple[JevElement, List[str]]]:
    index = raw.get("index")
    if not isinstance(index, int) or isinstance(index, bool):
        return None
    bounds = _parse_bounds(raw.get("bounds"), width, height)
    if bounds is None:
        return None
    info = raw.get("jev")
    if isinstance(info, dict):
        return _from_jev(raw, info, index, bounds)
    return _from_type(raw, index, bounds)


def _from_jev(
    raw: Dict[str, Any], info: Dict[str, Any], index: int, bounds: Bounds
) -> Tuple[JevElement, List[str]]:
    password = bool(info.get("password"))
    text = "" if password else _clean(info.get("text"))
    description = _clean(info.get("description"))
    element = JevElement(
        index=index,
        label=PASSWORD_LABEL if password else _clean(info.get("label")),
        clickable=bool(info.get("clickable")),
        editable=bool(info.get("editable")),
        scrollable=bool(info.get("scrollable")),
        enabled=info.get("enabled") is not False,
        focused=bool(info.get("focused")),
        password=password,
        checkable=bool(info.get("checkable")),
        checked=bool(info.get("checked")),
        selected=bool(info.get("selected")),
        value=text,
        bounds=bounds,
        keyboard=bool(info.get("keyboard")),
        hint=_clean(info.get("hint")),
        resource_id=str(raw.get("resourceId") or ""),
        class_name=str(raw.get("className") or ""),
        descendants=_int(info.get("descendants")),
    )
    return element, [t for t in (text, description) if t]


def _from_type(
    raw: Dict[str, Any], index: int, bounds: Bounds
) -> Tuple[JevElement, List[str]]:
    # Legacy iOS portal elements: flags are derived from the XCUI type.
    kind = str(raw.get("type") or raw.get("className") or "").removeprefix(_XCUI_PREFIX)
    password = kind == "SecureTextField"
    checkable = kind == "Switch"
    raw_value = _clean(raw.get("value")).strip("'\"")
    name = _clean(raw.get("label")) or _clean(raw.get("identifier"))
    hint = _clean(raw.get("placeholder"))
    # Read-only text views carry their text as the label, with no value or hint;
    # they stay tappable since an empty labeled input looks the same.
    static_text_view = (
        kind == "TextView"
        and bool(_clean(raw.get("label")))
        and not (raw_value or hint)
    )
    editable = kind in _IOS_EDITABLE and not static_text_view
    label = name if editable else name or hint
    value = "" if password or checkable else raw_value
    element = JevElement(
        index=index,
        label=PASSWORD_LABEL if password else label,
        clickable=kind in _IOS_CLICKABLE
        or (kind == "Image" and bool(label))
        or kind in _IOS_KEYBOARD
        or static_text_view,
        editable=editable,
        scrollable=False,
        enabled=True,
        focused=False,
        password=password,
        checkable=checkable,
        checked=checkable and raw_value.lower() in {"1", "true", "on"},
        selected=False,
        value=value,
        bounds=bounds,
        # The provider marks elements nested under the Keyboard line.
        keyboard=kind in _IOS_KEYBOARD or raw.get("in_keyboard") is True,
        hint=hint,
        resource_id=_clean(raw.get("identifier")),
        class_name=kind,
    )
    own = [] if password else [label, value]
    return element, [t for t in dict.fromkeys(own) if t]


def _subtree_summary(node: Dict[str, Any]) -> Tuple[int, str]:
    """Count a node's descendants and build its descendant-aware label."""
    parts: Dict[str, None] = {}
    length = 0
    descendants = -1
    root_keyboard = _is_keyboard_node(node)
    stack: List[Tuple[Dict[str, Any], bool]] = [(node, False)]
    while stack:
        current, in_keyboard = stack.pop()
        descendants += 1
        in_keyboard = in_keyboard or (
            current is not node and _is_keyboard_node(current)
        )
        collect = (
            length <= LABEL_MAX_CHARS
            and (root_keyboard or not in_keyboard)
            and _first(current, "isVisibleToUser") not in (False, "false")
        )
        if collect:
            password = _flag(current, "isPassword", "password")
            for key in (
                ("contentDescription",) if password else ("text", "contentDescription")
            ):
                value = _clean(current.get(key))
                if value and value not in parts:
                    parts[value] = None
                    length += len(value) + 3
        children = current.get("children") or []
        # Reverse so pre-order is kept when popping.
        stack.extend(
            (child, in_keyboard)
            for child in reversed(children)
            if isinstance(child, dict)
        )
    label = " / ".join(parts)
    if len(label) > LABEL_MAX_CHARS:
        label = label[: LABEL_MAX_CHARS - 1].rstrip() + "…"
    return descendants, label


def _is_keyboard_node(node: Dict[str, Any]) -> bool:
    if _ios_class(node.get("className") or "") in _IOS_KEYBOARD:
        return True
    package = node.get("packageName") or ""
    return (
        package.endswith("inputmethod")
        or ".inputmethod." in package
        or package in _ANDROID_KEYBOARD_PACKAGES
    )


def _ios_class(class_name: str) -> str:
    """Short XCUI type name, or "" for an Android (dotted) class."""
    if not class_name or "." in class_name:
        return ""
    return class_name.removeprefix(_XCUI_PREFIX)


def _class_key(class_name: str) -> str:
    # Formatted elements keep only the last dotted segment.
    return class_name.rsplit(".", 1)[-1].removeprefix(_XCUI_PREFIX)


def _walk(elements: List[Any]) -> Iterator[Dict[str, Any]]:
    for element in elements:
        if not isinstance(element, dict):
            continue
        yield element
        yield from _walk(element.get("children") or [])


def _parse_bounds(value: Any, width: int, height: int) -> Optional[Bounds]:
    if not isinstance(value, str):
        return None
    try:
        left, top, right, bottom = (int(float(part)) for part in value.split(","))
    except ValueError:
        return None
    if width > 0 and height > 0:
        left, top = max(0, left), max(0, top)
        right, bottom = min(width, right), min(height, bottom)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _first(node: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in node:
            return node[key]
    return None


def _flag(node: Dict[str, Any], *keys: str) -> bool:
    return any(node.get(key) in (True, "true") for key in keys)


def _clean(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    return " ".join(str(value).split())


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
