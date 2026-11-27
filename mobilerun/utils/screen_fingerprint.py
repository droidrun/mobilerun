"""Pure screen normalization and comparison shared by recording and replay."""

from collections.abc import Iterator, Mapping

from mobilerun.models.recording import Fingerprint


def _first_string(data: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and (trimmed := value.strip()):
            return trimmed
    return None


def _walk(tree: object) -> Iterator[Mapping[str, object]]:
    pending = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, Mapping):
            yield node
            pending.append(node.get("children"))
        elif isinstance(node, (list, tuple)):
            pending.extend(node)


def screen_fingerprint(state: Mapping[str, object]) -> Fingerprint:
    """Summarize a UI snapshot or raw tree without storing labels, text or values."""
    phone = state.get("phone_state")
    tree = next(
        (state[key] for key in ("a11y_tree", "elements", "nodes") if key in state),
        state,
    )
    sources = tuple(
        source for source in (phone, state, tree) if isinstance(source, Mapping)
    )
    app = next(
        (
            value
            for source in sources
            if (value := _first_string(source, "packageName", "package"))
        ),
        None,
    )
    activity = next(
        (
            value
            for source in sources
            if (value := _first_string(source, "activityName", "activity"))
        ),
        None,
    )

    keys: set[str] = set()
    for node in _walk(tree):
        resource = _first_string(
            node, "resourceId", "resource_id", "viewIdResourceName", "identifier"
        )
        klass = _first_string(node, "className", "class_name", "class", "type")
        if resource:
            keys.add(f"id:{resource}")
        elif klass:
            keys.add(f"class:{klass.rsplit('.', 1)[-1]}")
    return Fingerprint(app=app, activity=activity, elements=tuple(sorted(keys)))


def fingerprint_matches(
    expected: Fingerprint, observed: Fingerprint | None, threshold: float
) -> bool:
    """Require the same app and activity and enough shared structural keys."""
    if observed is None or not (observed.app or observed.activity or observed.elements):
        return False
    if (expected.app, expected.activity) != (observed.app, observed.activity):
        return False
    left, right = set(expected.elements), set(observed.elements)
    union = left | right
    score = len(left & right) / len(union) if union else 1.0
    return score >= threshold
