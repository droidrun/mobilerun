"""Compact, model-facing summary of an app's declared deep links."""

from __future__ import annotations

from typing import Any, Iterable

# How each <data> path attribute reads in a URL template ("*" = any text).
_PATH_FORMATS = (
    ("path", "{}"),
    ("pathPrefix", "{}*"),
    ("pathPattern", "{}"),
    ("pathAdvancedPattern", "{}"),
    ("pathSuffix", "*{}"),
)
_SSP_FORMATS = (
    ("ssp", "{}"),
    ("sspPrefix", "{}*"),
    ("sspPattern", "{}"),
    ("sspAdvancedPattern", "{}"),
    ("sspSuffix", "*{}"),
)
_MAX_TEMPLATES_PER_FILTER = 6
_MAX_LIST_ITEMS = 8


def _unique(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(v) for v in values if v))


def _formatted(data: list[dict], formats) -> list[str]:
    return _unique(
        fmt.format(element[key])
        for element in data
        for key, fmt in formats
        if element.get(key)
    )


def _short_list(values: list[str]) -> str:
    shown = ", ".join(values[:_MAX_LIST_ITEMS])
    extra = len(values) - _MAX_LIST_ITEMS
    return f"{shown} (+{extra} more)" if extra > 0 else shown


def _filter_line(intent_filter: dict) -> str | None:
    """Render one intent filter; Android combines its <data> elements."""
    data = [d for d in intent_filter.get("data") or [] if isinstance(d, dict)]
    types = _unique(d.get("mimeType") for d in data)
    schemes = _unique(d.get("scheme") for d in data)
    if types:
        # A typed filter needs the intent's MIME type, which open_deep_link never
        # sets; Android resolves one only for content:// URIs.
        schemes = [s for s in schemes if s == "content"]
    if not schemes:
        return None
    # Like Android, a port without a host adds no authority.
    hosts = _unique(
        f"{d['host']}:{d['port']}" if d.get("port") else d["host"]
        for d in data
        if d.get("host")
    )
    paths = _formatted(data, _PATH_FORMATS)
    parts = _formatted(data, _SSP_FORMATS)

    if hosts:
        tails = paths or [" (any path)"]
        templates = [f"{s}://{h}{t}" for s in schemes for h in hosts for t in tails]
    else:
        templates = [f"{s}:{p}" for s in schemes for p in (parts or ["*"])]

    if len(templates) <= _MAX_TEMPLATES_PER_FILTER:
        line = ", ".join(templates)
    else:
        line = f"scheme {_short_list(schemes)}"
        if hosts:
            line += f"; host {_short_list(hosts)}"
        if paths:
            line += f"; path {_short_list(paths)}"
        elif parts and not hosts:
            line += f"; {_short_list(parts)}"
    if types:
        line += f" [type {_short_list(types)}]"
    if intent_filter.get("additionalRules"):
        line += " [narrowed by extra rules on Android 15+; not every match opens]"
    return line


def format_deep_links(result: dict, *, max_chars: int = 4000) -> str:
    """Summarize a Portal ``app/deep-links`` result for the model."""
    package = result.get("package", "")
    filters = [f for f in result.get("filters") or [] if isinstance(f, dict)]
    lines = _unique(_filter_line(f) for f in filters if f.get("callable"))

    if not lines:
        summary = f"{package} declares no deep links that can be opened."
    else:
        summary = (
            f"Deep links declared by {package} (* = any text; .* in a pattern = any text). "
            "Fill in real values and open one with open_deep_link:"
        )
        shown: list[str] = []
        size = len(summary)
        for line in lines:
            entry = f"- {line}"
            if size + len(entry) + 1 > max_chars:
                break
            shown.append(entry)
            size += len(entry) + 1
        summary = "\n".join([summary, *shown])
        if len(shown) < len(lines):
            summary += f"\n... +{len(lines) - len(shown)} more"

    if not result.get("declarationsComplete", True):
        warnings = ", ".join(str(w) for w in result.get("warnings") or [])
        summary += (
            f"\n(Declarations may be incomplete: {warnings or 'unknown reason'}.)"
        )
    summary += "\nThe app may also handle links it does not declare."
    return summary
