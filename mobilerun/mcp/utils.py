"""Read, write and edit agent configuration files."""

import json
import os
import re
import stat
import tempfile
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomlkit


@dataclass
class _Token:
    value: str
    start: int
    end: int


def _tokens(text: str) -> list[_Token]:
    result = []
    i = 1 if text.startswith("\ufeff") else 0
    while i < len(text):
        if text[i].isspace():
            i += 1
        elif text.startswith("//", i):
            end = text.find("\n", i + 2)
            i = len(text) if end < 0 else end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise ValueError("unterminated comment")
            i = end + 2
        elif text[i] == '"':
            start = i
            i += 1
            while i < len(text):
                if text[i] == "\\":
                    i += 2
                elif text[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
            result.append(_Token(text[start:i], start, i))
        elif text[i] in "{}[]:,":
            result.append(_Token(text[i], i, i + 1))
            i += 1
        else:
            start = i
            while i < len(text) and not text[i].isspace() and text[i] not in "{}[]:,/":
                i += 1
            if i == start:
                raise ValueError("unexpected character")
            result.append(_Token(text[start:i], start, i))
    return result


def _loads(text: str) -> Any:
    ts = _tokens(text)
    parts = []
    for i, token in enumerate(ts):
        if token.value == "," and i + 1 < len(ts) and ts[i + 1].value in ("}", "]"):
            if i == 0 or ts[i - 1].value in ("{", "[", ",", ":"):
                raise ValueError("trailing comma without a value")
            continue
        parts.append(token.value)
    return json.loads(" ".join(parts))


def _members(
    ts: list[_Token], index: int
) -> tuple[list[tuple[str, int, int, int | None]], int]:
    """Return (key, key token, last value token, comma token) and closing token."""
    if ts[index].value != "{":
        raise ValueError("expected object")
    result = []
    i = index + 1
    while ts[i].value != "}":
        key = json.loads(ts[i].value)
        key_index = i
        i += 2
        if ts[i].value in ("{", "["):
            depth = 0
            while True:
                if ts[i].value in ("{", "["):
                    depth += 1
                elif ts[i].value in ("}", "]"):
                    depth -= 1
                i += 1
                if not depth:
                    break
        else:
            i += 1
        comma = i if ts[i].value == "," else None
        result.append((key, key_index, i - 1, comma))
        if comma is not None:
            i += 1
    return result, i


def check_path(text: str, path: list[str]) -> None:
    """Reject ambiguous members along the path to the server entry."""
    ts = _tokens(text)
    index = 0
    for key in path:
        items, _ = _members(ts, index)
        matches = [item for item in items if item[0] == key]
        if len(matches) > 1:
            raise ValueError(f"Duplicate JSONC key: {key}")
        if not matches:
            return
        index = matches[0][1] + 2


def edit(text: str, path: list[str], value: Any = None, *, remove: bool = False) -> str:
    """Edit a member, creating missing parent objects when installing."""
    ts = _tokens(text)
    index = 0
    for depth, key in enumerate(path):
        items, close = _members(ts, index)
        found = next((m for m in reversed(items) if m[0] == key), None)
        if found and depth < len(path) - 1:
            index = found[1] + 2
            continue
        if found:
            _, key_i, end_i, comma = found
            if not remove:
                start = ts[key_i + 2].start
                indent = re.match(
                    r"[ \t]*", text[text.rfind("\n", 0, ts[key_i].start) + 1 :]
                ).group()
                newline = "\r\n" if "\r\n" in text else "\n"
                replacement = json.dumps(value, indent=2, ensure_ascii=False).replace(
                    "\n", newline + indent
                )
                return text[:start] + replacement + text[ts[end_i].end :]
            start, end = ts[key_i].start, ts[end_i].end
            if comma is not None:
                end = ts[comma].end
            elif len(items) > 1:
                previous = items[items.index(found) - 1]
                comma_pos = ts[previous[3]].start
                line_start = text.rfind("\n", 0, start) + 1
                line_end = text.find("\n", end)
                if (
                    text[line_start:start].strip() == ""
                    and line_end >= 0
                    and text[end:line_end].strip() == ""
                ):
                    return (
                        text[:comma_pos]
                        + text[comma_pos + 1 : line_start]
                        + text[line_end + 1 :]
                    )
                return text[:comma_pos] + text[comma_pos + 1 : start] + text[end:]
            # Remove the whole generated line when no adjacent comments would be lost.
            line_start = text.rfind("\n", 0, ts[key_i].start) + 1
            line_end = text.find("\n", end)
            if (
                start == ts[key_i].start
                and text[line_start:start].strip() == ""
                and line_end >= 0
                and text[end:line_end].strip() == ""
            ):
                start, end = line_start, line_end + 1
            return text[:start] + text[end:]
        if remove:
            return text
        for parent in reversed(path[depth + 1 :]):
            value = {parent: value}
        close_token = ts[close]
        line_start = text.rfind("\n", 0, close_token.start) + 1
        close_indent = text[line_start : close_token.start]
        multiline = (
            "\n" in text[ts[index].end : close_token.start] and not close_indent.strip()
        )
        if multiline:
            if items:
                member_line = text.rfind("\n", 0, ts[items[0][1]].start) + 1
                indent = text[member_line : ts[items[0][1]].start]
                if indent.strip():
                    indent = close_indent + "  "
            else:
                indent = close_indent + "  "
            rendered = json.dumps(value, indent=2, ensure_ascii=False).replace(
                "\n", "\n" + indent
            )
            trailing = bool(items and items[-1][3] is not None)
            newline = "\r\n" if "\r\n" in text else "\n"
            rendered = rendered.replace("\n", newline)
            insertion = (
                indent
                + json.dumps(key)
                + ": "
                + rendered
                + ("," if trailing else "")
                + newline
            )
            if items and not trailing:
                pos = ts[items[-1][2]].end
                text = text[:pos] + "," + text[pos:]
                line_start += 1
            return text[:line_start] + insertion + text[line_start:]
        rendered = json.dumps(value, ensure_ascii=False)
        prefix = "" if not items else (" " if items[-1][3] is not None else ", ")
        return (
            text[: close_token.start]
            + prefix
            + json.dumps(key)
            + ": "
            + rendered
            + text[close_token.start :]
        )
    raise ValueError("empty path")


def read_file(path: Path) -> MutableMapping[str, Any]:
    if not path.exists():
        return tomlkit.document() if path.suffix == ".toml" else {}
    try:
        text = path.read_text(encoding="utf-8")
        data = (
            tomlkit.parse(text)
            if path.suffix == ".toml"
            else _loads(text) if path.suffix == ".jsonc" else json.loads(text)
        )
        if not isinstance(data, MutableMapping):
            raise ValueError("expected an object")
        return data
    except (ValueError, tomlkit.exceptions.TOMLKitError) as exc:
        raise ValueError(f"Invalid config {path}: {exc}") from exc


def write_file(
    path: Path, data: MutableMapping[str, Any], content: str | None = None
) -> None:
    content = (
        content
        if content is not None
        else (
            tomlkit.dumps(data)
            if path.suffix == ".toml"
            else json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        )
    )
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
            os.chmod(name, mode)
            file.write(content)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def jsonc_source(path: Path) -> str:
    if not path.exists():
        return "{}"
    with path.open(encoding="utf-8", newline="") as file:
        return file.read()
