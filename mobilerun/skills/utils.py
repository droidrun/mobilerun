"""Helpers for skill metadata and safe directory installation."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import tempfile
import textwrap
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bundle import Skill


def _description(text: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    for index, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            break
        if line.startswith("description:"):
            value = line.partition(":")[2].strip()
            if value in (">", ">-", ">+", "|", "|-", "|+"):
                parts = []
                for part in lines[index + 1 :]:
                    if part and not part[0].isspace():
                        break
                    parts.append(part.strip())
                return ("\n" if value.startswith("|") else " ").join(parts).strip()
            if value.startswith(('"', "'")):
                try:
                    return str(ast.literal_eval(value))
                except (ValueError, SyntaxError):
                    return value.strip("\"'")
            return value
    return ""


def _reference_description(text: str, file: Path, parent: Skill) -> str:
    description = _description(text)
    if description:
        return description
    relative = file.relative_to(parent.path).as_posix()
    destination = r"(?:\./)?" + re.escape(relative) + r"(?:#[^\s)>]*)?"
    pattern = re.compile(
        r"\]\((?:<"
        + destination
        + r">|"
        + destination
        + r")(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\).*?—\s*(.+)"
    )
    for line in (parent.path / "SKILL.md").read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            return re.sub(
                r"^read when:\s*", "Use when ", match.group(1).strip(), flags=re.I
            )
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        closing = next(
            (i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"), None
        )
        if closing is not None:
            lines = lines[closing + 1 :]
    paragraph = []
    for line in lines:
        line = line.strip()
        if not line:
            if paragraph:
                break
            continue
        if line.startswith("#"):
            if paragraph:
                break
            continue
        paragraph.append(line)
    return textwrap.shorten(" ".join(paragraph), width=200, placeholder="…")


_INSTALL_MARKER = ".mobilerun-installed"
DiscardChanges = bool | Callable[[str], bool]


def _tree(
    path: Path, *, legacy: bool = False, unsafe: list[str] | None = None
) -> dict[str, str | None] | None:
    """Fingerprint regular files and directories without following links."""
    entries: dict[str, str | None] = {}
    pending = [path]
    while pending:
        for item in pending.pop().iterdir():
            relative = item.relative_to(path).as_posix()
            if not item.is_symlink() and item.is_dir():
                entries[relative] = None
                pending.append(item)
            elif not item.is_symlink() and item.is_file():
                if not (legacy and item.name == ".DS_Store"):
                    with item.open("rb") as stream:
                        entries[relative] = hashlib.file_digest(
                            stream, "sha256"
                        ).hexdigest()
            elif unsafe is None:
                return None
            else:
                unsafe.append(relative)
    return entries


def _managed(
    path: Path, sources: dict[str, Path], discard_changes: DiscardChanges
) -> bool:
    """Refuse changed installs unless separately confirmed; never claim foreign trees."""
    if path.is_symlink() or not path.is_dir():
        return False
    marker = path / _INSTALL_MARKER
    source = sources.get(path.name)
    if marker.is_symlink():
        return False
    if marker.is_file():
        data = marker.read_bytes()
        if data in (b"mobilerun\n", b"mobilerun\r\n"):
            expected = (
                _tree(source)
                if source and source.is_dir() and not source.is_symlink()
                else None
            )
        else:
            try:
                manifest = json.loads(data)
            except ValueError:
                return False
            if not isinstance(manifest, dict) or not isinstance(
                manifest.get("files"), dict
            ):
                return False
            expected = manifest["files"]
        if discard_changes is True:
            return True
        changes: list[str] = []
        actual = _tree(path, unsafe=changes)
        assert actual is not None
        actual.pop(_INSTALL_MARKER)
        changes.extend(
            name
            for name, digest in actual.items()
            if expected is None or name not in expected or expected[name] != digest
        )
        if not changes:
            return True
        summary = ", ".join(ascii(name)[:120] for name in sorted(changes)[:10])
        if len(changes) > 10:
            summary += f", … ({len(changes) - 10} more)"
        warning = f"{path} contains local changes or unverified files: {summary}"
        if callable(discard_changes) and discard_changes(warning):
            return True
        raise ValueError(
            f"{warning}. Back them up or explicitly allow discarding changes."
        )
    if source is None or not source.is_dir() or source.is_symlink():
        return False
    actual = _tree(path, legacy=True)
    return actual is not None and actual == _tree(source, legacy=True)


def _installed_parent(path: Path, fallback: str | None) -> str | None:
    """Prefer the recorded dependency over a possibly newer release's catalog."""
    marker = path / _INSTALL_MARKER
    if path.is_symlink() or marker.is_symlink() or not marker.is_file():
        return fallback
    try:
        manifest = json.loads(marker.read_bytes())
    except ValueError:
        return fallback
    if isinstance(manifest, dict) and isinstance(manifest.get("files"), dict):
        parent = manifest.get("parent", fallback)
        return parent if isinstance(parent, str) else None
    return fallback


def _install_directory(dest: Path, *, source: Path, parent: str | None = None) -> None:
    temporary = Path(tempfile.mkdtemp(prefix=".mobilerun-install-", dir=dest.parent))
    backup = dest.parent / (".mobilerun-backup-" + uuid.uuid4().hex)
    moved = False
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        files = _tree(temporary)
        if files is None or _INSTALL_MARKER in files:
            raise ValueError(f"Cannot create install manifest for {source}")
        (temporary / _INSTALL_MARKER).write_text(
            json.dumps({"files": files, "parent": parent}, sort_keys=True),
            encoding="utf-8",
        )
        if dest.exists():
            dest.rename(backup)
            moved = True
        try:
            temporary.rename(dest)
        except BaseException:
            if moved:
                backup.rename(dest)
                moved = False
            raise
        if moved:
            shutil.rmtree(backup)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
