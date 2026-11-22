"""Helpers for skill metadata and safe directory installation."""

from __future__ import annotations

import ast
import re
import shutil
import tempfile
import textwrap
import uuid
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


def _tree(path: Path) -> dict[Path, bytes | None] | None:
    entries: dict[Path, bytes | None] = {}
    pending = [path]
    while pending:
        for item in pending.pop().iterdir():
            if item.name == ".DS_Store":
                continue
            relative = item.relative_to(path)
            if item.is_symlink():
                return None
            if item.is_dir():
                entries[relative] = None
                pending.append(item)
            elif item.is_file():
                entries[relative] = item.read_bytes()
            else:
                return None
    return entries


def _managed(path: Path, sources: dict[str, Path]) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    marker = path / _INSTALL_MARKER
    if marker.is_file() and not marker.is_symlink():
        return True
    source = sources.get(path.name)
    if source is None or not source.is_dir() or source.is_symlink():
        return False
    actual = _tree(path)
    return actual is not None and actual == _tree(source)


def _install_directory(dest: Path, *, source: Path) -> None:
    temporary = Path(tempfile.mkdtemp(prefix=".mobilerun-install-", dir=dest.parent))
    backup = dest.parent / (".mobilerun-backup-" + uuid.uuid4().hex)
    moved = False
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        (temporary / _INSTALL_MARKER).write_text("mobilerun\n", encoding="utf-8")
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
