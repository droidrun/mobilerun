"""Build and read versioned macro JSONL files."""

import os
import re
from collections.abc import Collection
from contextvars import ContextVar
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from mobilerun.macro.errors import MacroError
from mobilerun.macro.models import MacroStep
from mobilerun.macro.settings import get_macro_settings
from mobilerun.macro.utils import has_masked_arguments
from mobilerun.models import RunStep

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_VERSION = re.compile(r"v([1-9][0-9]*)\.jsonl")
_UNSUPPORTED = (
    "Unsupported macro schema. Build a macro from a current run's steps.jsonl."
)
_Row = TypeVar("_Row", bound=BaseModel)
_replay_snapshot: ContextVar[tuple[Path, list[MacroStep]] | None] = ContextVar(
    "macro_replay_snapshot", default=None
)


def _named_folder(name: str) -> Path:
    if not _NAME.fullmatch(name):
        raise MacroError(
            "Invalid macro name; use letters, digits, dots, underscores or hyphens"
        )
    return Path(get_macro_settings().macro_directory).expanduser().absolute() / name


def _versions(folder: Path, *, files_only: bool) -> dict[int, Path]:
    return {
        int(match[1]): path
        for path in folder.iterdir()
        if (match := _VERSION.fullmatch(path.name))
        and (not files_only or path.is_file())
    }


def resolve_macro_path(target: str | Path) -> Path:
    """Resolve a file or the latest numeric version of a macro name or folder."""
    path = Path(target).expanduser()
    if isinstance(target, str) and _NAME.fullmatch(target) and not path.exists():
        path = _named_folder(target)
    if path.is_dir():
        try:
            versions = _versions(path, files_only=True)
        except OSError:
            raise MacroError("Could not read macro directory") from None
        if versions:
            return versions[max(versions)].absolute()
        if (path / "macro.json").is_file():
            raise MacroError(_UNSUPPORTED)
        raise MacroError("No macro versions found in the directory")
    if not path.is_file():
        raise MacroError("Macro file or directory not found")
    return path.absolute()


def _read_rows(path: Path, model: type[_Row]) -> list[_Row]:
    rows = []
    try:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                try:
                    rows.append(model.model_validate_json(line))
                except ValidationError:
                    raise MacroError(
                        f"{_UNSUPPORTED} Invalid JSONL row {line_number}."
                    ) from None
    except (OSError, UnicodeError):
        raise MacroError("Could not read macro evidence") from None
    return rows


def build_macro(
    steps_path: Path,
    name: str,
    *,
    step_ids: Collection[int] | None = None,
    after_version: int = 0,
) -> Path:
    """Publish the successful, replayable run rows as a new immutable version."""
    folder = _named_folder(name)
    if type(after_version) is not int or after_version < 0:
        raise MacroError("after_version must be a nonnegative integer")
    rows = [
        MacroStep(
            i=row.i,
            action=row.action,
            args=row.args,
            before=row.before,
            after=row.after,
            note=row.note,
            target=row.target,
        )
        for row in _read_rows(steps_path, RunStep)
        if row.ok
        and (step_ids is None or row.i in step_ids)
        and not has_masked_arguments(row.args)
    ]
    if not rows:
        raise MacroError("No replayable steps to build a macro")

    temporary: Path | None = None
    try:
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        if folder.is_symlink():
            raise MacroError("Macro name cannot refer to a symbolic link")
        with NamedTemporaryFile(
            dir=folder, prefix=".build-", delete=False, mode="w", encoding="utf-8"
        ) as file:
            temporary = Path(file.name)
            for row in rows:
                file.write(row.model_dump_json(exclude_defaults=True) + "\n")
        version = max(
            _versions(folder, files_only=False),
            default=0,
        )
        version = max(version, after_version) + 1
        while True:
            target = folder / f"v{version}.jsonl"
            try:
                os.link(temporary, target)
            except FileExistsError:
                version += 1
            else:
                return target
    except (OSError, UnicodeError):
        raise MacroError("Could not publish macro version") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_macro(target: str | Path) -> list[MacroStep]:
    """Load a pinned file or the latest version of a macro name or folder."""
    snapshot = _replay_snapshot.get()
    if snapshot is not None and Path(target).expanduser().absolute() == snapshot[0]:
        return snapshot[1]
    rows = _read_rows(resolve_macro_path(target), MacroStep)
    if not rows:
        raise MacroError("Macro contains no steps")
    return rows
