"""Discover and install the release-pinned coding-agent skills."""

import json
import shutil
import tempfile
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

from mobilerun.config.settings import get_settings
from mobilerun.skills.utils import (
    _description,
    _install_directory,
    _managed,
    _reference_description,
)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    source: str = ""
    relative_path: Path = Path(".")
    parent: str | None = None
    source_file: Path | None = None


@dataclass(frozen=True)
class InstallResult:
    name: str
    status: str
    path: Path


def skills_root() -> Path:
    package = Path(__file__).resolve().parents[1]
    bundled = package / "assets" / "skills"
    if bundled.is_dir():
        return bundled
    checkout = package.parent / "skills"
    if checkout.is_dir() and any(checkout.rglob("SKILL.md")):
        return checkout
    raise FileNotFoundError(
        "Skills bundle missing; run `git submodule update --init` in the checkout."
    )


def skills_roots() -> list[Path]:
    return [skills_root()]


def list_skills() -> list[Skill]:
    result = []
    seen = {}
    for root in skills_roots():
        skill_files = sorted(root.rglob("SKILL.md"))
        parents = {}
        guides = sorted(root.rglob("GUIDE.md"))
        references = sorted(
            file
            for file in root.rglob("*.md")
            if file.parent.name == "references"
            and file.name not in {"SKILL.md", "GUIDE.md"}
        )
        for file in skill_files + guides + references:
            if not file.is_file():
                continue
            text = file.read_text(encoding="utf-8")
            is_reference = file in references
            name = file.parent.name if file.name == "SKILL.md" else ""
            parent = None
            parent_skill = None
            if file.name != "SKILL.md":
                parent_skill = next(
                    (
                        parents[p]
                        for p in (file.parent, *file.parent.parents)
                        if p in parents
                    ),
                    None,
                )
                if parent_skill is None:
                    continue
                if is_reference and file.parent != parent_skill.path / "references":
                    continue
                parent = parent_skill.name
                if is_reference:
                    name = f"{parent}-{file.stem}"
            lines = text.splitlines()
            if file.name == "GUIDE.md" and (
                not lines
                or lines[0].strip() != "---"
                or not any(line.strip() == "---" for line in lines[1:])
            ):
                continue
            if lines and lines[0].strip() == "---":
                for line in lines[1:]:
                    if line.strip() == "---":
                        break
                    if line.startswith("name:"):
                        name = line.partition(":")[2].strip().strip("\"'") or name
                        break
            if not name:
                continue
            _names([name])
            if name in {"bundle", "single"}:
                raise ValueError(f"reserved skill name: {name}")
            if name in seen:
                raise ValueError(
                    f"Duplicate skill name: {name} ({seen[name]}, {file.parent})"
                )
            seen[name] = file.parent
            skill = Skill(
                name,
                (
                    _reference_description(text, file, parent_skill)
                    if is_reference and parent_skill
                    else _description(text)
                ),
                file.parent,
                root.name,
                file.parent.relative_to(root),
                parent,
                file,
            )
            if file.name == "SKILL.md":
                parents[file.parent] = skill
            result.append(skill)
    return result


def bundle_folders() -> dict[str, Path]:
    folders = {}
    for root in skills_roots():
        for file in sorted(root.rglob("SKILL.md")):
            relative = file.relative_to(root)
            if len(relative.parts) < 2:
                continue
            name = relative.parts[0]
            folder = root / name
            if name in folders and folders[name] != folder:
                raise ValueError(f"Duplicate bundle folder: {name}")
            folders[name] = folder
    return folders


def unknown_skill_message(name: str, available: list[str]) -> str:
    message = (
        f"Unknown skill: {name}. Run `mobilerun skills list` to see available skills."
    )
    matches = get_close_matches(name, available, n=1)
    if matches:
        message += f" Did you mean '{matches[0]}'?"
    return message


def _target(agent: str, scope: str, directory: Path | None) -> Path:
    if agent not in get_settings().MOBILERUN_AGENTS:
        raise ValueError(f"Unknown agent: {agent}")
    if scope not in ("user", "project"):
        raise ValueError(f"Unknown scope: {scope}")
    if directory is not None:
        return Path(directory).expanduser()
    paths = get_settings().MOBILERUN_AGENTS[agent].skills
    relative = paths[scope == "project"] if paths else None
    if relative is None:
        raise ValueError(f"{agent} does not support skills in {scope} scope")
    return (Path.home() if scope == "user" else Path.cwd()) / relative


def _names(names: list[str]) -> None:
    if any(
        not name or name.startswith(".") or Path(name).name != name or "\\" in name
        for name in names
    ):
        raise ValueError("Invalid skill name")


def installed(
    agent: str, scope: str = "user", directory: Path | None = None
) -> list[str]:
    target = _target(agent, scope, directory)
    return (
        sorted(
            p.name
            for p in target.iterdir()
            if not p.name.startswith(".") and p.is_dir() and (p / "SKILL.md").is_file()
        )
        if target.is_dir()
        else []
    )


def _protect_bundle(target: Path, names: list[str]) -> None:
    for root in skills_roots():
        root = root.resolve()
        for name in names:
            dest = (target / name).resolve()
            if dest == root or dest in root.parents or root in dest.parents:
                raise ValueError("Target overlaps the bundled skills")


def install(
    skill_names: list[str],
    agent: str,
    scope: str = "user",
    dir: Path | None = None,
    force: bool = False,
) -> list[InstallResult]:
    _names(skill_names)
    available = {skill.name: skill for skill in list_skills()}
    missing = set(skill_names) - available.keys()
    if missing:
        raise ValueError(
            "\n".join(
                unknown_skill_message(name, list(available)) for name in sorted(missing)
            )
        )
    target = _target(agent, scope, dir)
    chosen = [available[name] for name in skill_names]
    expanded = []
    added = set()
    for skill in chosen:
        if skill.parent and skill.parent not in added:
            expanded.append(available[skill.parent])
            added.add(skill.parent)
        if skill.name not in added:
            expanded.append(skill)
            added.add(skill.name)
    return _install_skills(
        expanded,
        target,
        force,
        {s.parent for s in chosen if s.parent and s.parent not in skill_names},
    )


def _install_skills(
    skills: list[Skill], target: Path, force: bool, parents: set[str] | None = None
) -> list[InstallResult]:
    parents = parents or set()
    _protect_bundle(target, [skill.name for skill in skills])
    available = {skill.name: skill for skill in list_skills()}
    managed_sources = {name: skill.path for name, skill in available.items()}
    managed_sources.update(bundle_folders())
    for skill in skills:
        dest = target / skill.name
        if (
            force
            and skill.name not in parents
            and (dest.exists() or dest.is_symlink())
            and not _managed(dest, managed_sources)
        ):
            raise ValueError(f"Cannot replace {dest}: not managed")
    target.mkdir(parents=True, exist_ok=True)
    results = []
    for skill in skills:
        dest = target / skill.name
        if dest.exists() or dest.is_symlink():
            if not force or skill.name in parents:
                if skill.name in parents and not (dest / "SKILL.md").is_file():
                    raise ValueError(
                        f"Cannot use {dest}: parent skill is missing SKILL.md"
                    )
                results.append(InstallResult(skill.name, "exists", dest))
                continue
        if skill.parent:
            assert skill.source_file is not None
            text = skill.source_file.read_bytes().decode("utf-8")
            newline = "\r\n" if "\r\n" in text else "\n"
            lines = text.splitlines(keepends=True)
            end = (
                next(
                    (i for i in range(1, len(lines)) if lines[i].strip() == "---"), None
                )
                if lines and lines[0].strip() == "---"
                else None
            )
            had_frontmatter = end is not None
            if end is None:
                lines.insert(
                    0,
                    newline.join(
                        [
                            "---",
                            f"name: {json.dumps(skill.name, ensure_ascii=False)}",
                            f"description: {json.dumps(skill.description, ensure_ascii=False)}",
                            "---",
                            "",
                        ]
                    ),
                )
                end = 0
            elif not lines[end].endswith(("\n", "\r")):
                lines[end] += newline
            if had_frontmatter and skill.source_file.name != "GUIDE.md":
                fields = ("name", "description")
                additions = [
                    f"{field}: {json.dumps(getattr(skill, field), ensure_ascii=False)}{newline}"
                    for field in fields
                    if not any(line.startswith(f"{field}:") for line in lines[1:end])
                ]
                lines[end:end] = additions
                end += len(additions)
            relative = skill.path.relative_to(available[skill.parent].path).as_posix()
            original_dir = f"../{skill.parent}/" + (
                f"{relative}/" if relative != "." else ""
            )
            pointer = (
                f"> Part of the `{skill.parent}` skill. Resolve relative links in this "
                f"file against `{original_dir}`; skill-root paths such as `local/`, "
                f"`apps/` and `AGENTS.md` against `../{skill.parent}/`."
            )
            lines.insert(end + 1, pointer + newline * 2)
            with tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary)
                (source / "SKILL.md").write_bytes("".join(lines).encode("utf-8"))
                _install_directory(dest, source=source)
        else:
            _install_directory(dest, source=skill.path)
        results.append(InstallResult(skill.name, "installed", dest))
    return results


def install_bundles(
    names: list[str],
    agent: str,
    scope: str = "user",
    dir: Path | None = None,
    force: bool = False,
) -> list[InstallResult]:
    _names(names)
    available = bundle_folders()
    missing = set(names) - available.keys()
    if missing:
        raise ValueError("Unknown bundle folder: " + ", ".join(sorted(missing)))
    skills = list_skills()
    target = _target(agent, scope, dir)
    results = []
    for name in names:
        folder = available[name]
        _protect_bundle(target, [name])
        target.mkdir(parents=True, exist_ok=True)
        dest = target / name
        if dest.exists() or dest.is_symlink():
            if not force:
                results.append(InstallResult(name, "exists", dest))
                continue
            sources = {s.name: s.path for s in skills}
            sources.update(available)
            if not _managed(dest, sources):
                raise ValueError(f"Cannot replace {dest}: not managed")
        _install_directory(dest, source=folder)
        results.append(InstallResult(name, "installed", dest))
    return results


def uninstall(
    skill_names: list[str],
    agent: str,
    scope: str = "user",
    dir: Path | None = None,
) -> list[InstallResult]:
    _names(skill_names)
    target = _target(agent, scope, dir)
    _protect_bundle(target, skill_names)
    results = []
    managed_sources = {s.name: s.path for s in list_skills()}
    managed_sources.update(bundle_folders())
    for name in skill_names:
        dest = target / name
        if not dest.exists() and not dest.is_symlink():
            results.append(InstallResult(name, "not installed", dest))
        elif not _managed(dest, managed_sources):
            results.append(InstallResult(name, "not managed", dest))
        else:
            shutil.rmtree(dest)
            results.append(InstallResult(name, "removed", dest))
    return results
