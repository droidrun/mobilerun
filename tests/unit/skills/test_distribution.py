"""Skill distribution: release archives, catalog discovery and safe installation."""

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.skills import bundle
from tests.support.paths import REPO_ROOT


def test_skills_wheel_and_sdist_include_full_tree_and_installed_cli(tmp_path):
    pytest.importorskip("hatchling")
    source = REPO_ROOT / "skills"
    assert (source / "mobile-harness/SKILL.md").is_file()
    files = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for name in subprocess.check_output(["git", "ls-files", "-z"], cwd=source)
        .decode()
        .split("\0")
        if name
        for path in [source / name]
        if path.is_file()
        and not any(
            part.startswith(".") or part in {"tests", "__pycache__"}
            for part in path.relative_to(source).parts
        )
    }
    assert "apps/com.ebay.mobile/scrape_newest_listings.md" in files
    dist = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hatchling",
            "build",
            "-t",
            "wheel",
            "-t",
            "sdist",
            "-d",
            str(dist),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(dist.glob("*.whl"))
    with ZipFile(wheel) as archive:
        actual = {
            name.removeprefix("mobilerun/assets/skills/"): archive.read(name)
            for name in archive.namelist()
            if name.startswith("mobilerun/assets/skills/")
        }
        assert actual == files
        assert "mobilerun-framework/SKILL.md" in actual
    with tarfile.open(next(dist.glob("*.tar.gz"))) as archive:
        assert any(
            member.isfile() and member.name.partition("/")[2] == "SKILLS-RECORD"
            for member in archive.getmembers()
        )
        actual = {}
        for member in archive.getmembers():
            if member.isfile() and "/skills/" in member.name:
                prefix, relative = member.name.split("/skills/", 1)
                if "/" not in prefix:
                    actual[relative] = archive.extractfile(member).read()
        assert actual == files

    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = venv / "bin/python"
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
        check=True,
        capture_output=True,
        text=True,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = sysconfig.get_paths()["purelib"]
    command = "import sys,sysconfig; sys.path.insert(0,sysconfig.get_paths()['purelib']); from mobilerun.cli.app import cli; cli()"

    def run(*args):
        return subprocess.run(
            [str(python), "-c", command, *args],
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    entries = json.loads(run("skills", "list", "--json").stdout)
    assert {entry["source"] for entry in entries} == {"skills"}
    assert {entry["name"] for entry in entries} >= {"mobile-harness", "mobilerun-cloud"}
    target = tmp_path / "target"
    run(
        "skills",
        "install",
        "single",
        "mobilerun-cloud",
        "--agent",
        "codex",
        "--dir",
        str(target),
    )
    assert (target / "mobilerun-cloud/SKILL.md").is_file()


def test_untracked_skill_files_stay_out_of_release_archives(tmp_path):
    pytest.importorskip("hatchling")
    for name in ("pyproject.toml", "hatch_build.py", "README.md"):
        shutil.copy(REPO_ROOT / name, tmp_path)
    (tmp_path / "mobilerun").mkdir()
    (tmp_path / "mobilerun/__init__.py").touch()
    skills = tmp_path / "skills"
    example = skills / "example"
    example.mkdir(parents=True)
    (example / "SKILL.md").write_text("Example skill")
    subprocess.run(["git", "init", "-q"], cwd=skills, check=True)
    subprocess.run(["git", "add", "example/SKILL.md"], cwd=skills, check=True)
    for name in (".env", "private-notes.md"):
        (example / name).write_text("local only")

    dist = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-d", str(dist)],
        cwd=tmp_path,
        check=True,
    )
    with (
        ZipFile(next(dist.glob("*.whl"))) as wheel,
        tarfile.open(next(dist.glob("*.tar.gz"))) as sdist,
    ):
        for names in (wheel.namelist(), sdist.getnames()):
            assert [n.split("/skills/", 1)[1] for n in names if "/skills/" in n] == [
                "example/SKILL.md"
            ]
        sdist.extractall(tmp_path / "unpacked", filter="data")
    unpacked = next((tmp_path / "unpacked").iterdir())
    (unpacked / "skills/example/private-notes.md").touch()
    subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel"],
        cwd=unpacked,
        check=True,
    )
    with ZipFile(next((unpacked / "dist").glob("*.whl"))) as wheel:
        assert (
            "mobilerun/assets/skills/example/private-notes.md" not in wheel.namelist()
        )


@pytest.mark.parametrize(
    "missing_entry", [False, True], ids=["complete", "missing-file"]
)
def test_wheel_builds_from_skills_record_without_git_metadata(
    tmp_path: Path, missing_entry: bool
) -> None:
    """A Docker-style source tree bundles only the recorded skills files."""
    pytest.importorskip("hatchling")
    for name in ("pyproject.toml", "hatch_build.py", "README.md"):
        shutil.copy(REPO_ROOT / name, tmp_path)
    (tmp_path / "mobilerun").mkdir()
    (tmp_path / "mobilerun/__init__.py").touch()
    example = tmp_path / "skills/example"
    (example / "references").mkdir(parents=True)
    (example / "SKILL.md").write_text("Example skill")
    (example / "references/guide.md").write_text("Example guide")
    (example / "private-notes.md").write_text("local only")
    record = b"example/SKILL.md\0example/references/guide.md\0"
    if missing_entry:
        record += b"example/removed.md\0"
    (tmp_path / "SKILLS-RECORD").write_bytes(record)

    result = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    if missing_entry:
        assert result.returncode != 0
        assert "SKILLS-RECORD lists a missing file: example/removed.md" in result.stderr
        return

    assert result.returncode == 0, result.stderr
    prefix = "mobilerun/assets/skills/"
    with ZipFile(next((tmp_path / "dist").glob("*.whl"))) as wheel:
        assert {
            name.removeprefix(prefix): wheel.read(name)
            for name in wheel.namelist()
            if name.startswith(prefix)
        } == {
            "example/SKILL.md": b"Example skill",
            "example/references/guide.md": b"Example guide",
        }


def test_skills_install_and_uninstall(tmp_path, monkeypatch):
    source = tmp_path / "source"
    skill = source / "example"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        '---\nname: example\ndescription: "A skill"\n---\nBody\n'
    )
    monkeypatch.setattr(bundle, "skills_root", lambda: source)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    assert bundle.list_skills()[0].description == "A skill"
    assert bundle.install(["example"], "claude-code")[0].status == "installed"
    assert bundle.install(["example"], "claude-code")[0].status == "exists"
    assert (
        bundle.install(["example"], "claude-code", force=True)[0].status == "installed"
    )
    assert bundle.installed("claude-code") == ["example"]
    assert bundle.uninstall(["example"], "claude-code")[0].status == "removed"
    bundle.install(["example"], "opencode", scope="project")
    assert (tmp_path / ".opencode/skills/example/SKILL.md").exists()
    bundle.install(["example"], "codex")
    assert (tmp_path / "home/.agents/skills/example/SKILL.md").exists()
    bundle.install(["example"], "codex", scope="project")
    assert (tmp_path / ".agents/skills/example/SKILL.md").exists()
    target = tmp_path / "custom"
    result = CliRunner().invoke(
        cli, ["skills", "install", "example", "--agent", "codex", "--dir", str(target)]
    )
    assert result.exit_code == 0, result.output
    assert (target / "example/SKILL.md").exists()
    with pytest.raises(ValueError):
        bundle.install(["../bad"], "codex")


def test_resolver_prefers_installed_and_reports_missing(tmp_path, monkeypatch):
    package = tmp_path / "mobilerun"
    fake_module = package / "skills" / "bundle.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.touch()
    monkeypatch.setattr(bundle, "__file__", str(fake_module))
    checkout = tmp_path / "skills" / "one"
    checkout.mkdir(parents=True)
    (checkout / "SKILL.md").write_text(
        "---\ndescription: >\n  First line\n  second line\n---\n"
    )
    assert bundle.skills_root() == checkout.parent
    assert bundle.list_skills()[0].description == "First line second line"
    bundled = package / "assets" / "skills"
    bundled.mkdir(parents=True)
    assert bundle.skills_root() == bundled
    bundled.rmdir()
    (checkout / "SKILL.md").unlink()
    with pytest.raises(FileNotFoundError, match="git submodule update --init"):
        bundle.skills_root()


def test_force_cannot_delete_bundle(tmp_path, monkeypatch):
    source = tmp_path / "source"
    skill = source / "example"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\ndescription: hello\n---\n")
    monkeypatch.setattr(bundle, "skills_root", lambda: source)
    with pytest.raises(ValueError):
        bundle.install(["example"], "codex", dir=source, force=True)
    assert (skill / "SKILL.md").exists()


def test_nested_skills_use_frontmatter_name_and_detect_duplicates(
    tmp_path, monkeypatch
):
    root = tmp_path / "source"
    path = root / "apps" / "phone" / "card"
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text("---\nname: phone-card\ndescription: A card\n---\n")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    skills = bundle.list_skills()
    assert [(skill.name, skill.path) for skill in skills] == [("phone-card", path)]
    other = root / "core" / "phone"
    other.mkdir(parents=True)
    (other / "SKILL.md").write_text("---\nname: phone-card\n---\n")
    with pytest.raises(ValueError, match="Duplicate skill name"):
        bundle.list_skills()


def test_bundle_install_and_nested_installed(tmp_path, monkeypatch):
    root = tmp_path / "source"
    path = root / "apps" / "phone" / "card"
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text("---\nname: phone-card\n---\n")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    target = tmp_path / "target"
    bundle.install_bundles(["apps"], "codex", dir=target)
    assert (target / "apps/phone/card/SKILL.md").exists()
    assert bundle.installed("codex", directory=target) == []
    bundle.uninstall(["apps"], "codex", dir=target)
    bundle.install(["phone-card"], "codex", dir=target)
    assert (target / "phone-card/SKILL.md").exists()


@pytest.mark.parametrize("name", ["bundle", "single"])
def test_discovery_rejects_reserved_skill_name(tmp_path, monkeypatch, name):
    root = tmp_path / "skills"
    path = root / name
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(f"---\nname: {name}\n---\n")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    with pytest.raises(ValueError, match="reserved"):
        bundle.list_skills()


def test_uninstall_and_force_preserve_unmanaged_paths(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    (root / "known").mkdir(parents=True)
    (root / "known/SKILL.md").write_text("---\nname: known\n---\n")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    target = tmp_path / "target"
    (target / "my-notes").mkdir(parents=True)
    (target / "my-notes/data").write_text("keep")
    assert (
        bundle.uninstall(["my-notes"], "codex", dir=target)[0].status == "not managed"
    )
    assert (target / "my-notes/data").read_text() == "keep"
    for name in (".mobilerun-harness", ".private"):
        with pytest.raises(ValueError, match="Invalid"):
            bundle.uninstall([name], "codex", dir=target)
    (target / "known").mkdir()
    (target / "known/data").write_text("keep")
    with pytest.raises(ValueError, match="not managed"):
        bundle.install(["known"], "codex", dir=target, force=True)
    assert (target / "known/data").read_text() == "keep"


def test_force_rejects_plain_files_and_bundle_foreign_directories(
    tmp_path, monkeypatch
):
    root = tmp_path / "skills"
    (root / "known").mkdir(parents=True)
    (root / "known/SKILL.md").write_text("---\nname: known\n---\n")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    target = tmp_path / "target"
    target.mkdir()
    (target / "known").write_text("user file")
    with pytest.raises(ValueError, match="not managed"):
        bundle.install(["known"], "codex", dir=target, force=True)
    assert (target / "known").read_text() == "user file"
    assert bundle.uninstall(["known"], "codex", dir=target)[0].status == "not managed"
    (target / "known").unlink()
    (target / "known").mkdir()
    (target / "known/user.txt").write_text("user")
    with pytest.raises(ValueError, match="not managed"):
        bundle.install_bundles(["known"], "codex", dir=target, force=True)
    assert (target / "known/user.txt").read_text() == "user"


def test_same_name_foreign_skill_preserved_and_legacy_copies_managed(
    tmp_path, monkeypatch
):
    root = tmp_path / "skills"
    (root / "classic").mkdir(parents=True)
    content = "---\nname: classic\n---\nsource\n"
    (root / "classic/SKILL.md").write_text(content)
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    target = tmp_path / "target"
    (target / "classic").mkdir(parents=True)
    (target / "classic/SKILL.md").write_text("---\nname: classic\n---\nprivate\n")
    assert bundle.uninstall(["classic"], "codex", dir=target)[0].status == "not managed"
    with pytest.raises(ValueError, match="not managed"):
        bundle.install(["classic"], "codex", dir=target, force=True)
    with pytest.raises(ValueError, match="not managed"):
        bundle.install_bundles(["classic"], "codex", dir=target, force=True)
    assert "private" in (target / "classic/SKILL.md").read_text()
    (target / "classic/SKILL.md").write_text(content)
    assert (
        bundle.install(["classic"], "codex", dir=target, force=True)[0].status
        == "installed"
    )
    assert (target / "classic/.mobilerun-installed").is_file()
    (target / "classic/SKILL.md").write_text("modified managed content")
    assert (
        bundle.uninstall(["classic"], "codex", dir=target, discard_changes=True)[
            0
        ].status
        == "removed"
    )
    (target / "classic").mkdir()
    (target / "classic/SKILL.md").write_text(content)
    assert bundle.uninstall(["classic"], "codex", dir=target)[0].status == "removed"


@pytest.mark.parametrize("mode", ["single", "bundle"])
def test_copy_failure_keeps_destination_retryable(tmp_path, monkeypatch, mode):
    root = tmp_path / "skills"
    source = root / "classic"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("---\nname: classic\n---\nbody")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    target = tmp_path / "target"
    copytree = bundle.shutil.copytree

    def fail(src, dst, **kwargs):
        if src == source:
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "partial").write_text("partial")
            raise OSError("copy failed")
        return copytree(src, dst, **kwargs)

    def install(force=False):
        fn = bundle.install if mode == "single" else bundle.install_bundles
        return fn(["classic"], "codex", dir=target, force=force)

    monkeypatch.setattr(bundle.shutil, "copytree", fail)
    with pytest.raises(OSError, match="copy failed"):
        install()
    assert not (target / "classic").exists()
    monkeypatch.setattr(bundle.shutil, "copytree", copytree)
    assert install()[0].status == "installed"
    old = (target / "classic/SKILL.md").read_text()
    monkeypatch.setattr(bundle.shutil, "copytree", fail)
    with pytest.raises(OSError, match="copy failed"):
        install(force=True)
    assert (target / "classic/SKILL.md").read_text() == old
    monkeypatch.setattr(bundle.shutil, "copytree", copytree)
    assert install(force=True)[0].status == "installed"
    assert not list(target.glob(".mobilerun-install-*"))


def test_force_swap_failure_restores_old_install(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    (root / "classic").mkdir(parents=True)
    (root / "classic/SKILL.md").write_text("---\nname: classic\n---\nsource")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    target = tmp_path / "target"
    bundle.install(["classic"], "codex", dir=target)
    (target / "classic/SKILL.md").write_text("existing managed skill")
    rename = Path.rename

    def fail(self, destination):
        if self.name.startswith(".mobilerun-install-"):
            raise OSError("swap failed")
        return rename(self, destination)

    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(OSError, match="swap failed"):
        bundle.install(
            ["classic"], "codex", dir=target, force=True, discard_changes=True
        )
    assert (target / "classic/SKILL.md").read_text() == "existing managed skill"
    assert not list(target.glob(".mobilerun-install-*"))
    assert not list(target.glob(".mobilerun-backup-*"))
    monkeypatch.setattr(Path, "rename", rename)
    assert (
        bundle.install(
            ["classic"], "codex", dir=target, force=True, discard_changes=True
        )[0].status
        == "installed"
    )


def test_single_source_nested_discovery_and_bundle_copy(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    nested = root / "apps/example/card"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text("---\nname: example-card\n---\n")
    (root / "apps/example/guide.md").write_text("guide")
    other = tmp_path / "harness-skills/core/foreign"
    other.mkdir(parents=True)
    (other / "SKILL.md").write_text("---\nname: foreign\n---\n")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    assert [s.name for s in bundle.list_skills()] == ["example-card"]
    target = tmp_path / "target"
    bundle.install_bundles(["apps"], "codex", dir=target)
    assert (target / "apps/example/card/SKILL.md").is_file()
    assert (target / "apps/example/guide.md").is_file()
    assert (target / "apps/.mobilerun-installed").is_file()
    bundle.install(["example-card"], "codex", dir=target)
    assert (target / "example-card/SKILL.md").is_file()
    assert not (target / ".mobilerun-harness").exists()


def test_real_skill_source_has_unique_unreserved_names():
    from mobilerun.skills.bundle import list_skills, skills_root

    skills = list_skills()
    assert skills_root().name == "skills"
    assert skills
    assert len({skill.name for skill in skills}) == len(skills)


@pytest.mark.parametrize("mode", ["single", "bundle"])
def test_unmarked_legacy_tree_requires_exact_files(tmp_path, monkeypatch, mode):
    root = tmp_path / "skills"
    source = root / "classic"
    (source / "references").mkdir(parents=True)
    (source / "SKILL.md").write_text("---\nname: classic\n---\n")
    (source / "references" / "guide.md").write_bytes(b"original\x00content")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    target = tmp_path / "target"
    target.mkdir()
    dest = target / "classic"
    shutil.copytree(source, dest)
    (dest / ".DS_Store").write_text("ignored")
    assert bundle.uninstall(["classic"], "codex", dir=target)[0].status == "removed"

    shutil.copytree(source, dest)
    user_file = dest / "references" / "notes.txt"
    user_file.write_text("keep my notes")
    assert bundle.uninstall(["classic"], "codex", dir=target)[0].status == "not managed"
    installer = bundle.install if mode == "single" else bundle.install_bundles
    with pytest.raises(ValueError, match="not managed"):
        installer(["classic"], "codex", dir=target, force=True)
    assert user_file.read_text() == "keep my notes"


def test_unmarked_legacy_changed_file_is_not_managed(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    source = root / "classic"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("---\nname: classic\n---\n")
    (source / "guide").write_bytes(b"original")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    target = tmp_path / "target"
    target.mkdir()
    dest = target / "classic"
    shutil.copytree(source, dest)
    (dest / "guide").write_bytes(b"modified")
    assert bundle.uninstall(["classic"], "codex", dir=target)[0].status == "not managed"
    assert (dest / "guide").read_bytes() == b"modified"


def test_unmarked_nested_bundle_exact_tree_and_symlink(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    source = root / "apps"
    skill = source / "one"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: one\n---\n")
    (source / "guide.md").write_text("guide")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    target = tmp_path / "target"
    target.mkdir()
    dest = target / "apps"
    shutil.copytree(source, dest)
    assert bundle.uninstall(["apps"], "codex", dir=target)[0].status == "removed"
    shutil.copytree(source, dest)
    (dest / "link").symlink_to(source / "guide.md")
    assert bundle.uninstall(["apps"], "codex", dir=target)[0].status == "not managed"
    with pytest.raises(ValueError, match="not managed"):
        bundle.install_bundles(["apps"], "codex", dir=target, force=True)
    assert (dest / "link").is_symlink()


def test_guide_subskill_discovery_install_and_uninstall(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    parent = root / "harness"
    guide = parent / "core" / "memory"
    guide.mkdir(parents=True)
    (parent / "SKILL.md").write_text("---\nname: harness\n---\nParent\n")
    (guide / "GUIDE.md").write_text(
        "---\nname: memory\ndescription: guide\n---\nBody\n"
    )
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    skills = bundle.list_skills()
    assert [(s.name, s.parent) for s in skills] == [
        ("harness", None),
        ("memory", "harness"),
    ]
    assert list(bundle.bundle_folders()) == ["harness"]
    target = tmp_path / "target"
    assert [r.name for r in bundle.install(["memory"], "codex", dir=target)] == [
        "harness",
        "memory",
    ]
    text = (target / "memory/SKILL.md").read_text()
    assert "---\n> Part of the `harness` skill." in text
    (target / "harness/SKILL.md").write_text("custom")
    bundle.install(["memory"], "codex", dir=target, force=True)
    assert (target / "harness/SKILL.md").read_text() == "custom"
    assert bundle.uninstall(["memory"], "codex", dir=target)[0].status == "removed"
    assert (target / "harness/SKILL.md").read_text() == "custom"


def test_guide_name_collision(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    parent = root / "harness"
    guide = parent / "nested"
    guide.mkdir(parents=True)
    (parent / "SKILL.md").write_text("---\nname: harness\n---\n")
    (guide / "GUIDE.md").write_text("---\nname: harness\n---\n")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    with pytest.raises(ValueError, match="Duplicate skill name"):
        bundle.list_skills()


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_guide_closing_delimiter_at_eof_retains_line_endings(
    tmp_path, monkeypatch, newline
):
    root = tmp_path / "skills"
    guide = root / "harness" / "guide"
    guide.mkdir(parents=True)
    (guide.parent / "SKILL.md").write_text("---\nname: harness\n---\n")
    (guide / "GUIDE.md").write_bytes(newline.join([b"---", b"name: guide", b"---"]))
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    target = tmp_path / "target"
    bundle.install(["guide"], "codex", dir=target)
    result = (target / "guide/SKILL.md").read_bytes()
    assert newline + b"---" + newline + b"> Part of the `harness` skill." in result
    if newline == b"\r\n":
        assert b"\n" not in result.replace(b"\r\n", b"")


def test_guide_install_keeps_unmanaged_existing_parent(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    guide = root / "harness/guide"
    guide.mkdir(parents=True)
    (guide.parent / "SKILL.md").write_text("---\nname: harness\n---\n")
    (guide / "GUIDE.md").write_text("---\nname: guide\n---\nBody")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    target = tmp_path / "target"
    parent = target / "harness"
    parent.mkdir(parents=True)
    (parent / "SKILL.md").write_text("unmanaged")
    results = bundle.install(["guide"], "codex", dir=target, force=True)
    assert [(r.name, r.status) for r in results] == [
        ("harness", "exists"),
        ("guide", "installed"),
    ]
    assert (parent / "SKILL.md").read_text() == "unmanaged"
    assert not (parent / ".mobilerun-installed").exists()
    assert (target / "guide/SKILL.md").is_file()


def test_guide_discovery_ignores_orphans_and_malformed_frontmatter(
    tmp_path, monkeypatch
):
    root = tmp_path / "skills"
    orphan = root / "orphan/GUIDE.md"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("---\nname: orphan\n---\n")
    parent = root / "harness"
    parent.mkdir()
    (parent / "SKILL.md").write_text("---\nname: harness\n---\n")
    (parent / "GUIDE.md").write_text("---\nname: malformed\n")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    assert [s.name for s in bundle.list_skills()] == ["harness"]


def test_guide_discovery_uses_nearest_skill_ancestor(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    inner = root / "outer/inner"
    guide = inner / "guide"
    guide.mkdir(parents=True)
    (inner.parent / "SKILL.md").write_text("---\nname: outer\n---\n")
    (inner / "SKILL.md").write_text("---\nname: inner\n---\n")
    (guide / "GUIDE.md").write_text("---\nname: guide\n---\n")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    skills = {s.name: s for s in bundle.list_skills()}
    assert skills["guide"].parent == "inner"


def test_reference_discovery_and_install(tmp_path, monkeypatch):
    import yaml

    root = tmp_path / "skills"
    parent = root / "cloud"
    refs = parent / "references"
    refs.mkdir(parents=True)
    (parent / "SKILL.md").write_text(
        '---\nname: cloud\n---\n- [Trouble](./references/trouble.md) — Fix: "bad" devices.\n'
    )
    (refs / "trouble.md").write_text("# Trouble\n\nFallback body.\n")
    (refs / "fallback.md").write_text("# Title\n\nA helpful paragraph.\n")
    (refs / "named.md").write_text(
        "---\nname: custom\ndescription: Own description\n---\n\n# Title\n"
    )
    (refs / "partial.md").write_text(
        '---\ndescription: "Partial: details"\n---\nBody\n'
    )
    (parent / "README.md").write_text("Not a reference")
    nested = refs / "nested"
    nested.mkdir()
    (nested / "other.md").write_text("Ignored")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    found = {s.name: s for s in bundle.list_skills()}
    assert set(found) == {
        "cloud",
        "cloud-trouble",
        "cloud-fallback",
        "cloud-partial",
        "custom",
    }
    assert found["cloud-trouble"].description == 'Fix: "bad" devices.'
    assert found["cloud-fallback"].description == "A helpful paragraph."
    assert found["custom"].description == "Own description"
    assert all(
        found[n].parent == "cloud"
        for n in ("cloud-trouble", "cloud-fallback", "custom")
    )
    target = tmp_path / "target"
    results = bundle.install(["cloud-trouble"], "codex", dir=target)
    assert [(r.name, r.status) for r in results] == [
        ("cloud", "installed"),
        ("cloud-trouble", "installed"),
    ]
    content = (target / "cloud-trouble/SKILL.md").read_text()
    frontmatter = yaml.safe_load(content.split("---", 2)[1])
    assert frontmatter == {
        "name": "cloud-trouble",
        "description": 'Fix: "bad" devices.',
    }
    assert "../cloud/references/" in content
    assert "# Trouble\n\nFallback body." in content
    bundle.install(["cloud-partial"], "codex", dir=target)
    partial = (target / "cloud-partial/SKILL.md").read_text()
    assert yaml.safe_load(partial.split("---", 2)[1]) == {
        "name": "cloud-partial",
        "description": "Partial: details",
    }


def test_reference_discovery_only_direct_skill_references(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    parent = root / "cloud"
    for rel in (
        "references/public.md",
        "local/references/private.md",
        "apps/app/references/card.md",
    ):
        file = parent / rel
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("# Heading\n\nDetails.\n")
    (parent / "SKILL.md").write_text("---\nname: cloud\n---\n")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    assert {s.name for s in bundle.list_skills()} == {"cloud", "cloud-public"}


@pytest.mark.parametrize(
    "link",
    [
        "references/info.md#setup",
        './references/info.md "Title"',
        "<references/info.md>",
        '<./references/info.md#setup> "Title"',
    ],
)
@pytest.mark.parametrize(
    "description, expected",
    [
        ("read when: devices fail.", "Use when devices fail."),
        ("READ WHEN: devices fail.", "Use when devices fail."),
        ("Other guidance.", "Other guidance."),
    ],
)
def test_reference_index_link_variations(
    tmp_path, monkeypatch, link, description, expected
):
    root = tmp_path / "skills"
    parent = root / "cloud"
    refs = parent / "references"
    refs.mkdir(parents=True)
    (parent / "SKILL.md").write_text(
        f"---\nname: cloud\n---\n- [Info]({link}) — {description}\n"
    )
    (refs / "info.md").write_text("# Title\n\nFallback paragraph.\n")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    assert (
        next(s for s in bundle.list_skills() if s.name == "cloud-info").description
        == expected
    )


@pytest.mark.parametrize("version", ["current", "retired", "old-manifest"])
def test_parent_uninstall_requires_all_managed_children(tmp_path, monkeypatch, version):
    root = tmp_path / "source"
    refs = root / "parent/references"
    refs.mkdir(parents=True)
    (refs.parent / "SKILL.md").write_text("parent")
    (refs / "child.md").write_text("reference")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    target = tmp_path / "target"
    names = ["parent", "parent-child"]
    bundle.install(names, "codex", dir=target)
    if version == "retired":
        (refs / "child.md").unlink()  # A later release no longer ships this guide.
    elif version == "old-manifest":
        marker = target / "parent-child/.mobilerun-installed"
        manifest = json.loads(marker.read_text())
        del manifest["parent"]
        marker.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="parent-child"):
        bundle.uninstall(["parent"], "codex", dir=target, discard_changes=True)
    notes = target / "parent-child/notes.md"
    notes.write_text("keep")
    with pytest.raises(ValueError, match="local changes"):
        bundle.uninstall(names, "codex", dir=target)
    assert (target / "parent/SKILL.md").exists() and notes.exists()
    assert [
        r.status
        for r in bundle.uninstall(names, "codex", dir=target, discard_changes=True)
    ] == ["removed", "removed"]
