import hashlib
import json

import pytest

from mobilerun.skills import bundle


@pytest.fixture
def installed_skill(tmp_path, monkeypatch):
    root = tmp_path / "source"
    source = root / "example"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("original")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    target = tmp_path / "target"
    bundle.install(["example"], "codex", dir=target)
    return target / "example"


@pytest.mark.parametrize("operation", ["install", "install_bundles", "uninstall"])
@pytest.mark.parametrize("filename", ["user-notes.md", "SKILL.md"])
@pytest.mark.parametrize("legacy", [False, True])
def test_local_changes_block_removal(installed_skill, operation, filename, legacy):
    if legacy:
        (installed_skill / ".mobilerun-installed").write_text("mobilerun\n")
    private = installed_skill / filename
    private.write_text("keep this")
    options = {} if operation == "uninstall" else {"force": True}
    with pytest.raises(ValueError, match="local changes"):
        getattr(bundle, operation)(
            ["example"], "codex", dir=installed_skill.parent, **options
        )
    assert private.read_text() == "keep this"


def test_manifest_hashes_installed_content(installed_skill):
    marker = json.loads((installed_skill / ".mobilerun-installed").read_text())
    assert marker["files"] == {"SKILL.md": hashlib.sha256(b"original").hexdigest()}


@pytest.mark.parametrize("operation", ["install", "install_bundles", "uninstall"])
def test_explicit_confirmation_discards_local_changes(installed_skill, operation):
    notes = installed_skill / "user-notes.md"
    notes.write_text("discard this")
    options = {} if operation == "uninstall" else {"force": True}
    getattr(bundle, operation)(
        ["example"],
        "codex",
        dir=installed_skill.parent,
        discard_changes=True,
        **options,
    )
    assert not notes.exists()


def test_unchanged_legacy_marker_can_be_removed(installed_skill):
    (installed_skill / ".mobilerun-installed").write_text("mobilerun\n")
    assert (
        bundle.uninstall(["example"], "codex", dir=installed_skill.parent)[0].status
        == "removed"
    )


def test_missing_installed_file_is_not_local_data(installed_skill):
    (installed_skill / "SKILL.md").unlink()
    assert (
        bundle.uninstall(["example"], "codex", dir=installed_skill.parent)[0].status
        == "removed"
    )


@pytest.mark.parametrize(
    "command", [["uninstall"], ["install", "--force"], ["install", "bundle", "--force"]]
)
@pytest.mark.parametrize("decision", ["no", "confirm", "yes-flag", "discard-flag"])
def test_cli_requires_specific_consent(installed_skill, command, decision, monkeypatch):
    from click.testing import CliRunner

    from mobilerun.cli.base import menu, prompts
    from mobilerun.skills.cli import skills

    interactive = decision in ("no", "confirm")
    asked = []

    def confirm(message, *, default):
        asked.append(default)
        return decision == "confirm"

    monkeypatch.setattr(menu, "is_interactive", lambda: interactive)
    monkeypatch.setattr(prompts, "confirm_prompt", confirm)
    (installed_skill / "SKILL.md").write_text("my changes")
    for index in range(12):
        (installed_skill / f"notes-{index:02d}").write_text("keep me")
    flags = {"yes-flag": ["--yes"], "discard-flag": ["--discard-changes"]}.get(
        decision, []
    )
    result = CliRunner().invoke(
        skills,
        [
            *command,
            "example",
            "--agent",
            "codex",
            "--dir",
            str(installed_skill.parent),
            *flags,
        ],
    )
    assert asked == ([False] if interactive else [])
    if decision != "discard-flag":
        assert "SKILL.md" in result.output and "notes-00" in result.output
        assert "notes-11" not in result.output and "more" in result.output
    if decision in ("confirm", "discard-flag"):
        assert result.exit_code == 0, result.output
        assert not (installed_skill / "notes-00").exists()
    else:
        assert result.exit_code != 0
        assert (installed_skill / "notes-00").read_text() == "keep me"
        assert (installed_skill / "SKILL.md").read_text() == "my changes"
