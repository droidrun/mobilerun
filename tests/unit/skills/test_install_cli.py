"""Skill installation, picker behavior and safe removal of managed files."""

import hashlib
import json
from importlib import import_module
from pathlib import Path

import pytest
from click.testing import CliRunner

from mobilerun.skills import bundle
from mobilerun.skills.cli import skills

install_cli = import_module("mobilerun.skills.cli.install")


@pytest.fixture
def skill_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for name in ("alpha", "beta"):
        skill = source / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"---\ndescription: {name} description\n---\n")
    monkeypatch.setattr(bundle, "skills_root", lambda: source)
    monkeypatch.setattr(install_cli, "select_prompt", lambda message, choices: "single")
    return tmp_path / "target"


def invoke(target, *args):
    return CliRunner().invoke(
        skills, ["install", *args, "--agent", "codex", "--dir", str(target)]
    )


def test_interactive_selects_skills(skill_source, monkeypatch):
    monkeypatch.setattr(install_cli, "is_interactive", lambda: True, raising=False)
    seen = []

    def select(choices, **kwargs):
        seen.extend((choice.name, choice.description) for choice in choices)
        return ["beta"]

    monkeypatch.setattr(install_cli, "select_skills", select, raising=False)
    result = invoke(skill_source)
    assert result.exit_code == 0, result.output
    assert seen == [("alpha", "alpha description"), ("beta", "beta description")]
    assert (skill_source / "beta" / "SKILL.md").exists()
    assert not (skill_source / "alpha").exists()


def test_interactive_prompts_for_agent_then_skills(skill_source, monkeypatch):
    monkeypatch.setattr(install_cli, "is_interactive", lambda: True)
    prompts = []

    def agent_prompt(message, choices):
        prompts.append(message)
        return "codex" if message == "Select agent" else "single"

    def skills_prompt(choices, **kwargs):
        prompts.append("skills")
        return ["alpha"]

    monkeypatch.setattr(install_cli, "select_prompt", agent_prompt)
    monkeypatch.setattr(install_cli, "select_skills", skills_prompt)
    result = CliRunner().invoke(skills, ["install", "--dir", str(skill_source)])
    assert result.exit_code == 0, result.output
    assert prompts == ["Select agent", "skills"]
    assert (skill_source / "alpha" / "SKILL.md").exists()


def test_noninteractive_no_agent_reports_missing_skills(monkeypatch):
    monkeypatch.setattr(install_cli, "is_interactive", lambda: False)
    result = CliRunner().invoke(skills, ["install"])
    assert result.exit_code == 2
    assert "Specify skill names or --all." in result.output


def test_interactive_empty_selection(skill_source, monkeypatch):
    monkeypatch.setattr(install_cli, "is_interactive", lambda: True, raising=False)
    monkeypatch.setattr(
        install_cli, "select_skills", lambda choices, **kwargs: [], raising=False
    )
    result = invoke(skill_source)
    assert result.exit_code == 0
    assert "No skills selected." in result.output
    assert not skill_source.exists()


@pytest.mark.parametrize("cancel", [None, KeyboardInterrupt()])
def test_interactive_cancel(skill_source, monkeypatch, cancel):
    monkeypatch.setattr(install_cli, "is_interactive", lambda: True, raising=False)

    def select(choices, **kwargs):
        if cancel is not None:
            raise cancel
        return None

    monkeypatch.setattr(install_cli, "select_skills", select, raising=False)
    result = invoke(skill_source)
    assert result.exit_code in (0, 130)
    assert "Traceback" not in result.output
    assert not skill_source.exists()


def test_noninteractive_requires_names_or_all(skill_source, monkeypatch):
    monkeypatch.setattr(install_cli, "is_interactive", lambda: False, raising=False)
    result = invoke(skill_source)
    assert result.exit_code == 2
    assert "Specify skill names or --all." in result.output


def test_names_and_all_are_mutually_exclusive(skill_source):
    result = invoke(skill_source, "alpha", "--all")
    assert result.exit_code == 2
    assert "Specify skill names or --all, not both." in result.output


def test_all_installs_all_skills(skill_source):
    result = invoke(skill_source, "--all")
    assert result.exit_code == 0, result.output
    assert (skill_source / "alpha" / "SKILL.md").exists()
    assert (skill_source / "beta" / "SKILL.md").exists()


def test_names_install_only_requested_skill(skill_source):
    result = invoke(skill_source, "alpha")
    assert result.exit_code == 0, result.output
    assert (skill_source / "alpha" / "SKILL.md").exists()
    assert not (skill_source / "beta").exists()


def test_unknown_name_is_reported(skill_source):
    result = invoke(skill_source, "missing")
    assert result.exit_code != 0
    assert "Unknown skill: missing. Run `mobilerun skills list`" in result.output


def test_agent_choice_reads_environment_at_invocation(monkeypatch, skill_source):
    from mobilerun.skills.cli.install import install_command

    monkeypatch.setenv(
        "MOBILERUN_AGENTS", '{"custom":{"skills":["global","project"],"mcp":null}}'
    )
    result = CliRunner().invoke(
        install_command, ["--agent", "custom", "--dir", str(skill_source), "--all"]
    )
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("mode", ["everything", "bundle", "single"])
def test_interactive_modes(skill_source, monkeypatch, mode):
    monkeypatch.setattr(install_cli, "is_interactive", lambda: True)
    nested = skill_source.parent / "source" / "apps" / "nested"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text("---\nname: nested\n---\n")

    def choose_mode(message, choices):
        assert message == "Select install mode"
        assert [(choice.value, choice.label) for choice in choices] == [
            ("everything", "Everything"),
            ("bundle", "Bundle (pick folders)"),
            ("single", "Single (pick skills)"),
        ]
        return mode

    monkeypatch.setattr(install_cli, "select_prompt", choose_mode)
    monkeypatch.setattr(install_cli, "select_skills", lambda choices: ["alpha"])
    result = invoke(skill_source)
    assert result.exit_code == 0, result.output
    assert (skill_source / "alpha/SKILL.md").exists()
    assert (skill_source / "beta/SKILL.md").exists() == (mode == "everything")


@pytest.mark.parametrize("mode", [(), ("bundle",), ("single",)])
def test_all_modes_include_nested_skills(tmp_path, monkeypatch, mode):
    root = tmp_path / "skills"
    for path, name in [(root / "alpha", "alpha"), (root / "apps" / "card", "card")]:
        path.mkdir(parents=True)
        (path / "SKILL.md").write_text(f"---\nname: {name}\n---\n")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    target = tmp_path / "target"
    result = invoke(target, *mode, "--all")
    assert result.exit_code == 0, result.output
    assert (target / "alpha/SKILL.md").exists()
    assert (
        target / ("card/SKILL.md" if mode == ("single",) else "apps/card/SKILL.md")
    ).exists()


def capture_picker(monkeypatch, select):
    """Record the displayed choices and simulate a user selection."""
    seen = {}

    class Prompt:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def register_kb(self, key):
            return lambda function: function

        def execute(self):
            return select(seen["choices"])

    monkeypatch.setattr(install_cli, "FuzzyPrompt", Prompt)
    return seen


def test_single_picker_is_filterable_multiselect(monkeypatch, tmp_path):
    seen = capture_picker(monkeypatch, lambda choices: ["alpha"])
    selected = install_cli.select_skills(
        [bundle.Skill("alpha", "description", tmp_path, "source", Path("apps/card"))]
    )
    assert selected == ["alpha"]
    assert seen["multiselect"] is True
    assert "source/apps/card" in seen["choices"][0]["name"]


def test_list_reports_only_individually_installed_skill(tmp_path, monkeypatch):
    from types import SimpleNamespace

    module = import_module("mobilerun.skills.cli.list")
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: SimpleNamespace(
            MOBILERUN_AGENTS={"codex": SimpleNamespace(skills=("user", None))}
        ),
    )
    monkeypatch.setattr(module, "installed", lambda agent, scope: ["apps", "one"])
    monkeypatch.setattr(
        module,
        "list_skills",
        lambda: [
            bundle.Skill("one", "", tmp_path, "skills", Path("apps/one")),
            bundle.Skill("two", "", tmp_path, "skills", Path("apps/two")),
        ],
    )
    result = CliRunner().invoke(module.list_command, ["--json"])
    assert result.exit_code == 0, result.output
    one, two = json.loads(result.output)
    assert one["installed"] == ["codex:user"]
    assert two["installed"] == []


def test_flat_picker_offers_everything_and_installs_all(skill_source, monkeypatch):
    monkeypatch.setattr(install_cli, "is_interactive", lambda: True)
    seen = capture_picker(monkeypatch, lambda choices: [choices[0]["value"]])
    result = invoke(skill_source)
    assert result.exit_code == 0, result.output
    assert [choice["name"] for choice in seen["choices"]][
        0
    ] == "Everything (all skills)"
    assert (skill_source / "alpha/SKILL.md").exists()
    assert (skill_source / "beta/SKILL.md").exists()


def test_picker_transformer_echoes_names_only(monkeypatch, tmp_path):
    seen = capture_picker(monkeypatch, lambda choices: ["alpha", "beta"])
    skills = [
        bundle.Skill(name, "description", tmp_path, "source", Path("apps") / name)
        for name in ("alpha", "beta")
    ]
    install_cli.select_skills(skills)
    labels = [entry["name"] for entry in seen["choices"]]
    assert seen["transformer"](labels) == "alpha, beta"


def test_bundle_picker_transformer_echoes_bundle_names(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    for folder in ("apps", "tools"):
        skill = root / folder / "nested"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"---\nname: {folder}-skill\n---\n")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    monkeypatch.setattr(install_cli, "is_interactive", lambda: True)
    seen = capture_picker(
        monkeypatch, lambda choices: [choice["value"] for choice in choices]
    )
    result = invoke(tmp_path / "target", "bundle")
    assert result.exit_code == 0, result.output
    assert (
        seen["transformer"]([choice["name"] for choice in seen["choices"]])
        == "apps, tools"
    )


def test_guide_enables_mode_menu_and_bundle_does_not_install_guide_separately(
    skill_source, monkeypatch
):
    guide = skill_source.parent / "source/alpha/nested/GUIDE.md"
    guide.parent.mkdir(parents=True)
    guide.write_text("---\nname: alpha-guide\ndescription: guide\n---\nBody\n")
    monkeypatch.setattr(install_cli, "is_interactive", lambda: True)
    prompts = []

    def select(message, choices):
        prompts.append((message, [choice.value for choice in choices]))
        return "everything"

    monkeypatch.setattr(install_cli, "select_prompt", select)
    result = invoke(skill_source)
    assert result.exit_code == 0, result.output
    assert prompts == [("Select install mode", ["everything", "bundle", "single"])]
    assert (skill_source / "alpha/nested/GUIDE.md").is_file()
    assert not (skill_source / "alpha-guide").exists()


def test_guide_list_show_and_single_install(skill_source):
    guide = skill_source.parent / "source/alpha/nested/GUIDE.md"
    guide.parent.mkdir(parents=True)
    guide.write_text("---\nname: alpha-guide\ndescription: guide\n---\nBody\n")
    listing = CliRunner().invoke(skills, ["list"])
    assert "alpha-guide [alpha]" in listing.output
    shown = CliRunner().invoke(skills, ["show", "alpha-guide"])
    assert "alpha-guide [alpha]" in shown.output
    assert "GUIDE.md" in shown.output
    result = invoke(skill_source, "single", "alpha-guide")
    assert result.exit_code == 0, result.output
    assert "alpha: installed" in result.output
    assert "alpha-guide: installed" in result.output


def test_single_all_does_not_install_guides_separately(skill_source):
    guide = skill_source.parent / "source/alpha/nested/GUIDE.md"
    guide.parent.mkdir(parents=True)
    guide.write_text("---\nname: alpha-guide\n---\nBody\n")
    result = invoke(skill_source, "single", "--all")
    assert result.exit_code == 0, result.output
    assert (skill_source / "alpha/nested/GUIDE.md").is_file()
    assert not (skill_source / "alpha-guide").exists()


def test_reference_show_uses_source_file_and_single_all_excludes_references(
    skill_source,
):
    refs = skill_source.parent / "source/alpha/references"
    refs.mkdir()
    (refs / "info.md").write_text("# Info\n\nUseful info.\n")
    shown = CliRunner().invoke(skills, ["show", "alpha-info"])
    assert shown.exit_code == 0, shown.output
    assert str(refs / "info.md") in shown.output
    result = invoke(skill_source, "single", "--all")
    assert result.exit_code == 0, result.output
    assert not (skill_source / "alpha-info").exists()


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
    from mobilerun.cli.base import menu, prompts

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
