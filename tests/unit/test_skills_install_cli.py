from importlib import import_module
from pathlib import Path

import pytest
from click.testing import CliRunner

from mobilerun.skills import bundle
from mobilerun.skills.cli.group import skills

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
    from click.testing import CliRunner

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


def test_single_picker_is_filterable_multiselect(monkeypatch, tmp_path):
    seen = {}

    class Prompt:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def register_kb(self, key):
            return lambda function: function

        def execute(self):
            return ["alpha"]

    monkeypatch.setattr(install_cli, "FuzzyPrompt", Prompt)
    selected = install_cli.select_skills(
        [bundle.Skill("alpha", "description", tmp_path, "source", Path("apps/card"))]
    )
    assert selected == ["alpha"]
    assert seen["multiselect"] is True
    assert "source/apps/card" in seen["choices"][0]["name"]


def test_list_reports_only_individually_installed_skill(tmp_path, monkeypatch):
    import json
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
    seen = {}

    class Prompt:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def register_kb(self, key):
            return lambda function: function

        def execute(self):
            return [seen["choices"][0]["value"]]

    monkeypatch.setattr(install_cli, "FuzzyPrompt", Prompt)
    result = invoke(skill_source)
    assert result.exit_code == 0, result.output
    assert [choice["name"] for choice in seen["choices"]][
        0
    ] == "Everything (all skills)"
    assert (skill_source / "alpha/SKILL.md").exists()
    assert (skill_source / "beta/SKILL.md").exists()


def test_picker_transformer_echoes_names_only(monkeypatch, tmp_path):
    seen = {}

    class Prompt:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def register_kb(self, key):
            return lambda function: function

        def execute(self):
            return ["alpha", "beta"]

    monkeypatch.setattr(install_cli, "FuzzyPrompt", Prompt)
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
    seen = {}

    class Prompt:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def register_kb(self, key):
            return lambda function: function

        def execute(self):
            return [choice["value"] for choice in seen["choices"]]

    monkeypatch.setattr(install_cli, "FuzzyPrompt", Prompt)
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
