from pathlib import Path

import pytest
from click.testing import CliRunner


def test_macro_missing_explicit_and_implicit_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mobilerun.macro.cli import macro

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    missing = runner.invoke(macro, ["list", "missing"])
    assert missing.exit_code == 1
    assert "Macro directory not found: missing" in missing.output
    default = runner.invoke(macro, ["list"])
    assert default.exit_code == 0
    assert "No macros yet" in default.output
    as_json = runner.invoke(macro, ["list", "--json"])
    assert as_json.exit_code == 0
    assert as_json.output.strip() == "[]"


def test_unknown_skills_offer_list_and_close_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mobilerun.skills import bundle
    from mobilerun.skills.cli.group import skills

    source = tmp_path / "source"
    (source / "alpha").mkdir(parents=True)
    (source / "alpha" / "SKILL.md").write_text("---\ndescription: alpha\n---\n")
    monkeypatch.setattr(bundle, "skills_root", lambda: source)
    runner = CliRunner()
    for args in (["show", "alpa"], ["install", "alpa", "--agent", "codex"]):
        result = runner.invoke(skills, args)
        assert result.exit_code == 1
        assert (
            "Unknown skill: alpa. Run `mobilerun skills list` to see available skills."
            in result.output
        )
        assert "Did you mean 'alpha'?" in result.output
    result = runner.invoke(skills, ["show", "bogus"])
    assert result.exit_code == 1
    assert "Did you mean" not in result.output


def test_sandbox_visible_in_root_help() -> None:
    from mobilerun.cli.app import cli

    output = CliRunner().invoke(cli, ["--help"]).output
    assert "sandbox" in output and "Deploy and run cloud functions." in output


def test_sandbox_visible_in_root_menu() -> None:
    import click

    from mobilerun.cli.app import cli

    with click.Context(cli) as ctx:
        choice = next(
            choice for choice in cli.menu_choices(ctx) if choice.value == "sandbox"
        )
        assert choice.hint == "Deploy and run cloud functions."
