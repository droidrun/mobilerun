import click
from click.testing import CliRunner


def test_interactive_group_invokes_selection(monkeypatch):
    from mobilerun.cli.base import menu

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "select_prompt", lambda message, choices: "hello")

    @click.group(cls=menu.MenuGroup)
    def group():
        pass

    @group.command()
    def hello():
        click.echo("selected")

    result = CliRunner().invoke(group, [])
    assert result.exit_code == 0
    assert "selected" in result.output


def test_interactive_group_prints_epilog_before_prompt(monkeypatch):
    from mobilerun.cli.base import menu

    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    def select(message, choices):
        click.echo("PROMPT")
        return None

    monkeypatch.setattr(menu, "select_prompt", select)

    @click.group(cls=menu.MenuGroup, epilog="A helpful tip")
    def group():
        pass

    result = CliRunner().invoke(group, [])
    assert result.exit_code == 0
    assert result.output.index("A helpful tip") < result.output.index("PROMPT")


def test_interactive_group_without_epilog_prints_no_extra_hint(monkeypatch):
    from mobilerun.cli.base import menu

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "select_prompt", lambda message, choices: None)

    @click.group(cls=menu.MenuGroup)
    def group():
        pass

    result = CliRunner().invoke(group, [])
    assert result.exit_code == 0
    assert "Usage:" not in result.output


def test_noninteractive_group_still_shows_help():
    from mobilerun.cli.base.menu import MenuGroup

    @click.group(cls=MenuGroup)
    def group():
        pass

    @group.command()
    def hello():
        pass

    result = CliRunner().invoke(group, [])
    assert result.exit_code == 2
    assert "Usage:" in result.output


def test_workspace_apps_help_mentions_dashboard_app_store():
    from mobilerun.cli.workspace.group import workspace

    result = CliRunner().invoke(workspace, ["apps", "--help"])
    assert result.exit_code == 0
    assert "https://cloud.mobilerun.ai/apps" in result.output


def test_cancel_menu_exits_cleanly(monkeypatch):
    from mobilerun.cli.base import menu

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "select_prompt", lambda message, choices: None)

    @click.group(cls=menu.MenuGroup)
    def group():
        pass

    assert CliRunner().invoke(group, []).exit_code == 0


def test_root_menu_does_not_load_commands_until_selected(monkeypatch):
    from mobilerun.cli import app
    from mobilerun.cli.base import menu

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "select_prompt", lambda message, choices: None)
    monkeypatch.setattr(
        app, "import_module", lambda name: (_ for _ in ()).throw(AssertionError(name))
    )
    result = CliRunner().invoke(app.cli, [])
    assert result.exit_code == 0


def test_keyboard_interrupt_exits_130(monkeypatch):
    from mobilerun.cli.base import menu

    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    def interrupt(message, choices):
        raise KeyboardInterrupt

    monkeypatch.setattr(menu, "select_prompt", interrupt)

    @click.group(cls=menu.MenuGroup)
    def group():
        pass

    assert CliRunner().invoke(group, []).exit_code == 130


def test_required_choice_argument_is_selected_in_terminal(monkeypatch):
    from mobilerun.cli.base import menu

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "select_prompt", lambda message, choices: "home")

    @click.command()
    @click.argument(
        "button", cls=menu.ChoiceArgument, type=click.Choice(["back", "home"])
    )
    def press(button):
        click.echo(button)

    result = CliRunner().invoke(press, [])
    assert result.exit_code == 0
    assert "home" in result.output
