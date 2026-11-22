"""Feature helpers stay grouped by role while retaining public exports."""


def test_macro_utilities_expose_state_matching_and_logging():
    from mobilerun.macro.models import MACRO_SCHEMA_VERSION, normalize_ui_state
    from mobilerun.macro.utils import compare_states, configure_macro_logging

    assert MACRO_SCHEMA_VERSION
    assert callable(compare_states)
    assert callable(configure_macro_logging)
    assert callable(normalize_ui_state)


def test_skills_utils_parse_frontmatter_description():
    from mobilerun.skills.utils import _description

    assert _description("---\ndescription: Example\n---\n") == "Example"


def test_macro_runtime_defaults_read_environment(monkeypatch):
    from mobilerun.macro import MacroPlayer

    monkeypatch.setenv("MOBILERUN_MACRO_STATE_TIMEOUT", "12")
    monkeypatch.setenv("MOBILERUN_MACRO_STATE_THRESHOLD", "0.4")
    monkeypatch.setenv("MOBILERUN_MACRO_DELAY", "2")
    monkeypatch.setenv("MOBILERUN_MACRO_POLL_INTERVAL", "0.3")
    player = MacroPlayer()
    assert (player.delay_between_actions, player.state_timeout) == (2, 12)
    assert (player.state_threshold, player.state_poll_interval) == (0.4, 0.3)
    monkeypatch.setenv("MOBILERUN_MACRO_STATE_TIMEOUT", "13")
    assert MacroPlayer().state_timeout == 13
    assert MacroPlayer(state_timeout=7).state_timeout == 7


def test_macro_matcher_threshold_reads_environment(monkeypatch):
    from mobilerun.macro.utils import compare_states

    saved = {"phone_state": {"package": "a"}}
    current = {"phone_state": {"package": "b"}}
    monkeypatch.setenv("MOBILERUN_MACRO_STATE_THRESHOLD", "0.9")
    assert not compare_states(saved, current).matches
    monkeypatch.setenv("MOBILERUN_MACRO_STATE_THRESHOLD", "0.8")
    assert compare_states(saved, current).matches


def test_macro_list_default_directory_reads_environment(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from mobilerun.macro.cli import macro

    path = tmp_path / "not-a-directory"
    path.write_text("file")
    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(path))
    result = CliRunner().invoke(macro, ["list", "--json"])
    assert result.exit_code != 0
    assert "Not a directory" in result.output


def test_macro_replay_cli_defaults_read_environment(monkeypatch):
    import click

    from mobilerun.macro.cli.replay import replay

    monkeypatch.setenv("MOBILERUN_MACRO_DELAY", "3")
    monkeypatch.setenv("MOBILERUN_MACRO_STATE_TIMEOUT", "9")
    monkeypatch.setenv("MOBILERUN_MACRO_STATE_THRESHOLD", "0.7")
    options = {param.name: param for param in replay.params}
    with click.Context(replay) as context:
        assert options["delay"].get_default(context) == 3
        assert options["state_timeout"].get_default(context) == 9
        assert options["state_threshold"].get_default(context) == 0.7
