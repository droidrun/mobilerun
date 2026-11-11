"""User-facing help, output, and automation contracts for the CLI."""

import importlib
from unittest.mock import AsyncMock, Mock

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli.main import cli
from mobilerun.config_manager import ConfigLoader

run_module = importlib.import_module("mobilerun.cli.run")


def commands(command=cli, path=()):
    yield path, command
    if isinstance(command, click.Group):
        for name, child in command.commands.items():
            yield from commands(child, (*path, name))


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setattr(ConfigLoader, "get_user_config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setenv("MOBILERUN_TELEMETRY_ENABLED", "false")
    for name in ("MOBILERUN_CONFIG", "MOBILERUN_TARGET", "MOBILERUN_DEVICE", "MOBILERUN_DEBUG"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("path", [(), ("run",), ("device",), ("device", "list"),
                                 ("device", "ui"), ("device", "pair"), ("auth", "login"),
                                 ("configure",), ("macro", "replay")])
def test_help_has_copyable_examples(path):
    result = CliRunner().invoke(cli, [*path, "--help"])
    assert result.exit_code == 0, result.output
    assert "Examples:\n    mobilerun " in result.stdout
    assert "\n    mobilerun " in result.stdout
    if not path:
        assert result.stdout.rstrip().endswith("Docs: https://docs.mobilerun.ai/framework/guides/cli")


@pytest.mark.parametrize("path,command", list(commands()))
def test_help_options_and_summaries_are_complete(path, command):
    with click.Context(command, **command.context_settings) as ctx:
        params = command.get_params(ctx)
        for param in params:
            if isinstance(param, click.Option):
                assert param.help and param.help.endswith("."), (path, param.name)
                assert param.show_default is not False, (path, param.name)
        names = [param.name for param in params]
        shared_first = [name for name in ("device", "target", "platform", "config_path") if name in names]
        shared_last = [name for name in ("api_url", "json_output", "verbose", "help") if name in names]
        assert names[:len(shared_first)] == shared_first
        assert names[-len(shared_last):] == shared_last
    assert command.get_short_help_str(limit=1).endswith("."), path
    assert not command.get_short_help_str(limit=1).endswith("..."), path


def test_root_command_order_and_agent_discovery(monkeypatch):
    monkeypatch.setattr(run_module, "list_agents", Mock(side_effect=AssertionError("root must not scan")))
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    names = ["run", "device", "configure", "auth", "doctor", "env", "macro"]
    assert [result.stdout.index(f"  {name} ") for name in names] == sorted(
        result.stdout.index(f"  {name} ") for name in names
    )


def test_agent_option_tracks_installed_agents(monkeypatch):
    monkeypatch.setattr(run_module, "list_agents", lambda: [])
    runner = CliRunner()
    assert "--agent" not in runner.invoke(cli, ["run", "--help"]).stdout
    result = runner.invoke(cli, ["run", "TASK", "-a", "missing"])
    assert result.exit_code == 2
    assert "No external agents are installed; install one before using --agent." in result.stderr
    monkeypatch.setattr(run_module, "list_agents", lambda: ["installed"])
    assert "--agent [installed]" in runner.invoke(cli, ["run", "--help"]).stdout
    callback = AsyncMock(return_value=(True, "done", 1))
    monkeypatch.setattr(run_module, "_run_task", callback)
    assert runner.invoke(cli, ["run", "TASK", "-a", "installed"]).exit_code == 0
    assert callback.await_args.kwargs["agent"] == "installed"
    assert runner.invoke(cli, ["run", "TASK", "-a", "missing"]).exit_code == 2


@pytest.mark.parametrize("args,suggestion", [(["devcie"], "device"), (["device", "lst"], "list"),
                                             (["auth", "stauts"], "status")])
def test_command_typos_suggest_without_accepting(args, suggestion):
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 2
    assert f"Did you mean '{suggestion}'?" in result.stderr


def test_no_suggestion_for_unrelated_command():
    result = CliRunner().invoke(cli, ["unrelated-nonsense"])
    assert result.exit_code == 2
    assert "Did you mean" not in result.stderr


@pytest.mark.parametrize("path", [("run",), ("configure",), ("macro", "replay")])
def test_provider_help_wraps_and_configure_only_lists_supported(path):
    result = CliRunner().invoke(cli, [*path, "--help"])
    assert "-p, --provider PROVIDER" in result.stdout
    assert max(map(len, result.stdout.splitlines())) < 100
    if path == ("configure",):
        assert "openrouter" not in result.stdout and "deepseek" not in result.stdout
        for name in ("openrouter", "deepseek"):
            assert CliRunner().invoke(cli, ["configure", "-p", name]).exit_code == 2


def test_login_and_connection_usage():
    runner = CliRunner()
    assert "[SERVICE]" in runner.invoke(cli, ["auth", "login", "--help"]).stdout
    for command in ("connect", "disconnect"):
        assert "HOST:PORT" in runner.invoke(cli, ["device", command, "--help"]).stdout
    text = runner.invoke(cli, ["auth", "login", "--help"]).stdout
    assert "[default: 300.0]" in text
    assert "(openai, anthropic, gemini)" in " ".join(text.split())


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
def test_doctor_preserves_checks_once(monkeypatch, failed, json_output):
    import json

    doctor = importlib.import_module("mobilerun.cli.doctor_command")
    from mobilerun.cli.doctor import CheckResult, Status, console

    async def check(*args, **kwargs):
        console.print("Device: duplicate progress")
        console.print("Device: duplicate summary")
        return [CheckResult("Device", Status.FAIL if failed else Status.PASS, "observed", "detail")]

    monkeypatch.setattr(doctor, "run_doctor", check)
    result = CliRunner().invoke(cli, ["doctor", *(["-j"] if json_output else [])])
    assert result.exit_code == int(failed), result.output
    assert "duplicate" not in result.output
    assert "Error: 1" not in result.output
    if json_output:
        assert json.loads(result.stdout) == {
            "ok": not failed, "checks": [{"name": "Device", "status": "fail" if failed else "pass",
                                           "detail": "observed; detail"}],
        }
    else:
        assert result.output.count("Device:") == 1
        assert "observed; detail" in result.stdout


@pytest.mark.parametrize("message,hint", [
    ("device 'missing' not found", "mobilerun device list"),
    ("No connected Android devices found.", "mobilerun device pair"),
])
def test_local_error_next_step(monkeypatch, message, hint):
    from mobilerun.cli import device_lifecycle

    monkeypatch.setattr(device_lifecycle, "probe_device", Mock(side_effect=RuntimeError(message)))
    result = CliRunner().invoke(cli, ["device", "ping", "-d", "missing"])
    assert result.exit_code == 1
    assert hint in result.stderr
    if "No connected" in message:
        assert "USB" in result.stderr and "-t cloud" in result.stderr


@pytest.mark.parametrize("tty", [False, True])
def test_rows_only_include_terminal_header(tty):
    from io import StringIO

    from mobilerun.cli.options import emit_rows

    class Output(StringIO):
        def isatty(self):
            return tty

    output = Output()
    with click.Context(cli, obj={"stdout": output}):
        emit_rows(("NAME", "DETAIL"), [("first", "line\nbreak\tcolumn"), ("last", None)])
    expected = "first\tline break column\nlast\t\n"
    assert output.getvalue() == ("NAME\tDETAIL\n" if tty else "") + expected


@pytest.mark.parametrize("local_failure,cloud_failure", [(False, False), (False, True),
                                                         (True, False), (True, True)])
@pytest.mark.parametrize("json_output", [False, True])
def test_all_devices_retains_partial_results_and_closes(monkeypatch, local_failure, cloud_failure, json_output):
    import json
    from types import SimpleNamespace

    from mobilerun.cli import device_lifecycle

    clients, devices = [], []
    calls = []

    def factory(url, *, cloud):
        scope = "cloud" if cloud else "local"
        failure = cloud_failure if cloud else local_failure
        device = SimpleNamespace(device_id=scope, _conn=object(), capabilities={"backend": scope}, close=Mock())
        devices.append(device)

        def listing(**kwargs):
            calls.append(kwargs["scope"])
            if failure:
                raise RuntimeError(scope + " unavailable")
            return [device]

        client = SimpleNamespace(list_devices=listing, close=Mock())
        clients.append(client)
        return client

    monkeypatch.setattr(device_lifecycle, "core_client", factory)
    result = CliRunner().invoke(cli, ["device", "list", "--all", *(["-j"] if json_output else [])])
    assert result.exit_code == int(local_failure and cloud_failure), result.output
    assert calls == ["local", "cloud"]
    assert "Error: 1" not in result.output
    expected = [target for target, failed in (("local", local_failure), ("cloud", cloud_failure)) if not failed]
    if json_output:
        data = json.loads(result.stdout)
        assert [row["device_id"] for row in data["devices"]] == expected
        assert data["errors"] == [{"target": target, "error": target + " unavailable"}
                                  for target, failed in (("local", local_failure), ("cloud", cloud_failure)) if failed]
    else:
        assert [line.split("\t")[0] for line in result.stdout.splitlines()] == expected
    assert result.stderr.count("Warning: cloud:") == int(cloud_failure)
    assert result.stderr.count("Warning: local:") == int(local_failure)
    for client in clients:
        client.close.assert_called_once()
    for device, failed in zip(devices, (local_failure, cloud_failure), strict=True):
        assert device.close.call_count == int(not failed)


@pytest.mark.parametrize("json_output", [False, True])
def test_all_devices_with_no_key_and_empty_local_is_success(monkeypatch, json_output):
    import json
    from types import SimpleNamespace

    from mobilerun.cli import device_lifecycle, selection

    local_client = SimpleNamespace(list_devices=Mock(return_value=[]), close=Mock())
    monkeypatch.setattr(selection, "resolve_cloud_api_key", lambda: None)

    def factory(url, *, cloud):
        if cloud:
            selection.require_cloud_api_key()
        return local_client

    monkeypatch.setattr(device_lifecycle, "core_client", factory)
    result = CliRunner().invoke(cli, ["device", "list", "--all", *(["-j"] if json_output else [])])
    assert result.exit_code == 0, result.output
    assert result.stderr.count("Warning: cloud:") == 1
    if json_output:
        data = json.loads(result.stdout)
        assert data["devices"] == []
        assert "No cloud API key" in data["errors"][0]["error"]
    else:
        assert "USB" in result.stderr and "mobilerun device pair" in result.stderr and "-t cloud" in result.stderr


def test_connection_hint_preserves_port_and_hides_url_secrets():
    import httpx
    from mobilerun_sdk import APIConnectionError

    from mobilerun.cli.options import convert_api_error

    request = httpx.Request("GET", "http://user:password@[::1]:9123/v1/devices?token=hidden")
    text, _, _ = convert_api_error(APIConnectionError(request=request))
    assert "http://[::1]:9123" in text
    assert "--api-url" in text and "MOBILERUN_API_BASE_URL" in text
    assert all(word not in text for word in ("password", "hidden", "user:"))


@pytest.mark.parametrize("app_field", ["package_name", "app_id", "bundle_id", "package"])
def test_apps_are_tsv(monkeypatch, app_field):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from mobilerun.cli import device_commands

    @contextmanager
    def driver(*args, **kwargs):
        yield SimpleNamespace(list_apps=lambda **kw: [
            {app_field: "example.app", "label": "Example", "version_name": "1.0",
             "version_code": 1, "is_system_app": False},
        ])

    monkeypatch.setattr(device_commands, "device_driver", driver)
    result = CliRunner().invoke(cli, ["device", "apps"])
    assert result.exit_code == 0, result.output
    assert result.stdout == "example.app\tExample\t1.0\t1\tfalse\n"



def test_macro_list_is_tsv(tmp_path):
    import json

    from mobilerun.macro.state import MACRO_SCHEMA_VERSION

    folder = tmp_path / "sample"
    folder.mkdir()
    (folder / "macro.json").write_text(json.dumps({"macro_schema_version": MACRO_SCHEMA_VERSION, "description": "Example macro", "total_actions": 3}))
    result = CliRunner().invoke(cli, ["macro", "list", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert result.stdout == "sample\tExample macro\t3\n"


def test_env_text_uses_empty_values_lowercase_booleans_and_one_secret_status(monkeypatch):
    from mobilerun.cli import deviceauth
    from mobilerun.config_manager import env_keys

    monkeypatch.setattr(deviceauth, "load_credentials", lambda: None)
    monkeypatch.setattr(env_keys, "resolve_env_key", lambda *args: None)
    monkeypatch.setenv("MOBILERUN_ANDROID_PORTAL_TOKEN", "never-print-marker")
    monkeypatch.setenv("MOBILERUN_STREAM_SCREENSHOTS", "YES")
    monkeypatch.setenv("MOBILERUN_OAUTH_MANUAL", "0")
    monkeypatch.delenv("ANDROID_SERIAL", raising=False)
    result = CliRunner().invoke(cli, ["env"])
    assert result.exit_code == 0, result.output
    assert "MOBILERUN_DEVICE=  (not set)\n" in result.stdout
    assert "MOBILERUN_DEBUG=false  (default)\n" in result.stdout
    assert "MOBILERUN_STREAM_SCREENSHOTS=true  (env)\n" in result.stdout
    assert "MOBILERUN_OAUTH_MANUAL=false  (env)\n" in result.stdout
    assert "MOBILERUN_IOS_PORTAL_TOKEN=not set\n" in result.stdout
    assert "MOBILERUN_ANDROID_PORTAL_TOKEN=set  (env)\n" in result.stdout
    assert "never-print-marker" not in result.output
    assert "None" not in result.stdout and "False" not in result.stdout


@pytest.mark.parametrize("tty,no_input", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("flags,missing", [([], "-p / --provider"),
                                         (["-p", "openai"], "-m / --model"),
                                         (["-p", "openai", "-m", "MODEL"], "-k / --api-key"),
                                         (["-p", "openai-like", "-m", "MODEL", "-k", "dummy"], "-u / --llm-url")])
def test_configure_without_input_fails_before_loading_or_writing(monkeypatch, tty, no_input, flags, missing, tmp_path):
    from mobilerun.config_manager import env_keys

    configure = importlib.import_module("mobilerun.cli.configure")
    monkeypatch.setattr(configure, "input_allowed", lambda disabled: tty and not disabled)
    monkeypatch.setattr(env_keys, "resolve_env_key", lambda *args: "")
    monkeypatch.setattr(ConfigLoader, "load", Mock(side_effect=AssertionError("must not load or create config")))
    wizard = Mock(side_effect=AssertionError("must not prompt"))
    monkeypatch.setattr(configure, "run_configure_wizard", wizard)
    result = CliRunner().invoke(cli, ["configure", *flags, *(["--no-input"] if no_input else [])])
    assert result.exit_code == 2, result.output
    assert missing in result.stderr
    wizard.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("stdin", ["dummy-marker\nsecond-line\n", "dummy-marker\r\n", "dummy-marker"])
@pytest.mark.parametrize("command", ["configure", "token"])
def test_secret_from_stdin_is_one_line_and_never_printed(monkeypatch, stdin, command, tmp_path):
    from mobilerun.cli import auth

    secret = "dummy-marker"
    if command == "configure":
        module = importlib.import_module("mobilerun.cli.configure")
        callback = Mock()
        monkeypatch.setattr(module, "run_configure_wizard", callback)
        args = ["configure", "-p", "openai", "-m", "MODEL", "-k", "-", "--no-input", "-v"]
    else:
        callback = Mock()
        monkeypatch.setattr(auth, "save_anthropic_setup_token", callback)
        args = ["auth", "login", "anthropic", "--token", "-", "--credential-path", str(tmp_path / "auth.json"),
                "--no-input", "-v"]
    result = CliRunner().invoke(cli, args, input=stdin)
    assert result.exit_code == 0, result.output
    assert secret not in result.output and "second-line" not in result.output
    if command == "configure":
        assert callback.call_args.kwargs["api_key"] == secret
    else:
        assert callback.call_args.args[1] == secret


@pytest.mark.parametrize("stdin", ["", "\n", " \n"])
@pytest.mark.parametrize("args", [["configure", "-k", "-"], ["auth", "login", "anthropic", "--token", "-"]])
def test_empty_secret_stdin_is_usage_error(args, stdin):
    result = CliRunner().invoke(cli, args, input=stdin)
    assert result.exit_code == 2, result.output
    assert "stdin is empty" in result.stderr


@pytest.mark.parametrize("from_stdin", [False, True])
@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("command", ["configure", "token"])
def test_secrets_are_redacted_from_errors_and_verbose_logs(monkeypatch, from_stdin, verbose, command, tmp_path):
    import logging

    from mobilerun.cli import auth

    secret = "dummy-error-marker"

    def fail(*args, **kwargs):
        logging.getLogger("mobilerun").error("failure containing %s", secret)
        raise RuntimeError("upstream failure containing " + secret)

    if command == "configure":
        module = importlib.import_module("mobilerun.cli.configure")
        monkeypatch.setattr(module, "run_configure_wizard", fail)
        args = ["configure", "-p", "openai", "-m", "MODEL", "-k", "-" if from_stdin else secret, "--no-input"]
    else:
        monkeypatch.setattr(auth, "save_anthropic_setup_token", fail)
        args = ["auth", "login", "anthropic", "--token", "-" if from_stdin else secret,
                "--credential-path", str(tmp_path / "auth.json"), "--no-input"]
    result = CliRunner().invoke(cli, [*args, *(["-v"] if verbose else [])], input=secret + "\n" if from_stdin else None)
    assert result.exit_code == 1, result.output
    assert secret not in result.output
    assert "[redacted]" in result.stderr
    assert ("Traceback" in result.stderr) == verbose


@pytest.mark.parametrize("service,needs_input", [("mobilerun", False), ("openai", False), ("xai", False),
                                               ("anthropic", True), ("gemini", True)])
def test_login_only_blocks_flows_that_need_terminal_input(monkeypatch, service, needs_input):
    from mobilerun.cli import auth

    monkeypatch.setattr(auth, "login_requires_input", lambda name: name in {"anthropic", "gemini"})
    callbacks = []
    for name in ("_login_mobilerun", "_run_openai_oauth_login", "_run_grok_oauth_login",
                 "_run_anthropic_oauth_login", "_run_gemini_oauth_login"):
        callback = Mock()
        monkeypatch.setattr(auth, name, callback)
        callbacks.append(callback)
    result = CliRunner().invoke(cli, ["auth", "login", service, "--no-input"])
    assert result.exit_code == (2 if needs_input else 0), result.output
    assert sum(callback.call_count for callback in callbacks) == int(not needs_input)
    if service == "anthropic":
        assert "--token" in result.stderr
    if service == "gemini":
        assert "terminal" in result.stderr and "--no-input" in result.stderr


@pytest.mark.parametrize("service", ["anthropic", "gemini"])
def test_provider_callback_login_without_manual_prompt_is_allowed(monkeypatch, service):
    from mobilerun.cli import auth

    monkeypatch.setattr(auth, "login_requires_input", lambda _: False)
    callback = Mock()
    monkeypatch.setattr(auth, f"_run_{service}_oauth_login", callback)
    result = CliRunner().invoke(cli, ["auth", "login", service, "--no-input"])
    assert result.exit_code == 0, result.output
    callback.assert_called_once()


@pytest.mark.parametrize("failure,code", [(EOFError, 1), (KeyboardInterrupt, 130)])
def test_prompt_eof_and_interrupt_have_distinct_exit_codes(monkeypatch, failure, code):
    configure = importlib.import_module("mobilerun.cli.configure")
    monkeypatch.setattr(configure, "input_allowed", lambda _: True)
    monkeypatch.setattr(configure, "run_configure_wizard", Mock(side_effect=failure))
    result = CliRunner().invoke(cli, ["configure"])
    assert result.exit_code == code, result.output
    if code == 1:
        assert result.stderr == "Error: Input closed; run the command again in a terminal.\n"


@pytest.mark.parametrize("cause,code", [(EOFError, 1), (KeyboardInterrupt, 130)])
def test_click_prompt_abort_cause_preserves_exit_code(monkeypatch, cause, code):
    configure = importlib.import_module("mobilerun.cli.configure")
    monkeypatch.setattr(configure, "input_allowed", lambda _: True)

    def abort(*args, **kwargs):
        try:
            raise cause
        except cause as exc:
            raise click.Abort from exc

    monkeypatch.setattr(configure, "run_configure_wizard", abort)
    result = CliRunner().invoke(cli, ["configure"])
    assert result.exit_code == code, result.output


def test_no_input_configure_resolves_provider_environment_key(monkeypatch):
    configure = importlib.import_module("mobilerun.cli.configure")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-environment-marker")
    wizard = Mock()
    monkeypatch.setattr(configure, "run_configure_wizard", wizard)
    result = CliRunner().invoke(cli, ["configure", "-p", "openai", "-m", "MODEL", "--no-input"])
    assert result.exit_code == 0, result.output
    assert "dummy-environment-marker" not in result.output
    assert wizard.call_args.kwargs["api_key"] is None


def test_no_input_configure_never_starts_missing_oauth_login(monkeypatch):
    configure = importlib.import_module("mobilerun.cli.configure")
    monkeypatch.setattr(configure, "_oauth_credentials_present", lambda *args: False)
    wizard = Mock()
    monkeypatch.setattr(configure, "run_configure_wizard", wizard)
    result = CliRunner().invoke(cli, ["configure", "-p", "openai-oauth", "-m", "MODEL", "--no-input"])
    assert result.exit_code == 2, result.output
    assert "mobilerun auth login openai" in result.stderr
    wizard.assert_not_called()


def test_click_prompt_eof_is_one_failure_sentence(monkeypatch):
    configure = importlib.import_module("mobilerun.cli.configure")
    monkeypatch.setattr(configure, "input_allowed", lambda _: True)
    monkeypatch.setattr(configure, "run_configure_wizard", lambda *args, **kwargs: click.prompt("Value"))
    result = CliRunner().invoke(cli, ["configure"], input="")
    assert result.exit_code == 1, result.output
    assert result.stderr.count("Input closed; run the command again in a terminal.") == 1
    assert "Aborted!" not in result.output


def test_manual_login_detection_reuses_provider_flow_selection(monkeypatch):
    from mobilerun.cli import oauth_actions

    monkeypatch.setattr(oauth_actions, "anthropic_headless", lambda: False)
    monkeypatch.setattr(oauth_actions, "gemini_headless", lambda: False)
    monkeypatch.delenv("MOBILERUN_OAUTH_MANUAL", raising=False)
    assert not oauth_actions.login_requires_input("gemini")
    monkeypatch.setenv("MOBILERUN_OAUTH_MANUAL", "true")
    assert oauth_actions.login_requires_input("gemini")
    assert oauth_actions.login_requires_input("anthropic")
    assert not oauth_actions.login_requires_input("openai")
    monkeypatch.delenv("MOBILERUN_OAUTH_MANUAL")
    monkeypatch.setattr(oauth_actions, "gemini_headless", lambda: True)
    assert oauth_actions.login_requires_input("gemini")
