"""User-facing help, output, and automation contracts for the CLI."""

import importlib
import json
import logging
from contextlib import contextmanager
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import click
import httpx
import pytest
from click.shell_completion import BashComplete, get_completion_class
from click.testing import CliRunner
from mobilerun_sdk import APIConnectionError, APIStatusError

from mobilerun.cli import (
    auth,
    device_commands,
    device_lifecycle,
    deviceauth,
    oauth_actions,
    selection,
)
from mobilerun.cli.doctor import CheckResult, Status, console
from mobilerun.cli.main import cli
from mobilerun.cli.options import convert_api_error, emit_rows
from mobilerun.config_manager import ConfigLoader, env_keys
from mobilerun.macro.state import MACRO_SCHEMA_VERSION

run_module = importlib.import_module("mobilerun.cli.run")
configure = importlib.import_module("mobilerun.cli.configure")
doctor_command = importlib.import_module("mobilerun.cli.doctor_command")


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


@pytest.fixture
def allow_input(monkeypatch):
    def patch(tty=True):
        monkeypatch.setattr(configure, "input_allowed", lambda disabled: tty and not disabled)

    patch()
    return patch


@pytest.fixture(params=["configure", "token"])
def secret_command(request, monkeypatch, tmp_path):
    callback = Mock()
    if request.param == "configure":
        monkeypatch.setattr(configure, "run_configure_wizard", callback)
        args = ["configure", "-p", "openai", "-m", "MODEL", "-k"]

        def value():
            return callback.call_args.kwargs["api_key"]
    else:
        monkeypatch.setattr(auth, "save_anthropic_setup_token", callback)
        args = [
            "auth",
            "login",
            "anthropic",
            "--credential-path",
            str(tmp_path / "auth.json"),
            "--token",
        ]

        def value():
            return callback.call_args.args[1]

    return args, callback, value


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("run",),
        ("device",),
        ("device", "list"),
        ("device", "ui"),
        ("device", "pair"),
        ("auth", "login"),
        ("configure",),
        ("macro", "replay"),
    ],
)
def test_help_has_copyable_examples(path):
    result = CliRunner().invoke(cli, [*path, "--help"])
    assert result.exit_code == 0, result.output
    assert "Examples:\n    mobilerun " in result.stdout
    assert "\n    mobilerun " in result.stdout
    if not path:
        assert result.stdout.rstrip().endswith(
            "Docs: https://docs.mobilerun.ai/framework/guides/cli"
        )


@pytest.mark.parametrize("path,command", list(commands()))
def test_help_options_and_summaries_are_complete(path, command):
    with click.Context(command, **command.context_settings) as ctx:
        params = command.get_params(ctx)
        for param in params:
            if isinstance(param, click.Option):
                assert param.help and param.help.endswith("."), (path, param.name)
                assert param.show_default is not False, (path, param.name)
        names = [param.name for param in params]
        shared_first = [
            name for name in ("device", "target", "platform", "config_path") if name in names
        ]
        shared_last = [
            name for name in ("api_url", "json_output", "verbose", "help") if name in names
        ]
        assert names[: len(shared_first)] == shared_first
        assert names[-len(shared_last) :] == shared_last
    assert command.get_short_help_str(limit=1).endswith("."), path
    assert not command.get_short_help_str(limit=1).endswith("..."), path


def test_root_command_order_and_agent_discovery(monkeypatch):
    monkeypatch.setattr(
        run_module, "list_agents", Mock(side_effect=AssertionError("root must not scan"))
    )
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    names = ["run", "device", "configure", "auth", "doctor", "env", "macro", "completion"]
    assert [result.stdout.index(f"  {name} ") for name in names] == sorted(
        result.stdout.index(f"  {name} ") for name in names
    )


def test_agent_option_hidden_without_installed_agents(monkeypatch):
    monkeypatch.setattr(run_module, "list_agents", lambda: [])
    assert "--agent" not in CliRunner().invoke(cli, ["run", "--help"]).stdout


def test_agent_option_rejected_without_installed_agents(monkeypatch):
    monkeypatch.setattr(run_module, "list_agents", lambda: [])
    result = CliRunner().invoke(cli, ["run", "TASK", "-a", "missing"])
    assert result.exit_code == 2
    assert "No external agents are installed; install one before using --agent." in result.stderr


def test_installed_agent_is_listed_and_accepted(monkeypatch):
    monkeypatch.setattr(run_module, "list_agents", lambda: ["installed"])
    runner = CliRunner()
    assert "--agent [installed]" in runner.invoke(cli, ["run", "--help"]).stdout
    callback = AsyncMock(return_value=(True, "done", 1))
    monkeypatch.setattr(run_module, "_run_task", callback)
    assert runner.invoke(cli, ["run", "TASK", "-a", "installed"]).exit_code == 0
    assert callback.await_args.kwargs["agent"] == "installed"


def test_unknown_agent_is_rejected(monkeypatch):
    monkeypatch.setattr(run_module, "list_agents", lambda: ["installed"])
    assert CliRunner().invoke(cli, ["run", "TASK", "-a", "missing"]).exit_code == 2


@pytest.mark.parametrize(
    "args,suggestion",
    [
        (["devcie"], "mobilerun device"),
        (["device", "lst"], "mobilerun device list"),
        (["auth", "stauts"], "mobilerun auth status"),
    ],
)
def test_command_typos_suggest_without_accepting(args, suggestion):
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 2
    assert f"Did you mean '{suggestion}'?" in result.stderr


def test_no_suggestion_for_unrelated_command():
    result = CliRunner().invoke(cli, ["unrelated-nonsense"])
    assert result.exit_code == 2
    assert "Did you mean" not in result.stderr


@pytest.mark.parametrize("path", [("run",), ("configure",), ("macro", "replay")])
def test_provider_help_wraps(path):
    result = CliRunner().invoke(cli, [*path, "--help"])
    assert "-p, --provider PROVIDER" in result.stdout
    assert max(map(len, result.stdout.splitlines())) < 100


@pytest.mark.parametrize("name", ["openrouter", "deepseek"])
def test_configure_only_lists_supported_providers(name):
    result = CliRunner().invoke(cli, ["configure", "--help"])
    assert name not in result.stdout
    assert CliRunner().invoke(cli, ["configure", "-p", name]).exit_code == 2


def test_login_usage_and_service_options():
    runner = CliRunner()
    assert "[SERVICE]" in runner.invoke(cli, ["auth", "login", "--help"]).stdout
    text = runner.invoke(cli, ["auth", "login", "--help"]).stdout
    assert "[default: 300.0]" in text
    assert "(openai, anthropic, gemini)" in " ".join(text.split())


@pytest.mark.parametrize("command", ["connect", "disconnect"])
def test_connection_usage(command):
    result = CliRunner().invoke(cli, ["device", command, "--help"])
    assert "HOST:PORT" in result.stdout


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
def test_doctor_preserves_checks_once(monkeypatch, failed, json_output):

    async def check(*args, **kwargs):
        assert kwargs["render_results"] is False
        console.print("Please enable accessibility manually")
        return [CheckResult("Device", Status.FAIL if failed else Status.PASS, "observed", "detail")]

    monkeypatch.setattr(doctor_command, "run_doctor", check)
    result = CliRunner().invoke(cli, ["doctor", *(["-j"] if json_output else [])])
    assert result.exit_code == int(failed), result.output
    assert "Please enable accessibility manually" in result.stderr
    assert "Error: 1" not in result.output
    if json_output:
        assert json.loads(result.stdout) == {
            "ok": not failed,
            "checks": [
                {
                    "name": "Device",
                    "status": "fail" if failed else "pass",
                    "detail": "observed; detail",
                }
            ],
        }
    else:
        assert result.output.count("Device:") == 1
        assert "observed; detail" in result.stdout


@pytest.mark.parametrize(
    "message,hint",
    [
        ("device 'missing' not found", "mobilerun device list"),
        ("No connected Android devices found.", "mobilerun device pair"),
    ],
)
def test_local_error_next_step(monkeypatch, message, hint):

    monkeypatch.setattr(device_lifecycle, "probe_device", Mock(side_effect=RuntimeError(message)))
    result = CliRunner().invoke(cli, ["device", "ping", "-d", "missing"])
    assert result.exit_code == 1
    assert hint in result.stderr
    if "No connected" in message:
        assert "USB" in result.stderr and "-t cloud" in result.stderr


@pytest.mark.parametrize("tty", [False, True])
def test_rows_only_include_terminal_header(tty):

    class Output(StringIO):
        def isatty(self):
            return tty

    output = Output()
    with click.Context(cli, obj={"stdout": output}):
        emit_rows(
            ("NAME", "DETAIL"), [("first", "line\nbreak\tcolumn\x1b\u2028\u200b"), ("last", None)]
        )
    expected = "first\tline break column   \nlast\t\n"
    assert output.getvalue() == ("NAME\tDETAIL\n" if tty else "") + expected


@pytest.fixture(
    params=[
        (False, False, 0, ["local", "cloud"], [], "", [1, 1]),
        (
            False,
            True,
            0,
            ["local"],
            [{"target": "cloud", "error": "cloud unavailable"}],
            "Warning: cloud: cloud unavailable\n",
            [1, 0],
        ),
        (
            True,
            False,
            0,
            ["cloud"],
            [{"target": "local", "error": "local unavailable"}],
            "Warning: local: local unavailable\n",
            [0, 1],
        ),
        (
            True,
            True,
            1,
            [],
            [
                {"target": "local", "error": "local unavailable"},
                {"target": "cloud", "error": "cloud unavailable"},
            ],
            "Warning: local: local unavailable\nWarning: cloud: cloud unavailable\n",
            [0, 0],
        ),
    ]
)
def all_devices_case(request, monkeypatch):
    local_failure, cloud_failure, code, ids, errors, warnings, close_counts = request.param
    clients, devices, calls = [], [], []

    def factory(url, *, cloud):
        scope = "cloud" if cloud else "local"
        failure = cloud_failure if cloud else local_failure
        device = SimpleNamespace(
            device_id=scope, _conn=object(), capabilities={"backend": scope}, close=Mock()
        )
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
    yield code, ids, errors, warnings
    assert calls == ["local", "cloud"]
    for client in clients:
        client.close.assert_called_once()
    assert [device.close.call_count for device in devices] == close_counts


def test_all_devices_text_retains_partial_results(all_devices_case):
    code, ids, _, warnings = all_devices_case
    result = CliRunner().invoke(cli, ["device", "list", "--all"])
    assert result.exit_code == code, result.output
    assert [line.split("\t")[0] for line in result.stdout.splitlines()] == ids
    assert result.stderr == warnings


def test_all_devices_json_retains_partial_results(all_devices_case):
    code, ids, errors, warnings = all_devices_case
    result = CliRunner().invoke(cli, ["device", "list", "--all", "-j"])
    assert result.exit_code == code, result.output
    data = json.loads(result.stdout)
    assert [row["device_id"] for row in data["devices"]] == ids
    assert data["errors"] == errors
    assert result.stderr == warnings


@pytest.fixture
def empty_local_without_cloud_key(monkeypatch):
    local_client = SimpleNamespace(list_devices=Mock(return_value=[]), close=Mock())
    monkeypatch.setattr(selection, "resolve_cloud_api_key", lambda: None)

    def factory(url, *, cloud):
        if cloud:
            selection.require_cloud_api_key()
        return local_client

    monkeypatch.setattr(device_lifecycle, "core_client", factory)


def test_all_devices_text_with_no_key_and_empty_local_is_success(empty_local_without_cloud_key):
    result = CliRunner().invoke(cli, ["device", "list", "--all"])
    assert result.exit_code == 0, result.output
    assert result.stderr.count("Warning: cloud:") == 1
    assert (
        "USB" in result.stderr
        and "mobilerun device pair" in result.stderr
        and "-t cloud" in result.stderr
    )


def test_all_devices_json_with_no_key_and_empty_local_is_success(empty_local_without_cloud_key):
    result = CliRunner().invoke(cli, ["device", "list", "--all", "-j"])
    assert result.exit_code == 0, result.output
    assert result.stderr.count("Warning: cloud:") == 1
    data = json.loads(result.stdout)
    assert data["devices"] == []
    assert "No cloud API key" in data["errors"][0]["error"]


def test_connection_hint_preserves_port_and_hides_url_secrets():

    request = httpx.Request("GET", "http://user:password@[::1]:9123/v1/devices?token=hidden")
    text, _, _ = convert_api_error(APIConnectionError(request=request))
    assert "http://[::1]:9123" in text
    assert "--api-url" in text and "MOBILERUN_API_BASE_URL" in text
    assert all(word not in text for word in ("password", "hidden", "user:"))


@pytest.mark.parametrize("app_field", ["package_name", "app_id", "bundle_id", "package"])
def test_apps_are_tsv(monkeypatch, app_field):

    @contextmanager
    def driver(*args, **kwargs):
        yield SimpleNamespace(
            list_apps=lambda **kw: [
                {
                    app_field: "example.app",
                    "label": "Example",
                    "version_name": "1.0",
                    "version_code": 1,
                    "is_system_app": False,
                },
            ]
        )

    monkeypatch.setattr(device_commands, "device_driver", driver)
    result = CliRunner().invoke(cli, ["device", "apps"])
    assert result.exit_code == 0, result.output
    assert result.stdout == "example.app\tExample\t1.0\t1\tfalse\n"


def test_macro_list_is_tsv(tmp_path):

    folder = tmp_path / "sample"
    folder.mkdir()
    (folder / "macro.json").write_text(
        json.dumps(
            {
                "macro_schema_version": MACRO_SCHEMA_VERSION,
                "description": "Example macro",
                "total_actions": 3,
            }
        )
    )
    result = CliRunner().invoke(cli, ["macro", "list", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert result.stdout == "sample\tExample macro\t3\n"


def test_env_text_uses_empty_values_lowercase_booleans_and_one_secret_status(monkeypatch):

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
@pytest.mark.parametrize(
    "flags,missing",
    [
        ([], "-p / --provider"),
        (["-p", "openai"], "-m / --model"),
        (["-p", "openai", "-m", "MODEL"], "-k / --api-key"),
        (["-p", "openai-like", "-m", "MODEL", "-k", "dummy"], "-u / --llm-url"),
    ],
)
def test_configure_without_input_fails_before_loading_or_writing(
    monkeypatch,
    allow_input,
    tty,
    no_input,
    flags,
    missing,
    tmp_path,
):

    allow_input(tty)
    monkeypatch.setattr(env_keys, "resolve_env_key", lambda *args: "")
    monkeypatch.setattr(
        ConfigLoader, "load", Mock(side_effect=AssertionError("must not load or create config"))
    )
    wizard = Mock(side_effect=AssertionError("must not prompt"))
    monkeypatch.setattr(configure, "run_configure_wizard", wizard)
    result = CliRunner().invoke(cli, ["configure", *flags, *(["--no-input"] if no_input else [])])
    assert result.exit_code == 2, result.output
    assert missing in result.stderr
    wizard.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "stdin", ["dummy-marker\nsecond-line\n", "dummy-marker\r\n", "dummy-marker"]
)
def test_secret_from_stdin_is_one_line_and_never_printed(secret_command, stdin):
    args, callback, value = secret_command
    result = CliRunner().invoke(cli, [*args, "-", "--no-input", "-v"], input=stdin)
    assert result.exit_code == 0, result.output
    assert "dummy-marker" not in result.output and "second-line" not in result.output
    callback.assert_called_once()
    assert value() == "dummy-marker"


@pytest.mark.parametrize("stdin", ["", "\n", " \n"])
def test_empty_secret_stdin_is_usage_error(secret_command, stdin):
    args, callback, _ = secret_command
    result = CliRunner().invoke(cli, [*args, "-"], input=stdin)
    assert result.exit_code == 2, result.output
    assert "stdin is empty" in result.stderr
    callback.assert_not_called()


@pytest.mark.parametrize("from_stdin", [False, True])
@pytest.mark.parametrize("verbose", [False, True])
def test_secrets_are_redacted_from_errors_and_verbose_logs(secret_command, from_stdin, verbose):
    args, callback, _ = secret_command
    secret = "dummy-error-marker"

    def fail(*args, **kwargs):
        logging.getLogger("mobilerun").error("failure containing %s", secret)
        raise RuntimeError("upstream failure containing " + secret)

    callback.side_effect = fail
    result = CliRunner().invoke(
        cli,
        [*args, "-" if from_stdin else secret, "--no-input", *(["-v"] if verbose else [])],
        input=secret + "\n" if from_stdin else None,
    )
    assert result.exit_code == 1, result.output
    assert secret not in result.output
    assert "[redacted]" in result.stderr
    assert ("Traceback" in result.stderr) == verbose


@pytest.mark.parametrize(
    "service,code,calls,hints",
    [
        ("mobilerun", 0, 1, ()),
        ("openai", 0, 1, ()),
        ("xai", 0, 1, ()),
        ("anthropic", 2, 0, ("--token",)),
        ("gemini", 2, 0, ("terminal", "--no-input")),
    ],
)
def test_login_only_blocks_flows_that_need_terminal_input(monkeypatch, service, code, calls, hints):

    monkeypatch.setattr(auth, "login_requires_input", lambda name: name in {"anthropic", "gemini"})
    callbacks = []
    for name in (
        "_login_mobilerun",
        "_run_openai_oauth_login",
        "_run_grok_oauth_login",
        "_run_anthropic_oauth_login",
        "_run_gemini_oauth_login",
    ):
        callback = Mock()
        monkeypatch.setattr(auth, name, callback)
        callbacks.append(callback)
    result = CliRunner().invoke(cli, ["auth", "login", service, "--no-input"])
    assert result.exit_code == code, result.output
    assert sum(callback.call_count for callback in callbacks) == calls
    for hint in hints:
        assert hint in result.stderr


@pytest.mark.parametrize("service", ["anthropic", "gemini"])
def test_provider_callback_login_without_manual_prompt_is_allowed(monkeypatch, service):

    monkeypatch.setattr(auth, "login_requires_input", lambda _: False)
    callback = Mock()
    monkeypatch.setattr(auth, f"_run_{service}_oauth_login", callback)
    result = CliRunner().invoke(cli, ["auth", "login", service, "--no-input"])
    assert result.exit_code == 0, result.output
    callback.assert_called_once()


@pytest.mark.parametrize("failure,code", [(EOFError, 1), (KeyboardInterrupt, 130)])
def test_prompt_eof_and_interrupt_have_distinct_exit_codes(monkeypatch, allow_input, failure, code):
    monkeypatch.setattr(configure, "run_configure_wizard", Mock(side_effect=failure))
    result = CliRunner().invoke(cli, ["configure"])
    assert result.exit_code == code, result.output
    if code == 1:
        assert result.stderr == "Error: Input closed; run the command again in a terminal.\n"


def test_click_prompt_interrupt_exits_130(monkeypatch, allow_input):
    def prompt(*args, **kwargs):
        monkeypatch.setattr(
            click.termui, "visible_prompt_func", Mock(side_effect=KeyboardInterrupt)
        )
        return click.prompt("Value")

    monkeypatch.setattr(configure, "run_configure_wizard", prompt)
    result = CliRunner().invoke(cli, ["configure"])
    assert result.exit_code == 130, result.output
    assert "Input closed" not in result.output


def test_no_input_configure_resolves_provider_environment_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-environment-marker")
    wizard = Mock()
    monkeypatch.setattr(configure, "run_configure_wizard", wizard)
    result = CliRunner().invoke(cli, ["configure", "-p", "openai", "-m", "MODEL", "--no-input"])
    assert result.exit_code == 0, result.output
    assert "dummy-environment-marker" not in result.output
    assert wizard.call_args.kwargs["api_key"] is None


def test_no_input_configure_never_starts_missing_oauth_login(monkeypatch):
    monkeypatch.setattr(configure, "_oauth_credentials_present", lambda *args: False)
    wizard = Mock()
    monkeypatch.setattr(configure, "run_configure_wizard", wizard)
    result = CliRunner().invoke(
        cli, ["configure", "-p", "openai-oauth", "-m", "MODEL", "--no-input"]
    )
    assert result.exit_code == 2, result.output
    assert "mobilerun auth login openai" in result.stderr
    wizard.assert_not_called()


def test_click_prompt_eof_is_one_failure_sentence(monkeypatch, allow_input):
    monkeypatch.setattr(
        configure, "run_configure_wizard", lambda *args, **kwargs: click.prompt("Value")
    )
    result = CliRunner().invoke(cli, ["configure"], input="")
    assert result.exit_code == 1, result.output
    assert result.stderr.count("Input closed; run the command again in a terminal.") == 1
    assert "Aborted!" not in result.output


@pytest.mark.parametrize(
    "service,headless,manual,expected",
    [
        ("gemini", False, "", False),
        ("anthropic", False, "", False),
        ("gemini", True, "", True),
        ("anthropic", True, "", True),
        ("gemini", False, "true", True),
        ("anthropic", False, "1", True),
        ("openai", True, "true", False),
        ("xai", True, "true", False),
        ("mobilerun", True, "true", False),
        ("gemini", False, "0", False),
    ],
)
def test_manual_login_detection(monkeypatch, service, headless, manual, expected):
    monkeypatch.setattr(oauth_actions, "_is_headless_environment", lambda: headless)
    monkeypatch.setenv("MOBILERUN_OAUTH_MANUAL", manual)
    assert oauth_actions.login_requires_input(service) is expected


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_completion_is_click_source_without_installing(shell, tmp_path, monkeypatch):

    monkeypatch.setattr(BashComplete, "_check_version", staticmethod(lambda: None))
    monkeypatch.setenv("MOBILERUN_CONFIG", str(tmp_path / "missing.yaml"))
    result = CliRunner().invoke(cli, ["completion", shell])
    assert result.exit_code == 0, result.output
    expected = get_completion_class(shell)(cli, {}, "mobilerun", "_MOBILERUN_COMPLETE").source()
    assert result.stdout == expected + "\n"
    assert "_MOBILERUN_COMPLETE" in result.stdout
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []


def test_completion_suggests_device_subcommands():
    result = CliRunner().invoke(
        cli,
        [],
        prog_name="mobilerun",
        env={
            "_MOBILERUN_COMPLETE": "bash_complete",
            "COMP_WORDS": "mobilerun device l",
            "COMP_CWORD": "2",
        },
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == ["plain,list", "plain,long-press"]


def test_completion_rejects_unknown_shell():
    result = CliRunner().invoke(cli, ["completion", "unknown"])
    assert result.exit_code == 2, result.output


@pytest.mark.parametrize(
    "names,expected", [([], []), (["installed", "other"], ["plain,installed"])]
)
def test_agent_completion_discovers_installed_names(monkeypatch, names, expected):
    monkeypatch.setattr(run_module, "list_agents", lambda: names)
    result = CliRunner().invoke(
        cli,
        [],
        prog_name="mobilerun",
        env={
            "_MOBILERUN_COMPLETE": "bash_complete",
            "COMP_WORDS": "mobilerun run TASK -a i",
            "COMP_CWORD": "4",
        },
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.strip().splitlines() == expected


@pytest.mark.parametrize(
    "words,cword",
    [
        ("mobilerun configure -k - ", "4"),
        ("mobilerun auth login anthropic --token - ", "6"),
    ],
)
def test_completion_does_not_read_secret_stdin(monkeypatch, words, cword):
    reader = Mock(side_effect=AssertionError("completion must not access stdin"))
    monkeypatch.setattr(click, "get_text_stream", reader)
    result = CliRunner().invoke(
        cli,
        [],
        prog_name="mobilerun",
        input="dummy-secret\n",
        env={
            "_MOBILERUN_COMPLETE": "bash_complete",
            "COMP_WORDS": words,
            "COMP_CWORD": cword,
        },
    )
    assert result.exit_code == 0, result.output
    reader.assert_not_called()


def test_terminal_secret_input_is_hidden(monkeypatch, secret_command):
    args, callback, value = secret_command
    stream = Mock()
    stream.isatty.return_value = True
    original_stream = click.get_text_stream
    monkeypatch.setattr(
        click, "get_text_stream", lambda name: stream if name == "stdin" else original_stream(name)
    )
    original_prompt = click.prompt

    def hidden_prompt(*args, **kwargs):
        assert kwargs["hide_input"] is True and kwargs["err"] is True
        monkeypatch.setattr(
            click.termui, "hidden_prompt_func", Mock(return_value="dummy-hidden-marker")
        )
        return original_prompt(*args, **kwargs)

    monkeypatch.setattr(click, "prompt", hidden_prompt)
    result = CliRunner().invoke(cli, [*args, "-", "--no-input"])
    assert result.exit_code == 0, result.output
    callback.assert_called_once()
    stream.readline.assert_not_called()
    assert value() == "dummy-hidden-marker"
    assert "dummy-hidden-marker" not in result.output
    assert ":" in result.stderr


def test_cloud_empty_list_has_cloud_hint(monkeypatch):
    monkeypatch.setattr(device_lifecycle, "_list_target", lambda *a: [])
    result = CliRunner().invoke(cli, ["device", "list", "-t", "cloud"])
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "Create or start a device in Mobilerun Cloud" in result.stderr
    assert "sign-in" in result.stderr
    assert all(text not in result.stderr for text in ("USB", "pair", "-t cloud"))


def test_cloud_missing_device_has_cloud_listing_hint(monkeypatch):
    monkeypatch.setattr(
        device_lifecycle, "probe_device", Mock(side_effect=RuntimeError("device not found"))
    )
    result = CliRunner().invoke(cli, ["device", "ping", "-t", "cloud", "-d", "missing"])
    assert result.exit_code == 1
    assert "`mobilerun device list -t cloud`" in result.stderr


def test_all_devices_api_error_is_short_and_redacted(monkeypatch):
    monkeypatch.setenv("MOBILERUN_CLOUD_API_KEY", "dummy-api-marker")
    request = httpx.Request("GET", "http://127.0.0.1:9/v1/devices")
    response = httpx.Response(404, request=request)
    error = APIStatusError(
        "missing", response=response, body={"detail": "device\n dummy-api-marker not found"}
    )
    monkeypatch.setattr(device_lifecycle, "_list_target", Mock(side_effect=[[], error]))
    result = CliRunner().invoke(cli, ["device", "list", "--all", "-j"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "devices": [],
        "errors": [
            {"target": "cloud", "error": "device [redacted] not found"},
        ],
    }
    assert "dummy-api-marker" not in result.output


def test_env_json_booleans_match_text(monkeypatch):
    monkeypatch.setenv("MOBILERUN_STREAM_SCREENSHOTS", "YES")
    monkeypatch.setenv("MOBILERUN_OAUTH_MANUAL", "0")
    result = CliRunner().invoke(cli, ["env", "-j"])
    assert result.exit_code == 0, result.output
    values = {row["variable"]: row["value"] for row in json.loads(result.stdout)}
    assert values["MOBILERUN_STREAM_SCREENSHOTS"] is True
    assert values["MOBILERUN_OAUTH_MANUAL"] is False


@pytest.mark.parametrize("secret", ["", " "])
def test_terminal_empty_secret_is_usage_error(monkeypatch, secret_command, secret):
    args, callback, _ = secret_command
    monkeypatch.setattr(click, "get_text_stream", lambda name: SimpleNamespace(isatty=lambda: True))
    original_prompt = click.prompt

    def prompt(*args, **kwargs):
        monkeypatch.setattr(click.termui, "hidden_prompt_func", lambda _: secret)
        return original_prompt(*args, **kwargs)

    monkeypatch.setattr(click, "prompt", prompt)
    result = CliRunner().invoke(cli, [*args, "-"])
    assert result.exit_code == 2, result.output
    assert "stdin is empty" in result.stderr
    callback.assert_not_called()


@pytest.mark.parametrize("flags", [[], ["-j"]])
def test_doctor_runner_suppresses_results_and_keeps_instructions(monkeypatch, flags):
    module = importlib.import_module("mobilerun.cli.doctor")
    config = SimpleNamespace(device=SimpleNamespace(serial=None, platform="ios"))
    monkeypatch.setattr(
        module,
        "check_sdk_version",
        AsyncMock(return_value=CheckResult("SDK", Status.PASS, "ready")),
    )
    monkeypatch.setattr(
        module,
        "check_config",
        AsyncMock(
            return_value=(
                CheckResult("Config", Status.PASS, "ready"),
                config,
            )
        ),
    )
    monkeypatch.setattr(
        module, "check_adb", AsyncMock(return_value=CheckResult("ADB", Status.PASS, "ready"))
    )
    monkeypatch.setattr(selection, "select_device", AsyncMock(return_value=object()))
    monkeypatch.setattr(
        device_commands, "probe_device", lambda *a, **kw: console.print("Please enable manually")
    )
    result = CliRunner().invoke(cli, ["doctor", *flags])
    assert result.exit_code == 0, result.output
    assert result.stderr.strip() == "Please enable manually"
    assert "All checks passed" not in result.output
    assert "Mobilerun Doctor" not in result.output


@pytest.mark.parametrize(
    "args,message",
    [
        (["login"], "No such command 'login'. Did you mean 'mobilerun auth login'?"),
        (["logout"], "No such command 'logout'. Did you mean 'mobilerun auth logout'?"),
        (["connect", "X"], "No such command 'connect'. Did you mean 'mobilerun device connect'?"),
        (
            ["disconnect", "X"],
            "No such command 'disconnect'. Did you mean 'mobilerun device disconnect'?",
        ),
        (["setup"], "No such command 'setup'. Did you mean 'mobilerun device setup'?"),
        (["ping"], "No such command 'ping'. Did you mean 'mobilerun device ping'?"),
        (["pair"], "No such command 'pair'. Did you mean 'mobilerun device pair'?"),
        (["devices"], "No such command 'devices'. Did you mean 'mobilerun device list'?"),
        (["whoami"], "No such command 'whoami'. Did you mean 'mobilerun auth status'?"),
        (["lst"], "No such command 'lst'. Did you mean 'mobilerun device list'?"),
        (["devcie"], "No such command 'devcie'. Did you mean 'mobilerun device'?"),
        (["device", "status"], "No such command 'status'. Did you mean 'mobilerun auth status'?"),
        (["auth", "lst"], "No such command 'lst'. Did you mean 'mobilerun device list'?"),
        (["login", "-h"], "No such command 'login'. Did you mean 'mobilerun auth login'?"),
        (["login", "auth"], "No such command 'login'. Did you mean 'mobilerun auth login'?"),
        (
            ["list"],
            "No such command 'list'. Did you mean 'mobilerun device list', 'mobilerun macro list'?",
        ),
        (
            ["configure", "openai"],
            "Got unexpected extra argument (openai). Use 'mobilerun auth login openai' to sign in.",
        ),
    ],
)
def test_unknown_commands_suggest_paths_without_execution(monkeypatch, args, message):
    callbacks = []
    for _, command in commands():
        if not isinstance(command, click.Group):
            callback = Mock(side_effect=AssertionError("must not execute a command"))
            monkeypatch.setattr(command, "callback", callback)
            callbacks.append(callback)
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 2, result.output
    assert result.stdout == ""
    assert result.stderr.splitlines()[-1] == "Error: " + message
    assert result.stderr.count("Did you mean") <= 1
    for callback in callbacks:
        callback.assert_not_called()


@pytest.mark.parametrize(
    "words,cword",
    [
        ("mobilerun login ", "2"),
        ("mobilerun device status ", "3"),
        ("mobilerun devices ", "2"),
        ("mobilerun configure openai ", "3"),
    ],
)
def test_completion_never_suggests_command_corrections(words, cword):
    result = CliRunner().invoke(
        cli,
        [],
        prog_name="mobilerun",
        env={
            "_MOBILERUN_COMPLETE": "bash_complete",
            "COMP_WORDS": words,
            "COMP_CWORD": cword,
        },
    )
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert "Did you mean" not in result.stdout
    assert "to sign in" not in result.stdout
