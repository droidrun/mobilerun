import asyncio
import json
import subprocess
import sys
from contextlib import contextmanager
from importlib import import_module
from unittest.mock import ANY, AsyncMock, call

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.cli.base import menu, safety
from mobilerun.cli.devices.base import discovery
from mobilerun.models.billing import Billing
from mobilerun.sandbox import packaging
from mobilerun.sandbox.models import Artifact, DeploymentReceipt, Endpoint

deploy = import_module("mobilerun.sandbox.deployment")
sandbox_billing = import_module("mobilerun.sandbox.billing")
sandbox_execution = import_module("mobilerun.sandbox.execution")
sandbox_flows = import_module("mobilerun.sandbox.flows")


def invoke(*args):
    return CliRunner().invoke(cli, ["sandbox", *map(str, args)])


class BillingRunner(CliRunner):
    def __init__(self, interactive=True):
        super().__init__()
        self.interactive = interactive

    @contextmanager
    def isolation(self, *args, **kwargs):
        with super().isolation(*args, **kwargs) as streams:
            sys.stdin.isatty = sys.stderr.isatty = lambda: self.interactive
            yield streams


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("MOBILERUN_BILLING", "paid")
    monkeypatch.setattr(menu, "is_interactive", lambda: False)
    monkeypatch.setattr(
        packaging,
        "package",
        lambda path, *, source: Artifact(
            packaging.read_declaration(path, source=source),
            source,
            "test-digest",
            {},
            None,
        ),
    )
    monkeypatch.setattr(
        deploy.clients,
        "get_document",
        AsyncMock(side_effect=AssertionError("get_document")),
    )
    for module, names in (
        (deploy, ("deploy_file",)),
        (
            sandbox_execution,
            ("fire_remote", "recent_runs", "wait_result", "execution_logs"),
        ),
        (sandbox_flows, ("set_enabled", "list_deployments", "describe_remote")),
        (discovery, ("discover_devices",)),
    ):
        for name in names:
            monkeypatch.setattr(
                module, name, AsyncMock(side_effect=AssertionError(name))
            )


@pytest.fixture
def script(tmp_path):
    def create(options=""):
        path = tmp_path / "sandbox_script.py"
        path.write_text(
            "from mobilerun.sandbox import cloud\n"
            "raise AssertionError('source must not be imported')\n"
            f"@cloud('demo'{', ' + options if options else ''})\n"
            "def handler(): pass\n",
            encoding="utf-8",
        )
        return path

    return create


@pytest.fixture
def receipt():
    value = DeploymentReceipt(
        "demo", "fn-1", 3, "created", None, None, (), "abc", None, ()
    )
    deploy.deploy_file.side_effect = None
    deploy.deploy_file.return_value = value
    return value


def test_help_and_path_do_not_load_runtime_and_plugin_is_removable(tmp_path):
    script = """
import sys
from unittest.mock import patch
from click.testing import CliRunner
from mobilerun.addons import plugins
from mobilerun.cli.app import cli

sys.modules['mobilerun_functions'] = None
runner = CliRunner()
result = runner.invoke(cli, ['--help'])
assert result.exit_code == 0 and 'sandbox' in result.output, result.output
assert 'mobilerun.sandbox' not in sys.modules
for command in ([], *[[name] for name in ('deploy', 'list', 'fire', 'runs', 'enable', 'disable', 'path')]):
    result = runner.invoke(cli, ['sandbox', *command, '--help'])
    assert result.exit_code == 0, result.output
result = runner.invoke(cli, ['sandbox', 'path', '--json'])
assert result.exit_code == 0, result.output
assert 'mobilerun.sandbox._runtime' not in sys.modules
installed = plugins.discover
with patch.object(plugins, 'discover', lambda group: {
    name: entry for name, entry in installed(group).items() if name != 'sandbox'
}):
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0 and 'sandbox' not in result.output, result.output
    result = runner.invoke(cli, ['sandbox', '--help'])
    assert result.exit_code == 2 and 'No such command' in result.output, result.output
"""
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)


@pytest.mark.parametrize(
    "setting,relative,exists,source",
    [
        (None, "config/sandbox", False, "default"),
        ("", "config/sandbox", False, "default"),
        ("~/custom", "custom", False, "env"),
        ("~", ".", True, "env"),
    ],
    ids=["unset", "empty", "custom", "home"],
)
def test_path_default_env_empty_exists_and_no_creation(
    monkeypatch, tmp_path, setting, relative, exists, source
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path / "config"))
    if setting is None:
        monkeypatch.delenv("MOBILERUN_SANDBOX_DIRECTORY", raising=False)
    else:
        monkeypatch.setenv("MOBILERUN_SANDBOX_DIRECTORY", setting)
    path = tmp_path / relative
    result = invoke("path", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "path": str(path),
        "exists": exists,
        "source": source,
    }
    human = invoke("path")
    assert human.exit_code == 0, human.output
    assert str(path) in human.stdout
    assert path.exists() is exists


@pytest.mark.parametrize("command", ["deploy", "fire", "runs", "enable", "disable"])
def test_required_arguments_in_automation(command):
    result = invoke(command, "--yes" if command != "runs" else "--json")
    assert result.exit_code == 2
    metavar = "FILE" if command == "deploy" else "NAME"
    assert f"Missing argument '{metavar}" in result.stderr


def test_noninteractive_no_commerce_disable_requires_yes():
    result = invoke("disable", "demo", "--billing", "no_commerce")
    assert result.exit_code == 1
    assert "pass --yes" in result.stderr
    deploy.deploy_file.assert_not_awaited()
    sandbox_execution.fire_remote.assert_not_awaited()
    sandbox_flows.set_enabled.assert_not_awaited()


@pytest.mark.parametrize(
    ("command", "confirmation"),
    [
        ("deploy", "Deploy 'demo' (device-free)"),
        ("fire", "Fire sandbox endpoint 'demo'"),
        ("enable", "Enable sandbox flows 'demo', 'other'"),
    ],
)
@pytest.mark.parametrize(
    ("interactive", "answer", "flags", "error"),
    [
        pytest.param(True, False, [], "Cancelled.", id="declined"),
        pytest.param(True, True, [], None, id="accepted"),
        pytest.param(False, None, ["--yes"], None, id="yes"),
        pytest.param(False, None, [], "pass --yes", id="noninteractive"),
        pytest.param(True, True, ["--billing", "included"], None, id="escalation"),
    ],
)
def test_billable_commands_require_consent(
    command,
    confirmation,
    interactive,
    answer,
    flags,
    error,
    script,
    receipt,
    monkeypatch,
):
    prompt = AsyncMock(return_value=answer)
    monkeypatch.setattr(safety, "confirm_prompt_async", prompt)
    paid_calls = []

    async def guarded(*args, billing, **kwargs):
        sandbox_billing.require_paid(billing=billing)
        paid_calls.append((args, billing, prompt.await_count))
        return receipt if command == "deploy" else {"invocation_id": "inv-1"}

    path = script()
    operation, args, calls = {
        "deploy": (deploy.deploy_file, [str(path)], [(path,)]),
        "fire": (sandbox_execution.fire_remote, ["demo"], [("demo", {})]),
        "enable": (
            sandbox_flows.set_enabled,
            ["demo", "other"],
            [("demo", True), ("other", True)],
        ),
    }[command]
    operation.side_effect = guarded
    result = BillingRunner(interactive).invoke(
        cli, ["sandbox", command, *args, *flags, "--json"]
    )
    if error:
        assert result.exit_code == 1, result.output
        assert error in result.stderr
        operation.assert_not_awaited()
        assert result.stdout == ""
        assert paid_calls == []
    else:
        assert result.exit_code == 0, result.output
        assert paid_calls == [
            (args, Billing.PAID, int(answer is not None)) for args in calls
        ]
    if answer is None:
        prompt.assert_not_awaited()
    elif "included" in flags:
        prompt.assert_awaited_once()
        assert prompt.await_args.kwargs == {"default": False}
        assert "paid" in prompt.await_args.args[0]
    else:
        prompt.assert_awaited_once_with(
            f"{confirmation} with paid billing (charges may apply)?", default=False
        )


@pytest.mark.parametrize("command", ["deploy", "fire", "enable"])
def test_billable_picker_skips_remote_lookup_without_billing_terminal(
    command, script, monkeypatch
):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    args = [str(script("schedule='0 12 * * *'"))] if command == "deploy" else []
    result = BillingRunner(interactive=False).invoke(cli, ["sandbox", command, *args])
    assert result.exit_code == 1, result.output
    assert (
        result.stderr
        == "Error: Refusing to proceed without confirmation; pass --yes.\n"
    )
    discovery.discover_devices.assert_not_awaited()
    sandbox_flows.list_deployments.assert_not_awaited()


@pytest.mark.parametrize("command", ["fire", "enable"])
def test_billable_name_picker_honors_yes_before_lookup(command, monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    sandbox_flows.list_deployments.side_effect = None
    sandbox_flows.list_deployments.return_value = []
    result = BillingRunner(interactive=False).invoke(cli, ["sandbox", command, "--yes"])
    assert result.exit_code == 0, result.output
    assert "No deployed sandbox functions found." in result.stderr
    sandbox_flows.list_deployments.assert_awaited_once_with(billing=Billing.PAID)


@pytest.mark.parametrize("command", ["fire", "enable"])
def test_billable_name_picker_refuses_cleanly_with_running_loop(command, monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    async def run_cli():
        return BillingRunner(interactive=False).invoke(cli, ["sandbox", command])

    result = asyncio.run(run_cli())
    assert result.exit_code == 1, result.output
    assert (
        result.stderr
        == "Error: Refusing to proceed without confirmation; pass --yes.\n"
    )
    sandbox_flows.list_deployments.assert_not_awaited()


@pytest.mark.parametrize(
    ("billing", "interactive", "answer", "yes", "attempts", "exit_code"),
    [
        ("included", True, True, False, ["included", "paid"], 0),
        ("included", True, False, False, ["included"], 1),
        ("no_commerce", False, None, False, ["no_commerce"], 1),
        ("no_commerce", False, None, True, ["no_commerce", "paid"], 0),
    ],
)
def test_cli_escalates_only_after_guard(
    script, receipt, monkeypatch, billing, interactive, answer, yes, attempts, exit_code
):
    prompt = AsyncMock(return_value=answer)
    monkeypatch.setattr(safety, "confirm_prompt_async", prompt)
    mutations = []
    accepted = attempts[-1] == "paid"
    package = packaging.package
    packaged = []

    def inspect(path, *, source):
        assert accepted, "rejected deploy must not inspect source"
        assert prompt.await_count == int(answer is not None)
        packaged.append(package(path, source=source))
        return packaged[-1]

    monkeypatch.setattr(packaging, "package", inspect)

    async def guarded(*args, billing, **kwargs):
        sandbox_billing.require_paid(billing=billing)
        mutations.append(billing)
        return receipt

    deploy.deploy_file.side_effect = guarded
    args = ["sandbox", "deploy", str(script()), "--billing", billing, "--json"]
    if yes:
        args.append("--yes")
    result = BillingRunner(interactive).invoke(cli, args)
    assert result.exit_code == exit_code, result.output
    assert [c.kwargs["billing"].value for c in deploy.deploy_file.await_args_list] == (
        ["paid"] if accepted else []
    )
    assert len(packaged) == int(accepted)
    if accepted:
        assert deploy.deploy_file.await_args.kwargs["artifact"] is packaged[0]
    assert mutations == ([Billing.PAID] if attempts[-1] == "paid" else [])
    assert prompt.await_count == int(answer is not None)
    if answer is False:
        assert "Cancelled." in result.stderr
    elif exit_code:
        assert "--yes" in result.stderr and "MOBILERUN_BILLING=paid" in result.stderr


@pytest.mark.parametrize("consent", [False, True])
def test_enable_batch_authorizes_once_even_if_first_item_fails(monkeypatch, consent):
    prompt = AsyncMock(return_value=consent)
    monkeypatch.setattr(safety, "confirm_prompt_async", prompt)

    async def enable(name, enabled, *, billing):
        sandbox_billing.require_paid(billing=billing)
        if name == "first":
            raise RuntimeError("first failed")

    sandbox_flows.set_enabled.side_effect = enable
    result = BillingRunner().invoke(
        cli, ["sandbox", "enable", "first", "second", "--billing", "included", "--json"]
    )
    assert result.exit_code == 1, result.output
    prompt.assert_awaited_once()
    assert all(name in prompt.await_args.args[0] for name in ("first", "second"))
    if not consent:
        assert "Cancelled." in result.stderr
        sandbox_flows.set_enabled.assert_not_awaited()
    else:
        assert sandbox_flows.set_enabled.await_args_list == [
            call("first", True, billing=Billing.PAID),
            call("second", True, billing=Billing.PAID),
        ]
        assert json.loads(result.stdout) == [
            {"name": "first", "status": "failed"},
            {"name": "second", "status": "enabled"},
        ]


@pytest.mark.parametrize(
    ("options", "flags", "device", "device_free"),
    [
        ("", [], None, False),
        ("device='decorator'", [], "decorator", False),
        (
            "schedule='0 12 * * *', device='decorator'",
            ["--device", "cli-phone"],
            "cli-phone",
            False,
        ),
        ("device='decorator'", ["--device-free"], None, True),
    ],
)
def test_deploy_device_precedence(options, flags, device, device_free, script, receipt):
    path = script(options)
    result = invoke("deploy", path, *flags, "--yes", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["function_id"] == "fn-1"
    deploy.deploy_file.assert_awaited_once_with(
        path,
        device=device,
        device_free=device_free,
        billing=Billing.PAID,
        artifact=ANY,
    )
    discovery.discover_devices.assert_not_awaited()


def test_schedule_requires_explicit_device_choice_even_with_yes(script):
    result = invoke("deploy", script("schedule='0 12 * * *'"), "--yes")
    assert result.exit_code == 2
    assert "--device-free" in result.stderr
    deploy.deploy_file.assert_not_awaited()
    discovery.discover_devices.assert_not_awaited()


def test_conflicting_device_options(script):
    result = invoke("deploy", script(), "--device", "phone", "--device-free", "--yes")
    assert result.exit_code == 2
    assert "--device" in result.stderr and "--device-free" in result.stderr
    deploy.deploy_file.assert_not_awaited()


def test_endpoint_text_uses_required_schema_fields_and_never_prints_api_key(
    script, receipt, monkeypatch
):
    from dataclasses import replace

    from mobilerun.sandbox import packaging
    from mobilerun.sandbox.models import Artifact, CloudDeclaration

    deploy.deploy_file.return_value = replace(
        receipt,
        flow_id="flow-1",
        endpoint=Endpoint("https://example.test/v1/triggers/id/fire", "id"),
    )
    monkeypatch.setenv("MOBILERUN_API_KEY", "must-not-render-key")
    monkeypatch.setattr(
        packaging,
        "package",
        lambda path, *, source: Artifact(
            CloudDeclaration("demo"),
            "",
            "abc",
            {
                "type": "object",
                "required": ["count"],
                "properties": {
                    "count": {"type": "integer"},
                    "token": {"type": "string", "default": "must-not-render-default"},
                },
            },
            None,
        ),
    )
    result = invoke("deploy", script(), "--yes")
    assert result.exit_code == 0, result.output
    for text in (
        "https://example.test/v1/triggers/id/fire",
        "flow-1",
        "Authorization: Bearer <YOUR_API_KEY>",
        "MOBILERUN_API_KEY",
        '{"payload": {"count": 0}}',
        "invocationId",
        "mobilerun sandbox runs demo",
    ):
        assert text in result.stdout
    assert "must-not-render" not in result.output


@pytest.mark.parametrize("pick_phone", [True, False])
@pytest.mark.parametrize("selected", [True, False])
def test_paid_picker_prompts_stay_off_json_stdout(
    pick_phone, selected, script, receipt, monkeypatch
):
    from mobilerun.sandbox.cli import commands

    path = script("schedule='0 12 * * *'")
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setenv("MOBILERUN_BILLING", "no_commerce")

    def prompt_file(message):
        click.echo(message)
        return str(path)

    monkeypatch.setattr(menu, "text_prompt", prompt_file)
    discovery.discover_devices.side_effect = None
    discovery.discover_devices.return_value = [
        discovery.DiscoveredDevice("cloud-id", "cloud", "Cloud Phone", "ready", None),
        discovery.DiscoveredDevice("local-id", "local", "Local Phone", None, None),
    ]

    async def pick(message, choices):
        click.echo(message)
        assert [c.label for c in choices] == ["Cloud Phone", "Device-free"]
        return choices[0 if pick_phone else 1].value if selected else None

    picker = AsyncMock(side_effect=pick)
    monkeypatch.setattr(commands, "select_prompt_async", picker)

    async def confirm(message, *, default):
        click.echo(message)
        return True

    confirmation = AsyncMock(side_effect=confirm)
    monkeypatch.setattr(safety, "confirm_prompt_async", confirmation)
    result = BillingRunner().invoke(
        cli, ["sandbox", "deploy", "--billing", "paid", "--json"]
    )
    assert result.exit_code == 0, result.output
    discovery.discover_devices.assert_awaited_once_with(local=False, strict=True)
    assert "File" in result.stderr and "device" in result.stderr.lower()
    if selected:
        target = "device 'cloud-id'" if pick_phone else "device-free"
        assert "charges may apply" in result.stderr
        confirmation.assert_awaited_once_with(
            f"Deploy 'demo' ({target}) on schedule '0 12 * * *' with paid billing (charges may apply)?",
            default=False,
        )
        assert json.loads(result.stdout)["version"] == 3
        deploy.deploy_file.assert_awaited_once_with(
            path,
            device="cloud-id" if pick_phone else None,
            device_free=not pick_phone,
            billing=Billing.PAID,
            artifact=ANY,
        )
    else:
        confirmation.assert_not_awaited()
        assert result.stdout == ""
        deploy.deploy_file.assert_not_awaited()


def test_fire_prompts_for_name_and_does_not_poll(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    stub_flow_previews()

    def pick_name(message, choices):
        click.echo(message)
        assert [(c.value, c.label) for c in choices] == [
            ("demo", "demo · abcd1234… · 1 · ● ready"),
            ("other", "other · fn-2 · 2 · ● pending"),
        ]
        return choices[0].value

    fake_deployments()
    monkeypatch.setattr(menu, "select_prompt", pick_name)
    monkeypatch.setattr(safety, "confirm_prompt_async", AsyncMock(return_value=True))
    sandbox_execution.fire_remote.side_effect = None
    sandbox_execution.fire_remote.return_value = {
        "invocation_id": "inv-1",
        "ignored": 2,
    }
    result = BillingRunner().invoke(
        cli, ["sandbox", "fire", "--input", '{"billing": "payload"}', "--json"]
    )
    assert result.exit_code == 0, result.output
    assert "Name" in result.stderr
    assert json.loads(result.stdout) == {"invocation_id": "inv-1"}
    sandbox_execution.fire_remote.assert_awaited_once_with(
        "demo", {"billing": "payload"}, billing=Billing.PAID
    )
    sandbox_execution.wait_result.assert_not_awaited()


def stub_flow_previews():
    sandbox_flows.describe_remote.side_effect = lambda name, **kwargs: {"flow_id": name}
    deploy.clients.get_document.side_effect = lambda flow_id, **kwargs: (
        {"devices": ["cloud-device"], "on": {"custom": {}}}
        if flow_id == "demo"
        else {"on": {"schedule": {"cron": "0 12 * * *"}}}
    )


@pytest.mark.parametrize("device", [None, "picked-phone"])
def test_deploy_uses_same_artifact_after_file_changes(
    script, receipt, monkeypatch, device
):
    from mobilerun.sandbox.cli import commands

    path = script("schedule='0 12 * * *'")
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    events = []
    package = packaging.package

    async def prompt(message, *, default):
        events.append("billing")
        assert "'demo'" in message and "'changed'" not in message
        assert (device or "device-free") in message
        assert "0 12 * * *" in message
        return True

    def inspect(*args, **kwargs):
        events.append("package")
        return package(*args, **kwargs)

    async def pick_device(*, as_json, yes):
        events.append("device")
        path.write_text(path.read_text().replace("'demo'", "'changed'"))
        return device, device is None

    monkeypatch.setattr(commands, "_pick_device", pick_device)
    monkeypatch.setattr(safety, "confirm_prompt_async", prompt)
    monkeypatch.setattr(packaging, "package", inspect)
    result = BillingRunner().invoke(
        cli, ["sandbox", "deploy", str(path), "--billing", "included", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert events == ["device", "billing", "package"]
    artifact = deploy.deploy_file.await_args.kwargs["artifact"]
    assert artifact.declaration.name == "demo" and "'changed'" not in artifact.source


def test_runs_json_only_emits_summary():
    sandbox_execution.recent_runs.side_effect = None
    sandbox_execution.recent_runs.return_value = [
        {
            "id": "exec",
            "status": "completed",
            "invocationId": "inv",
            "startedAt": "2026-01-01T00:00:00Z",
            "finishedAt": "2026-01-01T00:00:02Z",
            "result": {"steps": [{"body": {"stderr": "secret-sentinel"}}]},
        }
    ]
    result = invoke("runs", "demo", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == [
        {
            "id": "exec",
            "status": "completed",
            "invocationId": "inv",
            "startedAt": "2026-01-01T00:00:00Z",
            "finishedAt": "2026-01-01T00:00:02Z",
            "duration": 2,
        }
    ]
    assert "secret-sentinel" not in result.output


def fake_deployments():
    sandbox_flows.list_deployments.side_effect = None
    sandbox_flows.list_deployments.return_value = [
        {
            "id": "abcd1234-5678-9012-3456-789012345678",
            "name": "demo",
            "latestVersion": {"version": 1, "status": "ready"},
        },
        {
            "id": "fn-2",
            "name": "other",
            "latestVersion": {"version": 2, "status": "pending"},
        },
    ]


def test_runs_uses_single_picker_with_resolved_billing(monkeypatch):
    fake_deployments()
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "select_prompt", lambda label, choices: choices[1].value)
    sandbox_execution.recent_runs.side_effect = None
    sandbox_execution.recent_runs.return_value = []
    result = invoke("runs", "--billing", "no_commerce", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []
    sandbox_flows.list_deployments.assert_awaited_once_with(billing=Billing.NO_COMMERCE)
    sandbox_execution.recent_runs.assert_awaited_once_with(
        "other", 10, billing=Billing.NO_COMMERCE
    )


def test_picker_load_failure_does_not_fall_back_to_text(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    sandbox_flows.list_deployments.side_effect = RuntimeError("secret response body")
    result = invoke("runs", "--json")
    assert result.exit_code == 1
    assert "unexpected failure" in result.stderr
    assert "secret response body" not in result.output
    sandbox_execution.recent_runs.assert_not_awaited()


def test_fire_reads_input_file(tmp_path):
    path = tmp_path / "payload.json"
    path.write_text('{"limit": 2}', encoding="utf-8")
    sandbox_execution.fire_remote.side_effect = None
    sandbox_execution.fire_remote.return_value = {"invocation_id": "inv-1"}
    result = invoke("fire", "demo", "--input-file", path, "--yes")
    assert result.exit_code == 0, result.output
    assert "inv-1" in result.stdout
    sandbox_execution.fire_remote.assert_awaited_once_with(
        "demo", {"limit": 2}, billing=Billing.PAID
    )


@pytest.mark.parametrize("payload", ["[]", "null", "", "bad-secret", '{"x": NaN}'])
def test_fire_requires_valid_json_object_without_echoing_input(payload):
    result = invoke("fire", "demo", "--input", payload, "--yes")
    assert result.exit_code == 2
    assert "JSON object" in result.stderr and "bad-secret" not in result.stderr
    sandbox_execution.fire_remote.assert_not_awaited()


def test_fire_rejects_both_payload_options(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("{}", encoding="utf-8")
    result = invoke("fire", "demo", "--input", "{}", "--input-file", path, "--yes")
    assert result.exit_code == 2
    assert "--input" in result.stderr and "--input-file" in result.stderr
    sandbox_execution.fire_remote.assert_not_awaited()


@pytest.mark.parametrize("limit", [None, 5])
def test_no_commerce_permits_reads(limit):
    sandbox_flows.list_deployments.side_effect = None
    sandbox_flows.list_deployments.return_value = [{"id": "fn-1", "name": "demo"}]
    sandbox_execution.recent_runs.side_effect = None
    sandbox_execution.recent_runs.return_value = [
        {"id": "exec-1", "status": "completed"}
    ]
    flags = ("--billing", "no_commerce", "--json")
    listed = invoke("list", *flags)
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.stdout) == [{"id": "fn-1", "name": "demo"}]
    args = ["--limit", limit] if limit is not None else []
    runs = invoke("runs", "demo", *flags, *args)
    assert runs.exit_code == 0, runs.output
    assert json.loads(runs.stdout) == [{"id": "exec-1", "status": "completed"}]
    sandbox_execution.recent_runs.assert_awaited_once_with(
        "demo", limit or 10, billing=Billing.NO_COMMERCE
    )


@pytest.mark.parametrize(
    ("command", "billing", "enabled"),
    [("enable", Billing.PAID, True), ("disable", Billing.NO_COMMERCE, False)],
)
def test_set_enabled_after_consent(command, billing, enabled):
    sandbox_flows.set_enabled.side_effect = None
    sandbox_flows.set_enabled.return_value = {"id": "flow-1", "enabled": enabled}
    result = invoke(command, "demo", "--billing", billing.value, "--yes", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == [
        {"name": "demo", "status": "enabled" if enabled else "disabled"}
    ]
    sandbox_flows.set_enabled.assert_awaited_once_with("demo", enabled, billing=billing)


@pytest.mark.parametrize(
    ("command", "enabled", "billing"),
    [
        ("enable", True, Billing.PAID),
        ("disable", False, Billing.NO_COMMERCE),
    ],
)
@pytest.mark.parametrize("consent", [True, False])
def test_batch_multiselect_confirms_all_once(
    monkeypatch, command, enabled, billing, consent
):
    fake_deployments()
    if enabled:
        stub_flow_previews()
    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    def pick_many(message, choices):
        click.echo(message)
        assert [c.value for c in choices] == ["demo", "other"]
        return [c.value for c in choices]

    monkeypatch.setattr(menu, "multi_select_prompt", pick_many)
    confirm = AsyncMock(return_value=consent)
    monkeypatch.setattr(safety, "confirm_prompt_async", confirm)
    sandbox_flows.set_enabled.side_effect = None
    sandbox_flows.set_enabled.return_value = {"id": "flow-1", "enabled": enabled}
    result = BillingRunner().invoke(
        cli, ["sandbox", command, "--billing", billing.value, "--json"]
    )
    assert result.exit_code == int(enabled and not consent), result.output
    confirm.assert_awaited_once()
    if enabled:
        assert all(name in confirm.await_args.args[0] for name in ("demo", "other"))
    else:
        assert "demo" in result.stderr and "other" in result.stderr
    sandbox_flows.list_deployments.assert_awaited_once_with(billing=billing)
    status = "enabled" if enabled else "disabled"
    if consent:
        assert json.loads(result.stdout) == [
            {"name": "demo", "status": status},
            {"name": "other", "status": status},
        ]
        assert sandbox_flows.set_enabled.await_args_list == [
            call(name, enabled, billing=billing) for name in ("demo", "other")
        ]
    else:
        if enabled:
            assert "Cancelled." in result.stderr and result.stdout == ""
        else:
            assert json.loads(result.stdout) == []
        sandbox_flows.set_enabled.assert_not_awaited()


def test_batch_reports_partial_failure_and_processes_later_names():
    sandbox_flows.set_enabled.side_effect = [
        RuntimeError("secret response body"),
        {"id": "flow-2", "enabled": False},
    ]
    result = invoke("disable", "demo", "other", "--yes", "--json")
    assert result.exit_code == 1
    assert json.loads(result.stdout) == [
        {"name": "demo", "status": "failed"},
        {"name": "other", "status": "disabled"},
    ]
    assert "secret response body" not in result.output
    assert sandbox_flows.set_enabled.await_args_list == [
        call(name, False, billing=Billing.PAID) for name in ("demo", "other")
    ]


def test_unexpected_failures_do_not_echo_raw_error_or_emit_json():
    sandbox_flows.list_deployments.side_effect = RuntimeError("secret response body")
    result = invoke("list", "--json")
    assert result.exit_code == 1
    assert "unexpected failure" in result.stderr
    assert "secret response body" not in result.output
    assert result.stdout == ""
