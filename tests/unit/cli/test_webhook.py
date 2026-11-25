import json
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest
from click.testing import CliRunner
from mobilerun_sdk import BadRequestError
from pydantic import SecretStr

from mobilerun.cli.app import cli
from mobilerun.core.webhooks import WebhookEndpoint
from mobilerun.errors.cloud import CloudConflict

URL = (
    "https://example.invalid:8443/path?token=private-token&opaque-sentinel&other=value"
)
DISPLAY_URL = "https://example.invalid:8443/path?token=•••&•••&other=•••"
SECRET = "test-only-signature"
ID = "0b177bf3-abda-4746-a7bc-559924eb9436"
USAGE = f'Use it with @cloud("my-function", webhook="{ID}") or as endpointId in a flow\'s webhooks.send step.'


def invoke(*args, **kwargs):
    return CliRunner().invoke(cli, ["webhook", *args], **kwargs)


def test_webhook_group():
    result = invoke("--help")
    assert result.exit_code == 0
    assert "register" in result.output


@pytest.fixture
def hooks(monkeypatch):
    from mobilerun.cli.webhook import group
    from mobilerun.core.webhooks import DEFAULT_WEBHOOK_EVENT

    endpoint = WebhookEndpoint(
        ID,
        "http",
        SecretStr(URL),
        "ACTIVE",
        "healthy",
        (DEFAULT_WEBHOOK_EVENT,),
        SecretStr(SECRET),
    )
    fake = AsyncMock()
    fake.find_by_url.return_value = None
    fake.register.return_value = fake.get.return_value = endpoint
    fake.list.return_value = [endpoint]
    fake.ping.return_value = (True, 200)
    monkeypatch.setattr(group, "CloudWebhooks", lambda: fake)
    return fake


@pytest.mark.parametrize("flags", [[], ["--json"]])
def test_register_then_duplicate(hooks, flags):
    hooks.register.return_value = replace(
        hooks.get.return_value, event_types=("workflow.step.emit", "sms.received")
    )
    args = [
        "register",
        URL,
        "--event-type",
        "workflow.step.emit",
        "--event-type",
        "sms.received",
        *flags,
    ]
    result = invoke(*args)
    assert result.exit_code == 0
    assert result.output.count(SECRET) == 1 and URL not in result.output
    assert "store" in result.output.lower()
    if flags:
        assert json.loads(result.stdout)["secret"] == SECRET
        assert json.loads(result.stdout)["url"] == DISPLAY_URL
    hooks.register.assert_awaited_once_with(
        URL, event_types=("workflow.step.emit", "sms.received")
    )
    hooks.find_by_url.return_value = hooks.register.return_value
    result = invoke(*args)
    assert result.exit_code == 0 and ID in result.output
    assert "already" in result.output.lower()
    assert USAGE in result.output
    assert SECRET not in result.output and URL not in result.output
    assert hooks.register.await_count == 1


@pytest.mark.parametrize(
    "args",
    [
        ("register",),
        ("get",),
        ("ping",),
        ("delete", "--yes"),
    ],
)
def test_missing_input(hooks, args):
    assert invoke(*args).exit_code == 2
    hooks.register.assert_not_awaited()
    hooks.delete.assert_not_awaited()


@pytest.mark.parametrize("args", [["list"], ["get", ID]])
@pytest.mark.parametrize("flags", [[], ["--json"]])
def test_read_redacts(hooks, args, flags):
    result = invoke(*args, *flags)
    assert result.exit_code == 0
    assert URL not in result.output and SECRET not in result.output
    assert "healthy" in result.output
    if flags:
        data = json.loads(result.stdout)
        row = data[0] if args == ["list"] else data
        assert row["id"] == ID and "secret" not in row
        assert row["url"] == DISPLAY_URL
    else:
        for label in ("ID", "URL", "Events", "State", "Health"):
            assert label in result.output


def test_delete_requires_confirmation(hooks, monkeypatch):
    from mobilerun.cli.webhook import group

    assert invoke("delete", ID).exit_code != 0
    monkeypatch.setattr(group, "is_interactive", lambda: True)
    monkeypatch.setattr(group, "confirm_prompt", lambda *a, **kw: False)
    assert invoke("delete", ID).exit_code != 0
    hooks.delete.assert_not_awaited()
    monkeypatch.setattr(group, "confirm_prompt", lambda *a, **kw: True)
    monkeypatch.setattr(group, "select_prompt", lambda _, choices: choices[0].value)
    assert invoke("delete").exit_code == 0
    hooks.delete.assert_awaited_once_with(ID)
    monkeypatch.setattr(group, "is_interactive", lambda: False)
    assert invoke("delete", ID, "--yes").exit_code == 0
    assert hooks.delete.await_count == 2


def test_url_is_the_only_prompt(hooks, monkeypatch):
    from mobilerun.cli.webhook import group

    monkeypatch.setattr(group, "is_interactive", lambda: True)

    result = invoke("register", input=f"http://bad.invalid\n{URL}\n")
    assert result.exit_code == 0 and URL in result.output
    assert result.output.count("Webhook URL (https://…)") == 2
    assert "must start with https://" in result.output
    hooks.find_by_url.assert_awaited_once_with(URL)
    hooks.register.assert_awaited_once_with(URL, event_types=("workflow.step.emit",))


def test_default_and_flow_hint(hooks):
    result = invoke("register", URL)
    assert result.exit_code == 0
    assert result.stdout.splitlines() == [
        f"Registered webhook: {ID}",
        f"Signing secret (shown once, store it now): {SECRET}",
        f"Test it: mobilerun webhook ping {ID}",
        USAGE,
    ]
    assert "workflow.step.emit" in invoke("register", "--help").output
    hooks.register.assert_awaited_once_with(URL, event_types=("workflow.step.emit",))


@pytest.mark.parametrize("flags", [[], ["--json"]])
@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"state": "DISABLED"}, "disabled"),
        ({"health": "blocked"}, "blocked"),
        ({"event_types": ("sms.received",)}, "filter excludes workflow.step.emit"),
    ],
)
def test_unusable_existing(hooks, flags, changes, reason):
    hooks.find_by_url.return_value = replace(hooks.get.return_value, **changes)
    result = invoke("register", URL, *flags)
    assert result.exit_code == 1
    lines = (result.stderr if flags else result.stdout).splitlines()
    assert len(lines) == 2 and lines[0] == f"Webhook already registered: {ID}"
    assert reason in lines[1]
    assert "@cloud" not in result.output and SECRET not in result.output
    if flags:
        data = json.loads(result.stdout)
        assert (
            data["error"] == lines[1] and data["created"] is False and data["id"] == ID
        )
    hooks.register.assert_not_awaited()
    hooks.delete.assert_not_awaited()


@pytest.mark.parametrize(("events", "exit_code"), [((), 0), (("sms.received",), 1)])
def test_duplicate_race_rechecks_once(hooks, events, exit_code):
    hooks.find_by_url.side_effect = [
        None,
        replace(hooks.get.return_value, event_types=events),
    ]
    hooks.register.side_effect = CloudConflict(
        operation="Register webhook", status=409, code="webhook_duplicate_url"
    )
    result = invoke("register", URL)
    assert result.exit_code == exit_code and "already registered" in result.stdout
    assert (USAGE in result.stdout) == (exit_code == 0)
    assert hooks.find_by_url.await_count == 2 and hooks.register.await_count == 1


def test_other_conflict_still_fails(hooks):
    hooks.register.side_effect = CloudConflict(
        operation="Register webhook",
        status=409,
        code="webhook_registration_in_progress",
    )
    result = invoke("register", URL)
    assert result.exit_code == 1 and "in progress" in result.output
    hooks.find_by_url.assert_awaited_once_with(URL)


@pytest.mark.parametrize("flags", [[], ["--json"]])
def test_replay_without_secret(hooks, flags):
    hooks.register.return_value = replace(hooks.get.return_value, secret=None)
    result = invoke("register", URL, *flags)
    assert result.exit_code == 0
    lines = (result.stderr if flags else result.stdout).splitlines()
    assert ID in lines[0] and lines[1].startswith(
        "Signing secret unavailable (already issued);"
    )
    assert "rotate_secret" in lines[1] and lines[-1] == USAGE
    if flags:
        assert "secret" not in json.loads(result.stdout)


@pytest.mark.parametrize("flags", [[], ["--json"]])
def test_custom_filter_has_no_flow_hint(hooks, flags):
    hooks.register.return_value = replace(
        hooks.get.return_value, event_types=("sms.received",)
    )
    result = invoke("register", URL, "--event-type", "sms.received", *flags)
    assert result.exit_code == 0 and "@cloud" not in result.output
    if flags:
        assert json.loads(result.stdout)["id"] == ID


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("endpoint limit reached (12 per org)", "endpoint limit"),
        (
            "eventTypes contains an unknown or non-webhook-deliverable type",
            "non-webhook-deliverable",
        ),
        (f"failed {URL} {SECRET}", "withheld"),
    ],
)
def test_400_redacts_untrusted_detail(hooks, message, expected):
    from mobilerun.clients.cloud.webhooks import _request

    async def rejected():
        raise BadRequestError(
            "bad request",
            response=httpx.Response(400, request=httpx.Request("POST", URL)),
            body={"message": message, "details": SECRET},
        )

    async def register(*a, **kw):
        return await _request(rejected(), "Register webhook")

    hooks.register.side_effect = register
    result = invoke("register", URL, "--event-type", "test")
    hooks.register.assert_awaited_once_with(URL, event_types=("test",))
    assert result.exit_code != 0 and expected in result.output
    assert URL not in result.output and SECRET not in result.output


@pytest.mark.parametrize("command", ["get", "delete", "ping"])
def test_select_endpoint_or_empty(hooks, monkeypatch, command):
    from mobilerun.cli.webhook import group

    monkeypatch.setattr(group, "is_interactive", lambda: True)

    def select(_, choices):
        label = choices[0].label
        for field in (ID[:8], DISPLAY_URL, "ACTIVE", "healthy", "workflow.step.emit"):
            assert field in label
        assert URL not in label
        return choices[0].value

    monkeypatch.setattr(group, "select_prompt", select)
    args = ["--yes"] if command == "delete" else []
    result = invoke(command, *args)
    assert result.exit_code == 0
    getattr(hooks, command).assert_awaited_once_with(ID)
    hooks.list.return_value = []
    result = invoke(command, *args)
    assert (
        result.exit_code == 0
        and "No webhooks. Create one: mobilerun webhook register <url>" in result.output
    )
    assert getattr(hooks, command).await_count == 1


@pytest.mark.parametrize(
    ("outcome", "text", "code"),
    [
        ((True, 200), "Delivered: HTTP 200 OK", 0),
        (
            (False, 405),
            "Failed: HTTP 405 Method Not Allowed (the receiver rejected the POST)",
            1,
        ),
        ((False, 599), "Failed: HTTP 599 (the receiver rejected the POST)", 1),
        (
            (False, None),
            "Failed: no HTTP response from the receiver (possible network error or timeout).",
            1,
        ),
    ],
)
def test_ping_result(hooks, outcome, text, code):
    hooks.ping.return_value = outcome
    result = invoke("ping", ID)
    assert (
        result.exit_code == code
        and f"Ping sent to {ID}" in result.output
        and text in result.output
    )
    assert URL not in result.output


def test_smoke_cleans_up_before_failing_on_missing_secret(hooks, monkeypatch, capsys):
    from tests.unit.smoke.test_webhook_cli import test_webhook_cli_lifecycle

    token = "f067e40a-0cc8-45ae-9005-d80d345e05ba"
    calls = []

    def receiver(client, method, path, **kwargs):
        calls.append((method, path))
        return httpx.Response(200, json={"uuid": token})

    monkeypatch.setattr(httpx.Client, "request", receiver)
    hooks.register.return_value = replace(hooks.get.return_value, secret=None)
    with pytest.raises(pytest.fail.Exception, match="signing secret"):
        test_webhook_cli_lifecycle(None)
    hooks.delete.assert_awaited_once_with(ID)
    assert calls == [("POST", "/token"), ("DELETE", f"/token/{token}")]
    output = capsys.readouterr()
    assert "receiver deleted" in output.out and token not in output.out + output.err
