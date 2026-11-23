import json
from pathlib import Path
from unittest.mock import AsyncMock

from click.testing import CliRunner

from mobilerun.cli.mailbox import group as mailbox
from mobilerun.cli.phone_number import group as number
from mobilerun.cli.proxy import group as proxy
from mobilerun.cli.workspace import group as workspace


def test_workspace_download_and_rm_alias(monkeypatch):
    download = AsyncMock(return_value=Path("/tmp/saved"))
    monkeypatch.setattr(workspace.Workspace, "download_file", download)
    result = CliRunner().invoke(
        workspace.workspace, ["files", "download", "f1", "/tmp/out"]
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "/tmp/saved"
    assert "Downloading" in result.stderr
    download.assert_awaited_once_with("f1", "/tmp/out")
    help_text = CliRunner().invoke(workspace.workspace, ["files", "--help"]).output
    assert "rm " in help_text and "remove " not in help_text
    delete = AsyncMock()
    monkeypatch.setattr(workspace.Workspace, "delete_file", delete)
    assert (
        CliRunner()
        .invoke(workspace.workspace, ["files", "remove", "f1", "--yes"])
        .exit_code
        == 0
    )
    delete.assert_awaited_once_with("f1")


def test_proxy_release_confirmation_and_json(monkeypatch):
    cancel = AsyncMock()
    monkeypatch.setattr(proxy, "cancel_proxy", cancel, raising=False)
    runner = CliRunner()
    assert runner.invoke(proxy.proxy, ["release", "p1"]).exit_code != 0
    cancel.assert_not_awaited()
    result = runner.invoke(proxy.proxy, ["release", "p1", "--yes", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"id": "p1", "status": "released"}
    cancel.assert_awaited_once()


def test_code_timeout_validation():
    runner = CliRunner()
    for command, args in (
        (number.number, ["code", "n1"]),
        (mailbox.mailbox, ["code", "m1"]),
    ):
        result = runner.invoke(command, [*args, "--timeout", "0"])
        assert result.exit_code == 2, result.output
    text = runner.invoke(number.number, ["--help"]).output
    assert "code " in text and "sms " not in text


def test_status_commands_show_resource(monkeypatch):
    from types import SimpleNamespace as N

    monkeypatch.setattr(
        number.cloud_numbers,
        "retrieve",
        AsyncMock(
            return_value=N(
                data=N(
                    id="n1",
                    state="active",
                    phone_number="+123",
                    device_binding=N(device_id="d1"),
                )
            )
        ),
    )
    from mobilerun.clients.cloud import mailboxes as cloud_mailboxes

    monkeypatch.setattr(
        cloud_mailboxes,
        "retrieve",
        AsyncMock(
            return_value=N(data=N(id="m1", status="active", address="a@example.com"))
        ),
        raising=False,
    )
    runner = CliRunner()
    n = runner.invoke(number.number, ["status", "n1", "--json"])
    m = runner.invoke(mailbox.mailbox, ["status", "m1", "--json"])
    assert n.exit_code == 0, n.output
    assert json.loads(n.stdout) == {
        "id": "n1",
        "status": "active",
        "phone_number": "+123",
        "device_id": "d1",
    }
    assert m.exit_code == 0, m.output
    assert json.loads(m.stdout) == {
        "id": "m1",
        "status": "active",
        "address": "a@example.com",
    }


def test_code_timeout_is_actionable(monkeypatch):
    from types import SimpleNamespace as N

    for module, group, resource in (
        (number, number.number, "n1"),
        (mailbox, mailbox.mailbox, "m1"),
    ):
        fake = N(wait_for_code=AsyncMock(side_effect=TimeoutError()))
        monkeypatch.setattr(
            module,
            "PhoneNumber" if module is number else "Mailbox",
            lambda *a, fake=fake, **k: fake,
        )
        result = CliRunner().invoke(
            group, ["code", resource, "--timeout", "2", "--sender", "example"]
        )
        assert result.exit_code != 0
        assert (
            resource in result.output
            and "example" in result.output
            and "2" in result.output
            and "retry" in result.output.lower()
        )
        assert "Waiting" in result.stderr


def test_number_messages_countries_and_send(monkeypatch):
    from types import SimpleNamespace as N

    from pydantic import BaseModel

    class Message(BaseModel):
        body: str

    messages = AsyncMock(return_value=[Message(body="hello")])
    countries = AsyncMock(return_value=N(data=N(items=[Message(body="US")])))
    send = AsyncMock()
    monkeypatch.setattr(number.cloud_numbers, "messages", messages)
    monkeypatch.setattr(number.cloud_numbers, "countries", countries)
    monkeypatch.setattr(number.cloud_numbers, "send", send)
    runner = CliRunner()
    result = runner.invoke(number.number, ["messages", "n1", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == [{"body": "hello"}]
    result = runner.invoke(number.number, ["countries", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == [{"body": "US"}]
    assert runner.invoke(number.number, ["send", "n1", "+123", "Hi"]).exit_code != 0
    send.assert_not_awaited()
    result = runner.invoke(
        number.number, ["send", "n1", "+123", "Hi", "--yes", "--billing", "paid"]
    )
    assert result.exit_code == 0, result.output
    send.assert_awaited_once()


def test_send_requires_paid_billing(monkeypatch):
    send = AsyncMock()
    monkeypatch.setattr(number.cloud_numbers, "send", send)
    result = CliRunner().invoke(
        number.number, ["send", "n1", "+123", "Hi", "--yes", "--billing", "no_commerce"]
    )
    assert result.exit_code != 0
    send.assert_not_awaited()


def test_create_reports_address_and_state(monkeypatch):
    from types import SimpleNamespace as N

    for module, group, name, resource, data in (
        (
            number,
            number.number,
            "PhoneNumber",
            "n1",
            N(id="n1", state="active", phone_number="+123"),
        ),
        (
            mailbox,
            mailbox.mailbox,
            "Mailbox",
            "m1",
            N(id="m1", status="active", address="a@example.com"),
        ),
    ):
        monkeypatch.setattr(module, name, lambda *a, **k: object())
        monkeypatch.setattr(
            module, "complete_checkout", AsyncMock(return_value=N(id=resource))
        )
        client = number.cloud_numbers if module is number else mailbox.cloud_mailboxes
        monkeypatch.setattr(client, "retrieve", AsyncMock(return_value=N(data=data)))
        result = CliRunner().invoke(group, ["create", "--yes", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["id"] == resource and payload["status"] == "active"
        assert (
            payload.get("phone_number") == "+123"
            or payload.get("address") == "a@example.com"
        )


def test_release_json(monkeypatch):
    from types import SimpleNamespace as N

    for module, group, name, resource in (
        (number, number.number, "PhoneNumber", "n1"),
        (mailbox, mailbox.mailbox, "Mailbox", "m1"),
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda *a, **k: N(
                release=AsyncMock(
                    return_value=N(
                        data=N(state="cancel_scheduled", status="cancel_scheduled")
                    )
                )
            ),
        )
        result = CliRunner().invoke(group, ["release", resource, "--yes", "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == {
            "id": resource,
            "status": "cancel_scheduled",
        }


def test_workspace_json_operations_and_interactive_confirmation(monkeypatch, tmp_path):
    from types import SimpleNamespace as N

    from mobilerun.cli.base import menu

    path = tmp_path / "input.txt"
    path.write_text("hello")
    monkeypatch.setattr(
        workspace.Workspace,
        "upload_file",
        AsyncMock(return_value=N(id="f1", name="input.txt")),
    )
    monkeypatch.setattr(
        workspace.Workspace,
        "download_file",
        AsyncMock(return_value=tmp_path / "input.txt"),
    )
    monkeypatch.setattr(workspace.Workspace, "delete_file", AsyncMock())
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    from mobilerun.cli.base import safety

    async def confirm(message, **kwargs):
        print(message)
        return True

    monkeypatch.setattr(safety, "confirm_prompt_async", confirm)
    runner = CliRunner()
    for args, status in (
        (["upload", str(path)], "uploaded"),
        (["download", "f1"], "downloaded"),
        (["rm", "f1"], "deleted"),
    ):
        result = runner.invoke(
            workspace.workspace, ["files", *args, "--json"], input="y\n"
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert (payload[0] if status == "uploaded" else payload)["status"] == status
        if status == "deleted":
            assert "Continue?" in result.stderr


def test_proxy_json_buy_bind(monkeypatch):
    from types import SimpleNamespace as N

    from mobilerun.cli.base import menu

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    from mobilerun.cli.base import safety

    async def confirm(message, **kwargs):
        print(message)
        return True

    monkeypatch.setattr(safety, "confirm_prompt_async", confirm)

    async def buy(*a, **k):
        print("Complete payment: example")
        return N(id="p1", status="active")

    monkeypatch.setattr(proxy, "buy_proxy", buy)
    phone = N(close=AsyncMock())
    extension = N(bind=AsyncMock(), status=AsyncMock(return_value=N(connected=True)))
    monkeypatch.setattr(proxy, "CloudDevice", lambda **k: phone)
    monkeypatch.setattr(proxy, "CloudFixedProxy", lambda *a: extension)
    runner = CliRunner()
    result = runner.invoke(
        proxy.proxy,
        ["buy", "--country", "US", "--billing", "paid", "--json"],
        input="y\n",
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["id"] == "p1"
    assert "Continue?" in result.stderr and "Complete payment" in result.stderr
    result = runner.invoke(proxy.proxy, ["bind", "p1", "d1", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "id": "p1",
        "status": "connected",
        "device_id": "d1",
    }
    assert "Waiting" in result.stderr


def test_create_retrieval_failure_preserves_success(monkeypatch):
    from types import SimpleNamespace as N

    for module, group, name, client in (
        (number, number.number, "PhoneNumber", number.cloud_numbers),
        (mailbox, mailbox.mailbox, "Mailbox", mailbox.cloud_mailboxes),
    ):
        monkeypatch.setattr(module, name, lambda *a, **k: object())
        monkeypatch.setattr(
            module, "complete_checkout", AsyncMock(return_value=N(id="r1"))
        )
        monkeypatch.setattr(
            client,
            "retrieve",
            AsyncMock(side_effect=RuntimeError("network unavailable")),
        )
        result = CliRunner().invoke(group, ["create", "--yes", "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == {"id": "r1", "status": "unknown"}
        assert (
            "r1" in result.stderr
            and "created; details unavailable" in result.stderr
            and "RuntimeError" in result.stderr
        )


def test_timeout_resolves_address_and_falls_back(monkeypatch):
    from types import SimpleNamespace as N

    for module, group, name, client, attr, address in (
        (
            number,
            number.number,
            "PhoneNumber",
            number.cloud_numbers,
            "phone_number",
            "+123",
        ),
        (
            mailbox,
            mailbox.mailbox,
            "Mailbox",
            mailbox.cloud_mailboxes,
            "address",
            "a@example.com",
        ),
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda *a, **k: N(wait_for_code=AsyncMock(side_effect=TimeoutError())),
        )
        monkeypatch.setattr(
            client, "retrieve", AsyncMock(return_value=N(data=N(**{attr: address})))
        )
        result = CliRunner().invoke(group, ["code", "r1", "--timeout", "2"])
        assert address in result.output
        monkeypatch.setattr(
            client, "retrieve", AsyncMock(side_effect=RuntimeError("offline"))
        )
        result = CliRunner().invoke(group, ["code", "r1", "--timeout", "2"])
        assert "r1" in result.output and "Retry" in result.output


def test_proxy_cancel_wrapper_sdk_and_error_mapping(monkeypatch):
    import asyncio
    from types import SimpleNamespace as N

    import httpx
    import pytest
    from mobilerun_sdk import APIStatusError

    from mobilerun.clients.cloud import proxies as client_proxies
    from mobilerun.errors.cloud import CloudResourceNotFound
    from mobilerun.models.billing import Billing

    cancel = AsyncMock()
    context = AsyncMock()
    context.__aenter__.return_value = N(connect=N(proxies=N(cancel=cancel)))
    monkeypatch.setattr(client_proxies, "cloud_client", lambda *a, **k: context)
    asyncio.run(client_proxies.cancel_proxy("p1", billing=Billing.NO_COMMERCE))
    cancel.assert_awaited_once_with("p1")
    response = httpx.Response(
        404, request=httpx.Request("DELETE", "https://example.test/proxy")
    )
    cancel.side_effect = APIStatusError(
        "secret server response", response=response, body=None
    )
    with pytest.raises(CloudResourceNotFound) as exc:
        asyncio.run(client_proxies.cancel_proxy("p1", billing=Billing.NO_COMMERCE))
    assert "secret server response" not in str(exc.value)


def test_confirmed_number_and_mailbox_release_json_is_clean(monkeypatch):
    from types import SimpleNamespace as N

    from mobilerun.cli.base import menu

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    from mobilerun.cli.base import safety

    async def confirm(message, **kwargs):
        print(message)
        return True

    monkeypatch.setattr(safety, "confirm_prompt_async", confirm)
    for module, group, name in (
        (number, number.number, "PhoneNumber"),
        (mailbox, mailbox.mailbox, "Mailbox"),
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda *a, **k: N(
                release=AsyncMock(
                    return_value=N(
                        data=N(state="cancel_scheduled", status="cancel_scheduled")
                    )
                )
            ),
        )
        result = CliRunner().invoke(group, ["release", "r1", "--json"], input="y\n")
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == {"id": "r1", "status": "cancel_scheduled"}
        assert "Continue?" in result.stderr


def test_workspace_apps_mutation_json(monkeypatch, tmp_path):
    from types import SimpleNamespace as N

    from mobilerun.cli.base import menu

    apk = tmp_path / "a.apk"
    apk.write_bytes(b"apk")
    monkeypatch.setattr(
        workspace.Workspace,
        "upload_app",
        AsyncMock(return_value=N(id="a1", name="App", package="pkg")),
    )
    monkeypatch.setattr(workspace.Workspace, "delete_app", AsyncMock())
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    from mobilerun.cli.base import safety

    async def confirm(message, **kwargs):
        print(message)
        return True

    monkeypatch.setattr(safety, "confirm_prompt_async", confirm)
    runner = CliRunner()
    result = runner.invoke(workspace.workspace, ["apps", "upload", str(apk), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["id"] == "a1"
    assert "Uploading" in result.stderr
    result = runner.invoke(
        workspace.workspace, ["apps", "remove", "a1", "--json"], input="y\n"
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"id": "a1", "status": "deleted"}
    assert "Continue?" in result.stderr


def test_create_warning_does_not_expose_unexpected_exception_text(monkeypatch):
    from types import SimpleNamespace as N

    monkeypatch.setattr(number, "PhoneNumber", lambda **k: object())
    monkeypatch.setattr(number, "complete_checkout", AsyncMock(return_value=N(id="n1")))
    monkeypatch.setattr(
        number.cloud_numbers,
        "retrieve",
        AsyncMock(side_effect=RuntimeError("secret-token")),
    )
    result = CliRunner().invoke(number.number, ["create", "--yes", "--json"])
    assert result.exit_code == 0
    assert "secret-token" not in result.output
    assert "RuntimeError" in result.stderr


def test_confirmed_number_mailbox_create_json_and_checkout_stderr(monkeypatch):
    from types import SimpleNamespace as N

    from mobilerun.cli.base import menu

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    from mobilerun.cli.base import safety

    async def confirm(message, **kwargs):
        print(message)
        return True

    monkeypatch.setattr(safety, "confirm_prompt_async", confirm)
    for module, group, name, client in (
        (number, number.number, "PhoneNumber", number.cloud_numbers),
        (mailbox, mailbox.mailbox, "Mailbox", mailbox.cloud_mailboxes),
    ):

        async def purchase():
            print("Complete payment: example")
            return N(id="r1")

        monkeypatch.setattr(module, name, lambda **k: purchase())
        monkeypatch.setattr(
            client,
            "retrieve",
            AsyncMock(
                return_value=N(
                    data=N(
                        id="r1",
                        state="active",
                        status="active",
                        phone_number="+123",
                        address="a@example.com",
                    )
                )
            ),
        )
        result = CliRunner().invoke(group, ["create", "--json"], input="y\n")
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["id"] == "r1"
        assert "Continue?" in result.stderr and "Complete payment" in result.stderr
