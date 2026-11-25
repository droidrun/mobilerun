from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import test_cli_doctor_repairs
from click.testing import CliRunner

from mobilerun.diagnostics.checks import ReadyTcp
from mobilerun.diagnostics.results import CheckResult, Status

checks = test_cli_doctor_repairs.checks


@pytest.mark.parametrize("lost_forward", [True, False])
def test_plain_doctor_uses_only_existing_tcp_connection(
    checks, monkeypatch, lost_forward
):
    from mobilerun.diagnostics import checks as diagnostics

    module, ok, _ = checks
    device = module.check_device.return_value[1]

    async def forwards():
        yield SimpleNamespace(serial=device.serial, local="tcp:1234", remote="tcp:8080")

    device.forward_list = forwards
    device.forward_port = AsyncMock()
    device.shell = AsyncMock()
    connect = AsyncMock(side_effect=AssertionError("must not repair"))
    monkeypatch.setattr(diagnostics.PortalClient, "connect", connect)
    monkeypatch.setattr(
        diagnostics.PortalClient,
        "_fetch_auth_token",
        AsyncMock(return_value="test-token"),
    )
    paths = []

    def respond(request):
        assert request.url.port == 1234
        assert request.headers["Authorization"] == "Bearer test-token"
        paths.append(request.url.path)
        if request.url.path == "/ping":
            return httpx.Response(200)
        if lost_forward:
            raise httpx.ConnectError("forward disappeared")
        if request.url.path == "/state_full":
            return httpx.Response(
                200,
                json={
                    "result": dict.fromkeys(
                        ("a11y_tree", "phone_state", "device_context")
                    )
                },
            )
        assert request.url.path == "/screenshot"
        return httpx.Response(200, json={"status": "success", "result": "cG5n"})

    client = httpx.AsyncClient
    monkeypatch.setattr(
        diagnostics.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(respond), **kw),
    )

    async def state(device, debug, **kwargs):
        if not kwargs:
            return ok, None
        return await diagnostics.check_portal_state(device, debug, **kwargs)

    module.check_portal_state.side_effect = state
    module.check_portal_installed.return_value = (ok, True)
    monkeypatch.setattr(module, "check_tcp", diagnostics.check_tcp)
    result = CliRunner().invoke(module.doctor, ["-d", device.serial])
    assert result.exit_code == int(lost_forward), result.output
    assert paths == ["/ping", "/state_full", *([] if lost_forward else ["/screenshot"])]
    connect.assert_not_awaited()
    device.forward_port.assert_not_awaited()
    device.shell.assert_not_awaited()


def test_plain_doctor_reports_broken_tcp_screenshot(checks, monkeypatch):
    module, ok, _ = checks
    tcp_portal = object()
    module.check_portal_installed.return_value = (ok, True)
    module.check_tcp.return_value = ReadyTcp(1234, {}, "ping ok")
    module.check_portal_state.side_effect = [(ok, None), (ok, tcp_portal)]
    screenshot = AsyncMock(
        return_value=CheckResult("Screenshot", Status.FAIL, "empty response")
    )
    monkeypatch.setattr(module, "check_screenshot", screenshot)
    result = CliRunner().invoke(module.doctor, ["-d", "serial"])
    assert result.exit_code == 1, result.output
    assert "Screenshot (tcp): empty response" in result.output
    screenshot.assert_awaited_once_with(tcp_portal, False)
    module.check_tcp.assert_awaited_once_with(
        module.check_device.return_value[1], False, fix=False
    )
