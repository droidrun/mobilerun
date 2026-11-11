"""Terminal QR rendering and local wireless debugging at the CLI boundary."""

import io
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner

from mobilerun.cli import auth, device_lifecycle, device_pair, deviceauth, qr
from mobilerun.cli.main import cli
from mobilerun.config_manager import ConfigLoader


class TerminalRunner(CliRunner):
    @contextmanager
    def isolation(self, *args, **kwargs):
        with super().isolation(*args, **kwargs) as streams:
            sys.stderr.isatty = lambda: True
            yield streams


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setattr(ConfigLoader, "get_user_config_path", lambda: tmp_path / "config.yaml")
    for name in ("MOBILERUN_CONFIG", "MOBILERUN_DEBUG", "MOBILERUN_AUTH_URL"):
        monkeypatch.delenv(name, raising=False)


def test_print_qr_skips_nonterminal(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(qr.sys, "stderr", output)
    assert qr.print_qr("https://example.test/verify") is False
    assert output.getvalue() == ""


def test_print_qr_uses_stderr_and_preserves_quiet_zone(monkeypatch):
    output = io.StringIO()
    output.isatty = lambda: True
    monkeypatch.setattr(qr.sys, "stderr", output)
    payload = "https://example.test/verify?user_code=ABCD-EFGH"
    code = qr.segno.make_qr(payload)
    make = Mock(return_value=code)
    monkeypatch.setattr(qr.segno, "make_qr", make)
    assert qr.print_qr(payload) is True
    make.assert_called_once_with(payload)
    lines = output.getvalue().splitlines()
    assert len(lines) > 10
    assert all(line.startswith("\x1b[97;40m") and line.endswith("\x1b[0m") for line in lines)
    # Two compact rows preserve the four-module light quiet zone.
    assert lines[:2] == ["\x1b[97;40m" + "█" * code.symbol_size()[0] + "\x1b[0m"] * 2


@pytest.mark.parametrize("error", [ValueError("cannot encode"), UnicodeError("cannot render")])
def test_print_qr_encoding_failure_returns_false(monkeypatch, error):
    output = io.StringIO()
    output.isatty = lambda: True
    monkeypatch.setattr(qr.sys, "stderr", output)
    monkeypatch.setattr(qr.segno, "make_qr", Mock(side_effect=error))
    assert qr.print_qr("https://example.test/verify") is False
    assert output.getvalue() == ""


@pytest.mark.parametrize("error", [OSError("cannot write"), UnicodeError("unsupported encoding")])
def test_print_qr_write_failure_returns_false(monkeypatch, error):
    output = io.StringIO()
    output.isatty = lambda: True
    output.write = Mock(side_effect=error)
    monkeypatch.setattr(qr.sys, "stderr", output)
    assert qr.print_qr("https://example.test/verify") is False


@pytest.fixture
def login(monkeypatch, tmp_path):
    import webbrowser

    link = "https://example.test/verify?user_code=ABCD-EFGH"
    code = deviceauth.CodeResponse("device-code", "ABCD-EFGH", "https://example.test/verify",
                                   link, 120, 5)
    request = Mock(return_value=code)
    poll = Mock(return_value="test-session")
    save = Mock(return_value=tmp_path / "credentials.json")
    browser = Mock()
    monkeypatch.setattr(deviceauth, "request_code", request)
    monkeypatch.setattr(deviceauth, "poll_token", poll)
    monkeypatch.setattr(deviceauth, "save_token", save)
    monkeypatch.setattr(deviceauth, "fetch_session", lambda *_: None)
    monkeypatch.setattr(webbrowser, "open", browser)
    printed = Mock(side_effect=qr.print_qr)
    monkeypatch.setattr(auth, "print_qr", printed)
    return SimpleNamespace(code=code, request=request, poll=poll, save=save,
                           browser=browser, printed=printed)


def test_login_prints_complete_link_qr_and_preserves_flow(login):
    result = TerminalRunner().invoke(cli, ["auth", "login"])
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "ABCD-EFGH" in result.stderr and "Visit:" in result.stderr
    login.printed.assert_called_once_with("https://example.test/verify?user_code=ABCD-EFGH")
    assert "\x1b[97;40m" in result.stderr
    login.browser.assert_called_once_with("https://example.test/verify?user_code=ABCD-EFGH")
    login.request.assert_called_once_with(auth.DEFAULT_AUTH_URL, auth.DEFAULT_DEVICE_CLIENT_ID)
    login.poll.assert_called_once_with(auth.DEFAULT_AUTH_URL, auth.DEFAULT_DEVICE_CLIENT_ID, login.code)
    login.save.assert_called_once_with("test-session", auth.DEFAULT_AUTH_URL)


def test_login_no_qr_preserves_browser_and_polling(login):
    result = TerminalRunner().invoke(cli, ["auth", "login", "--no-qr"])
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    login.printed.assert_not_called()
    login.browser.assert_called_once_with("https://example.test/verify?user_code=ABCD-EFGH")
    login.poll.assert_called_once_with(auth.DEFAULT_AUTH_URL, auth.DEFAULT_DEVICE_CLIENT_ID, login.code)


def test_login_without_complete_link_skips_qr_and_browser(login):
    login.code.verification_uri_complete = ""
    result = TerminalRunner().invoke(cli, ["auth", "login"])
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    login.printed.assert_not_called()
    login.browser.assert_not_called()
    login.poll.assert_called_once_with(auth.DEFAULT_AUTH_URL, auth.DEFAULT_DEVICE_CLIENT_ID, login.code)


def test_login_continues_after_qr_render_failure(monkeypatch, login):
    monkeypatch.setattr(qr.segno, "make_qr", Mock(side_effect=ValueError("cannot encode")))
    result = TerminalRunner().invoke(cli, ["auth", "login"])
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "Visit:" in result.stderr and "ABCD-EFGH" in result.stderr
    login.browser.assert_called_once_with("https://example.test/verify?user_code=ABCD-EFGH")
    login.poll.assert_called_once_with(auth.DEFAULT_AUTH_URL, auth.DEFAULT_DEVICE_CLIENT_ID, login.code)
    login.save.assert_called_once_with("test-session", auth.DEFAULT_AUTH_URL)


@pytest.mark.parametrize("service", ["openai", "anthropic", "gemini", "xai"])
def test_no_qr_is_not_a_provider_option(service):
    result = CliRunner().invoke(cli, ["auth", "login", service, "--no-qr"])
    assert result.exit_code == 2
    assert f"--no-qr is not supported for {service}" in result.stderr
    assert result.stdout == ""


def test_mdns_parser_reads_services_and_ignores_header():
    output = (
        "List of discovered mdns services\n"
        "other-device\t_adb-tls-pairing._tcp\t192.0.2.2:30001\n"
        "mobilerun-a1b2c3\t_adb-tls-pairing._tcp.\t192.0.2.1:30002\n"
        "adb-device\t_adb-tls-connect._tcp\t192.0.2.1:30003\n"
        "ipv6\t_adb-tls-connect._tcp\t[2001:db8::1]:30004\n"
        "legacy\t_adb._tcp\t192.0.2.1:5555\n"
        "broken\t_adb-tls-pairing._tcp\tnot-an-address\n"
    )
    services = device_pair.parse_mdns_services(output)
    assert services == [
        ("other-device", "_adb-tls-pairing._tcp", "192.0.2.2:30001"),
        ("mobilerun-a1b2c3", "_adb-tls-pairing._tcp", "192.0.2.1:30002"),
        ("adb-device", "_adb-tls-connect._tcp", "192.0.2.1:30003"),
        ("legacy", "_adb._tcp", "192.0.2.1:5555"),
        ("ipv6", "_adb-tls-connect._tcp", "[2001:db8::1]:30004"),
    ]
    assert services[-1].ip == "[2001:db8::1]"


@pytest.mark.parametrize("address", [
    "2001:db8::1:30004", "fe80::1%en0:30004", "[fe80::1%en0]:30004",
    "169.254.1.2:30004", "192.0.2.1:0", "192.0.2.1:65536", "invalid:30004",
])
def test_mdns_parser_skips_unsafe_addresses(address):
    assert device_pair.parse_mdns_services(
        f"mobilerun-a1b2c3\t_adb-tls-pairing._tcp\t{address}"
    ) == []


@pytest.fixture
def pairing(monkeypatch):
    monkeypatch.setattr(device_pair.shutil, "which", lambda _: "/fake/adb")
    printed = Mock(return_value=True)
    monkeypatch.setattr(device_pair, "print_qr", printed)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(device_pair, "time", SimpleNamespace(
        monotonic=lambda: clock.now, sleep=lambda seconds: setattr(clock, "now", clock.now + seconds)
    ))
    choices = iter("a1b2c3" + "Ab12Cd34Ef")
    monkeypatch.setattr(device_pair.secrets, "choice", lambda alphabet: next(choices))
    responses = [
        (0, "mdns daemon version [Openscreen discovery 0.0.0]", ""),
        (0, "other\t_adb-tls-pairing._tcp\t192.0.2.2:31001\n"
            "mobilerun-a1b2c3-extra\t_adb-tls-pairing._tcp\t192.0.2.2:31002", ""),
        (0, "mobilerun-a1b2c3\t_adb-tls-pairing._tcp\t192.0.2.1:32001", ""),
        (0, "Enter pairing code: Successfully paired to 192.0.2.1:32001 [guid=adb-device]", ""),
        (0, "unrelated\t_adb-tls-connect._tcp\t192.0.2.2:33001", ""),
        (0, "adb-device\t_adb-tls-connect._tcp\t192.0.2.1:33002", ""),
        (0, "connected to 192.0.2.1:33002", ""),
    ]
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        status, out, err = responses.pop(0)
        return subprocess.CompletedProcess(["adb", *args], status, out, err)

    monkeypatch.setattr(device_pair, "_run_adb", run)
    return SimpleNamespace(responses=responses, calls=calls, printed=printed, clock=clock)


def test_pair_sequences_discovery_pair_and_connect_without_exposing_password(pairing):
    result = TerminalRunner().invoke(cli, ["device", "pair", "-v"])
    assert result.exit_code == 0, result.output
    assert result.stdout == "192.0.2.1:33002\n"
    assert "Paired and connected" in result.stderr
    assert "mobilerun device setup -d 192.0.2.1:33002" in result.stderr
    assert pairing.printed.call_args.args == ("WIFI:T:ADB;S:mobilerun-a1b2c3;P:Ab12Cd34Ef;;",)
    assert [args for args, _ in pairing.calls] == [
        ("mdns", "check"), ("mdns", "services"), ("mdns", "services"),
        ("pair", "192.0.2.1:32001"), ("mdns", "services"), ("mdns", "services"),
        ("connect", "192.0.2.1:33002"),
    ]
    assert pairing.calls[3][1]["input"] == "Ab12Cd34Ef\n"
    assert all("Ab12Cd34Ef" not in " ".join(args) for args, _ in pairing.calls)
    assert all(kwargs["input"] is None for _, kwargs in pairing.calls[:3] + pairing.calls[4:])
    assert "Ab12Cd34Ef" not in result.output
    assert pairing.clock.now == 2


def test_pair_prefers_ipv4_when_ipv6_is_listed_first(pairing):
    pairing.responses[2] = (0,
        "mobilerun-a1b2c3\t_adb-tls-pairing._tcp\t[2001:db8::1]:32001\n"
        "mobilerun-a1b2c3\t_adb-tls-pairing._tcp\t192.0.2.1:32001", "")
    pairing.responses[5] = (0,
        "adb-device\t_adb-tls-connect._tcp\t[2001:db8::1]:33002\n"
        "adb-device\t_adb-tls-connect._tcp\t192.0.2.1:33002", "")
    result = TerminalRunner().invoke(cli, ["device", "pair"])
    assert result.exit_code == 0, result.output
    assert result.stdout == "192.0.2.1:33002\n"
    assert pairing.calls[3][0] == ("pair", "192.0.2.1:32001")
    assert pairing.calls[-1][0] == ("connect", "192.0.2.1:33002")


def test_pair_accepts_bracketed_global_ipv6_when_no_ipv4_exists(pairing):
    pairing.responses[2] = (0,
        "mobilerun-a1b2c3\t_adb-tls-pairing._tcp\t[2001:db8::1]:32001", "")
    pairing.responses[5] = (0,
        "adb-device\t_adb-tls-connect._tcp\t[2001:db8::1]:33002", "")
    result = TerminalRunner().invoke(cli, ["device", "pair"])
    assert result.exit_code == 0, result.output
    assert result.stdout == "[2001:db8::1]:33002\n"
    assert pairing.calls[3][0] == ("pair", "[2001:db8::1]:32001")
    assert pairing.calls[-1][0] == ("connect", "[2001:db8::1]:33002")


CONNECT_REPLIES = [
    ("connected to 192.0.2.1:33002", 0),
    ("already connected to 192.0.2.1:33002", 0),
    ("daemon started\nconnected to 192.0.2.1:33002", 0),
    ("not connected to 192.0.2.1:33002", 1),
    ("error: connected to wrong service", 1),
]


@pytest.mark.parametrize("reply,exit_code", CONNECT_REPLIES)
def test_pair_connect_reply(reply, exit_code, pairing):
    pairing.responses[-1] = (0, reply, "")
    result = TerminalRunner().invoke(cli, ["device", "pair"])
    assert result.exit_code == exit_code, result.output


@pytest.mark.parametrize("reply,exit_code", CONNECT_REPLIES)
def test_device_connect_reply(monkeypatch, reply, exit_code):
    monkeypatch.setattr(device_lifecycle.adb, "connect", AsyncMock(return_value=reply))
    result = CliRunner().invoke(cli, ["device", "connect", "192.0.2.1:33002"])
    assert result.exit_code == exit_code, result.output


def test_pair_fails_if_qr_cannot_be_rendered(monkeypatch, pairing):
    monkeypatch.setattr(device_pair, "print_qr", qr.print_qr)
    monkeypatch.setattr(qr.segno, "make_qr", Mock(side_effect=ValueError("cannot encode")))
    result = TerminalRunner().invoke(cli, ["device", "pair"])
    assert result.exit_code == 1
    assert "Could not display the pairing QR code." in result.stderr
    assert result.stdout == ""
    assert [args for args, _ in pairing.calls] == [("mdns", "check")]


@pytest.mark.parametrize("response_count,limit", [(2, 1), (5, 2)])
def test_pair_timeout(pairing, response_count, limit):
    pairing.responses[:] = pairing.responses[:response_count]
    result = TerminalRunner().invoke(cli, ["device", "pair", "--timeout", str(limit)])
    assert result.exit_code == 1
    assert "Timed out" in result.stderr
    assert result.stdout == ""
    assert pairing.clock.now == limit


@pytest.mark.parametrize("stage,index,message,expected", [
    ("pair", 3, "Failed pair: Ab12Cd34Ef", "Failed pair: [redacted]"),
    ("connect", 6, "Failed connect: refused", "Failed connect: refused"),
])
@pytest.mark.parametrize("returncode", [0, 1])
def test_pair_adb_failure_redacts_password(pairing, stage, index, message, expected, returncode):
    pairing.responses[index] = (returncode, message, "")
    result = TerminalRunner().invoke(cli, ["device", "pair", "-v"])
    assert result.exit_code == 1
    assert expected in result.stderr
    assert "Ab12Cd34Ef" not in result.output
    assert result.stdout == ""
    assert pairing.calls[-1][0][0] == stage


def test_pair_requires_mdns(pairing):
    pairing.responses[0] = (0, "ERROR: mdns discovery disabled", "")
    result = TerminalRunner().invoke(cli, ["device", "pair"])
    assert result.exit_code == 1
    assert "mDNS is unavailable" in result.stderr
    pairing.printed.assert_not_called()


def test_pair_reports_mdns_adb_failure(pairing):
    pairing.responses[0] = (1, "", "adb: mDNS discovery is disabled")
    result = TerminalRunner().invoke(cli, ["device", "pair"])
    assert result.exit_code == 1
    assert "adb: mDNS discovery is disabled" in result.stderr
    pairing.printed.assert_not_called()


def test_pair_bounds_adb_calls_by_timeout(monkeypatch, pairing):
    run = Mock(side_effect=subprocess.TimeoutExpired(["adb", "mdns", "check"], 1))
    monkeypatch.setattr(device_pair, "_run_adb", run)
    result = TerminalRunner().invoke(cli, ["device", "pair", "--timeout", "1"])
    assert result.exit_code == 1
    assert "Timed out" in result.stderr
    assert run.call_args.kwargs["timeout"] == 1


def test_pair_requires_adb(monkeypatch, pairing):
    monkeypatch.setattr(device_pair.shutil, "which", lambda _: None)
    result = TerminalRunner().invoke(cli, ["device", "pair"])
    assert result.exit_code == 1
    assert "adb is missing" in result.stderr
    assert not pairing.calls


def test_pair_requires_terminal(pairing):
    result = CliRunner().invoke(cli, ["device", "pair"])
    assert result.exit_code == 2
    assert "needs a terminal" in result.stderr
    assert not pairing.calls
    pairing.printed.assert_not_called()


@pytest.mark.parametrize("seconds", ["0", "-1", "nan", "inf"])
def test_pair_timeout_must_be_positive(seconds):
    result = CliRunner().invoke(cli, ["device", "pair", "--timeout", seconds])
    assert result.exit_code == 2


def test_pair_interrupt_exits_130(monkeypatch, pairing):
    monkeypatch.setattr(device_pair, "_run_adb", Mock(side_effect=KeyboardInterrupt))
    result = TerminalRunner().invoke(cli, ["device", "pair"])
    assert result.exit_code == 130
    assert result.stdout == ""


def test_adb_wrapper_passes_password_only_on_stdin(monkeypatch):
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "paired", ""))
    monkeypatch.setattr(device_pair.subprocess, "run", run)
    device_pair._run_adb("pair", "192.0.2.1:32001", input="test-password\n", timeout=3)
    run.assert_called_once_with(
        ["adb", "pair", "192.0.2.1:32001"], input="test-password\n", capture_output=True,
        text=True, timeout=3, check=False,
    )
