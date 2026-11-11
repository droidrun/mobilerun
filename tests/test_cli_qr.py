"""Terminal QR rendering and local wireless debugging at the CLI boundary."""

import io
import re
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from mobilerun.cli import auth, device_pair, deviceauth, options, qr
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
    # Segno's compact output renders light modules as full blocks: two
    # solid lines preserve the four-module quiet zone above the symbol.
    assert lines[:2] == ["█" * code.symbol_size()[0]] * 2
    assert any("█" in line or "▀" in line or "▄" in line for line in lines)


@pytest.mark.parametrize("no_qr,complete", [(False, True), (True, True), (False, False)])
def test_login_qr_keeps_existing_flow_and_stdout(monkeypatch, tmp_path, no_qr, complete):
    import webbrowser

    link = "https://example.test/verify?user_code=ABCD-EFGH" if complete else ""
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
    args = ["auth", "login"] + (["--no-qr"] if no_qr else [])
    result = TerminalRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "ABCD-EFGH" in result.stderr and "Visit:" in result.stderr
    if complete and not no_qr:
        printed.assert_called_once_with(link)
        assert "█" in result.stderr or "▀" in result.stderr or "▄" in result.stderr
    else:
        printed.assert_not_called()
    if complete:
        browser.assert_called_once_with(link)
    else:
        browser.assert_not_called()
    request.assert_called_once_with(auth.DEFAULT_AUTH_URL, auth.DEFAULT_DEVICE_CLIENT_ID)
    poll.assert_called_once_with(auth.DEFAULT_AUTH_URL, auth.DEFAULT_DEVICE_CLIENT_ID, code)
    save.assert_called_once_with("test-session", auth.DEFAULT_AUTH_URL)


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
        ("ipv6", "_adb-tls-connect._tcp", "[2001:db8::1]:30004"),
    ]
    assert services[-1].ip == "[2001:db8::1]"


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
        (0, "Successfully paired to 192.0.2.1:32001", ""),
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
    assert re.fullmatch(r"WIFI:T:ADB;S:mobilerun-[a-z0-9]{6};P:[a-zA-Z0-9]{10};;",
                        pairing.printed.call_args.args[0])
    assert [args for args, _ in pairing.calls] == [
        ("mdns", "check"), ("mdns", "services"), ("mdns", "services"),
        ("pair", "192.0.2.1:32001"), ("mdns", "services"), ("mdns", "services"),
        ("connect", "192.0.2.1:33002"),
    ]
    assert pairing.calls[3][1]["input"] == "Ab12Cd34Ef\n"
    assert all("Ab12Cd34Ef" not in " ".join(args) for args, _ in pairing.calls)
    assert all(kwargs["input"] is None for i, (_, kwargs) in enumerate(pairing.calls) if i != 3)
    assert "Ab12Cd34Ef" not in result.output
    assert pairing.clock.now == 2


@pytest.mark.parametrize("stage", ["pairing", "connection"])
def test_pair_timeout(pairing, stage):
    if stage == "pairing":
        pairing.responses[:] = pairing.responses[:2]
        limit = 1
    else:
        pairing.responses[:] = pairing.responses[:5]
        limit = 2
    result = TerminalRunner().invoke(cli, ["device", "pair", "--timeout", str(limit)])
    assert result.exit_code == 1
    assert "Timed out" in result.stderr
    assert result.stdout == ""
    assert pairing.clock.now == limit


@pytest.mark.parametrize("stage,index", [("pair", 3), ("connect", 6)])
@pytest.mark.parametrize("returncode", [0, 1])
def test_pair_adb_failure_redacts_password(pairing, stage, index, returncode):
    pairing.responses[index] = (returncode, f"Failed {stage}: Ab12Cd34Ef", "")
    result = TerminalRunner().invoke(cli, ["device", "pair", "-v"])
    assert result.exit_code == 1
    assert f"Failed {stage}: [redacted]" in result.stderr
    assert "Ab12Cd34Ef" not in result.output
    assert result.stdout == ""
    assert pairing.calls[-1][0][0] == stage


def test_pair_errors_use_shared_command_handler(monkeypatch, pairing):
    pairing.responses[3] = (1, "", "Pairing failed: Ab12Cd34Ef")
    convert = Mock(wraps=options.convert_api_error)
    monkeypatch.setattr(options, "convert_api_error", convert)
    result = TerminalRunner().invoke(cli, ["device", "pair"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Pairing failed: [redacted]" in result.stderr
    assert "Ab12Cd34Ef" not in result.output
    convert.assert_called_once()
    error, context = convert.call_args.args
    assert error.message == "Pairing failed: [redacted]"
    assert context.command is cli.commands["device"].commands["pair"]
    assert isinstance(context.command, options.Command)


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
