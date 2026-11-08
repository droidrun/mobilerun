import asyncio

import pytest

from mobilerun.agent.utils.portal_setup import adb_install_verifier_enabled
from mobilerun.cli import doctor, main

VERIFIER_SETTING = "settings get global verifier_verify_adb_installs"


class FakeDevice:
    serial = "emulator-5554"

    def __init__(self, verifier="0", install_error=None):
        self.verifier = verifier
        self.install_error = install_error
        self.shell_calls = []

    async def shell(self, cmd):
        self.shell_calls.append(cmd)
        if cmd == VERIFIER_SETTING:
            if isinstance(self.verifier, Exception):
                raise self.verifier
            return self.verifier
        return ""

    async def install(self, path, **kwargs):
        if self.install_error:
            raise self.install_error


@pytest.fixture
def run_setup(monkeypatch, capsys):
    def run(device, setup_ok=False, **kwargs):
        async def fake_device(serial):
            return device

        async def fake_setup_portal(*args, **kwargs):
            return setup_ok

        monkeypatch.setattr(main.adb, "device", fake_device)
        monkeypatch.setattr(main, "setup_portal", fake_setup_portal)
        kwargs.setdefault("path", None)
        asyncio.run(main._setup_portal(device=device.serial, debug=False, **kwargs))
        # Rich wraps long lines; compare on single spaces.
        return " ".join(capsys.readouterr().out.split())

    return run


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("1\n", True), ("0", False), ("null", False), ("", False)],
)
def test_verifier_setting_is_read(value, expected):
    device = FakeDevice(verifier=value)

    assert asyncio.run(adb_install_verifier_enabled(device)) is expected


def test_unreadable_verifier_setting_counts_as_off():
    device = FakeDevice(verifier=RuntimeError("device offline"))

    assert asyncio.run(adb_install_verifier_enabled(device)) is False


def test_failed_install_with_verifier_on_explains_play_protect(run_setup):
    out = run_setup(FakeDevice(verifier="1"))

    assert "Watch your device" in out
    assert "Google Play Protect may have blocked the install" in out
    assert "Verify apps over USB" in out
    assert "Run 'mobilerun doctor' for diagnostics." in out


def test_failed_install_with_verifier_off_keeps_generic_message(run_setup):
    out = run_setup(FakeDevice(verifier="0"))

    assert "Setup failed. Run 'mobilerun doctor' for diagnostics." in out
    assert "Play Protect" not in out


def test_successful_install_with_verifier_on_shows_only_the_notice(run_setup):
    out = run_setup(FakeDevice(verifier="1"), setup_ok=True)

    assert "Watch your device" in out
    assert "Setup complete!" in out
    assert "may have blocked" not in out


def test_failure_from_doctor_does_not_suggest_doctor(run_setup):
    out = run_setup(FakeDevice(verifier="1"), suggest_doctor=False)

    assert "Setup failed." in out
    assert "mobilerun doctor" not in out
    assert "may have blocked" in out


def test_explicit_apk_install_failure_explains_play_protect(run_setup, tmp_path):
    apk = tmp_path / "portal.apk"
    apk.write_bytes(b"apk")
    device = FakeDevice(
        verifier="1",
        install_error=RuntimeError("INSTALL_FAILED_VERIFICATION_FAILURE"),
    )

    out = run_setup(device, path=str(apk))

    assert "Installation failed: INSTALL_FAILED_VERIFICATION_FAILURE" in out
    assert "Google Play Protect may have blocked the install" in out


def test_doctor_runs_setup_without_suggesting_doctor(monkeypatch):
    device = FakeDevice()
    calls = []

    async def passing(name):
        return doctor.CheckResult(name, doctor.Status.PASS, "ok")

    async def fake_setup_portal(**kwargs):
        calls.append(kwargs)

    async def portal_missing(device, debug):
        return doctor.CheckResult("Portal", doctor.Status.FAIL, "not installed"), False

    async def fake_check_sdk_version(debug):
        return await passing("SDK Version")

    async def fake_check_config(debug):
        return await passing("Config"), None

    async def fake_check_adb(debug):
        return await passing("ADB")

    async def fake_check_device(serial, debug):
        return await passing("Device"), device

    monkeypatch.setattr(doctor, "check_sdk_version", fake_check_sdk_version)
    monkeypatch.setattr(doctor, "check_config", fake_check_config)
    monkeypatch.setattr(doctor, "check_adb", fake_check_adb)
    monkeypatch.setattr(doctor, "check_device", fake_check_device)
    monkeypatch.setattr(doctor, "check_portal_installed", portal_missing)
    monkeypatch.setattr(main, "_setup_portal", fake_setup_portal)

    asyncio.run(doctor.run_doctor(device.serial))

    assert len(calls) == 1
    assert calls[0]["suggest_doctor"] is False
