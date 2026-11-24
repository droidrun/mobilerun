"""Portal commands built by the CLI target the foreground Android user."""

import asyncio
import json

import pytest
from mobilerun_core_local.driver.android import AndroidDriver
from mobilerun_core_local.driver.android.portal import PORTAL_PACKAGE_NAME

import mobilerun.agent.utils.android_user as android_user
import mobilerun.cli.doctor as doctor
import mobilerun.cli.setup as setup
import mobilerun.tools.ui.provider as provider_module
from mobilerun.tools.ui.provider import AndroidStateProvider

PORTAL_URI = f"content://{PORTAL_PACKAGE_NAME}/"


def provider_row(**result):
    return f"Row: 0 result={json.dumps(result, separators=(',', ':'))}"


class MultiUserDevice:
    """A device with one Portal per Android user.

    Only the foreground user's Portal has a bound accessibility service, like
    on a real device after ``am switch-user``.
    """

    serial = "emulator-5560"

    def __init__(self, current_user=0, packages=None):
        self.current_user = current_user
        self.packages = packages or {0: [PORTAL_PACKAGE_NAME], 10: []}
        self.commands = []
        self.package_filters = []

    @staticmethod
    def _target_user(cmd):
        parts = cmd.split()
        if "--user" in parts:
            return int(parts[parts.index("--user") + 1])
        return 0

    async def shell(self, cmd, **kwargs):
        self.commands.append(cmd)
        if cmd == "am get-current-user":
            return f"{self.current_user}\n"
        if cmd.startswith("content "):
            if self._target_user(cmd) != self.current_user:
                return provider_row(
                    status="error", error="Accessibility service not available"
                )
            if cmd.endswith("/version"):
                return provider_row(status="success", result="0.6.2")
            if cmd.endswith("/ping"):
                return provider_row(status="success", result="pong")
            if cmd.endswith("/state"):
                return provider_row(status="success", result="{}")
        return ""

    async def list_packages(self, filter_list=None):
        self.package_filters.append(filter_list)
        user = 0
        if filter_list and "--user" in filter_list:
            user = int(filter_list[filter_list.index("--user") + 1])
        return list(self.packages.get(user, []))

    def portal_commands(self):
        return [c for c in self.commands if PORTAL_URI in c]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def sleep(_):
        return None

    for module in (doctor, setup, provider_module):
        monkeypatch.setattr(module.asyncio, "sleep", sleep)


# ── helpers ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "user, flag",
    [(None, ""), (0, ""), (10, " --user 10"), (True, ""), (-1, ""), ("10", "")],
)
@pytest.mark.parametrize("user_flag", [android_user.user_flag, android_user._user_flag])
def test_user_flag_only_for_secondary_users(user_flag, user, flag):
    assert user_flag(user) == flag


class _ShellReturning:
    def __init__(self, output=None, error=None):
        self.output, self.error = output, error

    async def shell(self, cmd):
        assert cmd == "am get-current-user"
        if self.error:
            raise self.error
        return self.output


@pytest.mark.parametrize(
    "read", [android_user.get_foreground_user, android_user._read_foreground_user]
)
@pytest.mark.parametrize(
    "device, expected",
    [
        (_ShellReturning("10\n"), 10),
        (_ShellReturning("0"), 0),
        (_ShellReturning("Error: unknown command"), None),
        (_ShellReturning(None), None),
        (_ShellReturning(error=RuntimeError("offline")), None),
    ],
)
def test_foreground_user_is_read_from_am(read, device, expected):
    assert asyncio.run(read(device)) == expected


def test_list_foreground_packages_queries_the_foreground_user():
    device = MultiUserDevice(current_user=10, packages={10: ["a.b"]})
    assert asyncio.run(android_user.list_foreground_packages(device)) == ["a.b"]
    assert device.package_filters == [["--user", "10"]]


def test_list_foreground_packages_keeps_user_0_call_shape():
    device = MultiUserDevice(current_user=0)
    asyncio.run(android_user.list_foreground_packages(device))
    assert device.package_filters == [None]


# ── mobilerun doctor ─────────────────────────────────────────────


def test_doctor_checks_the_foreground_users_portal():
    device = MultiUserDevice(
        current_user=10, packages={0: [PORTAL_PACKAGE_NAME], 10: []}
    )

    # Portal installed for user 0 only is not installed for user 10.
    result, installed = asyncio.run(doctor.check_portal_installed(device, False))
    assert (result.status, installed) == (doctor.Status.FAIL, False)

    device.packages[10] = [PORTAL_PACKAGE_NAME]
    result, installed = asyncio.run(doctor.check_portal_installed(device, False))
    assert (result.status, installed) == (doctor.Status.PASS, True)

    result = asyncio.run(doctor.check_content_provider(device, False))
    assert result.status == doctor.Status.PASS

    _, installed_version, _, _ = asyncio.run(doctor.check_portal_version(device, False))
    assert installed_version == "0.6.2"

    assert device.portal_commands() == [
        f"content query --user 10 --uri {PORTAL_URI}state",
        f"content query --user 10 --uri {PORTAL_URI}version",
    ]


def test_doctor_commands_unchanged_for_user_0():
    device = MultiUserDevice(current_user=0)
    asyncio.run(doctor.check_portal_installed(device, False))
    asyncio.run(doctor.check_content_provider(device, False))
    asyncio.run(doctor.check_portal_version(device, False))

    assert device.package_filters == [None]
    assert device.portal_commands() == [
        f"content query --uri {PORTAL_URI}state",
        f"content query --uri {PORTAL_URI}version",
    ]


class _FakePortalClient:
    instances = []

    def __init__(self, device, prefer_tcp=False):
        self.device = device
        self.events = []
        _FakePortalClient.instances.append(self)

    async def refresh_user(self):
        self.events.append("refresh_user")
        return True

    async def _fetch_auth_token(self):
        self.events.append("fetch_auth_token")
        return "token"


class _LegacyPortalClient(_FakePortalClient):
    """A core-local PortalClient without multi-user support."""

    refresh_user = None


class _NoForwardDevice(MultiUserDevice):
    def forward_list(self):
        raise RuntimeError("stop after the token step")


@pytest.mark.parametrize(
    "user, toggle_flag", [(10, " --user 10"), (0, "")], ids=["user10", "user0"]
)
def test_doctor_tcp_check_targets_the_foreground_user(monkeypatch, user, toggle_flag):
    _FakePortalClient.instances = []
    monkeypatch.setattr(doctor, "PortalClient", _FakePortalClient)
    device = _NoForwardDevice(current_user=user)

    asyncio.run(doctor.check_tcp(device, False))

    assert device.portal_commands() == [
        f"content insert{toggle_flag} --uri {PORTAL_URI}toggle_socket_server "
        "--bind enabled:b:true"
    ]
    # The token comes from the foreground user's Portal.
    (portal,) = _FakePortalClient.instances
    assert portal.events == ["refresh_user", "fetch_auth_token"]


def test_doctor_tcp_check_works_with_legacy_portal_client(monkeypatch):
    _FakePortalClient.instances = []
    monkeypatch.setattr(doctor, "PortalClient", _LegacyPortalClient)

    asyncio.run(doctor.check_tcp(_NoForwardDevice(current_user=10), False))

    (portal,) = _FakePortalClient.instances
    assert portal.events == ["fetch_auth_token"]


# ── mobilerun setup ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "user, flag", [(10, " --user 10"), (0, "")], ids=["user10", "user0"]
)
def test_setup_waits_for_the_foreground_users_portal(user, flag):
    device = MultiUserDevice(current_user=user)

    assert asyncio.run(setup._wait_for_portal_service(device, timeout=5)) is True
    assert device.portal_commands() == [f"content query{flag} --uri {PORTAL_URI}ping"]


def test_setup_wait_resolves_the_ping_user_once():
    class NeverReady(MultiUserDevice):
        async def shell(self, cmd, **kwargs):
            if cmd.startswith("content "):
                self.commands.append(cmd)
                return provider_row(status="error", error="not bound")
            return await super().shell(cmd, **kwargs)

    device = NeverReady(current_user=10)

    ready = asyncio.run(setup._wait_for_portal_service(device, timeout=0.05))

    assert ready is False
    pings = device.portal_commands()
    assert len(pings) > 1
    # Without core-local multi-user support only the ping needs the user.
    if android_user.get_foreground_user is android_user._read_foreground_user:
        assert device.commands.count("am get-current-user") == 1
    assert all(" --user 10 " in c for c in device.portal_commands())


# ── UI provider Portal recovery ──────────────────────────────────


class _TcpPortal:
    tcp_available = True

    def __init__(self, user=None):
        if user is not None:
            self.user = user

    async def _fetch_auth_token(self):
        return "token"


def _provider(portal):
    driver = object.__new__(AndroidDriver)
    driver.device = MultiUserDevice()
    driver.portal = portal
    driver._portal_keyboard_available = False
    provider = object.__new__(AndroidStateProvider)
    provider.driver = driver
    return provider, driver.device


@pytest.mark.parametrize(
    "portal, flag",
    [
        (_TcpPortal(user=10), " --user 10"),
        (_TcpPortal(user=None), ""),
        (_TcpPortal(user=0), ""),
    ],
    ids=["user10", "legacy-client", "user0"],
)
def test_recover_portal_restarts_the_targeted_users_server(portal, flag):
    provider, device = _provider(portal)
    asyncio.run(provider._recover_portal())

    toggle = f"content insert{flag} --uri {PORTAL_URI}toggle_socket_server"
    assert device.portal_commands() == [
        f"{toggle} --bind enabled:b:false",
        f"{toggle} --bind enabled:b:true",
    ]
    # The client's cached user is used; no extra adb round trip.
    assert "am get-current-user" not in device.commands
