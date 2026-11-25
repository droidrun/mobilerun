"""Android device interface."""

import asyncio
import logging
import shlex
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Unpack

from async_adbutils import AdbDevice
from mobilerun_core_local.driver.android import AndroidDriver
from mobilerun_core_local.transport.android.portal_client import PortalClient

from mobilerun.clients.local import emulator
from mobilerun.clients.local.apk import inspect_apk
from mobilerun.config.settings import get_settings
from mobilerun.core.devices.base import Device, InstallOptions
from mobilerun.core.devices.components import environment
from mobilerun.core.devices.components import files as device_files
from mobilerun.core.devices.components.identifiers import _sanitize_identifier
from mobilerun.core.devices.components.raw import _backend_operation, _BackendIO
from mobilerun.core.operator import Operator
from mobilerun.core.proxy import LocalProxy, ProxyStatus
from mobilerun.core.proxy.target import AndroidProxyTarget
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.options import (
    StateProviderFactory,
    StateProviderOptions,
    android_state_provider,
)
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.models.apps import App
from mobilerun.models.files import DeviceFile


class AndroidDevice(_BackendIO, Device):
    """Android device backed by the local Android driver."""

    # Local file operations.
    async def push_file(self, path: str | Path, dest: str | None = None) -> DeviceFile:
        return await device_files.android_push_file(self, path, dest)

    async def list_files(self, directory: str) -> list[DeviceFile]:
        return await device_files.android_list_files(self, directory)

    async def pull_file(self, remote_path: str, local_path: str | Path) -> Path:
        return await device_files.android_pull_file(self, remote_path, local_path)

    async def delete_file(self, remote_path: str) -> None:
        await device_files.android_delete_file(self, remote_path)

    # Local environment controls.
    async def set_location(self, latitude: float, longitude: float) -> None:
        await environment.android_set_location(self, latitude, longitude)

    async def reset_location(self) -> None:
        raise DeviceOperationNotSupported(
            "reset_location is unsupported on local Android; emulators keep the last fixed location; set a new one instead"
        )

    async def set_timezone(self, timezone: str) -> None:
        environment.validate_timezone(timezone)
        await environment.android_shell(
            self, f"cmd alarm set-timezone {shlex.quote(timezone)}"
        )

    async def open_url(self, url: str) -> None:
        await environment.android_shell(
            self, f"am start -a android.intent.action.VIEW -d {shlex.quote(url)}"
        )

    _backend = AndroidDriver

    @property
    def platform(self) -> str:
        return self._driver.platform

    @property
    def supported_operations(self) -> set[str]:
        return self._driver.supported | {"http_proxy", "open_url"}

    @property
    def supported_buttons(self) -> set[str]:
        return self._driver.supported_buttons

    @property
    def _io_driver(self) -> AndroidDriver:
        if self._pending_avd:
            raise RuntimeError(
                "Await AndroidDevice(avd=...) or use async with before using it"
            )
        self._driver._serial = self._serial
        return self._driver

    @property
    def _adb_device(self) -> AdbDevice | None:
        return self._driver.device

    @_adb_device.setter
    def _adb_device(self, value: AdbDevice | None) -> None:
        self._driver.device = value

    @property
    def _portal(self) -> PortalClient | None:
        return self._driver.portal

    @_portal.setter
    def _portal(self, value: PortalClient | None) -> None:
        self._driver.portal = value

    @property
    def _portal_keyboard_available(self) -> bool:
        return self._driver._portal_keyboard_available

    @_portal_keyboard_available.setter
    def _portal_keyboard_available(self, value: bool) -> None:
        self._driver._portal_keyboard_available = value

    @property
    def identifier(self) -> str | None:
        return _sanitize_identifier(self._serial)

    @property
    def _state_recovery(self) -> Callable[[], Awaitable[None]] | None:
        return self._recover_portal

    async def _recover_portal(self) -> None:
        from mobilerun_core_local.driver.android.portal import (
            PORTAL_PACKAGE_NAME,
            portal_a11y_service,
            portal_content_uri,
            setup_keyboard,
        )

        device = self._adb_device
        if device is None:
            return
        logger = logging.getLogger("mobilerun")
        a11y = portal_a11y_service(PORTAL_PACKAGE_NAME)
        await device.shell("settings put secure accessibility_enabled 0")
        await asyncio.sleep(0.5)
        await device.shell(
            f"settings put secure enabled_accessibility_services {shlex.quote(a11y)}"
        )
        await device.shell("settings put secure accessibility_enabled 1")
        portal = self._portal
        if portal is not None and portal.tcp_available:
            toggle_uri = portal_content_uri(PORTAL_PACKAGE_NAME, "toggle_socket_server")
            try:
                await device.shell(
                    f"content insert --uri {shlex.quote(toggle_uri)} --bind enabled:b:false"
                )
                await asyncio.sleep(0.3)
                await device.shell(
                    f"content insert --uri {shlex.quote(toggle_uri)} --bind enabled:b:true"
                )
                new_token = await portal._fetch_auth_token()
                if new_token:
                    portal._auth_token = new_token
            except Exception as exc:
                logger.debug("TCP server restart failed: %s", exc)
        if self._portal_keyboard_available:
            try:
                await setup_keyboard(device)
            except Exception as exc:
                logger.debug("Portal keyboard restore failed: %s", exc)
        await asyncio.sleep(1.5)

    async def _swipe_humanized(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
    ) -> None:
        from mobilerun_core_local.driver.stealth import generate_curved_path

        await self._ensure_connected()
        device = self._adb_device
        if device is None:
            raise RuntimeError("Android device is not connected")
        points = generate_curved_path(x1, y1, x2, y2)
        x, y = points[0]
        await device.shell(f"input motionevent DOWN {x} {y}")
        for x, y in points[1:]:
            await asyncio.sleep(duration_ms / 1000 / len(points))
            await device.shell(f"input motionevent MOVE {x} {y}")
        x, y = points[-1]
        await device.shell(f"input motionevent UP {x} {y}")

    def __init__(
        self,
        serial: str | None = None,
        *,
        avd: str | None = None,
        headless: bool = False,
        timeout: float = 300,
        keep: bool = False,
        state_provider: StateProvider | None = None,
        state_provider_factory: StateProviderFactory | None = None,
        state_options: StateProviderOptions | None = None,
        operator: Operator | None = None,
        **kwargs: Any,
    ) -> None:
        if serial is not None and avd is not None:
            raise ValueError("Pass a serial or avd, not both")
        if "credentials" in kwargs:
            raise TypeError("Use await provider.bind(device), not credentials=...")
        self._avd = avd
        self._headless = headless
        self._boot_timeout = timeout
        self._keep = keep
        self._started_emulator = False
        self._pending_avd = avd is not None
        self._emulator_lock = asyncio.Lock()
        self._serial = serial
        self._driver = AndroidDriver(serial, **kwargs)
        super().__init__(
            state_provider=state_provider,
            operator=operator,
            state_options=state_options,
            state_provider_factory=state_provider_factory,
        )

    async def close(self) -> None:
        async with self._closing_extensions():
            if self._adb_device:
                from mobilerun_core_local.driver.android.portal import (
                    PORTAL_PACKAGE_NAME,
                    portal_ime_id,
                )

                try:
                    await self._adb_device.shell(
                        f"ime disable {shlex.quote(portal_ime_id(PORTAL_PACKAGE_NAME))}"
                    )
                except Exception:
                    pass
            if self._started_emulator and not self._keep:
                await self.terminate()

    async def _ensure_connected(self) -> None:
        if self._pending_avd:
            raise RuntimeError(
                "Await AndroidDevice(avd=...) or use async with before using it"
            )
        await super()._ensure_connected()

    @_backend_operation
    async def _input_text_raw(
        self, text: str, clear: bool = False, stealth: bool = False, wpm: int = 0
    ) -> bool:
        # ADB's typed backend only accepts text/clear; humanization is a wrapper.
        return await self._driver.input_text(text, clear)

    async def install_apk(
        self, path: str | Path, **kwargs: Unpack[InstallOptions]
    ) -> App:
        path = Path(path)
        package, code, version, name = await asyncio.to_thread(inspect_apk, path)
        return await self.install_app(
            App(package, version, name, version_code=code, path=path), **kwargs
        )

    @_backend_operation
    async def install_app(self, app: App, **kwargs: Unpack[InstallOptions]) -> App:
        if app.path is None:
            raise ValueError("This app has no local APK; use install_apk with a path")
        await self._io_driver.install_app(str(app.path), **kwargs)
        device = self._adb_device
        if device is None:
            raise RuntimeError("Android device is not connected")
        result = await device.shell(
            f"pm path {shlex.quote(app.package)}", encoding="utf-8"
        )
        if not isinstance(result, str):
            raise RuntimeError("ADB returned a non-text package list")
        if not any(line.startswith("package:") for line in result.splitlines()):
            raise RuntimeError(f"App {app.package} is not installed after adb install")
        return app

    async def start_app(self, package: str, activity: str | None = None) -> str:
        return await super().start_app(package, activity)

    async def stop_app(self, package: str) -> str:
        return await super().stop_app(package)

    async def uninstall_app(self, package: str) -> str:
        return await super().uninstall_app(package)

    @property
    def _proxy_target(self) -> AndroidProxyTarget:
        return AndroidProxyTarget(
            self._set_http_proxy, self._proxy_status, self._disconnect_proxy
        )

    async def _set_http_proxy(self, host: str, port: int) -> ProxyStatus:
        address = LocalProxy(host, port)
        await self._ensure_connected()
        device = self._adb_device
        if device is None:
            raise RuntimeError("Android device is not connected")
        await device.shell(
            f"settings put global http_proxy {shlex.quote(str(address))}"
        )
        return await self._proxy_status()

    async def _proxy_status(self) -> ProxyStatus:
        await self._ensure_connected()
        device = self._adb_device
        if device is None:
            raise RuntimeError("Android device is not connected")
        result = await device.shell("settings get global http_proxy", encoding="utf-8")
        if not isinstance(result, str):
            raise RuntimeError("ADB returned a non-text proxy setting")
        address = result.strip()
        connected = address not in ("", "null", ":0", "0.0.0.0:0")
        return ProxyStatus(connected, address if connected else None)

    async def _disconnect_proxy(self) -> None:
        await self._ensure_connected()
        device = self._adb_device
        if device is None:
            raise RuntimeError("Android device is not connected")
        await device.shell("settings put global http_proxy :0")

    async def connect(self) -> None:
        async with self._emulator_lock:
            if self._pending_avd:
                await self._boot()
            try:
                await self._ensure_connected()
            except BaseException as exc:
                if self._started_emulator and not self._keep:
                    try:
                        await self.terminate()
                    except Exception as cleanup_error:
                        exc.add_note(
                            f"Could not stop started emulator: {cleanup_error}"
                        )
                raise

    async def _boot(self) -> None:
        """Reuse or start an AVD; only a newly started emulator is owned."""
        name = (
            self._avd
            if self._avd is not None
            else get_settings().MOBILERUN_ANDROID_DEVICE_TYPE
        )
        if not name:
            available = await emulator.list_avds()
            raise ValueError(
                f"Set MOBILERUN_ANDROID_DEVICE_TYPE or pass device_type; available AVDs: {', '.join(available) or 'none'}"
            )
        serial = await emulator.find_running(name)
        started = serial is None
        if started:
            available = await emulator.list_avds()
            if name not in available:
                raise ValueError(
                    f"Unknown AVD {name!r}; available AVDs: {', '.join(available) or 'none'}"
                )
            serial = await emulator.start_avd(name, headless=self._headless)
        try:
            await emulator.wait_boot(serial, timeout=self._boot_timeout)
        except BaseException as exc:
            if started:
                try:
                    await emulator.stop(serial)
                except Exception as cleanup_error:
                    exc.add_note(
                        f"Could not stop started emulator {serial}: {cleanup_error}"
                    )
            raise
        self._serial = serial
        self._started_emulator = started
        self._pending_avd = False

    async def terminate(self) -> None:
        """Stop only an emulator started by this instance."""
        if not self._started_emulator:
            raise ValueError(
                "Cannot terminate an emulator not started by this device or a physical device"
            )
        serial = self._serial
        if serial is None:
            raise RuntimeError("Started emulator has no serial")
        await emulator.stop(serial)
        self._started_emulator = False

    def _create_state_provider(
        self,
        options: StateProviderOptions | None = None,
        *,
        device: Device | None = None,
    ) -> StateProvider:
        return android_state_provider(device or self, options or StateProviderOptions())
