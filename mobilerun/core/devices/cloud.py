"""Cloud device driver facade; lifecycle, apps and operator live alongside it."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar, cast

from mobilerun_core_local.driver.cloud import CloudDriver

from mobilerun.clients.cloud import proxies
from mobilerun.clients.cloud.client import _device_request
from mobilerun.clients.cloud.proxies import ProxyConnectParams
from mobilerun.clients.local.login_store import resolve_cloud_api_key
from mobilerun.config.settings import get_settings
from mobilerun.core.devices.base import Device
from mobilerun.core.devices.components import environment
from mobilerun.core.devices.components import files as device_files
from mobilerun.core.devices.components.cloud_apps import _CloudApps
from mobilerun.core.devices.components.cloud_lifecycle import _create_params
from mobilerun.core.devices.components.raw import _BackendIO
from mobilerun.core.operator.cloud import CloudTaskOperator
from mobilerun.core.proxy import CloudRotatingProxy, ProxyStatus
from mobilerun.core.proxy.base import CloudProxy
from mobilerun.core.proxy.target import CloudProxyTarget
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.options import (
    StateProviderFactory,
    StateProviderOptions,
    android_state_provider,
)
from mobilerun.errors import CloudAuthRequired, ExtensionConflict, ProxyNotActive
from mobilerun.errors.cloud import CloudAPIError
from mobilerun.models.billing import Billing, Plan
from mobilerun.models.files import DeviceFile
from mobilerun.models.location import Location

_UNSET = object()
_DEFAULT_PROXY = cast(CloudProxy, _UNSET)
T = TypeVar("T")


class CloudDevice(_CloudApps, _BackendIO, Device):
    """Cloud device backed by the cloud driver."""

    # Cloud device file operations.
    async def push_file(self, path: str | Path, dest: str | None = None) -> DeviceFile:
        return await device_files.cloud_push_file(self, path, dest)

    async def list_files(self, directory: str) -> list[DeviceFile]:
        return await device_files.cloud_list_files(self, directory)

    async def pull_file(self, remote_path: str, local_path: str | Path) -> Path:
        return await device_files.cloud_pull_file(self, remote_path, local_path)

    async def delete_file(self, remote_path: str) -> None:
        await device_files.cloud_delete_file(self, remote_path)

    # Cloud environment controls.
    async def get_location(self) -> Location | None:
        return await environment.cloud_get_location(self)

    async def set_location(self, latitude: float, longitude: float) -> None:
        await environment.cloud_set_location(self, latitude, longitude)

    async def reset_location(self) -> None:
        await environment.cloud_reset_location(self)

    async def set_language(self, locale: str) -> None:
        await environment.cloud_set_language(self, locale)

    async def get_language(self) -> str:
        return await environment.cloud_get_language(self)

    async def get_timezone(self) -> str | None:
        return await environment.cloud_get_timezone(self)

    async def set_timezone(self, timezone: str) -> None:
        await environment.cloud_set_timezone(self, timezone)

    async def set_clipboard(self, text: str) -> None:
        await environment.cloud_set_clipboard(self, text)

    async def get_clipboard(self) -> str:
        return await environment.cloud_get_clipboard(self)

    async def open_url(self, url: str) -> None:
        await environment.cloud_open_url(self, url)

    _backend = CloudDriver

    @property
    def platform(self) -> str:
        return self._driver.platform

    @property
    def supported_operations(self) -> set[str]:
        return (
            self._driver.supported
            | {"proxy_switch", "stop_app"}
            | ({"cloud_telephony"} if self._cloud_telephony else set())
        )

    @property
    def cloud_id(self) -> str | None:
        return self.device_id

    @property
    def supported_buttons(self) -> set[str]:
        return self._driver.supported_buttons

    @property
    def _io_driver(self) -> CloudDriver:
        # Raw I/O sees only an initialized id and the current clients (also when
        # tests or embedding applications replace a client).
        self._driver.device_id = self._require_device_id()
        self._driver._client = self._client
        self._driver._write_client = self._write_client
        return self._driver

    @property
    def _is_cloud(self) -> bool:
        return True

    @property
    def identifier(self) -> str | None:
        return self.device_id

    def __init__(
        self,
        device_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        *args: Any,
        state_provider: StateProvider | None = None,
        state_provider_factory: StateProviderFactory | None = None,
        state_options: StateProviderOptions | None = None,
        device_type: str | None = None,
        keep: bool = False,
        billing: Billing | str | None = None,
        plan: Plan | str | None = None,
        name: Any = _UNSET,
        country: Any = _UNSET,
        locale: Any = _UNSET,
        timezone: Any = _UNSET,
        android_version: Any = _UNSET,
        apps: Any = _UNSET,
        template: Any = _UNSET,
        proxy: CloudProxy = _DEFAULT_PROXY,
        **kwargs: Any,
    ) -> None:
        if "number" in kwargs:
            raise TypeError("Use await number.bind(phone), not CloudDevice(number=...)")
        if "credentials" in kwargs:
            raise TypeError("Use await provider.bind(phone), not credentials=...")
        driver_names = {"operator", "display_id", "user_id", "owner_id", "stealth"}
        unknown = kwargs.keys() - driver_names
        if unknown:
            raise TypeError(
                f"unexpected CloudDevice keyword argument(s): {', '.join(sorted(unknown))}"
            )
        creation = {
            key: value
            for key, value in {
                "name": name,
                "country": country,
                "locale": locale,
                "timezone": timezone,
                "android_version": android_version,
                "apps": apps,
                "template": template,
                "proxy": proxy,
            }.items()
            if value is not _UNSET
        }
        if device_id is not None and (
            creation or device_type is not None or plan is not None
        ):
            raise TypeError("Creation options are only used when creating a phone")
        api_key = resolve_cloud_api_key() if api_key is None else api_key
        if not api_key:
            raise CloudAuthRequired(code="credentials_required")
        base_url = (
            get_settings().MOBILERUN_CLOUD_BASE_URL if base_url is None else base_url
        )
        if device_id is None and "proxy" not in creation:
            creation["proxy"] = CloudRotatingProxy._from_settings()
        self._create_options = _create_params(creation) if device_id is None else {}
        self._name = None
        self._device_type = None
        if device_type is not None:
            if device_type not in ("android_cloud_phone", "ios_phone", "device_slot"):
                raise ValueError(f"Invalid cloud device type: {device_type}")
            self._device_type = device_type
        self._keep = keep
        self._owned = False
        self._billing = Billing(
            get_settings().MOBILERUN_BILLING if billing is None else billing
        )
        self._plan = plan
        self._ready = device_id is not None
        self._number_loaded = False
        self._cloud_telephony = False
        self._task_started_callback: Callable[[str], None] | None = None
        self._connect_lock = asyncio.Lock()
        operator = kwargs.pop("operator", None)
        self.device_id = device_id
        self._driver = CloudDriver(
            device_id or "",
            *args,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
        self._client = self._driver._client
        self._write_client = self._driver._write_client
        self._driver._call = self._call
        super().__init__(
            state_provider=state_provider,
            operator=operator,
            state_options=state_options,
            state_provider_factory=state_provider_factory,
        )
        self._creation_proxy: CloudProxy | None = (
            creation["proxy"] if device_id is None else None
        )
        self._creation_proxy_published = False
        self._remote_proxy_loaded = False
        self._owned_client = self._client

    async def connect(self) -> None:
        proxy = self._creation_proxy
        if proxy is None or self._creation_proxy_published:
            await super().connect()
            if proxy is None:
                from mobilerun.core.proxy.adoption import _adopt_remote_proxy

                async with self._extension_state.lock:
                    bindings = tuple(
                        binding
                        for binding in self._extension_state.bindings.values()
                        if binding.extension.exclusive == "proxy"
                    )
                    uncertain = any(binding.uncertain for binding in bindings)
                    if not self._remote_proxy_loaded or uncertain:
                        if not bindings or uncertain:
                            await _adopt_remote_proxy(self)
                        self._remote_proxy_loaded = True
            return
        from mobilerun.core.extensions.registry import Binding

        async with proxy._lifecycle_lock:
            async with self._extension_state.lock:
                if self._creation_proxy_published:
                    await super().connect()
                    return
                if proxy._binding is not None or proxy._closed:
                    raise ExtensionConflict("Creation proxy is already bound or closed")
                if self._extension_state.closing:
                    raise ExtensionConflict("Cloud phone is closing")
                proxy._check_name(self)
                proxy._action_entries()
                await super().connect()
                # Creation applied the exact supplied proxy. Do not issue a
                # second provisioning or switching request.
                await proxy.check_bind(self)
                binding = Binding(proxy, self, False)
                self._extension_state.bindings[proxy.name] = binding
                proxy._binding = binding
                self._creation_proxy_published = True

    @property
    def name(self) -> str | None:
        """Server display name from the last info/create/rename response, or None."""
        return self._name

    async def _call(self, coro: Awaitable[T]) -> T:
        return await _device_request(coro)

    def _create_state_provider(
        self,
        options: StateProviderOptions | None = None,
        *,
        device: Device | None = None,
    ) -> StateProvider:
        return android_state_provider(device or self, options or StateProviderOptions())

    def _default_operator(self) -> CloudTaskOperator:
        return CloudTaskOperator(billing=self._billing)

    @property
    def _proxy_target(self) -> CloudProxyTarget:
        return CloudProxyTarget(
            self._connect_proxy,
            self._proxy_status,
            self._disconnect_proxy,
            resolve_reference=self._resolve_proxy_reference,
        )

    async def _resolve_proxy_reference(self, reference: str) -> str | None:
        try:
            resolved = await proxies.get_proxy(
                reference,
                billing=self._billing,
                api_key=self._client.api_key,
                base_url=str(self._client.base_url),
            )
        except CloudAPIError:
            # Status is authoritative about connection, even when this account
            # cannot resolve the reference to reusable Connect credentials.
            return None
        return resolved.id

    async def _proxy_status(self) -> ProxyStatus:
        await self._ensure_connected()
        response = await proxies.device_status(
            self._require_device_id(),
            billing=self._billing,
            api_key=self._client.api_key,
            base_url=str(self._client.base_url),
        )
        return ProxyStatus(response.connected, response.name, response.protocol)

    async def _connect_proxy(self, params: ProxyConnectParams) -> ProxyStatus:
        await self._ensure_connected()
        await proxies.connect_device(
            self._require_device_id(),
            params,
            billing=self._billing,
            api_key=self._client.api_key,
            base_url=str(self._client.base_url),
        )
        info = await self.info()
        if info.state != "ready":
            await self.wait_ready()
        status = await self._proxy_status()
        if not status.connected:
            raise ProxyNotActive("The cloud phone has no active proxy after switching")
        return status

    async def _disconnect_proxy(self) -> None:
        await self._ensure_connected()
        await proxies.disconnect_device(
            self._require_device_id(),
            billing=self._billing,
            api_key=self._client.api_key,
            base_url=str(self._client.base_url),
        )

    async def tap(self, x: int, y: int) -> None:
        await self._ensure_connected()
        return await super().tap(x, y)

    async def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
    ) -> None:
        await self._ensure_connected()
        return await super().swipe(x1, y1, x2, y2, duration_ms)

    async def _input_text_raw(
        self, text: str, clear: bool = False, stealth: bool = False, wpm: int = 0
    ) -> bool:
        await self._ensure_connected()
        return await super()._input_text_raw(text, clear, stealth, wpm)

    async def press_button(self, button: str) -> None:
        await self._ensure_connected()
        return await super().press_button(button)

    async def start_app(self, package: str, activity: str | None = None) -> str:
        await self._ensure_connected()
        return await super().start_app(package, activity)

    async def stop_app(self, package: str) -> str:
        await self._ensure_connected()
        await _device_request(
            self._write_client.devices.apps.stop(
                package,
                device_id=self._require_device_id(),
                x_device_display_id=self._driver.display_id,
            )
        )
        return f"Stopped {package}"

    async def list_apps(self, include_system: bool = False) -> list[dict[str, Any]]:
        await self._ensure_connected()
        return await super().list_apps(include_system)

    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        await self._ensure_connected()
        return await super().screenshot(hide_overlay)

    async def _get_ui_tree(self) -> dict[str, Any]:
        await self._ensure_connected()
        return await super()._get_ui_tree()

    async def _get_date(self) -> str:
        await self._ensure_connected()
        return await super()._get_date()
