"""Root input atomicity, close refusal and remote identity regressions."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from test_proxy_extensions import ProxyMemory

from mobilerun.core.devices.base import Device
from mobilerun.core.devices.components.wrapper import DeviceWrapper
from mobilerun.core.devices.pool import DevicePool
from mobilerun.core.devices.recording import RecordingDevice
from mobilerun.core.extensions import Extension, action, device_action
from mobilerun.core.phone_number import CloudPhoneNumber
from mobilerun.core.proxy import CloudFixedProxy, ExistingCloudProxy
from mobilerun.core.proxy.target import ProxyStatus
from mobilerun.errors import ExtensionClosed, ExtensionConflict
from tests.support.devices import MemoryDevice


@pytest.mark.parametrize("secret_view", ["root", "wrapper", "pool"])
@pytest.mark.parametrize("other_view", ["root", "wrapper", "pool", "catalog"])
@pytest.mark.parametrize("operation", ["tap", "input_text"])
def test_focus_precheck_and_secret_typing_are_atomic_across_views(
    secret_view, other_view, operation
):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        class FocusDevice(MemoryDevice):
            focus = "password"

            def __init__(self):
                super().__init__(supported_operations=frozenset({"tap", "input_text"}))
                self.timeline = []

            async def _input_text_raw(
                self,
                text: str,
                clear: bool = False,
                stealth: bool = False,
                wpm: int = 0,
            ) -> bool:
                if text == "secret-canary":
                    entered.set()
                    await release.wait()
                self.timeline.append((text, self.focus))
                return True

            async def tap(self, x: int, y: int) -> None:
                self.timeline.append(("tap", self.focus))
                self.focus = "chat"

        class Vault(Extension):
            @device_action
            async def type_field(self, device: Device) -> bool:
                """Type a non-secret test canary through the bound facade."""
                return await device._input_secret("secret-canary")

        root = FocusDevice()
        wrapper = DeviceWrapper(RecordingDevice(root))
        for view in (root, wrapper):
            view.state.refresh = AsyncMock(
                return_value=SimpleNamespace(
                    phone_state={
                        "focusedElement": {"id": "password"},
                        "isEditable": True,
                    }
                )
            )
        pool = DevicePool([wrapper])
        vault = Vault(name="vault")
        await vault.bind(wrapper)
        if secret_view == "pool":
            secret = asyncio.create_task(pool.actions.call("vault.type_field", {}))
        else:
            secret = asyncio.create_task(
                (root if secret_view == "root" else wrapper)._input_secret(
                    "secret-canary"
                )
            )
        await entered.wait()
        arguments = {"x": 1, "y": 1} if operation == "tap" else {"text": "plain"}
        if other_view == "catalog":
            other = asyncio.create_task(
                root.actions.call(f"device.{operation}", arguments)
            )
        else:
            view = {"root": root, "wrapper": wrapper, "pool": pool}[other_view]
            other = asyncio.create_task(getattr(view, operation)(**arguments))
        for _ in range(3):
            await asyncio.sleep(0)
        assert not other.done()
        assert root.focus == "password"
        release.set()
        await secret
        await other
        assert root.timeline[0] == ("secret-canary", "password")
        assert root.timeline[1][0] == ("tap" if operation == "tap" else "plain")
        await root.close()
        await vault.close()

    asyncio.run(asyncio.wait_for(run(), timeout=2))


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("swipe", (0, 0, 1, 1)),
        ("drag", (0, 0, 1, 1)),
        ("press_button", ("home",)),
        ("press_key_code", (1,)),
        ("start_app", ("app",)),
        ("stop_app", ("app",)),
        ("open_url", ("https://example.com",)),
        ("tap_element", (1,)),
    ],
)
def test_other_focus_changing_operations_share_the_root_lock(name, arguments):
    async def run():
        root = MemoryDevice()
        wrapper = DeviceWrapper(root)
        async with root._input_operations.hold():
            queued = asyncio.create_task(getattr(wrapper, name)(*arguments))
            await asyncio.sleep(0)
            assert not queued.done()
        # The in-memory backend may not implement the operation. It must only
        # reach its body after the root input lock has been released.
        await asyncio.gather(queued, return_exceptions=True)

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_child_tasks_do_not_inherit_input_lock_ownership_and_cancel_releases_it():
    async def run():
        root = MemoryDevice()
        async with root._input_operations.hold():
            # Same-task nested delegation is reentrant.
            assert await DeviceWrapper(root).input_text("first")
            child = asyncio.create_task(root.input_text("cancelled"))
            await asyncio.sleep(0)
            assert not child.done()
            child.cancel()
            with pytest.raises(asyncio.CancelledError):
                await child
        assert await root.input_text("last")
        assert root.inputs == ["first", "last"]

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_inherited_mixin_operations_and_independent_roots_use_the_correct_lock():
    class NavigationMixin:
        async def open_url(self, url: str) -> None:
            self.opened = url

    class MixedDevice(NavigationMixin, MemoryDevice):
        pass

    async def run():
        root = MixedDevice()
        independent = MemoryDevice()
        async with root._input_operations.hold():
            pending = asyncio.create_task(
                DeviceWrapper(root).open_url("https://example.com")
            )
            await asyncio.sleep(0)
            assert not pending.done()
            assert await independent.input_text("independent")
        await pending
        assert root.opened == "https://example.com"

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_cancelling_active_input_releases_the_root_lock():
    async def run():
        started = asyncio.Event()

        class BlockingDevice(MemoryDevice):
            async def tap(self, x: int, y: int) -> None:
                started.set()
                await asyncio.Event().wait()

        root = BlockingDevice()
        tapping = asyncio.create_task(DeviceWrapper(root).tap(1, 1))
        await started.wait()
        tapping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tapping
        assert await root.input_text("after cancellation")

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_refused_cloud_proxy_close_does_not_poison_the_handle():
    async def run():
        phone = ProxyMemory(True)
        proxy = CloudFixedProxy("old")
        await proxy.bind(phone)
        with pytest.raises(ExtensionConflict):
            await proxy.close()
        assert proxy.device is phone
        assert (await proxy.status()).connected
        assert not proxy._closing
        await phone.close()

    asyncio.run(run())


@pytest.mark.parametrize("mutating", [False, True])
def test_cancelled_close_only_stays_closed_after_mutation_started(mutating):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        class Resource(Extension):
            @action
            async def read(self) -> bool:
                """Safely read the resource."""
                return True

            async def on_close(self) -> None:
                entered.set()
                await release.wait()

        resource = Resource()
        if not mutating:
            await resource._action_lock.acquire()
        closing = asyncio.create_task(resource.close())
        if mutating:
            await entered.wait()
        else:
            await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        if not mutating:
            resource._action_lock.release()
            assert await resource.read()
        else:
            with pytest.raises(ExtensionClosed):
                await resource.read()
        release.set()
        await resource.close()

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_successful_rollback_with_different_unresolvable_reference_does_not_preserve_handle():
    async def run():
        phone = ProxyMemory(True)
        old = CloudFixedProxy("old")
        await old.bind(phone)
        original = RuntimeError("switch failed")

        async def connect(name):
            phone.target.current = ProxyStatus(True, "different-reference")
            if name == "new":
                raise original
            return ProxyStatus(True, "mrc:old")

        phone.target.connect.side_effect = connect
        phone.target._resolve_reference = AsyncMock(return_value=None)
        with pytest.raises(RuntimeError) as raised:
            await CloudFixedProxy("new").bind(phone, force=True)
        assert raised.value is original
        assert isinstance(phone.proxy, ExistingCloudProxy)
        assert old.device is None
        await phone.close()

    asyncio.run(run())


@pytest.mark.parametrize("connected", [False, True])
def test_number_readback_uses_extension_owned_identity_and_update_hooks(connected):
    async def run():
        phone = MemoryDevice(
            cloud_id="device", supported_operations=frozenset({"cloud_telephony"})
        )

        class ThirdPartyNumber(Extension):
            role = "phone_number"
            exclusive = "phone_number"
            observed = "not-called"

            def _matches_remote(self, observed: object) -> bool:
                return getattr(observed, "number_id", None) == "third-party"

            def _observe_remote_binding(
                self, device: Device, observed: object | None
            ) -> None:
                assert device is phone
                self.observed = observed

        original = ThirdPartyNumber(name="sms")
        await original.bind(phone, close_on_device_close=True)
        binding = original._binding
        remote = (
            SimpleNamespace(number_id="third-party", phone_number="+123")
            if connected
            else None
        )
        with patch(
            "mobilerun.core.phone_number.cloud.devices.number",
            AsyncMock(return_value=SimpleNamespace(binding=remote)),
        ):
            await CloudPhoneNumber("incoming")._reconcile_bind_failure(
                phone, RuntimeError("response lost")
            )
        if connected:
            assert original.observed.number_id == remote.number_id
            assert original.observed.phone_number == remote.phone_number
            assert phone.number is original
            assert original._binding is binding
            assert binding.owned
        else:
            assert not phone.extensions
            assert original.device is None
        await phone.close()

    asyncio.run(run())
