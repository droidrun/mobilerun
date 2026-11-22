"""Repository assertions for extension contracts.

Factories must return fresh, isolated resources with mocked/free transports.
The checker only executes its own safe probe actions, never resource actions or purchases.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import AsyncIterator
from unittest.mock import AsyncMock, patch

from mobilerun.core.devices.base import Device
from mobilerun.core.devices.components.wrapper import DeviceWrapper
from mobilerun.core.devices.pool import DevicePool
from mobilerun.core.extensions import Extension, action, device_action
from mobilerun.errors.operators import (
    ExtensionConflict,
    ExtensionNotBound,
    ExtensionNotSupported,
)
from mobilerun.models.cloud_task import CloudTaskInputs
from mobilerun.models.context import ContextItem


async def check_extension_contract(
    make_extension: Callable[[str], Extension],
    make_device: Callable[[], Awaitable[Device]],
    *,
    incompatible_device: Callable[[], Awaitable[Device]] | None = None,
    unbind_requires_replacement: bool = False,
) -> None:
    """Check relationship invariants and safe standalone/device action dispatch."""

    from tests.support.devices import MemoryDevice

    class Probe(Extension):
        ran = False

        @action
        async def standalone(self) -> bool:
            """Safely probe device-independent dispatch."""
            return True

        @device_action
        async def attached(self, device: Device) -> bool:
            """Safely probe facade injection without touching any resource."""
            self.ran = True
            assert (
                device is self.device
            ), "J: device action did not receive bound facade"
            return True

    probe = Probe(name="contract_probe")
    assert await probe.standalone(), "J: unbound action failed"
    assert await probe.actions.call(
        "contract_probe.standalone", {}
    ), "J: unbound catalog action failed"
    for invoke in (
        probe.attached,
        lambda: probe.actions.call("contract_probe.attached", {}),
    ):
        try:
            await invoke()
        except ExtensionNotBound as error:
            assert "await contract_probe.bind(device) first" in str(
                error
            ), "J: unclear device requirement"
        else:
            raise AssertionError("J: unbound device action did not reject")
    assert not probe.ran, "J: unbound device action executed its body"
    facade = DeviceWrapper(MemoryDevice())
    try:
        await probe.bind(facade)
        assert await probe.attached(), "J: bound device action failed"
        assert await probe.actions.call(
            "contract_probe.attached", {}
        ), "J: bound catalog device action failed"
    finally:
        await facade.close()
        await probe.close()

    @asynccontextmanager
    async def scenario() -> AsyncIterator[tuple[Device, Extension]]:
        device = await make_device()
        extension = make_extension("first")
        try:
            yield device, extension
        finally:
            await device.close()
            await extension.close()

    async with scenario() as (device, extension):
        wrapper = DeviceWrapper(device)
        pool = DevicePool([device])
        views = (device.extensions, wrapper.extensions, pool.extensions[0])
        await extension.bind(wrapper)
        assert all(
            view[extension.name] is extension for view in views
        ), "A: shared registry differs"
        assert (
            device.__getattr__(extension.name) is extension
        ), "A: named lookup differs"
        actions = extension.actions.list()
        for entry in actions:
            assert entry.description and entry.name.startswith(
                "first."
            ), "G: invalid action metadata"
            json.dumps(dict(entry.input_schema), allow_nan=False)
            json.dumps(dict(entry.output_schema), allow_nan=False)
            assert (
                device.actions.get(entry.name).owner == extension.name
            ), "G: action missing"
        for item in await extension.context():
            assert isinstance(item, ContextItem), "G: context item has wrong type"
            assert item.source == extension.name, "G: context source differs"
            json.dumps(dict(item.data), allow_nan=False)
        task_inputs = await extension.cloud_task_inputs()
        assert isinstance(
            task_inputs, CloudTaskInputs
        ), "G: cloud task inputs have wrong type"
        reference_data = asdict(task_inputs)
        assert set(reference_data) == {
            "credential_references"
        }, "G: cloud task inputs must contain references only"
        assert all(
            set(reference) == {"package", "name"}
            for reference in reference_data["credential_references"]
        ), "G: cloud credential references must not contain values"
        json.dumps(reference_data, allow_nan=False)
        if unbind_requires_replacement:
            for unbind in (
                extension.unbind,
                lambda: device.unbind_extension(extension.name),
            ):
                try:
                    await unbind()
                except ExtensionConflict:
                    pass
                else:
                    raise AssertionError(
                        "B: required extension detached without replacement"
                    )
            await device.close()
        else:
            await extension.unbind()
        assert all(not view for view in views), "B: unbind left a shared relationship"
        if not unbind_requires_replacement:
            await extension.bind(device)
            await device.unbind_extension(extension.name)
        assert all(
            not view for view in views
        ), "B: device-side unbind left relationship"
        try:
            device.__getattr__(extension.name)
        except AttributeError:
            pass
        else:
            raise AssertionError("B: missing extension attribute did not fail")
        assert not any(
            entry.owner == extension.name for entry in device.actions.list()
        ), "G: actions survived unbind"

    for hook in ("check_bind", "on_bind"):
        async with scenario() as (device, extension):
            wrapper = DeviceWrapper(device)
            pool = DevicePool([device])
            views = (device.extensions, wrapper.extensions, pool.extensions[0])
            snapshots = tuple(dict(view) for view in views)
            binding_snapshots = {
                name: (binding, binding.facade, binding.owned, binding.uncertain)
                for name, binding in device._extension_state.bindings.items()
            }
            failure = ExtensionNotSupported("injected contract failure")
            with (
                patch.object(extension, hook, AsyncMock(side_effect=failure)),
                patch.object(extension, "_reconcile_bind_failure", AsyncMock()),
            ):
                try:
                    await extension.bind(device)
                except ExtensionNotSupported as exc:
                    assert exc is failure, "C: hook error changed"
                else:
                    raise AssertionError(f"C: {hook} failure was swallowed")
            for view, snapshot in zip(views, snapshots, strict=True):
                assert tuple(view) == tuple(snapshot) and all(
                    view[name] is value for name, value in snapshot.items()
                ), f"C: {hook} failure changed a registry snapshot"
                for name, observed in view.items():
                    binding = observed._binding
                    assert (
                        binding is not None
                        and binding.extension is observed
                        and binding.facade._extension_state is device._extension_state
                    ), f"C: {hook} registry has an orphan binding"
                    assert (
                        device._extension_state.bindings[name] is binding
                    ), f"C: {hook} binding points to a different registry entry"
            assert (
                extension._binding is None
            ), f"C: {hook} failed handle retained a binding outside the registry"
            for name, (before, facade, owned, uncertain) in binding_snapshots.items():
                after = device._extension_state.bindings[name]
                assert (
                    after is before
                    and after.facade is facade
                    and after.owned == owned
                    and after.uncertain == uncertain
                ), f"C: {hook} failure modified a registry binding snapshot"

    if incompatible_device is not None:
        device = await incompatible_device()
        extension = make_extension("incompatible")
        try:
            with patch.object(
                extension, "on_bind", AsyncMock(wraps=extension.on_bind)
            ) as bind_hook:
                try:
                    await extension.bind(device, force=True)
                except ExtensionNotSupported:
                    pass
                else:
                    raise AssertionError("E: incompatible device was accepted")
                assert bind_hook.await_count == 0, "E: incompatibility reached backend"
            assert not device.extensions, "E: incompatible binding published"
        finally:
            await extension.close()
            await device.close()

    async with scenario() as (device, first):
        second = make_extension("second")
        try:
            await first.bind(device)
            with patch.object(
                second, "on_bind", AsyncMock(wraps=second.on_bind)
            ) as hook:
                if first.exclusive is None:
                    await second.bind(device)
                    assert len(device.extensions) == 2, "D: nonexclusive binding lost"
                else:
                    try:
                        await second.bind(DeviceWrapper(device))
                    except ExtensionConflict as exc:
                        assert first.name in str(
                            exc
                        ), "D: conflict doesn't identify bound extension"
                    else:
                        raise AssertionError("D: exclusivity was bypassed")
                    assert hook.await_count == 0, "D: conflict reached backend"
                    await second.bind(device, force=True)
                    assert tuple(device.extensions) == (
                        second.name,
                    ), "I: force didn't replace"
                    assert first._binding is None, "I: old extension still bound"
        finally:
            if unbind_requires_replacement:
                await device.close()
            await second.close()

    async with scenario() as (device, first):
        second = make_extension("second")
        try:
            outcomes = await asyncio.wait_for(
                asyncio.gather(
                    first.bind(device),
                    second.bind(DeviceWrapper(device)),
                    return_exceptions=True,
                ),
                timeout=5,
            )
            if first.exclusive is not None:
                assert (
                    sum(isinstance(value, ExtensionConflict) for value in outcomes) == 1
                ), "F: exclusive concurrent binds didn't have one winner"
                assert (
                    len(device.extensions) == 1
                ), "F: concurrent registry inconsistent"
                await asyncio.wait_for(
                    asyncio.gather(
                        first.bind(device, force=True), second.bind(device, force=True)
                    ),
                    timeout=5,
                )
                assert (
                    len(device.extensions) == 1
                ), "I: concurrent force left multiple bindings"
            else:
                assert not any(
                    isinstance(value, BaseException) for value in outcomes
                ), "F: nonexclusive concurrent bind failed"
        finally:
            if unbind_requires_replacement:
                await device.close()
            await second.close()

    for owned in (False, True):
        async with scenario() as (device, extension):
            with (
                patch.object(
                    extension, "on_unbind", AsyncMock(wraps=extension.on_unbind)
                ) as unbind,
                patch.object(
                    extension, "on_close", AsyncMock(wraps=extension.on_close)
                ) as close,
            ):
                await extension.bind(device, close_on_device_close=owned)
                await device.close()
                await device.close()
                assert unbind.await_count == 1, "H: unbind must run once"
                assert close.await_count == int(owned), "H: ownership ignored"
                assert not device.extensions, "H: close left relationships"

    async with scenario() as (device, extension):
        await extension.bind(device, close_on_device_close=True)
        with (
            patch.object(
                extension, "on_unbind", AsyncMock(side_effect=RuntimeError("injected"))
            ),
            patch.object(extension, "on_close", AsyncMock()) as close,
            patch.object(extension, "_reconcile_bind_failure", AsyncMock()),
        ):
            try:
                await device.close()
            except ExceptionGroup:
                pass
            else:
                raise AssertionError("H: failed unbind was swallowed")
            assert (
                device.extensions[extension.name] is extension
            ), "H: failed unbind lost relationship"
            assert close.await_count == 0, "H: resource closed despite failed unbind"
