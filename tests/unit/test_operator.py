import asyncio
from typing import cast

import pytest

from mobilerun.core.devices import AndroidDevice, Device
from mobilerun.core.operator import Operator


def test_operator_default_check_is_noop():
    assert Operator().check(Device()) is None


def test_device_initializes_extension_registry():
    from mobilerun.core.devices.base import Device as DeviceMixin

    device = DeviceMixin()
    assert not device.extensions
    assert not hasattr(device, "_mailbox")
    assert not hasattr(device, "_placeholder_resources")


def test_unwrap_device_stops_before_non_device_driver():
    from mobilerun.core.devices.base import Device as DeviceMixin
    from mobilerun.core.devices.base import _unwrap_device
    from mobilerun.core.devices.components.wrapper import DeviceWrapper

    class WithInner(DeviceMixin):
        inner: object

    inner = WithInner()
    inner._inner = object()
    outer = DeviceWrapper(inner)
    assert _unwrap_device(outer) is inner


def test_device_constructor_checks_operator():
    from mobilerun.core.devices.base import Device as DeviceMixin

    class RejectingOperator(Operator):
        def check(self, device):
            raise ValueError("invalid pairing")

    with pytest.raises(ValueError, match="invalid pairing"):
        DeviceMixin(operator=RejectingOperator())


def test_device_rejects_operator_without_check():
    from mobilerun.core.devices.base import Device as DeviceMixin

    with pytest.raises(AttributeError, match="check"):
        DeviceMixin(operator=cast(Operator, object()))


def test_android_requires_operator():
    from mobilerun.errors import OperatorNotAvailable

    device = AndroidDevice(serial="test", portal_mode="disabled")
    with pytest.raises(OperatorNotAvailable):
        asyncio.run(device.do("hello"))


def test_operator_precedence():
    from mobilerun.core.devices.base import Device as DeviceMixin

    class Op(Operator):
        def __init__(self, value):
            self.value = value

        async def execute(
            self, device, instruction, *, output=None, max_steps=None, timeout=None
        ):
            return self.value

    class Device(DeviceMixin):
        def _default_operator(self):
            return Op("default")

    device = Device(operator=Op("attached"))
    assert asyncio.run(device.do("hello", operator=Op("call"))) == "call"
    assert asyncio.run(device.do("hello")) == "attached"
    assert asyncio.run(Device().do("hello")) == "default"


def test_operator_base_raises():
    from mobilerun.core.operator import Operator
    from mobilerun.errors import OperatorNotAvailable

    with pytest.raises(OperatorNotAvailable, match="No operator for object"):
        asyncio.run(Operator().execute(cast(Device, object()), "hello"))


def test_android_expect_uses_attached_operator():
    from mobilerun.models.results import ExecutionResult, ExpectationVerdict

    class FakeOperator(Operator):
        async def execute(
            self, device, instruction, *, output=None, max_steps=None, timeout=None
        ):
            assert output is ExpectationVerdict
            return ExecutionResult[ExpectationVerdict](
                success=True,
                output=ExpectationVerdict(passed=True, reason="yes"),
                device="AndroidDevice",
            )

    device = AndroidDevice(
        serial="test", portal_mode="disabled", operator=FakeOperator()
    )
    assert asyncio.run(device.expect("visible")).passed


def test_cloud_default_operator():
    from mobilerun.core.devices import CloudDevice
    from mobilerun.core.operator.cloud import CloudTaskOperator

    assert isinstance(
        CloudDevice("id", api_key="test")._default_operator(), CloudTaskOperator
    )


def test_core_has_no_agent_imports():
    import ast
    from pathlib import Path

    for path in Path("mobilerun/core").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("mobilerun.agents"), path
            elif isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("mobilerun.agents")
                    for alias in node.names
                ), path
