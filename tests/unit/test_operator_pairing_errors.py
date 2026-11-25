import asyncio
from types import SimpleNamespace

import httpx
import pytest
from mobilerun_sdk import NotFoundError, UnprocessableEntityError

from mobilerun.core.devices import (
    AndroidDevice,
    CloudDevice,
    DevicePool,
    RecordingDevice,
    StealthDevice,
)
from mobilerun.core.operator.cloud import CloudTaskOperator


def test_constructor_and_attachment_reject_local():
    from mobilerun.errors import OperatorNotSupported

    with pytest.raises(OperatorNotSupported, match="is not a cloud device"):
        AndroidDevice("emulator-5554", operator=CloudTaskOperator())
    assert "attach_operator" not in dir(AndroidDevice)


def test_per_call_and_pool_validate():
    from mobilerun.errors import OperatorNotSupported

    device = AndroidDevice("emulator-5554")
    with pytest.raises(OperatorNotSupported):
        asyncio.run(device.do("noop", operator=CloudTaskOperator()))
    with pytest.raises(OperatorNotSupported):
        asyncio.run(device.expect("noop", operator=CloudTaskOperator()))
    device._operator = CloudTaskOperator()
    with pytest.raises(OperatorNotSupported):
        DevicePool([device])


def test_cloud_pairing_works():
    device = CloudDevice("id", api_key="test", operator=CloudTaskOperator())
    assert device._cloud_client is not None
    DevicePool([device])


@pytest.mark.parametrize("wrapper", [RecordingDevice, StealthDevice])
def test_cloud_capability_delegates_through_wrappers(wrapper):
    local = AndroidDevice("serial")
    cloud = CloudDevice("id", api_key="test")
    assert local._is_cloud is False
    assert cloud._is_cloud is True
    assert wrapper(local)._is_cloud is False
    assert wrapper(cloud)._is_cloud is True
    CloudTaskOperator().check(wrapper(cloud))


@pytest.mark.parametrize(
    "status,body,error_name",
    [
        (404, {"detail": "Device 123 not found"}, "CloudDeviceNotFound"),
        (
            422,
            {"detail": [{"type": "uuid_parsing", "loc": ["body", "deviceId"]}]},
            "InvalidCloudDeviceId",
        ),
    ],
)
def test_cloud_task_maps_device_errors(status, body, error_name):
    import mobilerun.errors as errors
    from mobilerun.clients.cloud.tasks import run_task

    exception_type = NotFoundError if status == 404 else UnprocessableEntityError
    error = exception_type(
        "failure",
        response=httpx.Response(
            status, request=httpx.Request("POST", "https://example.invalid")
        ),
        body=body,
    )

    async def run(**kwargs):
        raise error

    with pytest.raises(getattr(errors, error_name), match=f"HTTP {status}"):
        asyncio.run(
            run_task(
                SimpleNamespace(tasks=SimpleNamespace(run=run)),
                device_id="123",
                task="noop",
                billing="included",
            )
        )


def test_other_422_is_preserved():
    from mobilerun.clients.cloud.tasks import run_task

    error = UnprocessableEntityError(
        "failure",
        response=httpx.Response(
            422, request=httpx.Request("POST", "https://example.invalid")
        ),
        body={"detail": [{"type": "other", "loc": ["body", "task"]}]},
    )

    async def run(**kwargs):
        raise error

    from mobilerun.errors.cloud import CloudValidationError

    with pytest.raises(CloudValidationError):
        asyncio.run(
            run_task(
                SimpleNamespace(tasks=SimpleNamespace(run=run)),
                device_id="123",
                task="noop",
                billing="included",
            )
        )


def test_wrappers_validate_inner_device():
    from mobilerun.core.devices import RecordingDevice, StealthDevice
    from mobilerun.errors import OperatorNotSupported

    for wrapper in (RecordingDevice, StealthDevice):
        with pytest.raises(OperatorNotSupported):
            wrapper(AndroidDevice("emulator-5554"), operator=CloudTaskOperator())
        device = wrapper(AndroidDevice("emulator-5554"))
        expected = rf"{wrapper.__name__}\(AndroidDevice\)"
        with pytest.raises(OperatorNotSupported, match=expected):
            asyncio.run(device.do("noop", operator=CloudTaskOperator()))
        device._inner._operator = CloudTaskOperator()
        device._operator = device._inner._operator
        with pytest.raises(OperatorNotSupported, match=expected):
            DevicePool([device])


def test_other_404_is_preserved():
    from mobilerun.clients.cloud.tasks import run_task

    error = NotFoundError(
        "failure",
        response=httpx.Response(
            404, request=httpx.Request("POST", "https://example.invalid")
        ),
        body={"detail": "Model not found"},
    )

    async def run(**kwargs):
        raise error

    from mobilerun.errors.cloud import CloudResourceNotFound

    with pytest.raises(CloudResourceNotFound):
        asyncio.run(
            run_task(
                SimpleNamespace(tasks=SimpleNamespace(run=run)),
                device_id="123",
                task="noop",
                billing="included",
            )
        )
