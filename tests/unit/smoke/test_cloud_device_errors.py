import asyncio

import pytest

from mobilerun.core.devices import CloudDevice
from mobilerun.errors import CloudDeviceNotFound


def test_unknown_cloud_device_has_clear_error(real_env):
    device = CloudDevice("00000000-0000-0000-0000-000000000000")
    with pytest.raises(CloudDeviceNotFound):
        asyncio.run(device.do("noop", max_steps=1))
