from mobilerun.core.devices.base import Device


def test_device_protocol_declares_app_installation():
    assert callable(Device.__dict__.get("install_app"))
    assert callable(Device.__dict__.get("uninstall_app"))
