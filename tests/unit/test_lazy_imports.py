"""Fresh interpreters keep unrelated device backends out of simple imports."""

import subprocess
import sys

import pytest


def run_python(code: str) -> None:
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "statement",
    ["import mobilerun", "import mobilerun.models", "from mobilerun import Workspace"],
)
def test_simple_import_does_not_load_backends(statement: str) -> None:
    run_python(f"""
import sys
{statement}
for prefix in ('mobilerun_sdk', 'adbutils', 'async_adbutils', 'httpx', 'PIL',
               'mobilerun.core.devices.android', 'mobilerun.clients.cloud'):
    loaded = [name for name in sys.modules if name == prefix or name.startswith(prefix + '.')]
    assert not loaded, (prefix, loaded)
""")


def test_public_exports_resolve_to_original_objects() -> None:
    run_python("""
import mobilerun
from importlib import import_module
modules = {
    'Device': 'devices.base', 'AndroidDevice': 'devices.android',
    'CloudDevice': 'devices.cloud', 'IOSDevice': 'devices.ios',
    'RemoteDevice': 'devices.remote', 'DevicePool': 'devices.pool',
    'RecordingDevice': 'devices.recording', 'StealthDevice': 'devices.stealth',
    'Workspace': 'workspace', 'CloudFixedProxy': 'proxy.cloud_fixed',
    'CloudRotatingProxy': 'proxy.cloud_rotating',
    'CloudSocks5Proxy': 'proxy.cloud_socks5', 'LocalProxy': 'proxy.local',
}
assert set(mobilerun._CORE_EXPORTS) == set(modules)
assert all(getattr(mobilerun, name) is not None for name in mobilerun.__all__)
for name, module in modules.items():
    assert getattr(mobilerun, name) is getattr(import_module('mobilerun.core.' + module), name)
assert set(mobilerun.__all__) <= set(dir(mobilerun))
namespace = {}
exec('from mobilerun import *', namespace)
assert all(namespace[name] is getattr(mobilerun, name) for name in mobilerun.__all__)
""")


def test_generic_device_does_not_import_concrete_backends() -> None:
    run_python("""
import sys
from mobilerun import Device
assert Device.__name__ == 'Device'
assert 'mobilerun.core.devices.android' not in sys.modules
assert 'mobilerun.clients.cloud' not in sys.modules
""")
