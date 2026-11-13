"""Compat-surface tests for mobilerun.tools.driver.

The concrete driver implementations live in mobilerun-core-local (CloudDriver
behind the [cloud] extra). Framework code imports them from those
packages directly; the ``mobilerun.tools.driver`` package is the single
remaining public compat surface. These tests assert it keeps working and
resolves to the exact same classes as the core packages.
"""

import unittest

import mobilerun_core_local.driver.android
import mobilerun_core_local.driver.base
import mobilerun_core_local.driver.cloud
import mobilerun_core_local.driver.humanize
import mobilerun_core_local.driver.ios
import mobilerun_core_local.driver.recording
import mobilerun_core_local.driver.visual_remote

import mobilerun.tools
from mobilerun.tools.driver import (
    AndroidDriver,
    CloudDriver,
    DeviceDisconnectedError,
    DeviceDriver,
    IOSDriver,
    RecordingDriver,
    StealthDriver,
    VisualRemoteDriver,
)


class DriverCompatIdentityTest(unittest.TestCase):
    """Compat names must be the exact same objects as the core implementations."""

    def test_android_driver_is_core_local_implementation(self):
        self.assertIs(
            AndroidDriver,
            mobilerun_core_local.driver.android.AndroidDriver,
        )

    def test_device_disconnected_error_is_core_local_implementation(self):
        self.assertIs(
            DeviceDisconnectedError,
            mobilerun_core_local.driver.base.DeviceDisconnectedError,
        )

    def test_device_driver_is_core_local_implementation(self):
        self.assertIs(
            DeviceDriver,
            mobilerun_core_local.driver.base.DeviceDriver,
        )

    def test_ios_driver_is_core_local_implementation(self):
        self.assertIs(
            IOSDriver,
            mobilerun_core_local.driver.ios.IOSDriver,
        )

    def test_recording_driver_is_core_local_implementation(self):
        self.assertIs(
            RecordingDriver,
            mobilerun_core_local.driver.recording.RecordingDriver,
        )

    def test_stealth_driver_is_core_local_implementation(self):
        self.assertIs(
            StealthDriver,
            mobilerun_core_local.driver.humanize.HumanizeDriver,
        )

    def test_visual_remote_driver_is_core_local_implementation(self):
        self.assertIs(
            VisualRemoteDriver,
            mobilerun_core_local.driver.visual_remote.VisualRemoteDriver,
        )

    def test_cloud_driver_is_core_local_implementation(self):
        self.assertIs(
            CloudDriver,
            mobilerun_core_local.driver.cloud.CloudDriver,
        )


class DriverCompatPackageLevelTest(unittest.TestCase):
    """mobilerun.tools re-exports a subset of driver names; verify those hold identity too."""

    def test_recording_driver_reexported_from_tools_package(self):
        # mobilerun/tools/__init__.py explicitly re-exports RecordingDriver.
        self.assertIs(mobilerun.tools.RecordingDriver, RecordingDriver)
        self.assertIs(
            mobilerun.tools.RecordingDriver,
            mobilerun_core_local.driver.recording.RecordingDriver,
        )

    def test_stealth_driver_not_reexported_from_tools_package(self):
        # mobilerun/tools/__init__.py does not re-export StealthDriver; only
        # mobilerun.tools.driver does. Documented here so this doesn't
        # silently regress if someone assumes otherwise.
        self.assertFalse(hasattr(mobilerun.tools, "StealthDriver"))

    def test_visual_remote_driver_not_reexported_from_tools_package(self):
        self.assertFalse(hasattr(mobilerun.tools, "VisualRemoteDriver"))

    def test_cloud_driver_not_reexported_from_tools_package(self):
        self.assertFalse(hasattr(mobilerun.tools, "CloudDriver"))


if __name__ == "__main__":
    unittest.main()
