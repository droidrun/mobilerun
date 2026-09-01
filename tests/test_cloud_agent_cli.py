import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from mobilerun.cli.main import cli, run_command
from mobilerun.config_manager.config_manager import MobileConfig

DEVICE_ID = "123e4567-e89b-12d3-a456-426614174000"


class FakeHandler:
    async def stream_events(self):
        if False:
            yield None

    def __await__(self):
        async def done():
            return SimpleNamespace(success=True)

        return done().__await__()


class CloudAgentCliTest(unittest.TestCase):
    def test_cli_forwards_cloud_options(self):
        async_run_command = AsyncMock(return_value=True)

        with patch("mobilerun.cli.main.run_command", async_run_command):
            result = CliRunner().invoke(
                cli,
                [
                    "run",
                    "Check iOS version",
                    "--cloud",
                    "-d",
                    DEVICE_ID,
                    "--cloud-base-url",
                    "https://cloud.example/v1",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        kwargs = async_run_command.await_args.kwargs
        self.assertTrue(kwargs["cloud"])
        self.assertEqual(kwargs["device"], DEVICE_ID)
        self.assertEqual(kwargs["cloud_base_url"], "https://cloud.example/v1")

    def test_run_command_injects_cloud_driver(self):
        created_drivers = []
        created_agents = []
        cloud_config = MobileConfig()
        cloud_config.device.use_tcp = True

        class FakeCloudDriver:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.connect = AsyncMock()
                created_drivers.append(self)

        class FakeAgent:
            def __init__(self, **kwargs):
                created_agents.append(kwargs)

            def run(self):
                return FakeHandler()

        with (
            patch("mobilerun.cli.main.ConfigLoader.load", return_value=cloud_config),
            patch("mobilerun.cli.main.MobileAgent", FakeAgent),
            patch("mobilerun.cli.main.resolve_cloud_api_key", return_value="token"),
            patch("mobilerun_core_local.driver.cloud.CloudDriver", FakeCloudDriver),
            patch("mobilerun.cli.main.adb.device", AsyncMock()) as adb_device,
        ):
            success = asyncio.run(
                run_command(
                    "Check iOS version",
                    cloud=True,
                    device=DEVICE_ID,
                    cloud_base_url="https://cloud.example/v1",
                    debug=False,
                )
            )

        self.assertTrue(success)
        driver = created_drivers[0]
        self.assertEqual(driver.kwargs["device_id"], DEVICE_ID)
        self.assertEqual(driver.kwargs["api_key"], "token")
        self.assertEqual(driver.kwargs["base_url"], "https://cloud.example/v1")
        driver.connect.assert_awaited_once()
        self.assertIs(created_agents[0]["driver"], driver)
        config = created_agents[0]["config"]
        self.assertEqual(config.device.platform, "android")
        self.assertEqual(config.device.device_id, DEVICE_ID)
        self.assertIsNone(config.device.serial)
        self.assertFalse(config.device.use_tcp)
        self.assertFalse(config.device.auto_setup)
        adb_device.assert_not_called()

    def test_run_command_requires_cloud_credential(self):
        with (
            patch("mobilerun.cli.main.ConfigLoader.load", return_value=MobileConfig()),
            patch("mobilerun.cli.main.resolve_cloud_api_key", return_value=None),
            patch("mobilerun.cli.main.MobileAgent") as agent,
        ):
            success = asyncio.run(
                run_command(
                    "Check iOS version",
                    cloud=True,
                    device=DEVICE_ID,
                    debug=False,
                )
            )

        self.assertFalse(success)
        agent.assert_not_called()

    def test_cloud_rejects_conflicting_backends(self):
        for extra in (
            {"ios": True},
            {"tcp": True},
            {"control_backend": "visual-remote"},
        ):
            with (
                self.subTest(extra=extra),
                patch(
                    "mobilerun.cli.main.ConfigLoader.load", return_value=MobileConfig()
                ),
                patch("mobilerun.cli.main.MobileAgent") as agent,
            ):
                success = asyncio.run(
                    run_command(
                        "Check iOS version",
                        cloud=True,
                        device=DEVICE_ID,
                        debug=False,
                        **extra,
                    )
                )

            self.assertFalse(success)
            agent.assert_not_called()

    def test_cloud_rejects_configured_control_backend(self):
        config = MobileConfig.from_dict(
            {"device": {"control_backend": "visual-remote"}}
        )

        with (
            patch("mobilerun.cli.main.ConfigLoader.load", return_value=config),
            patch("mobilerun.cli.main.MobileAgent") as agent,
        ):
            success = asyncio.run(
                run_command(
                    "Check iOS version",
                    cloud=True,
                    device=DEVICE_ID,
                    debug=False,
                )
            )

        self.assertFalse(success)
        agent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
