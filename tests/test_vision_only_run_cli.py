import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from mobilerun.cli.main import cli, run_command
from mobilerun.config_manager.config_manager import MobileConfig


class RunCliVisionOnlyTest(unittest.TestCase):
    def test_vision_only_forwards_normal_run_options(self):
        async_run_command = AsyncMock(return_value=True)
        runner = CliRunner()

        with patch("mobilerun.cli.main.run_command", async_run_command):
            result = runner.invoke(
                cli,
                [
                    "run",
                    "Check Wi-Fi",
                    "--vision-only",
                    "--ios",
                    "--control-backend",
                    "visual-remote",
                    "--device",
                    "http://localhost:8090",
                    "--device-id",
                    "phone-1",
                    "--provider",
                    "OpenAIResponses",
                    "--model",
                    "gpt-5.1",
                    "--steps",
                    "8",
                    "--no-stream",
                    "--debug",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        async_run_command.assert_awaited_once()
        kwargs = async_run_command.await_args.kwargs
        self.assertEqual(kwargs["command"], "Check Wi-Fi")
        self.assertTrue(kwargs["vision_only"])
        self.assertTrue(kwargs["ios"])
        self.assertEqual(kwargs["control_backend"], "visual-remote")
        self.assertEqual(kwargs["device"], "http://localhost:8090")
        self.assertEqual(kwargs["device_id"], "phone-1")
        self.assertEqual(kwargs["provider"], "OpenAIResponses")
        self.assertEqual(kwargs["model"], "gpt-5.1")
        self.assertEqual(kwargs["steps"], 8)
        self.assertFalse(kwargs["stream"])
        self.assertTrue(kwargs["debug"])

    def test_omitted_control_backend_preserves_default_backend_behavior(self):
        async_run_command = AsyncMock(return_value=True)
        runner = CliRunner()

        with (
            patch("mobilerun.cli.main.run_command", async_run_command),
            patch("mobilerun.cli.main.adb.device", AsyncMock(return_value=None)),
        ):
            result = runner.invoke(cli, ["run", "Check Wi-Fi", "--vision-only"])

        self.assertEqual(result.exit_code, 0, result.output)
        kwargs = async_run_command.await_args.kwargs
        self.assertTrue(kwargs["vision_only"])
        self.assertIsNone(kwargs["control_backend"])

    def test_run_help_documents_public_flags_only(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--vision-only", result.output)
        self.assertIn("--control-backend", result.output)
        self.assertIn("visual-remote", result.output)
        self.assertNotIn("--" + "connection", result.output)

    def test_run_command_skips_ios_portal_discovery_for_visual_remote(self):
        created_agents = []

        class FakeHandler:
            async def stream_events(self):
                if False:
                    yield None

            def __await__(self):
                async def done():
                    return SimpleNamespace(success=True)

                return done().__await__()

        class FakeAgent:
            def __init__(self, **kwargs):
                created_agents.append(kwargs)

            def run(self):
                return FakeHandler()

        with (
            patch(
                "mobilerun.cli.main.ConfigLoader.load",
                return_value=MobileConfig(),
            ),
            patch("mobilerun.cli.main.MobileAgent", FakeAgent),
            patch("mobilerun.cli.main.discover_ios_portal", AsyncMock()) as discover,
            patch("mobilerun.cli.main.validate_ios_portal_url") as validate,
        ):
            success = asyncio.run(
                run_command(
                    "Check Wi-Fi",
                    ios=True,
                    vision_only=True,
                    control_backend="visual-remote",
                    device="http://localhost:8090",
                    device_id="phone-1",
                    debug=False,
                )
            )

        self.assertTrue(success)
        discover.assert_not_called()
        validate.assert_not_called()
        config = created_agents[0]["config"]
        self.assertEqual(config.device.platform, "ios")
        self.assertEqual(config.device.control_backend, "visual-remote")
        self.assertEqual(config.device.serial, "http://localhost:8090")
        self.assertEqual(config.device.device_id, "phone-1")
        self.assertTrue(config.agent.vision_only)
        self.assertTrue(config.agent.manager.vision)
        self.assertTrue(config.agent.executor.vision)
        self.assertTrue(config.agent.fast_agent.vision)

    def test_run_command_visual_remote_forces_vision_without_vision_only(self):
        created_agents = []

        class FakeHandler:
            async def stream_events(self):
                if False:
                    yield None

            def __await__(self):
                async def done():
                    return SimpleNamespace(success=True)

                return done().__await__()

        class FakeAgent:
            def __init__(self, **kwargs):
                created_agents.append(kwargs)

            def run(self):
                return FakeHandler()

        with (
            patch(
                "mobilerun.cli.main.ConfigLoader.load",
                return_value=MobileConfig(),
            ),
            patch("mobilerun.cli.main.MobileAgent", FakeAgent),
        ):
            success = asyncio.run(
                run_command(
                    "Check Wi-Fi",
                    control_backend="visual-remote",
                    device="http://localhost:8090",
                    debug=False,
                )
            )

        self.assertTrue(success)
        config = created_agents[0]["config"]
        self.assertEqual(config.device.control_backend, "visual-remote")
        self.assertEqual(config.device.serial, "http://localhost:8090")
        self.assertTrue(config.agent.vision_only)
        self.assertTrue(config.agent.manager.vision)
        self.assertTrue(config.agent.executor.vision)
        self.assertTrue(config.agent.fast_agent.vision)

    def test_run_command_skips_adb_cleanup_for_config_visual_remote(self):
        created_agents = []

        class FakeHandler:
            async def stream_events(self):
                if False:
                    yield None

            def __await__(self):
                async def done():
                    return SimpleNamespace(success=True)

                return done().__await__()

        class FakeAgent:
            def __init__(self, **kwargs):
                created_agents.append(kwargs)

            def run(self):
                return FakeHandler()

        config = MobileConfig.from_dict(
            {
                "device": {
                    "control_backend": "visual-remote",
                    "serial": "http://localhost:8090",
                    "platform": "android",
                },
            }
        )

        with (
            patch("mobilerun.cli.main.ConfigLoader.load", return_value=config),
            patch("mobilerun.cli.main.MobileAgent", FakeAgent),
            patch("mobilerun.cli.main.adb.device", AsyncMock()) as adb_device,
        ):
            success = asyncio.run(run_command("Check Wi-Fi", debug=False))

        self.assertTrue(success)
        config = created_agents[0]["config"]
        self.assertEqual(config.device.control_backend, "visual-remote")
        self.assertTrue(config.agent.vision_only)
        self.assertTrue(config.agent.manager.vision)
        self.assertTrue(config.agent.executor.vision)
        self.assertTrue(config.agent.fast_agent.vision)
        adb_device.assert_not_called()


if __name__ == "__main__":
    unittest.main()
