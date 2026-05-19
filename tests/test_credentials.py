import asyncio
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mobilerun.agent.utils.actions import type_secret
from mobilerun.agent.utils.signatures import build_tool_registry
from mobilerun.credential_manager import FileCredentialManager


class DummyDriver:
    def __init__(self):
        self.taps = []
        self.inputs = []

    async def tap(self, x, y):
        self.taps.append((x, y))

    async def input_text(self, text):
        self.inputs.append(text)
        return True


class DummyUI:
    def get_element_coords(self, index):
        return (index * 10, index * 20)


class FileCredentialManagerTest(unittest.TestCase):
    def test_loads_direct_env_and_file_sources_from_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / "token.txt"
            token_path.write_text("file-token\n")

            with patch.dict(os.environ, {"APP_USERNAME": "env-user"}, clear=False):
                manager = FileCredentialManager(
                    {
                        "DIRECT": "plain-secret",
                        "ENV_USER": {"env": "APP_USERNAME"},
                        "FILE_TOKEN": {"file": str(token_path)},
                        "DISABLED": {"value": "ignored", "enabled": False},
                    }
                )

            self.assertEqual(
                asyncio.run(manager.get_keys()),
                ["DIRECT", "ENV_USER", "FILE_TOKEN"],
            )
            self.assertEqual(asyncio.run(manager.resolve_key("DIRECT")), "plain-secret")
            self.assertEqual(asyncio.run(manager.resolve_key("ENV_USER")), "env-user")
            self.assertEqual(
                asyncio.run(manager.resolve_key("FILE_TOKEN")), "file-token"
            )

    def test_loads_env_and_file_sources_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            secret_path = tmp_path / "password.txt"
            secret_path.write_text("file-password\n")
            credentials_path = tmp_path / "credentials.yaml"
            credentials_path.write_text(
                textwrap.dedent(
                    f"""
                    secrets:
                      APP_USER:
                        env: LOGIN_USER
                      APP_PASSWORD:
                        file: {secret_path}
                      INLINE:
                        value: inline-secret
                      SKIPPED:
                        env: MISSING_LOGIN_USER
                    """
                ).strip()
            )

            with (
                patch.dict(os.environ, {"LOGIN_USER": "yaml-user"}, clear=False),
                self.assertLogs("mobilerun", level="WARNING") as captured,
            ):
                manager = FileCredentialManager(str(credentials_path))

            self.assertEqual(
                asyncio.run(manager.get_keys()),
                ["APP_USER", "APP_PASSWORD", "INLINE"],
            )
            self.assertIn("MISSING_LOGIN_USER", "\n".join(captured.output))
            self.assertEqual(asyncio.run(manager.resolve_key("APP_USER")), "yaml-user")
            self.assertEqual(
                asyncio.run(manager.resolve_key("APP_PASSWORD")), "file-password"
            )

    def test_type_secret_uses_value_without_exposing_it_in_result(self):
        manager = FileCredentialManager({"PASSWORD": "super-secret"})
        driver = DummyDriver()
        ctx = SimpleNamespace(
            credential_manager=manager,
            ui=DummyUI(),
            driver=driver,
        )

        result = asyncio.run(type_secret("PASSWORD", 3, ctx=ctx))

        self.assertTrue(result.success)
        self.assertEqual(driver.taps, [(30, 60)])
        self.assertEqual(driver.inputs, ["super-secret"])
        self.assertIn("PASSWORD", result.summary)
        self.assertNotIn("super-secret", result.summary)

    def test_tool_registry_description_does_not_include_secret_values(self):
        manager = FileCredentialManager({"PASSWORD": "super-secret"})

        registry, standard_tool_names = asyncio.run(
            build_tool_registry(credential_manager=manager)
        )

        self.assertIn("type_secret", registry.tools)
        self.assertIn("type_secret", standard_tool_names)
        descriptions = registry.get_tool_descriptions_text()
        self.assertIn("type_secret", descriptions)
        self.assertNotIn("super-secret", descriptions)


if __name__ == "__main__":
    unittest.main()
