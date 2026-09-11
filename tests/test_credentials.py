import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.tool_registry import ToolRegistry
from mobilerun.config_manager.config_manager import CredentialsConfig
from mobilerun.credential_manager.file_credential_manager import FileCredentialManager
from mobilerun.credential_manager.placeholders import resolve_credential_placeholders


class FakeWorkflowContext:
    def __init__(self):
        self.events = []

    def write_event_to_stream(self, event):
        self.events.append(event)


class CredentialsTest(unittest.TestCase):
    def test_credentials_config_loads_env_keys_and_overrides_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "credentials.yaml"
            path.write_text(
                "secrets:\n"
                "  PASSWORD:\n"
                "    value: file-secret\n"
                "    enabled: true\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"MOBILERUN_PASSWORD": "env-secret"}):
                manager = FileCredentialManager(
                    CredentialsConfig(
                        enabled=True,
                        file_path=str(path),
                        env_keys=["PASSWORD"],
                        env_prefix="MOBILERUN_",
                    )
                )

            self.assertEqual(asyncio.run(manager.resolve_key("PASSWORD")), "env-secret")

    def test_yaml_env_reference_loads_secret_from_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "credentials.yaml"
            path.write_text(
                "secrets:\n"
                "  API_TOKEN:\n"
                "    env: MOBILERUN_API_TOKEN\n"
                "    enabled: true\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"MOBILERUN_API_TOKEN": "token-secret"}):
                manager = FileCredentialManager(str(path))

            self.assertEqual(asyncio.run(manager.resolve_key("API_TOKEN")), "token-secret")

    def test_credentials_config_can_load_env_prefix_without_explicit_keys(self):
        with patch.dict(os.environ, {"MOBILERUN_USERNAME": "alice"}, clear=False):
            manager = FileCredentialManager(
                CredentialsConfig(
                    enabled=True,
                    file_path="",
                    env_prefix="MOBILERUN_",
                )
            )

        self.assertEqual(asyncio.run(manager.resolve_key("USERNAME")), "alice")

    def test_credentials_config_accepts_single_env_key_string(self):
        with patch.dict(os.environ, {"MOBILERUN_PASSWORD": "secret"}):
            config = CredentialsConfig(
                enabled=True,
                file_path="",
                env_prefix="MOBILERUN_",
            )
            config.env_keys = "PASSWORD"
            manager = FileCredentialManager(config)

        self.assertEqual(asyncio.run(manager.resolve_key("PASSWORD")), "secret")

    def test_placeholder_resolver_replaces_known_secrets_only(self):
        async def run():
            manager = FileCredentialManager(
                {"USERNAME": "alice@example.com", "PASSWORD": "secret"}
            )
            return await resolve_credential_placeholders(
                {
                    "text": "login {{ USERNAME }} with {{PASSWORD}}",
                    "secret_id": "{{PASSWORD}}",
                    "unknown": "{{MISSING}}",
                    "items": ["{{PASSWORD}}"],
                },
                manager,
            )

        result = asyncio.run(run())

        self.assertEqual(result["text"], "login alice@example.com with secret")
        self.assertEqual(result["secret_id"], "{{PASSWORD}}")
        self.assertEqual(result["unknown"], "{{MISSING}}")
        self.assertEqual(result["items"], ["secret"])

    def test_tool_registry_resolves_placeholders_without_emitting_secret_values(self):
        async def run():
            seen = {}

            async def capture(text, *, ctx):
                seen["text"] = text
                return ActionResult(success=True, summary="captured")

            registry = ToolRegistry()
            registry.register(
                "capture",
                fn=capture,
                params={"text": {"type": "string", "required": True}},
                description="Capture text",
            )
            ctx = SimpleNamespace(
                credential_manager=FileCredentialManager({"PASSWORD": "secret"})
            )
            workflow_ctx = FakeWorkflowContext()
            result = await registry.execute(
                "capture",
                {"text": "{{PASSWORD}}"},
                ctx=ctx,
                workflow_ctx=workflow_ctx,
            )
            return result, seen, workflow_ctx.events

        result, seen, events = asyncio.run(run())

        self.assertTrue(result.success)
        self.assertEqual(seen["text"], "secret")
        self.assertEqual(events[0].tool_args, {"text": "{{PASSWORD}}"})
        self.assertNotIn("secret", str(events[0].tool_args))


if __name__ == "__main__":
    unittest.main()
