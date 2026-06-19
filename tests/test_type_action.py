import asyncio
import unittest
from types import SimpleNamespace

from mobilerun.agent.utils.actions import type_text
from mobilerun.agent.utils.signatures import build_tool_registry
from mobilerun.credential_manager.file_credential_manager import FileCredentialManager


class FakeDriver:
    def __init__(self):
        self.taps = []
        self.inputs = []

    async def tap(self, x, y):
        self.taps.append((x, y))

    async def input_text(self, text, clear=False):
        self.inputs.append((text, clear))
        return True


class FakeUI:
    def __init__(self):
        self.requested_indices = []

    def get_element_coords(self, index):
        self.requested_indices.append(index)
        return (123, 456)


class TypeActionTest(unittest.TestCase):
    def test_omitted_index_types_into_focused_input_without_tap(self):
        driver = FakeDriver()
        ui = FakeUI()
        ctx = SimpleNamespace(driver=driver, ui=ui)

        result = asyncio.run(type_text("usb c cable", clear=True, ctx=ctx))

        self.assertTrue(result.success)
        self.assertEqual(driver.inputs, [("usb c cable", True)])
        self.assertEqual(driver.taps, [])
        self.assertEqual(ui.requested_indices, [])

    def test_provided_index_taps_element_before_typing(self):
        driver = FakeDriver()
        ui = FakeUI()
        ctx = SimpleNamespace(driver=driver, ui=ui)

        result = asyncio.run(type_text("usb c cable", index=5, clear=True, ctx=ctx))

        self.assertTrue(result.success)
        self.assertEqual(ui.requested_indices, [5])
        self.assertEqual(driver.taps, [(123, 456)])
        self.assertEqual(driver.inputs, [("usb c cable", True)])

    def test_minus_one_index_keeps_backward_compatible_direct_typing(self):
        driver = FakeDriver()
        ui = FakeUI()
        ctx = SimpleNamespace(driver=driver, ui=ui)

        result = asyncio.run(type_text("usb c cable", index=-1, clear=True, ctx=ctx))

        self.assertTrue(result.success)
        self.assertEqual(driver.inputs, [("usb c cable", True)])
        self.assertEqual(driver.taps, [])
        self.assertEqual(ui.requested_indices, [])

    def test_type_text_resolves_credential_placeholders_without_leaking_summary(self):
        driver = FakeDriver()
        ui = FakeUI()
        ctx = SimpleNamespace(
            driver=driver,
            ui=ui,
            credential_manager=FileCredentialManager({"PASSWORD": "secret"}),
        )

        result = asyncio.run(type_text("{{PASSWORD}}", clear=True, ctx=ctx))

        self.assertTrue(result.success)
        self.assertEqual(driver.inputs, [("secret", True)])
        self.assertNotIn("secret", result.summary)

    def test_type_schema_makes_index_optional_and_explains_focused_input(self):
        async def run():
            registry, _ = await build_tool_registry(supported_buttons={"enter"})
            return registry.tools["type"]

        tool = asyncio.run(run())

        self.assertFalse(tool.params["index"]["required"])
        self.assertIsNone(tool.params["index"]["default"])
        self.assertIn("already focused", tool.description)
        self.assertIn("without index", tool.description)
        self.assertIn(
            'Usage Example: {"action": "type", "text": "example.com", "index": element_index, "clear": true}',
            tool.description,
        )
        self.assertNotIn("generic full-screen containers", tool.description)
        self.assertNotIn("Typing does not submit", tool.description)


if __name__ == "__main__":
    unittest.main()
