import asyncio
import unittest
from types import SimpleNamespace

from mobilerun.agent.utils.actions import type_text, type_text_direct
from mobilerun.agent.utils.signatures import build_tool_registry


class FakeDriver:
    def __init__(self, accepted=True):
        self.taps = []
        self.inputs = []
        self.accepted = accepted

    async def tap(self, x, y):
        self.taps.append((x, y))

    async def input_text(self, text, clear=False):
        self.inputs.append((text, clear))
        return self.accepted


class FakeUI:
    def __init__(self):
        self.requested_indices = []

    def get_element_coords(self, index):
        self.requested_indices.append(index)
        return (123, 456)


class FakeStateProvider:
    def __init__(self, state):
        self.state = state

    async def get_state(self):
        return self.state


def observable_state(text, *, is_password=False, is_hint=False, is_editable=True):
    return SimpleNamespace(
        phone_state={
            "isEditable": is_editable,
            "focusedElement": {
                "text": text,
                "isPassword": is_password,
                "isShowingHintText": is_hint,
            },
        }
    )


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

    def test_clear_requires_exact_focused_text(self):
        driver = FakeDriver()
        ui = FakeUI()
        ctx = SimpleNamespace(
            driver=driver,
            ui=ui,
            state_provider=FakeStateProvider(observable_state("different")),
        )

        result = asyncio.run(type_text("expected", clear=True, ctx=ctx))

        self.assertFalse(result.success)
        self.assertIn("did not contain the expected text", result.summary)

    def test_clear_ignores_whatsapp_zero_width_sentinel(self):
        driver = FakeDriver()
        ctx = SimpleNamespace(
            driver=driver,
            ui=FakeUI(),
            state_provider=FakeStateProvider(observable_state("\u200b9911656022")),
        )

        result = asyncio.run(type_text("9911656022", clear=True, ctx=ctx))

        self.assertTrue(result.success)
        self.assertIn("verified", result.summary)

    def test_append_requires_requested_text_to_appear(self):
        driver = FakeDriver()
        ctx = SimpleNamespace(
            driver=driver,
            ui=FakeUI(),
            state_provider=FakeStateProvider(observable_state("before and after")),
        )

        result = asyncio.run(type_text("after", clear=False, ctx=ctx))

        self.assertTrue(result.success)

    def test_hint_text_counts_as_empty(self):
        driver = FakeDriver()
        ctx = SimpleNamespace(
            driver=driver,
            ui=FakeUI(),
            state_provider=FakeStateProvider(
                observable_state("Search...", is_hint=True)
            ),
        )

        result = asyncio.run(type_text("query", clear=True, ctx=ctx))

        self.assertFalse(result.success)

    def test_password_field_preserves_transport_success_without_readback(self):
        driver = FakeDriver()
        ctx = SimpleNamespace(
            driver=driver,
            ui=FakeUI(),
            state_provider=FakeStateProvider(
                observable_state("••••", is_password=True)
            ),
        )

        result = asyncio.run(type_text("secret", clear=True, ctx=ctx))

        self.assertTrue(result.success)
        self.assertIn("verification unavailable", result.summary)

    def test_driver_rejection_fails_without_readback(self):
        driver = FakeDriver(accepted=False)
        provider = FakeStateProvider(observable_state("expected"))
        ctx = SimpleNamespace(driver=driver, ui=FakeUI(), state_provider=provider)

        result = asyncio.run(type_text("expected", clear=True, ctx=ctx))

        self.assertFalse(result.success)

    def test_direct_typing_uses_same_verification(self):
        driver = FakeDriver()
        ctx = SimpleNamespace(
            driver=driver,
            ui=FakeUI(),
            state_provider=FakeStateProvider(observable_state("wrong")),
        )

        result = asyncio.run(type_text_direct("expected", clear=True, ctx=ctx))

        self.assertFalse(result.success)

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
