import unittest

from mobilerun.agent.fast_agent.xml_parser import format_tool_calls, parse_tool_calls


class FastAgentXmlParserTest(unittest.TestCase):
    def test_drops_adjacent_exact_duplicate_tool_calls(self):
        text = """
I will tap the target.
<function_calls>
<invoke name="click_at">
<parameter name="x">128</parameter>
<parameter name="y">1560</parameter>
</invoke>
</function_calls>
<function_calls>
<invoke name="click_at">
<parameter name="x">128</parameter>
<parameter name="y">1560</parameter>
</invoke>
</function_calls>
"""

        thought, calls = parse_tool_calls(text, {"x": "number", "y": "number"})

        self.assertIn("I will tap", thought)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "click_at")
        self.assertEqual(calls[0].parameters, {"x": 128, "y": 1560})

    def test_keeps_non_identical_sequential_calls(self):
        text = """
I will tap two different targets.
<function_calls>
<invoke name="click_at">
<parameter name="x">128</parameter>
<parameter name="y">1560</parameter>
</invoke>
<invoke name="click_at">
<parameter name="x">200</parameter>
<parameter name="y">1560</parameter>
</invoke>
</function_calls>
"""

        _, calls = parse_tool_calls(text, {"x": "number", "y": "number"})

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].parameters, {"x": 128, "y": 1560})
        self.assertEqual(calls[1].parameters, {"x": 200, "y": 1560})

    def test_keeps_identical_invokes_inside_one_block(self):
        text = """
I will press back twice.
<function_calls>
<invoke name="system_button">
<parameter name="button">back</parameter>
</invoke>
<invoke name="system_button">
<parameter name="button">back</parameter>
</invoke>
</function_calls>
"""

        _, calls = parse_tool_calls(text)

        self.assertEqual(len(calls), 2)
        self.assertEqual([call.name for call in calls], ["system_button", "system_button"])
        self.assertEqual(calls[0].parameters, {"button": "back"})
        self.assertEqual(calls[1].parameters, {"button": "back"})

    def test_keeps_intentional_mixed_batch(self):
        text = """
I will focus the field and type.
<function_calls>
<invoke name="click_at">
<parameter name="x">261</parameter>
<parameter name="y">1888</parameter>
</invoke>
<invoke name="type_text">
<parameter name="text">Android version</parameter>
<parameter name="clear">true</parameter>
</invoke>
</function_calls>
"""

        _, calls = parse_tool_calls(
            text,
            {"x": "number", "y": "number", "clear": "boolean"},
        )

        self.assertEqual([call.name for call in calls], ["click_at", "type_text"])
        self.assertEqual(calls[0].parameters, {"x": 261, "y": 1888})
        self.assertEqual(
            calls[1].parameters,
            {"text": "Android version", "clear": True},
        )

    def test_duplicate_complete_blocks_execute_once(self):
        text = """
The task is done.
<function_calls>
<invoke name="complete">
<parameter name="success">true</parameter>
<parameter name="message">Done</parameter>
</invoke>
</function_calls>
<function_calls>
<invoke name="complete">
<parameter name="success">true</parameter>
<parameter name="message">Done</parameter>
</invoke>
</function_calls>
"""

        _, calls = parse_tool_calls(text, {"success": "boolean"})

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "complete")
        self.assertEqual(calls[0].parameters, {"success": True, "message": "Done"})

    def test_formatted_tool_calls_use_deduped_calls(self):
        text = """
Tap once.
<function_calls>
<invoke name="click_at">
<parameter name="x">128</parameter>
<parameter name="y">1560</parameter>
</invoke>
</function_calls>
<function_calls>
<invoke name="click_at">
<parameter name="x">128</parameter>
<parameter name="y">1560</parameter>
</invoke>
</function_calls>
"""

        _, calls = parse_tool_calls(text, {"x": "number", "y": "number"})
        formatted = format_tool_calls(calls)

        self.assertEqual(formatted.count('<invoke name="click_at">'), 1)
        self.assertIn('<parameter name="x">128</parameter>', formatted)
        self.assertIn('<parameter name="y">1560</parameter>', formatted)


if __name__ == "__main__":
    unittest.main()
