import asyncio
import unittest

from pydantic import BaseModel, Field

from mobilerun import MobileAgent
from mobilerun.agent.oneflows.structured_output_agent import (
    StructuredOutputAgent,
    coerce_structured_output_from_text,
)
from mobilerun.config_manager import MobileConfig


class ContactInfo(BaseModel):
    name: str = Field(description="Full name")
    phone: str
    email: str | None = None


class StructuredOutputCoercionTest(unittest.TestCase):
    def test_validates_raw_json_answer(self):
        result = coerce_structured_output_from_text(
            ContactInfo,
            '{"name": "Grace Liu", "phone": "+1 555 0100", "email": "grace@example.com"}',
        )

        self.assertIsInstance(result, ContactInfo)
        self.assertEqual(result.name, "Grace Liu")
        self.assertEqual(result.phone, "+1 555 0100")

    def test_validates_fenced_json_answer(self):
        result = coerce_structured_output_from_text(
            ContactInfo,
            """
Done.

```json
{"name": "Ada Lovelace", "phone": "+44 20 7946 0958"}
```
""",
        )

        self.assertIsInstance(result, ContactInfo)
        self.assertEqual(result.name, "Ada Lovelace")
        self.assertIsNone(result.email)

    def test_ignores_plain_text_without_json_shape(self):
        result = coerce_structured_output_from_text(
            ContactInfo,
            "I found Grace Liu's phone number, but this is not JSON.",
        )

        self.assertIsNone(result)

    def test_structured_output_agent_accepts_json_without_llm(self):
        async def run_agent():
            handler = StructuredOutputAgent(
                llm=None,
                pydantic_model=ContactInfo,
                answer_text='{"name": "Grace Liu", "phone": "+1 555 0100"}',
            ).run()
            return await handler

        result = asyncio.run(run_agent())

        self.assertTrue(result["success"])
        self.assertIsInstance(result["structured_output"], ContactInfo)
        self.assertEqual(result["structured_output"].name, "Grace Liu")

    def test_structured_output_agent_reports_missing_llm_for_plain_text(self):
        async def run_agent():
            handler = StructuredOutputAgent(
                llm=None,
                pydantic_model=ContactInfo,
                answer_text="Grace Liu can be reached at +1 555 0100.",
            ).run()
            return await handler

        result = asyncio.run(run_agent())

        self.assertFalse(result["success"])
        self.assertIsNone(result["structured_output"])
        self.assertIn("No structured output LLM", result["error_message"])


class MobileAgentOutputSchemaTest(unittest.TestCase):
    def test_no_schema_keeps_unstructured_mode(self):
        config = MobileConfig.from_dict({"agent": {"name": "external-agent"}})
        agent = MobileAgent("Find contact info", config=config)

        self.assertIsNone(agent.output_model)
        self.assertIsNone(agent.structured_output_llm)

    def test_set_output_schema_configures_model(self):
        config = MobileConfig.from_dict({"agent": {"name": "external-agent"}})
        agent = MobileAgent("Find contact info", config=config)

        returned = agent.set_output_schema(ContactInfo)

        self.assertIs(returned, agent)
        self.assertIs(agent.output_model, ContactInfo)

    def test_set_output_schema_rejects_non_model(self):
        config = MobileConfig.from_dict({"agent": {"name": "external-agent"}})
        agent = MobileAgent("Find contact info", config=config)

        with self.assertRaises(TypeError):
            agent.set_output_schema(dict)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
