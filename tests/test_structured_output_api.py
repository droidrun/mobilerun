import asyncio
import unittest
from types import SimpleNamespace

from pydantic import BaseModel

from mobilerun.agent.droid.droid_agent import MobileAgent, _StructuredOutputHandler


class AgentResponse(BaseModel):
    success: bool
    output: str | None = None


class FakeHandler:
    def __init__(self, result):
        self.result = result
        self.delegated_attribute = "delegated"

    async def stream_events(self):
        yield "event"

    def __await__(self):
        async def done():
            return self.result

        return done().__await__()


async def _await(awaitable):
    return await awaitable


class StructuredOutputApiTest(unittest.TestCase):
    def test_wrapper_returns_model_directly_and_delegates_handler_api(self):
        model = AgentResponse(success=True, output="done")
        result = SimpleNamespace(structured_output=model, reason="done")
        handler = _StructuredOutputHandler(FakeHandler(result), AgentResponse)

        async def collect_events():
            return [event async for event in handler.stream_events()]

        self.assertEqual(handler.delegated_attribute, "delegated")
        self.assertEqual(asyncio.run(collect_events()), ["event"])
        self.assertIs(asyncio.run(_await(handler)), model)

    def test_wrapper_raises_when_structured_output_is_missing(self):
        result = SimpleNamespace(structured_output=None, reason="not enough data")
        handler = _StructuredOutputHandler(FakeHandler(result), AgentResponse)

        with self.assertRaisesRegex(ValueError, "no structured output was produced"):
            asyncio.run(_await(handler))

    def test_wrapper_raises_when_structured_output_has_wrong_type(self):
        result = SimpleNamespace(
            structured_output=SimpleNamespace(success=True),
            reason="done",
        )
        handler = _StructuredOutputHandler(FakeHandler(result), AgentResponse)

        with self.assertRaisesRegex(TypeError, "does not match AgentResponse"):
            asyncio.run(_await(handler))

    def test_set_output_schema_enables_direct_model_return_and_updates_manager(self):
        manager_agent = SimpleNamespace(output_model=None)
        agent = object.__new__(MobileAgent)
        agent.manager_agent = manager_agent
        agent.output_model = None
        agent._return_structured_output_directly = False

        returned = MobileAgent.set_output_schema(agent, AgentResponse)

        self.assertIs(returned, agent)
        self.assertIs(agent.output_model, AgentResponse)
        self.assertIs(manager_agent.output_model, AgentResponse)
        self.assertTrue(agent._return_structured_output_directly)

    def test_set_output_schema_rejects_non_pydantic_models(self):
        agent = object.__new__(MobileAgent)
        agent.manager_agent = None

        with self.assertRaisesRegex(TypeError, "BaseModel subclass"):
            MobileAgent.set_output_schema(agent, dict)


if __name__ == "__main__":
    unittest.main()
