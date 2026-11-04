import json
import logging

from llama_index.core.llms.llm import LLM

from mobilerun.agent.selectors.base import ActionSelection, BaseActionSelector, SelectionRequest
from mobilerun.agent.usage import get_usage_from_response
from mobilerun.agent.utils.errors import describe_error
from mobilerun.agent.utils.inference import acall_with_retries

logger = logging.getLogger("mobilerun")


class LlamaIndexLLMSelector(BaseActionSelector):
    abstention_reason = "LLM response is None. This is a critical error."

    def __init__(self, llm: LLM):
        if llm is None:
            raise ValueError("LlamaIndexLLMSelector requires an llm")
        self.llm = llm

    @property
    def shared_llm(self) -> LLM:
        return self.llm

    @property
    def llms(self) -> tuple[LLM, ...]:
        return (self.llm,)

    @property
    def supports_native_tool_calls(self) -> bool:
        return bool(getattr(self.llm, "supports_native_tool_calls", False))

    async def select(self, request: SelectionRequest) -> ActionSelection | None:
        from mobilerun.agent.executor.prompts import parse_executor_response
        from mobilerun.agent.fast_agent.native_tools import (
            native_tool_kwargs,
            parse_native_tool_calls,
        )
        from mobilerun.agent.fast_agent.xml_parser import ToolCall, parse_tool_calls_detailed

        if not request.messages:
            raise ValueError("LLM selection requires messages in the request")
        if request.protocol not in {"executor", "xml", "native"}:
            raise ValueError(f"Unknown selection protocol: {request.protocol}")
        chat_kwargs = {}
        if request.protocol == "native":
            chat_kwargs = native_tool_kwargs(
                self.llm, request.tool_specs or [], request.parallel_tools
            )
        try:
            response = await acall_with_retries(
                self.llm, request.messages, stream=request.stream, **chat_kwargs
            )
        except ValueError as e:
            if request.protocol != "executor":
                raise
            logger.warning(f"Executor LLM returned empty response: {e}")
            text = (
                "### Thought\nExecutor failed to respond, try again\n"
                '### Action\n{"action": "invalid"}\n'
                "### Description\nExecutor failed to respond, try again"
            )
            return ActionSelection(
                actions=[ToolCall("invalid")],
                thought="Executor failed to respond, try again",
                description="Executor failed to respond, try again",
                action_json='{"action": "invalid"}',
                response_text=text,
            )
        except Exception as e:
            if request.protocol != "executor":
                raise
            raise RuntimeError(f"Error calling LLM in executor: {describe_error(e)}") from e
        if response is None and request.protocol != "executor":
            return None
        usage = None
        try:
            usage = get_usage_from_response(self.llm.class_name(), response)
        except Exception as e:
            logger.warning(f"Could not get usage: {e}")

        if request.protocol == "executor":
            try:
                parsed = parse_executor_response(str(response))
            except Exception as e:
                return ActionSelection(
                    actions=[ToolCall("invalid")],
                    thought=f"Failed to parse response: {str(e)}",
                    description="Invalid response format from LLM",
                    response=response,
                    response_text=str(response),
                    usage=usage,
                    parse_error=str(e),
                )
            actions = []
            try:
                action = json.loads(parsed["action"])
                actions = [
                    ToolCall(action["action"], {
                        k: v for k, v in action.items() if k != "action"
                    })
                ]
            except (ValueError, KeyError, TypeError):
                pass  # Keep the original JSON for the executor's error reporting.
            return ActionSelection(
                actions=actions,
                thought=parsed["thought"],
                description=parsed["description"],
                action_json=parsed["action"],
                response=response,
                usage=usage,
                response_text=str(response),
            )

        if request.protocol == "native":
            parsed = parse_native_tool_calls(response.message, request.param_types)
        else:
            parsed = parse_tool_calls_detailed(response.message.content, request.param_types)
        return ActionSelection(
            actions=parsed.calls,
            thought=parsed.thought,
            response=response,
            usage=usage,
            parse_result=parsed,
            response_text=response.message.content,
        )
