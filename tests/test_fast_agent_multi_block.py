import asyncio
from types import SimpleNamespace

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    TextBlock,
    ThinkingBlock,
)

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.fast_agent.fast_agent import FastAgent
from mobilerun.agent.selectors import LlamaIndexLLMSelector
from mobilerun.tools.ui.provider import StateProvider

FABRICATED = "The app has loaded."
IGNORED_NOTICE = "later block(s) were ignored"


def _block(*invokes):
    return "<function_calls>\n" + "\n".join(invokes) + "\n</function_calls>"


def _click(index):
    return f'<invoke name="click"><parameter name="index">{index}</parameter></invoke>'


def _complete(message):
    return (
        '<invoke name="complete"><parameter name="success">true</parameter>'
        f'<parameter name="message">{message}</parameter></invoke>'
    )


FIRST = "I will tap Open app.\n" + _block(_click(1))
MULTI_BLOCK = f"{FIRST}\n{FABRICATED}\n{_block(_click(2))}\nDone.\n{_block(_complete('fabricated summary'))}"
FINAL = "The results confirm it worked.\n" + _block(_complete("real"))


def _assistant(*blocks, **additional_kwargs):
    return ChatMessage(
        role="assistant", blocks=list(blocks), additional_kwargs=additional_kwargs
    )


class _FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def class_name(self):
        return "fake"

    async def achat(self, messages, **_kwargs):
        self.requests.append(list(messages))
        return ChatResponse(message=self.replies.pop(0))

    async def astream_chat(self, messages, **_kwargs):
        self.requests.append(list(messages))
        reply = self.replies.pop(0)

        async def gen():
            blocks = []
            for block in reply.blocks:
                if not isinstance(block, TextBlock):
                    blocks.append(block)
                    continue
                streamed = TextBlock(text="")
                blocks.append(streamed)
                for i in range(0, len(block.text), 16):
                    delta = block.text[i : i + 16]
                    streamed.text += delta
                    yield ChatResponse(
                        message=_assistant(*blocks, **reply.additional_kwargs),
                        delta=delta,
                    )
            yield ChatResponse(
                message=_assistant(*blocks, **reply.additional_kwargs), delta=""
            )

        return gen()


class _PromptResolver:
    def get_prompt(self, name):
        if name == "fast_agent_system":
            return "Use the documented XML tools: {{ tool_descriptions }}"
        if name == "fast_agent_user":
            return "Goal: {{ goal }}"
        return None


class _StateProvider(StateProvider):
    requires_coordinate_tools = False

    def __init__(self):
        super().__init__(None)

    async def get_state(self):
        return SimpleNamespace(
            formatted_text="1. Button 'Open app'",
            focused_text="",
            elements=[],
            phone_state={"packageName": "com.example", "currentApp": "Main"},
        )


class _Registry:
    def __init__(self, shared_state):
        self.shared_state = shared_state
        self.executed = []
        self.tools = {"click": object(), "complete": object()}

    def get_tool_descriptions_xml(self):
        return '<tool name="click"/><tool name="complete"/>'

    def get_param_types(self):
        return {"index": "number", "success": "boolean"}

    def get_signatures(self):
        return {name: {} for name in self.tools}

    async def execute(self, name, parameters, _action_ctx, workflow_ctx=None):
        self.executed.append((name, parameters))
        if name == "complete":
            await self.shared_state.complete(
                parameters.get("success", False),
                message=parameters.get("message", ""),
            )
        return ActionResult(success=True, summary=f"{name} executed")


def _run(replies, streaming=False, pending_message=None):
    async def _go():
        shared_state = MobileAgentState()
        if pending_message:
            shared_state.queue_user_message(pending_message)
        registry = _Registry(shared_state)
        llm = _FakeLLM(replies)
        config = SimpleNamespace(
            fast_agent=SimpleNamespace(vision=False, parallel_tools=True),
            max_steps=10,
            streaming=streaming,
            after_sleep_action=0,
        )
        agent = FastAgent(
            action_selector=LlamaIndexLLMSelector(llm),
            agent_config=config,
            registry=registry,
            action_ctx=SimpleNamespace(
                driver=SimpleNamespace(), credential_manager=None, ui=None
            ),
            state_provider=_StateProvider(),
            shared_state=shared_state,
            prompt_resolver=_PromptResolver(),
        )
        result = await agent.run(input="Open the app")
        return result, llm, registry, shared_state

    return asyncio.run(_go())


def _request_text(request):
    return "\n".join(message.content or "" for message in request)


@pytest.mark.parametrize("streaming", [False, True])
def test_only_first_block_runs_and_later_text_is_not_replayed(streaming):
    thinking = ThinkingBlock(content="private reasoning")
    reply = _assistant(thinking, TextBlock(text=MULTI_BLOCK), thinking={"sig": "abc"})

    result, llm, registry, shared_state = _run(
        [reply, _assistant(TextBlock(text=FINAL))], streaming=streaming
    )

    assert registry.executed == [
        ("click", {"index": 1}),
        ("complete", {"success": True, "message": "real"}),
    ]
    assert result["success"] is True
    assert result["reason"] == "real"
    assert len(llm.requests) == 2

    stored = shared_state.message_history[1]
    assert stored.blocks[0] is thinking
    assert [type(block) for block in stored.blocks] == [ThinkingBlock, TextBlock]
    assert stored.content == FIRST
    assert stored.additional_kwargs == {"thinking": {"sig": "abc"}}

    second_request = _request_text(llm.requests[1])
    assert FABRICATED not in second_request
    assert "fabricated summary" not in second_request
    assert f"2 {IGNORED_NOTICE}" in llm.requests[1][-1].content


def test_later_text_blocks_are_dropped_and_non_text_blocks_kept_in_order():
    first_thinking = ThinkingBlock(content="first")
    later_thinking = ThinkingBlock(content="later")
    reply = _assistant(
        first_thinking,
        TextBlock(text=FIRST),
        TextBlock(text=f"{FABRICATED}\n{_block(_click(2))}"),
        later_thinking,
    )

    _result, llm, registry, shared_state = _run(
        [reply, _assistant(TextBlock(text=FINAL))]
    )

    assert registry.executed[0] == ("click", {"index": 1})
    stored = shared_state.message_history[1]
    assert stored.blocks[0] is first_thinking
    assert stored.blocks[1].text == FIRST
    assert stored.blocks[2] is later_thinking
    assert len(stored.blocks) == 3
    assert FABRICATED not in _request_text(llm.requests[1])
    assert f"1 {IGNORED_NOTICE}" in llm.requests[1][-1].content


def test_cut_inside_later_text_block_keeps_exact_prefix():
    reply = _assistant(
        TextBlock(text="I will tap Open app."),
        TextBlock(text=f"{_block(_click(1))}\n{FABRICATED}\n{_block(_click(2))}"),
    )

    _result, llm, registry, shared_state = _run(
        [reply, _assistant(TextBlock(text=FINAL))]
    )

    assert registry.executed[0] == ("click", {"index": 1})
    assert shared_state.message_history[1].content == FIRST
    assert FABRICATED not in _request_text(llm.requests[1])


def test_notice_is_sent_when_complete_waits_for_pending_message():
    reply = _assistant(
        TextBlock(
            text=f"Done.\n{_block(_complete('early'))}\n{FABRICATED}\n"
            f"{_block(_click(2))}"
        )
    )

    result, llm, registry, _shared_state = _run(
        [reply, _assistant(TextBlock(text=FINAL))],
        pending_message="Also check the clock",
    )

    assert registry.executed == [
        ("complete", {"success": True, "message": "early"}),
        ("complete", {"success": True, "message": "real"}),
    ]
    assert result["reason"] == "real"
    assert f"1 {IGNORED_NOTICE}" in llm.requests[1][-1].content
    assert FABRICATED not in _request_text(llm.requests[1])


def test_trailing_text_after_only_block_is_cut_without_notice():
    reply = _assistant(TextBlock(text=f"{FIRST}\n{FABRICATED}"))

    _result, llm, _registry, shared_state = _run(
        [reply, _assistant(TextBlock(text=FINAL))]
    )

    assert shared_state.message_history[1].content == FIRST
    assert FABRICATED not in _request_text(llm.requests[1])
    assert IGNORED_NOTICE not in _request_text(llm.requests[1])


@pytest.mark.parametrize(
    "text",
    [
        FIRST + "\n",
        "I will tap twice.\n" + _block(_click(1), _click(2)),
    ],
    ids=["single_invoke", "multi_invoke"],
)
def test_single_block_reply_runs_every_invoke_and_is_stored_unchanged(text):
    reply = _assistant(ThinkingBlock(content="plan"), TextBlock(text=text))

    _result, llm, registry, shared_state = _run(
        [reply, _assistant(TextBlock(text=FINAL))]
    )

    invoked = text.count('<invoke name="click">')
    assert [name for name, _ in registry.executed] == ["click"] * invoked + ["complete"]
    assert shared_state.message_history[1] is reply
    assert IGNORED_NOTICE not in _request_text(llm.requests[1])


def test_complete_in_ignored_block_does_not_end_the_run():
    reply = _assistant(
        TextBlock(text=f"{FIRST}\nDone.\n{_block(_complete('fabricated'))}")
    )

    result, llm, registry, _shared_state = _run(
        [reply, _assistant(TextBlock(text=FINAL))]
    )

    assert len(llm.requests) == 2
    assert result["reason"] == "real"
    assert registry.executed == [
        ("click", {"index": 1}),
        ("complete", {"success": True, "message": "real"}),
    ]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    ("malformed", "kept_after"),
    [
        (
            "<function_calls>\n"
            '<invoke name="click"><parameter name="index">1</parameter>\n'
            "</function_calls>",
            "",
        ),
        (
            "<function_calls>\n"
            '<｜DSML｜invoke name="click">'
            '<｜DSML｜parameter name="index">1</｜DSML｜parameter>'
            "</｜DSML｜invoke>\n"
            "</function_calls>",
            "",
        ),
        # Without a close tag the block runs up to the next opening tag.
        (
            "<function_calls>\n"
            '<invoke name="click"><parameter name="index">1</parameter></invoke>',
            f"\n{FABRICATED}\n",
        ),
        (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="click">'
            '<｜DSML｜parameter name="index">1</｜DSML｜parameter>'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>",
            f"\n{FABRICATED}\n",
        ),
    ],
    ids=["xml_error", "dsml", "unclosed", "dsml_unwrapped"],
)
def test_malformed_first_block_runs_nothing_and_later_blocks_are_dropped(
    malformed, kept_after, streaming
):
    first = f"I will tap Open app.\n{malformed}"
    reply = _assistant(
        TextBlock(
            text=f"{first}\n{FABRICATED}\n{_block(_click(2))}\n"
            f"Done.\n{_block(_complete('fabricated summary'))}"
        )
    )

    result, llm, registry, shared_state = _run(
        [reply, _assistant(TextBlock(text=FINAL))], streaming=streaming
    )

    assert registry.executed == [("complete", {"success": True, "message": "real"})]
    assert result["reason"] == "real"
    assert len(llm.requests) == 2

    assert shared_state.message_history[1].content == first + kept_after
    second_request = _request_text(llm.requests[1])
    assert (FABRICATED in second_request) == bool(kept_after)
    assert "fabricated summary" not in second_request
    assert "could not be parsed" in llm.requests[1][-1].content
    assert IGNORED_NOTICE not in second_request


def test_prose_mention_of_open_tag_does_not_trigger_notice():
    reply = _assistant(
        TextBlock(text=f"{FIRST}\nI will wait for this <function_calls> result.")
    )

    _result, llm, registry, shared_state = _run(
        [reply, _assistant(TextBlock(text=FINAL))]
    )

    assert registry.executed[0] == ("click", {"index": 1})
    assert shared_state.message_history[1].content == FIRST
    assert IGNORED_NOTICE not in _request_text(llm.requests[1])


def test_tag_mentioned_before_first_block_gets_a_correction():
    reply = _assistant(
        TextBlock(text=f"I will use one <function_calls> block now.\n{FIRST}")
    )

    _result, llm, registry, _shared_state = _run(
        [reply, _assistant(TextBlock(text=FINAL))]
    )

    assert ("click", {"index": 1}) not in registry.executed
    correction = llm.requests[1][-1].content
    assert "could not be parsed" in correction
    assert "only inside the call" in correction


def test_add_memory_inside_first_block_gets_a_memory_hint():
    reply = _assistant(
        TextBlock(
            text="I will tap Open app.\n<function_calls>\n"
            "<add_memory>Open app is at index 1.</add_memory>\n"
            f"{_click(1)}\n</function_calls>"
        )
    )

    _result, llm, registry, shared_state = _run(
        [reply, _assistant(TextBlock(text=FINAL))]
    )

    assert ("click", {"index": 1}) not in registry.executed
    correction = llm.requests[1][-1].content
    assert "before the <function_calls> block, never inside it" in correction
    assert "memory written inside the block was not saved" in correction
    assert "Open app is at index 1." not in shared_state.agent_memory
