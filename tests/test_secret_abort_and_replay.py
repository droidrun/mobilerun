import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse

from mobilerun.agent.action_context import ActionContext
from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.droid.droid_agent import MobileAgent
from mobilerun.agent.droid.events import (
    ExecutorInputEvent,
    FinalizeEvent,
    ManagerInputEvent,
    ManagerPlanEvent,
)
from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.fast_agent.fast_agent import FastAgent
from mobilerun.agent.utils import actions
from mobilerun.agent.utils.actions import type_secret
from mobilerun.credential_manager import (
    ApprovalProvider,
    CredentialManager,
    SecretSpec,
)
from mobilerun.macro.replay import APPROVAL_REQUIRED, MacroPlayer
from mobilerun.macro.state import MACRO_SCHEMA_VERSION, normalize_ui_state
from mobilerun.tools.driver.secret_safe import SecretRedactor, SecretSafeDriver

CARD = "4111 1111 1111 1111"


@pytest.fixture(autouse=True)
def no_settle(monkeypatch):
    monkeypatch.setattr(actions, "_SECRET_FOCUS_SETTLE_SECONDS", 0)


def run(coro):
    return asyncio.run(coro)


# -- shared device fake for the real type_secret path -------------------------


class Device:
    def __init__(self):
        self.typed = []
        self.taps = []
        self.focused = False

    async def get_ui_tree(self):
        node = {
            "resourceId": "com.example.shop:id/card",
            "className": "android.widget.EditText",
            "text": "",
            "isFocused": self.focused,
            "boundsInScreen": {"left": 0, "top": 0, "right": 500, "bottom": 100},
        }
        return {
            "a11y_tree": {"packageName": "com.example.shop", "children": [node]},
            "phone_state": {"packageName": "com.example.shop", "isEditable": True},
        }

    async def tap(self, x, y):
        self.taps.append((x, y))
        self.focused = True

    async def input_text(self, text, clear=False, **kwargs):
        self.typed.append(text)
        return True


class UI:
    def get_element(self, index):
        return {
            "index": 1,
            "resourceId": "com.example.shop:id/card",
            "className": "android.widget.EditText",
            "bounds": "0,0,500,100",
        }

    def get_element_coords(self, index):
        return (250, 50)


class Provider:
    async def get_state(self):
        return UI()


class Denier(ApprovalProvider):
    def __init__(self, decision="denied"):
        self.decision = decision
        self.count = 0

    async def request(self, kind, details):
        self.count += 1
        return self.decision


class Secrets(CredentialManager):
    async def resolve_key(self, key):
        return CARD

    async def get_keys(self):
        return ["card"]

    async def get_secret_specs(self):
        return [SecretSpec("card", requires_approval=True, approval_label="Card")]


def real_ctx(shared_state, provider):
    return ActionContext(
        driver=SecretSafeDriver(Device(), SecretRedactor()),
        ui=UI(),
        shared_state=shared_state,
        state_provider=Provider(),
        credential_manager=Secrets(),
        approval_provider=provider,
    )


# -- FastAgent: sticky abort ------------------------------------------------


def _call(name, **params):
    body = "".join(f'<parameter name="{k}">{v}</parameter>' for k, v in params.items())
    return f'<invoke name="{name}">{body}</invoke>'


def _reply(*invokes):
    return (
        "Acting now.\n<function_calls>\n" + "\n".join(invokes) + "\n</function_calls>"
    )


class _Responder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = 0

    async def __call__(self, _llm, messages, *, stream):
        self.requests += 1
        if not self.responses:
            raise AssertionError("an LLM turn happened after the run was aborted")
        content = self.responses.pop(0)
        return ChatResponse(message=ChatMessage(role="assistant", content=content))


class _PromptResolver:
    def get_prompt(self, name):
        return {
            "fast_agent_system": "Tools: {{ tool_descriptions }}",
            "fast_agent_user": "Goal: {{ goal }}",
        }.get(name)


class _StateProvider:
    requires_coordinate_tools = False

    async def get_state(self):
        return SimpleNamespace(
            formatted_text="screen",
            focused_text="",
            elements=[],
            phone_state={"packageName": "com.example.shop"},
        )


class _Registry:
    """Runs the real type_secret; every other tool is a recorded no-op."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.executed = []
        self.tools = {n: object() for n in ("type_secret", "click", "complete")}

    def get_tool_descriptions_xml(self):
        return "<tool/>"

    def get_param_types(self):
        return {"index": "number", "success": "boolean", "secret_id": "string"}

    async def execute(self, name, parameters, action_ctx, workflow_ctx=None):
        self.executed.append(name)
        if name == "type_secret":
            return await type_secret(
                parameters["secret_id"], int(parameters["index"]), ctx=self.ctx
            )
        if name == "complete":
            await self.ctx.shared_state.complete(True, message="done")
        return ActionResult(success=True, summary=f"{name} ok")


def _run_fast(responses, *, queue_message=False, decision="denied", provider=True):
    async def go():
        state = MobileAgentState()
        if queue_message:
            state.queue_user_message("please continue anyway")
        denier = Denier(decision) if provider else None
        ctx = real_ctx(state, denier)
        registry = _Registry(ctx)
        responder = _Responder(responses)
        agent = FastAgent(
            llm=SimpleNamespace(),
            agent_config=SimpleNamespace(
                fast_agent=SimpleNamespace(vision=False, parallel_tools=True),
                max_steps=20,
                streaming=False,
                after_sleep_action=0,
            ),
            registry=registry,
            action_ctx=SimpleNamespace(
                driver=ctx.driver, credential_manager=None, ui=None
            ),
            state_provider=_StateProvider(),
            shared_state=state,
            prompt_resolver=_PromptResolver(),
        )
        with (
            patch(
                "mobilerun.agent.fast_agent.fast_agent.acall_with_retries",
                new=responder,
            ),
            patch(
                "mobilerun.agent.fast_agent.fast_agent.get_usage_from_response",
                return_value=None,
            ),
        ):
            result = await agent.run(input="Pay")
        return result, responder, registry, ctx, state

    return asyncio.run(go())


@pytest.mark.parametrize(
    ("decision", "provider", "reason"),
    [
        ("denied", True, "approval_denied"),
        ("expired", True, "approval_expired"),
        ("approved", False, "approval_unavailable"),
    ],
)
@pytest.mark.parametrize("queue_message", [False, True])
def test_fast_agent_ends_the_run_without_another_tool_or_llm_call(
    decision, provider, reason, queue_message
):
    tool_block = _reply(
        _call("type_secret", secret_id="card", index=1),
        _call("click", index=2),
        _call("complete", success="true", message="paid"),
    )

    result, responder, registry, ctx, state = _run_fast(
        [tool_block], queue_message=queue_message, decision=decision, provider=provider
    )

    assert result["success"] is False
    assert result["reason"] == reason
    assert responder.requests == 1  # no further LLM turn
    assert registry.executed == ["type_secret"]  # no further tool call
    assert ctx.driver.inner.typed == []
    if queue_message:  # the queued message neither revived nor lost the abort
        assert len(state.pending_user_messages) == 1
    assert state.terminal_abort == reason


def test_abort_takes_priority_over_a_completion_in_the_same_block():
    result, _, registry, _, state = _run_fast(
        [_reply(_call("type_secret", secret_id="card", index=1), _call("complete"))]
    )

    assert result["success"] is False and registry.executed == ["type_secret"]


def test_approved_secret_run_still_completes_normally():
    result, responder, registry, ctx, _ = _run_fast(
        [
            _reply(_call("type_secret", secret_id="card", index=1)),
            _reply(_call("complete", success="true", message="paid")),
        ],
        decision="approved",
    )

    assert result["success"] is True and result["reason"] == "done"
    assert ctx.driver.inner.typed == [CARD]
    assert responder.requests == 2


# -- Manager/Executor: sticky abort ------------------------------------------


def _agent_with_executor(on_run):
    class Handler:
        async def stream_events(self):
            if False:
                yield None

        def __await__(self):
            async def finish():
                on_run()
                return {
                    "action": {"name": "type_secret"},
                    "summary": "refused",
                    "outcome": False,
                    "error": "refused",
                }

            return finish().__await__()

    agent = object.__new__(MobileAgent)
    agent.shared_state = MobileAgentState()
    agent.executor_agent = SimpleNamespace(run=lambda subgoal: Handler())
    agent.handle_stream_event = lambda *_: None
    return agent


@pytest.mark.parametrize("reason", ["approval_denied", "approval_expired"])
def test_manager_mode_finalizes_before_the_next_manager_turn(reason):
    agent = _agent_with_executor(lambda: agent.shared_state.abort_run(reason))
    agent.shared_state.queue_user_message("keep going")

    result = run(
        agent.run_executor(SimpleNamespace(), ExecutorInputEvent(current_subgoal="pay"))
    )

    assert isinstance(result, FinalizeEvent)
    assert result.success is False and result.reason == reason
    assert len(agent.shared_state.pending_user_messages) == 1  # not consumed


def test_manager_mode_without_abort_proceeds_as_before():
    agent = _agent_with_executor(lambda: None)

    result = run(
        agent.run_executor(SimpleNamespace(), ExecutorInputEvent(current_subgoal="x"))
    )

    assert not isinstance(result, FinalizeEvent)


def test_manager_turn_never_starts_after_an_abort():
    agent = object.__new__(MobileAgent)
    agent.shared_state = MobileAgentState()
    agent.shared_state.abort_run("approval_denied")
    agent.config = SimpleNamespace(agent=SimpleNamespace(max_steps=5))
    agent.manager_agent = SimpleNamespace(
        run=lambda: pytest.fail("manager must not run after an abort")
    )

    result = run(agent.run_manager(SimpleNamespace(), ManagerInputEvent()))

    assert isinstance(result, FinalizeEvent) and result.reason == "approval_denied"


def test_fast_mode_result_is_overridden_by_the_abort_marker():
    agent = object.__new__(MobileAgent)
    agent.shared_state = MobileAgentState()
    agent.shared_state.abort_run("approval_expired")
    agent.config = SimpleNamespace(logging=SimpleNamespace(debug=False))

    result = run(
        agent.handle_fast_agent_result(
            SimpleNamespace(),
            SimpleNamespace(success=True, reason="paid", instruction="x"),
        )
    )

    assert result.success is False and result.reason == "approval_expired"


# -- Macro replay -------------------------------------------------------------


class ReplayDriver:
    platform = "android"

    def __init__(self):
        self.taps = []
        self.inputs = []
        self.tree = {
            "a11y_tree": [{"text": "Name", "className": "android.widget.EditText"}],
            "phone_state": {"package": "com.example", "activity": ".Main"},
        }

    async def connect(self):
        return None

    async def get_ui_tree(self):
        return self.tree

    async def tap(self, x, y):
        self.taps.append((x, y))

    async def input_text(self, text, clear=False):
        self.inputs.append((text, clear))
        self.tree = {
            "a11y_tree": [{"text": text, "className": "android.widget.EditText"}],
            "phone_state": {"package": "com.example", "activity": ".Main"},
        }
        return True


class ReplaySecrets(CredentialManager):
    def __init__(self, specs=None):
        self.specs = specs

    async def resolve_key(self, key):
        return {"card": CARD, "login": "hunter22"}[key]

    async def get_keys(self):
        return ["card", "login"]

    async def get_secret_specs(self):
        if self.specs is None:
            return await super().get_secret_specs()
        return self.specs


def _macro(*steps):
    actions_ = []
    for step in steps:
        actions_.append(
            {
                "recorded_at_ms": 1,
                "elapsed_since_previous_ms": 0,
                "pre_state": normalize_ui_state(
                    {
                        "a11y_tree": [
                            {"text": "Name", "className": "android.widget.EditText"}
                        ],
                        "phone_state": {"package": "com.example", "activity": ".Main"},
                    }
                ),
                **step,
            }
        )
    return {"macro_schema_version": MACRO_SCHEMA_VERSION, "actions": actions_}


def _player(manager, handoff=None):
    player = MacroPlayer(
        delay_between_actions=0,
        state_timeout=0,
        state_poll_interval=0,
        handoff_runner=handoff,
    )
    player.driver = ReplayDriver()
    player.credential_manager = manager
    return player


GATED = [SecretSpec("card", requires_approval=True), SecretSpec("login")]


def test_replay_stops_on_a_gated_secret_before_any_action():
    player = _player(ReplaySecrets(GATED))
    macro = _macro(
        {"action_type": "tap", "x": 1, "y": 2},
        {"action_type": "type_secret", "secret_id": "card"},
        {"action_type": "tap", "x": 9, "y": 9},
    )

    assert run(player.replay_macro(macro)) is False

    assert player.abort_reason == APPROVAL_REQUIRED
    assert player.driver.taps == [] and player.driver.inputs == []


def test_recorded_protected_marker_wins_over_the_replay_managers_specs():
    player = _player(ReplaySecrets(None))  # default specs: nothing gated
    macro = _macro(
        {"action_type": "type_secret", "secret_id": "card", "protected": True},
        {"action_type": "tap", "x": 9, "y": 9},
    )

    assert run(player.replay_macro(macro)) is False

    assert player.abort_reason == APPROVAL_REQUIRED
    assert player.driver.inputs == [] and player.driver.taps == []


def test_replay_action_alone_refuses_a_gated_secret():
    player = _player(ReplaySecrets(GATED))

    ok = run(player.replay_action({"action_type": "type_secret", "secret_id": "card"}))

    assert ok is False and player.driver.inputs == []
    assert player.abort_reason == APPROVAL_REQUIRED


def test_approval_required_ends_replay_before_the_agent_handoff():
    handoffs = []

    async def handoff(**kwargs):
        handoffs.append(kwargs)
        return True

    player = _player(ReplaySecrets(GATED), handoff)
    player.on_mismatch = "agent"
    macro = _macro({"action_type": "type_secret", "secret_id": "card"})
    player.driver.tree = {"a11y_tree": [], "phone_state": {"package": "other"}}

    assert run(player.replay_macro(macro)) is False

    assert handoffs == []


def test_start_from_step_after_the_gated_secret_still_replays():
    player = _player(ReplaySecrets(GATED))
    macro = _macro(
        {"action_type": "type_secret", "secret_id": "card"},
        {"action_type": "tap", "x": 5, "y": 6},
    )

    assert run(player.replay_macro(macro, start_from_step=1)) is True
    assert player.driver.taps == [(5, 6)]


def test_ordinary_secret_replays_and_later_snapshots_are_masked():
    player = _player(ReplaySecrets(GATED))
    macro = _macro(
        {"action_type": "type_secret", "secret_id": "login"},
        {
            "action_type": "wait",
            "duration": 0,
            # recorded from masked states: only matches if replay masks too
            "pre_state": normalize_ui_state(
                {
                    "a11y_tree": [
                        {
                            "text": "[secret:login]",
                            "className": "android.widget.EditText",
                        }
                    ],
                    "phone_state": {"package": "com.example", "activity": ".Main"},
                }
            ),
        },
    )

    assert run(player.replay_macro(macro)) is True

    assert player.driver.inputs == [("hunter22", False)]
    # after the run the registry is cleared; inside it the state was masked
    player._redactor.register("login", "hunter22")
    snapshot = run(player.get_current_state_snapshot())
    assert "hunter22" not in str(snapshot)
    assert "[secret:login]" in str(snapshot)
    player._redactor.clear()


def test_replay_logs_do_not_contain_values(caplog):
    player = _player(ReplaySecrets(GATED))
    macro = _macro({"action_type": "type_secret", "secret_id": "login"})

    with caplog.at_level("DEBUG"):
        run(player.replay_macro(macro))

    assert "hunter22" not in caplog.text


def test_replay_input_error_never_logs_the_value(caplog):
    player = _player(ReplaySecrets(GATED))

    async def boom(text, clear=False):
        raise ValueError(f"no way to type {text}")

    player.driver.input_text = boom

    with caplog.at_level("DEBUG"):
        ok = run(
            player.replay_action({"action_type": "type_secret", "secret_id": "login"})
        )

    assert ok is False and "hunter22" not in caplog.text


def test_plan_event_import_surface_is_unchanged():
    assert ManagerPlanEvent and FinalizeEvent
