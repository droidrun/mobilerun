"""Action output, private diagnostics, logger configuration and library presentation rules."""

import ast
import asyncio
import logging
import traceback
from types import SimpleNamespace

import pytest

from mobilerun.clients.cloud.tasks import _progress
from mobilerun.config.user_config import LoggingConfig
from mobilerun.core.extensions import Extension, action
from mobilerun.errors import (
    CloudServiceUnavailable,
    CredentialFieldNotFound,
    ExtensionActionError,
)
from mobilerun.logs.actions import agent_line, format_action
from mobilerun.logs.handlers import VERBOSE, configure_logging, resolve_log_level
from tests.support.paths import REPO_ROOT

PACKAGE = REPO_ROOT / "mobilerun"


@pytest.mark.parametrize(
    ("device_args", "lines", "expected"),
    [
        pytest.param(
            ("86f8d29f12345678",),
            (
                'data: {"type":"progress"}',
                'data: {"step":3,"action":{"name":"click","args":{"index":10}}}',
            ),
            "86f8d29f      · task  · step 3  tap #10",
            id="skip-empty-show-device",
        ),
        pytest.param(
            (),
            (
                'data: {"type":"tool_execution", "tool_name":"click_at", "tool_args":{"x":540,"y":1200,"text":"sensitive"}}',
                'data: {"type":"unknown", "token":"sensitive"}',
                "data: not-json",
            ),
            "unknown       · task  · step 1  tap (540, 1200) <redacted>",
            id="redact-sensitive-ignore-invalid",
        ),
    ],
)
def test_cloud_progress_reports_only_safe_actions(
    device_args, lines, expected, caplog, monkeypatch
):
    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def iter_lines(self):
            for line in lines:
                yield line

    client = SimpleNamespace(
        tasks=SimpleNamespace(
            with_streaming_response=SimpleNamespace(attach=lambda *a, **kw: Stream())
        )
    )
    # CLI setup can route records to stderr instead of pytest's root handler.
    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)
    with caplog.at_level(logging.INFO, logger="mobilerun.actions"):
        asyncio.run(_progress(client, "task", *device_args, billing="no_commerce"))
    assert [record.message for record in caplog.records] == [expected]


def test_action_never_shows_text_or_description():
    for field in ("text", "contentDescription", "description"):
        for label in ("Your code ABCDEF", "secret passphrase", "+", "Settings"):
            ui = SimpleNamespace(
                get_element=lambda index, field=field, label=label: {
                    field: label,
                    "className": "android.widget.Button",
                    "resourceId": "com.example:id/increment",
                }
            )
            assert format_action("tap", {"index": 8}, ui) == "tap Button[increment] #8"


def test_action_target_uses_available_class_id_and_index():
    for element, expected in (
        ({"className": "android.widget.EditText"}, "tap EditText #3"),
        ({"resourceId": "com.example:id/increment"}, "tap [increment] #3"),
        ({"text": "secret"}, "tap #3"),
    ):
        ui = SimpleNamespace(get_element=lambda index, element=element: element)
        assert format_action("tap", {"index": 3}, ui) == expected
    assert format_action("tap", {"x": 1, "y": 2}) == "tap (1, 2)"


def test_action_logging_accepts_context_without_optional_state_or_ui():
    ctx = SimpleNamespace(driver=SimpleNamespace(identifier="device"))
    assert agent_line(ctx, "tap", {"x": 1, "y": 2}) == ("device · step 1  tap (1, 2)")


def test_agent_action_uses_device_identifier_not_legacy_attributes():
    from mobilerun.core.devices.base import Device

    class ExampleDevice(Device):
        @property
        def identifier(self):
            return "typed-id"

        serial = "stale-secret"
        device_id = "old-id"

    ctx = SimpleNamespace(
        driver=ExampleDevice(), shared_state=SimpleNamespace(step_number=1), ui=None
    )
    assert agent_line(ctx, "tap", {"x": 1, "y": 2}) == "typed-id · step 1  tap (1, 2)"


def test_agent_action_with_no_identifier_has_safe_unknown_label():
    from mobilerun.core.devices.base import Device

    ctx = SimpleNamespace(
        driver=Device(), shared_state=SimpleNamespace(step_number=1), ui=None
    )
    assert agent_line(ctx, "tap", {"x": 1, "y": 2}) == "unknown · step 1  tap (1, 2)"


@pytest.mark.parametrize("timeout", [180, 180.0])
def test_timeout_reports_type_and_configured_nonsensitive_timeout(timeout, caplog):
    caplog.set_level(logging.INFO, logger="mobilerun")

    class Mailbox(Extension):
        @action
        async def wait_for_code(self, timeout: float = 180) -> str:
            """Wait for a test verification code."""
            raise TimeoutError("BACKEND-CANARY")

    async def run():
        mailbox = Mailbox(name="email")
        for invoke in (
            mailbox.wait_for_code,
            lambda: mailbox.wait_for_code(timeout),
            lambda: mailbox.actions.call("email.wait_for_code", {"timeout": timeout}),
            lambda: mailbox.actions.call("email.wait_for_code", {}),
        ):
            with pytest.raises(ExtensionActionError) as raised:
                await invoke()
            assert (
                str(raised.value)
                == "Action email.wait_for_code failed: TimeoutError (timed out after 180s)"
            )
            assert raised.value.cause_type == "TimeoutError"
            assert raised.value.__cause__ is None
            assert "BACKEND-CANARY" not in "".join(
                traceback.format_exception(raised.value)
            )

    asyncio.run(run())


def test_timeout_without_a_configured_timeout_does_not_guess(caplog):
    caplog.set_level(logging.INFO, logger="mobilerun")

    class Resource(Extension):
        @action
        async def run(self) -> None:
            """Fail with an unknown backend timeout."""
            raise TimeoutError("PRIVATE-CANARY")

    with pytest.raises(ExtensionActionError) as raised:
        asyncio.run(Resource().run())
    assert str(raised.value) == "Action extension.run failed: TimeoutError (timed out)"
    assert raised.value.cause_type == "TimeoutError"


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_sensitive_action_never_exposes_backend_message_or_chain(level, caplog):
    caplog.set_level(level, logger="mobilerun")
    canary = "PRIVATE-CANARY"

    class Vault(Extension):
        @action(sensitive_parameters=frozenset({"password"}))
        async def login(self, password: str, timeout: float = 180) -> None:
            """Perform a sensitive SDK operation."""
            raise TimeoutError(f"secret={password}")

    with pytest.raises(ExtensionActionError) as raised:
        asyncio.run(Vault().login(canary))
    assert raised.value.cause_type == "TimeoutError"
    assert "180s" in str(raised.value)
    assert raised.value.__cause__ is None
    assert "PRIVATE-CANARY" not in "".join(traceback.format_exception(raised.value))


def test_sensitive_timeout_is_not_added_to_message(caplog):
    caplog.set_level(logging.INFO, logger="mobilerun")

    class Resource(Extension):
        @action(sensitive_parameters=frozenset({"timeout"}))
        async def run(self, timeout: float) -> None:
            """Treat the timeout itself as confidential."""
            raise TimeoutError("PRIVATE-CANARY")

    with pytest.raises(ExtensionActionError) as raised:
        asyncio.run(Resource().run(987654))
    assert "TimeoutError (timed out)" in str(raised.value)
    assert "987654" not in str(raised.value)


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_cloud_error_propagates_unchanged(level, caplog):
    caplog.set_level(level, logger="mobilerun")
    original = CloudServiceUnavailable(operation="List SMS messages", status=503)

    class Resource(Extension):
        @action
        async def run(self) -> None:
            """Exercise opt-in debug diagnostics."""
            raise original

    with pytest.raises(CloudServiceUnavailable) as raised:
        asyncio.run(Resource().run())
    assert raised.value is original


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_nonsensitive_debug_action_chains_only_when_debug_enabled(level, caplog):
    caplog.set_level(level, logger="mobilerun")
    original = RuntimeError("BACKEND-CANARY")

    class Resource(Extension):
        @action
        async def run(self) -> None:
            """Exercise opt-in debug diagnostics for ordinary exceptions."""
            raise original

    with pytest.raises(ExtensionActionError) as raised:
        asyncio.run(Resource().run())
    assert raised.value.cause_type == "RuntimeError"
    assert "RuntimeError" in str(raised.value)
    assert "BACKEND-CANARY" not in str(raised.value)
    assert raised.value.__cause__ is (original if level == logging.DEBUG else None)


def test_nested_nonsensitive_action_cannot_escape_sensitive_debug_scope(caplog):
    caplog.set_level(logging.DEBUG, logger="mobilerun")
    canary = "PRIVATE-CANARY"

    class Resource(Extension):
        @action
        async def inner(self) -> None:
            """Fail inside the parent sensitive scope."""
            raise RuntimeError(canary)

        @action(sensitive_parameters=frozenset({"password"}))
        async def outer(self, password: str) -> None:
            """Delegate within a sensitive call."""
            await self.inner()

    with pytest.raises(ExtensionActionError) as raised:
        asyncio.run(Resource().outer(canary))
    assert raised.value.cause_type == "RuntimeError"
    assert raised.value.__cause__ is None
    assert "PRIVATE-CANARY" not in "".join(traceback.format_exception(raised.value))


def test_value_safe_error_still_propagates_unchanged(caplog):
    caplog.set_level(logging.DEBUG, logger="mobilerun")
    original = CredentialFieldNotFound("Credential field not found")

    class Resource(Extension):
        @action
        async def run(self) -> None:
            """Expose a public value-safe typed failure."""
            raise original

    with pytest.raises(CredentialFieldNotFound) as raised:
        asyncio.run(Resource().run())
    assert raised.value is original


@pytest.mark.parametrize(
    ("cli", "env", "config", "debug", "expected"),
    [
        (None, None, "normal", False, "normal"),
        (None, None, "normal", True, "debug"),
        (None, "verbose", "quiet", True, "verbose"),
        ("quiet", "debug", "verbose", True, "quiet"),
    ],
)
def test_level_precedence(cli, env, config, debug, expected):
    assert (
        resolve_log_level(cli, env, LoggingConfig(level=config, debug=debug))
        == expected
    )


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("quiet", ["result"]),
        ("normal", ["goal", "step 1", "step 2", "result"]),
        ("verbose", ["goal", "step 1", "step 2", "reasoning", "result"]),
        ("debug", ["goal", "step 1", "step 2", "reasoning", "debug", "result"]),
    ],
)
def test_filtering(level, expected):
    class Capture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    handler = Capture()
    logger = logging.getLogger("mobilerun")
    old_handlers, old_level, old_propagate = (
        logger.handlers[:],
        logger.level,
        logger.propagate,
    )
    try:
        configure_logging(debug=False, handler=handler, level=level)
        logger.info("goal")
        logger.info("step 1")
        logger.info("step 2")
        logger.log(VERBOSE, "reasoning")
        logger.debug("debug")
        logger.warning("result")
        assert handler.messages == expected
    finally:
        logger.handlers = old_handlers
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def test_log_level_is_registered(monkeypatch):
    from mobilerun.config.settings import get_settings

    monkeypatch.setenv("MOBILERUN_LOG_LEVEL", "verbose")
    assert get_settings().MOBILERUN_LOG_LEVEL == "verbose"


def test_default_device_environment_setting_removed(monkeypatch):
    from mobilerun.config.settings import get_settings

    monkeypatch.setenv("MOBILERUN_DEFAULT_DEVICE", "obsolete")
    assert "default_device" not in get_settings().model_fields_set
    assert "default_device" not in type(get_settings()).model_fields


def test_library_log_messages_have_no_emoji():
    violations = []
    for path in PACKAGE.rglob("*.py"):
        if path == PACKAGE / "logs" / "actions.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            if node.func.attr not in {
                "debug",
                "info",
                "warning",
                "error",
                "exception",
                "critical",
            }:
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id not in {
                "logger",
                "log",
                "logging",
            }:
                continue
            for arg in node.args[:1]:
                for part in ast.walk(arg):
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        if any(
                            0x1F300 <= ord(c) <= 0x1FAFF
                            or 0x2300 <= ord(c) <= 0x23FF
                            or 0x2600 <= ord(c) <= 0x27BF
                            for c in part.value
                        ):
                            violations.append(
                                f"{path.relative_to(PACKAGE)}:{node.lineno}"
                            )
    assert not violations, violations


def test_shared_library_has_no_browser_or_stdout_presentation():
    violations = []
    for directory in ("utils", "clients", "core"):
        for path in (PACKAGE / directory).rglob("*.py"):
            # Opt-in checkout handlers are the only library presentation boundary.
            if path == PACKAGE / "clients" / "cloud" / "checkout.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    modules = (
                        [alias.name for alias in node.names]
                        if isinstance(node, ast.Import)
                        else [node.module or ""]
                    )
                    if any(module.split(".")[0] == "webbrowser" for module in modules):
                        violations.append(
                            f"{path.relative_to(PACKAGE)}:{node.lineno}: webbrowser"
                        )
                if isinstance(node, ast.Call) and ast.unparse(node.func) in {
                    "print",
                    "sys.stdout.write",
                }:
                    violations.append(
                        f"{path.relative_to(PACKAGE)}:{node.lineno}: stdout"
                    )
    assert not violations, violations


def test_library_does_not_call_print():
    violations = []
    for path in PACKAGE.rglob("*.py"):
        # Opt-in checkout handlers deliberately print payment instructions.
        if path == PACKAGE / "clients" / "cloud" / "checkout.py":
            continue
        if "cli" in path.relative_to(PACKAGE).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                violations.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
    assert not violations, violations
