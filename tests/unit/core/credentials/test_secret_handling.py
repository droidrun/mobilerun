"""Offline contracts for credential providers, secret input and observation masking."""

import asyncio
import io
import json
import logging
from contextvars import copy_context
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from mobilerun_core_local.driver import cloud as cloud_module
from mobilerun_sdk import AsyncMobilerun

from mobilerun.core.credentials import (
    CloudCredentials,
    CredentialPolicy,
    DictCredentials,
    EnvCredentials,
    FileCredentials,
)
from mobilerun.core.credentials.masking import CredentialMasks
from mobilerun.core.devices.android import AndroidDevice
from mobilerun.core.devices.base import Device
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.devices.pool import DevicePool
from mobilerun.core.devices.recording import RecordingDevice
from mobilerun.core.devices.stealth import StealthDevice
from mobilerun.core.extensions import Extension
from mobilerun.core.state.options import StateProviderOptions, android_state_provider
from mobilerun.errors import (
    CredentialNotFoundError,
    DeviceError,
    DeviceOperationNotSupported,
    ExtensionActionError,
    ExtensionNotBound,
    NoFocusedInput,
)
from mobilerun.macro.errors import MacroError
from mobilerun.macro.models import normalize_ui_state
from mobilerun.macro.player import MacroPlayer
from mobilerun.macro.utils import compare_states
from tests.support.devices import MemoryDevice
from tests.support.extensions import check_extension_contract


class FakeDevice(Device):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_text = AsyncMock(return_value=True)
        self._input_text_raw = self.input_text
        self.state.refresh = AsyncMock(
            return_value=SimpleNamespace(
                phone_state={
                    "focusedElement": {"className": "EditText"},
                    "isEditable": True,
                }
            )
        )


def test_dict_normalizes_flat_and_nested_and_redacts():
    provider = DictCredentials(
        {
            "gmail_password": "private-value",
            "com.gmail": {
                "work": {"email": "private-email", "password": "private-password"}
            },
        }
    )
    asyncio.run(provider.bind(FakeDevice()))
    items = asyncio.run(provider.list())
    assert [(item["package"], item["name"], item["fields"]) for item in items] == [
        (None, "gmail_password", ["password"]),
        ("com.gmail", "work", ["email", "password"]),
    ]
    assert "private-" not in repr(items)
    assert "private-" not in str(items[0])
    device = provider.device
    asyncio.run(provider.type_field("com.gmail", "email", "work"))
    device.input_text.assert_awaited_once_with("private-email")


@pytest.mark.parametrize("extension", ["yaml", "json"])
def test_file_normalizes_flat_and_nested(tmp_path, extension):
    path = tmp_path / f"credentials.{extension}"
    path.write_text(
        json.dumps(
            {
                "gmail_password": "private-value",
                "com.gmail": {"work": {"username": "private-name"}},
            }
        )
    )
    path.chmod(0o600)
    provider = FileCredentials(path)
    device = FakeDevice()
    asyncio.run(provider.bind(device))
    asyncio.run(provider.type_field(None, "password", "gmail_password"))
    asyncio.run(provider.type_field("com.gmail", "username", "work"))
    assert [call.args[0] for call in device.input_text.await_args_list] == [
        "private-value",
        "private-name",
    ]


def test_yaml_flat(tmp_path):
    path = tmp_path / "credentials.yaml"
    path.write_text("gmail_password: private-value\n")
    path.chmod(0o600)
    provider = FileCredentials(path)
    asyncio.run(provider.bind(FakeDevice()))
    assert asyncio.run(provider.list())[0]["name"] == "gmail_password"


def test_env_normalizes_flat(monkeypatch):
    monkeypatch.setenv("MOBILERUN_SECRET_GMAIL_PASSWORD", "private-value")
    provider = EnvCredentials()
    device = FakeDevice()
    asyncio.run(provider.bind(device))
    asyncio.run(provider.type_field(None, "password", "GMAIL_PASSWORD"))
    device.input_text.assert_awaited_once_with("private-value")


def test_cloud_paginated_cached_and_login_key(monkeypatch):

    calls = []

    async def list_page(*, page):
        calls.append(page)
        return SimpleNamespace(
            items=[
                SimpleNamespace(
                    package_name=f"com.app{page}",
                    credential_name="work",
                    fields=[
                        SimpleNamespace(field_type="password", value="private-value")
                    ],
                    source="mobilerun" if page == 1 else "onepassword",
                )
            ],
            pagination=SimpleNamespace(has_next=page < 2),
        )

    async def list_package(package):
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    package_name=package,
                    credential_name="work",
                    fields=[
                        SimpleNamespace(field_type="password", value="private-value")
                    ],
                )
            ]
        )

    client = SimpleNamespace(
        credentials=SimpleNamespace(
            list=list_page, packages=SimpleNamespace(list=list_package)
        ),
        close=AsyncMock(),
    )
    constructor = []

    def make_client(**kwargs):
        constructor.append(kwargs)
        return client

    from mobilerun.clients.cloud import client as transport

    monkeypatch.setattr(transport, "AsyncMobilerun", make_client)
    monkeypatch.setattr(transport, "resolve_cloud_api_key", lambda: "fake-login-key")
    provider = CloudCredentials(base_url="https://fake.invalid")

    async def run():
        await provider.bind(FakeDevice())
        assert len(await provider.list()) == 2
        assert (await provider.list("com.app2"))[0]["name"] == "work"
        assert "private-value" not in repr(await provider.list())
        await provider.close()

    asyncio.run(run())
    assert calls == [1, 2]
    assert constructor == [
        {"api_key": "fake-login-key", "base_url": "https://fake.invalid"}
    ]


def test_device_order_recording_wrappers_pool_and_no_log(caplog):
    caplog.set_level("DEBUG")
    first = DictCredentials({"login": "private-first"})
    second = DictCredentials(
        {"login": "private-second", "other": "private-other"}, name="second"
    )
    device = FakeDevice()
    asyncio.run(first.bind(device))
    asyncio.run(second.bind(device))
    recording = RecordingDevice(device)
    stealth = StealthDevice(recording)
    assert dict(recording.extensions) == {"vault": first, "second": second}
    assert dict(stealth.extensions) == dict(recording.extensions)

    async def run():
        assert await first.type_field(None, "password", name="login") is None
        assert await second.type_field(None, "password", name="other") is None
        assert DevicePool([recording]).extensions[0]["vault"] is first
        assert await first.type_field(None, "password", name="login") is None

    asyncio.run(run())
    assert [call.args[0] for call in device.input_text.await_args_list] == [
        "private-first",
        "private-other",
        "private-first",
    ]
    assert recording.recorded_actions[0]["name"] == "vault.type_field"
    assert "private-" not in repr(recording.recorded_actions)
    assert "private-" not in caplog.text


@pytest.mark.parametrize(
    "reference",
    [
        {"package": "com.gmail", "field": "email", "name": "work"},
        {"secret": "login"},
        {"secret_id": "login"},
    ],
)
def test_macro_uses_device_credentials(reference, caplog):
    caplog.set_level("DEBUG")
    device = FakeDevice()
    provider = DictCredentials(
        {
            "login": "private-value",
            "com.gmail": {"work": {"email": "private-value"}},
        }
    )
    asyncio.run(provider.bind(device))
    assert asyncio.run(
        MacroPlayer(device).replay_action({"action_type": "type_secret", **reference})
    )
    device.input_text.assert_awaited_once_with("private-value")
    assert "private-value" not in caplog.text


@pytest.fixture
def focused_state(monkeypatch):
    from mobilerun.core.state.view import UIStateView

    monkeypatch.setattr(
        UIStateView,
        "refresh",
        AsyncMock(
            return_value=SimpleNamespace(
                phone_state={
                    "focusedElement": {"className": "EditText"},
                    "isEditable": True,
                }
            )
        ),
    )


@pytest.mark.usefixtures("focused_state")
def test_vault_exposes_only_metadata_and_types_secret():
    asyncio.run(_check_vault())


async def _check_vault():
    secret = "canary-vault-private-value"
    vault = DictCredentials({"app": {"account": {"password": secret}}})
    device = MemoryDevice()
    assert isinstance(vault, Extension)
    await vault.bind(device)
    metadata = await vault.list()
    assert secret not in repr(metadata)
    assert secret not in repr(await vault.context())
    assert await vault.type_field("app", "password") is None
    assert device.inputs == [secret]


@pytest.mark.usefixtures("focused_state")
def test_dict_vault_extension_contract():
    async def device():
        return MemoryDevice()

    asyncio.run(
        check_extension_contract(lambda name: DictCredentials({}, name=name), device)
    )


@pytest.mark.usefixtures("focused_state")
def test_file_and_env_vault_extension_contract(tmp_path):
    path = tmp_path / "credentials.yml"
    path.write_text("{}")
    path.chmod(0o600)

    async def device():
        return MemoryDevice()

    for factory in (
        lambda name: FileCredentials(path, name=name),
        lambda name: EnvCredentials("NONEXISTENT_TEST_PREFIX_", name=name),
    ):
        asyncio.run(check_extension_contract(factory, device))


@pytest.mark.usefixtures("focused_state")
def test_cloud_vault_extension_contract():
    async def device():
        return MemoryDevice()

    with patch(
        "mobilerun.core.credentials.cloud.secrets.list_secrets",
        AsyncMock(return_value=[]),
    ):
        asyncio.run(
            check_extension_contract(
                lambda name: CloudCredentials(api_key="test", name=name), device
            )
        )


@pytest.mark.usefixtures("focused_state")
def test_type_field_sanitizes_secret_input_failure(caplog):
    async def run():
        secret = "secret-canary-must-not-escape"
        device = MemoryDevice()
        device._input_secret = AsyncMock(side_effect=RuntimeError(secret))
        vault = DictCredentials({"app": {"one": {"password": secret}}})
        await vault.bind(device)
        with pytest.raises(ExtensionActionError) as exc:
            await vault.type_field("app", "password")
        assert secret not in str(exc.value)
        assert secret not in caplog.text
        device._input_secret.assert_awaited_once_with(secret)

    asyncio.run(run())


@pytest.mark.usefixtures("focused_state")
def test_type_field_reports_unsuccessful_input():
    async def run():
        device = MemoryDevice()
        device._input_secret = AsyncMock(return_value=False)
        vault = DictCredentials({"app": {"one": {"password": "secret"}}})
        await vault.bind(device)
        with pytest.raises(ExtensionActionError) as error:
            await vault.type_field("app", "password")
        assert isinstance(error.value, DeviceError)
        assert "secret" not in str(error.value)

    asyncio.run(run())


@pytest.mark.usefixtures("focused_state")
def test_type_field_uses_typed_missing_and_unbound_errors():
    async def run():
        vault = DictCredentials({})
        with pytest.raises(ExtensionNotBound):
            await vault.type_field("app", "password")
        await vault.bind(MemoryDevice())
        with pytest.raises(CredentialNotFoundError) as error:
            await vault.type_field("app", "password")
        from mobilerun.errors import CredentialFieldNotFound

        assert isinstance(error.value, CredentialFieldNotFound)
        assert not isinstance(error.value, ExtensionActionError)
        assert "password-value-canary" not in str(error.value)

    asyncio.run(run())


@pytest.mark.usefixtures("focused_state")
def test_type_field_device_action_rejects_unbound_before_lookup():
    async def run():
        vault = DictCredentials({})
        vault._find = AsyncMock()
        assert vault.actions.get("vault.type_field").requires_device
        with pytest.raises(ExtensionNotBound):
            await vault.type_field("app", "password")
        vault._find.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize(
    "phone_state",
    [
        {"focusedElement": None, "isEditable": False},
        {"focusedElement": {"className": "Button"}, "isEditable": False},
    ],
)
def test_secret_refuses_unfocused_or_noneditable_input(phone_state):
    async def run():
        device = Device()
        device.state.refresh = AsyncMock(
            return_value=SimpleNamespace(phone_state=phone_state)
        )
        device._input_text_raw = AsyncMock(return_value=True)
        with pytest.raises(NoFocusedInput, match="No text field is focused") as error:
            await device._input_secret("secret-canary")
        assert "secret-canary" not in str(error.value)
        device._input_text_raw.assert_not_awaited()

    asyncio.run(run())


def test_secret_types_only_when_focused_and_editable():
    async def run():
        device = Device()
        device.state.refresh = AsyncMock(
            return_value=SimpleNamespace(
                phone_state={
                    "focusedElement": {"className": "EditText"},
                    "isEditable": True,
                }
            )
        )
        device._input_text_raw = AsyncMock(return_value=True)
        assert await device._input_secret("demo-user")
        device._input_text_raw.assert_awaited_once_with("demo-user")

    asyncio.run(run())


@pytest.mark.parametrize(
    ("app_nodes", "overrides", "types_secret"),
    [
        pytest.param([{}], {}, True, id="edit-text-without-editable-flag"),
        pytest.param(
            [{}], {"focusedElement": {}, "isEditable": True}, True, id="empty-summary"
        ),
        pytest.param([{"isFocused": False}], {}, False, id="ime-only"),
        pytest.param([{}, {}], {}, False, id="ambiguous-focus"),
        pytest.param([{"packageName": "com.other"}], {}, False, id="other-package"),
        pytest.param([{"isEnabled": False}], {}, False, id="disabled"),
        pytest.param([{}], {"keyboardVisible": False}, False, id="keyboard-hidden"),
        pytest.param([{}], {"degraded": True}, False, id="degraded-cache"),
        pytest.param(
            [], {"degraded": True, "isEditable": True}, True, id="degraded-strict"
        ),
        pytest.param([{"windowId": 2}], {}, False, id="same-package-other-window"),
        pytest.param([{"windowId": None}], {}, False, id="unknown-node-window"),
        pytest.param([{}], {"rootWindowId": None}, False, id="unknown-root-window"),
        pytest.param(
            [{}], {"omitWindowId": True}, True, id="single-window-without-ids"
        ),
        pytest.param(
            [
                {},
                {
                    "className": "android.widget.FrameLayout",
                    "isFocused": False,
                    "children": [{"className": "android.widget.TextView"}],
                },
            ],
            {"omitWindowId": True},
            False,
            id="two-same-package-roots-without-ids",
        ),
        pytest.param(
            [
                {
                    "className": "android.widget.LinearLayout",
                    "isFocused": False,
                    "children": [
                        {
                            "className": "android.widget.EditText",
                            "packageName": "com.app",
                            "isEnabled": True,
                            "isFocused": True,
                        }
                    ],
                },
                {
                    "className": "android.view.View",
                    "resourceId": "com.app:id/overlay",
                    "isFocused": False,
                },
            ],
            {"omitWindowId": True},
            True,
            id="windowless-chrome-shape",
        ),
        pytest.param([{"className": "android.widget.Button"}], {}, False, id="button"),
        pytest.param(
            [{"className": "custom.NotAutoCompleteTextView"}],
            {},
            False,
            id="autocomplete-lookalike",
        ),
        pytest.param([{"isEditable": False}], {}, False, id="explicitly-not-editable"),
        pytest.param(
            [{"isEditable": True, "className": "custom.Input"}],
            {},
            True,
            id="explicitly-editable",
        ),
        pytest.param(
            [{"className": "android.widget.AutoCompleteTextView"}],
            {},
            True,
            id="autocomplete-without-editable-flag",
        ),
        pytest.param(
            [{}], {"omitPackage": True}, False, id="missing-foreground-package"
        ),
    ],
)
def test_secret_input_checks_raw_app_focus(
    app_nodes: list[dict[str, object]], overrides: dict[str, object], types_secret: bool
) -> None:
    async def run() -> None:
        device = MemoryDevice()
        device._set_state_provider(
            android_state_provider(device, StateProviderOptions())
        )
        device._get_ui_tree = AsyncMock(
            return_value={
                "device_context": {"screen_bounds": {"width": 100, "height": 100}},
                "phone_state": {
                    **(
                        {}
                        if overrides.get("omitPackage")
                        else {"packageName": "com.app"}
                    ),
                    "focusedElement": overrides.get(
                        "focusedElement", {"className": "android.widget.FrameLayout"}
                    ),
                    "keyboardVisible": overrides.get("keyboardVisible", True),
                    "isEditable": overrides.get("isEditable", False),
                },
                "a11y_tree": {
                    "className": "android.widget.FrameLayout",
                    "packageName": "com.app",
                    **(
                        {}
                        if overrides.get("omitWindowId")
                        else {"windowId": overrides.get("rootWindowId", 1)}
                    ),
                    "children": [
                        {
                            "className": "android.widget.EditText",
                            "packageName": "com.app",
                            **(
                                {} if overrides.get("omitWindowId") else {"windowId": 1}
                            ),
                            "isEnabled": True,
                            "isFocused": True,
                            **node,
                        }
                        for node in app_nodes
                    ],
                },
                "ime_tree": {
                    "className": "android.widget.FrameLayout",
                    # Package match alone must not let an IME node count.
                    "packageName": "com.app",
                    "windowId": 1,
                    "isEnabled": True,
                    "isFocused": True,
                },
                **(
                    {"degraded": True, "snapshotAgeMs": 500}
                    if overrides.get("degraded")
                    else {}
                ),
            }
        )
        vault = DictCredentials({"com.app": {"demo": {"password": "demo-value"}}})
        await vault.bind(device)
        if types_secret:
            await vault.type_field("com.app", "password", name="demo")
            assert device.inputs == ["demo-value"]
        else:
            with pytest.raises(NoFocusedInput):
                await vault.type_field("com.app", "password", name="demo")
            assert device.inputs == []
        device._get_ui_tree.assert_awaited_once()

    asyncio.run(run())


def test_secret_rejects_unknown_focus_capability():
    async def run():
        device = Device()
        device.state.refresh = AsyncMock(
            return_value=SimpleNamespace(
                phone_state={"observationMode": "screenshot_only"}
            )
        )
        device._input_text_raw = AsyncMock(return_value=True)
        with pytest.raises(DeviceOperationNotSupported):
            await device._input_secret("secret-canary")
        device._input_text_raw.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.usefixtures("focused_state")
def test_sensitive_extra_fields_and_to_thread_are_redacted():
    from mobilerun.utils.redaction import secret_scope

    secret = "extra-thread-canary"
    stream = io.StringIO()
    logger = logging.getLogger("thirdparty.extra-field-test")
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(request)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.WARNING)

    async def run():
        with secret_scope(secret):
            await asyncio.to_thread(
                logger.warning, "request", extra={"request": {"secret": secret}}
            )
            await asyncio.get_running_loop().run_in_executor(
                None,
                copy_context().run,
                partial(logger.warning, "executor", extra={"request": secret}),
            )

    try:
        asyncio.run(run())
    finally:
        logger.removeHandler(handler)
        logger.propagate = True
    assert secret not in stream.getvalue()
    assert "***" in stream.getvalue()


@pytest.mark.usefixtures("focused_state")
def test_secret_record_factory_chains_and_is_idempotent():
    from mobilerun.utils.redaction import secret_scope

    original = logging.getLogRecordFactory()
    calls = []

    def previous(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    logging.setLogRecordFactory(previous)
    try:
        with secret_scope("factory-canary"):
            installed = logging.getLogRecordFactory()
            with secret_scope("nested"):
                assert logging.getLogRecordFactory() is installed
                record = logging.getLogger("factory-test").makeRecord(
                    "factory-test",
                    logging.INFO,
                    __file__,
                    1,
                    "%s",
                    ("factory-canary",),
                    None,
                )
                assert record.getMessage() == "***"
                assert calls == [True]
    finally:
        logging.setLogRecordFactory(original)


@pytest.mark.usefixtures("focused_state")
def test_new_child_and_isolated_logger_redact_at_record_creation(caplog):
    secret = "new-child-secret-7492"
    device = AndroidDevice("device")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    isolated = logging.getLogger("external.isolated-secret-test")
    isolated.addHandler(handler)
    isolated.propagate = False
    isolated.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG)

    async def raw(value, *args, **kwargs):
        child = logging.getLogger("mobilerun_sdk.created_during_secret_7492")
        child.warning("new child %s", value)
        try:
            raise ValueError(value)
        except ValueError:
            isolated.exception("isolated %s", value, stack_info=True)
        return True

    device._input_text_raw = raw
    try:
        asyncio.run(device._input_secret(secret))
    finally:
        isolated.removeHandler(handler)
        isolated.propagate = True
    assert "new child ***" in caplog.text
    assert secret not in caplog.text
    assert "isolated ***" in stream.getvalue()
    assert secret not in stream.getvalue()


@pytest.mark.usefixtures("focused_state")
@pytest.mark.parametrize(
    ("device_factory", "logger_name"),
    [
        (lambda: CloudDevice("device", api_key="test"), "mobilerun_sdk._base_client"),
        (lambda: AndroidDevice("device"), "mobilerun_core_local"),
    ],
)
def test_secret_input_redacts_logs_exceptions_and_recording(
    device_factory, logger_name, caplog
):
    secret = "SENSITIVE-CANARY-9321"
    logger = logging.getLogger(logger_name)
    caplog.set_level(logging.DEBUG, logger=logger_name)
    device = device_factory()
    recorded = RecordingDevice(device)

    async def raw(value, *args, **kwargs):
        try:
            raise ValueError(f"backend {value}")
        except ValueError:
            logger.debug("request %s", {"text": value}, exc_info=True)
            raise RuntimeError(f"failure {value}") from None

    device._input_text_raw = raw

    async def run():
        with pytest.raises(RuntimeError) as exc:
            await recorded._input_secret(secret)
        assert secret not in str(exc.value)
        assert exc.value.__context__ is None

    asyncio.run(run())
    assert secret not in caplog.text
    assert all(secret not in repr(record.__dict__) for record in caplog.records)
    assert secret not in repr(recorded.recorded_actions)


@pytest.mark.usefixtures("focused_state")
def test_redaction_context_is_task_local_and_is_reset(caplog):
    secret = "context-canary"
    logger = logging.getLogger("mobilerun_sdk._base_client")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    device = AndroidDevice("device")
    entered = asyncio.Event()
    resume = asyncio.Event()

    async def raw(value, *args, **kwargs):
        logger.debug("inside %s", value)
        entered.set()
        await resume.wait()
        return True

    device._input_text_raw = raw

    async def run():
        task = asyncio.create_task(device._input_secret(secret))
        await entered.wait()
        logger.debug("outside %s", secret)
        resume.set()
        await task
        logger.debug("after %s", secret)

    asyncio.run(run())
    assert caplog.records[0].getMessage() == "inside ***"
    assert caplog.records[1].getMessage() == f"outside {secret}"
    assert caplog.records[2].getMessage() == f"after {secret}"


@pytest.mark.usefixtures("focused_state")
@pytest.mark.parametrize(
    "secret", ["cloud-sdk-secret-canary", "sdk-secret\nwith'quote"]
)
def test_real_cloud_sdk_debug_options_are_redacted(monkeypatch, caplog, secret):
    caplog.set_level(logging.DEBUG, logger="mobilerun_sdk._base_client")
    clients = []

    def sdk_factory(**kwargs):
        client = AsyncMobilerun(
            **kwargs,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: httpx.Response(204))
            ),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(cloud_module, "AsyncMobilerun", sdk_factory)

    async def run():
        device = CloudDevice(
            "device", api_key="test", base_url="https://example.invalid"
        )
        recording = RecordingDevice(device)
        assert await recording._input_secret(secret)
        assert not recording.recorded_actions
        for client in clients:
            await client.close()

    asyncio.run(run())
    assert any("Request options" in record.getMessage() for record in caplog.records)
    assert all(secret not in repr(record.__dict__) for record in caplog.records)
    assert repr(secret)[1:-1] not in caplog.text


@pytest.mark.usefixtures("focused_state")
def test_android_driver_failed_adb_shell_log_is_redacted(caplog):
    secret = "android-adb-secret-canary"
    caplog.set_level(logging.DEBUG, logger="mobilerun_core_local")
    device = AndroidDevice("device")
    device._ensure_connected = AsyncMock()
    device._driver.device = SimpleNamespace(
        shell=AsyncMock(side_effect=RuntimeError(secret))
    )

    async def input_text(value, *args, **kwargs):
        await device._driver._try_shell(f"input text {value}")
        raise ValueError(value)

    device._driver.input_text = input_text

    async def run():
        recording = RecordingDevice(device)
        with pytest.raises(ValueError) as error:
            await recording._input_secret(secret)
        assert secret not in str(error.value)
        assert not recording.recorded_actions

    asyncio.run(run())
    assert any(
        "ADB shell command failed" in record.getMessage() for record in caplog.records
    )
    assert all(secret not in repr(record.__dict__) for record in caplog.records)


def test_reordered_split_echo_masks_only_bounded_fragments():
    masks = CredentialMasks()
    masks.register("pin", "123456")
    state = normalize_ui_state(
        [
            {"text": text, "contentDescription": f"Order 1234; group {echo}"}
            for text, echo in [("123", "456"), ("456", "123")]
        ]
    )
    assert [node["content_description"] for node in masks.snapshot(state)["nodes"]] == [
        "Order 1234; group [secret:pin]"
    ] * 2


def test_split_pin_masks_echoes_without_erasing_description_text():
    masks = CredentialMasks()
    masks.register("pin", "123456")
    state = normalize_ui_state(
        [
            {"text": p, "contentDescription": f"Digit group: {p}; done"}
            for p in ("123", "456")
        ]
    )
    for node in masks.snapshot(state)["nodes"]:
        assert node["text"] == "[secret:pin]"
        assert node["content_description"] == "Digit group: [secret:pin]; done"


def snapshot(masks, *texts):
    return masks.snapshot(normalize_ui_state([{"text": t} for t in texts]))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_overlapping_pin_and_card_mask_outer_span(monkeypatch, reverse, split):
    masks = CredentialMasks()
    masks.register("pin", "4111")
    masks.mask_card_numbers = True
    if reverse:
        rules = tuple(masks._rules())
        monkeypatch.setattr(masks, "_rules", lambda: reversed(rules))
    card = "4111 1111 1111 1111"
    texts = card.split() if split else [card]
    assert [node["text"] for node in snapshot(masks, *texts)["nodes"]] == [
        "[secret:card_number]"
    ] * len(texts)


def test_package_independent_mask_registers_every_matching_field():
    masks = CredentialMasks()
    vault = DictCredentials(
        {p: {"login": {"password": v}} for p, v in [("a", "first"), ("b", "second")]}
    )
    vault.policies = (CredentialPolicy("login", mask_from_start=True),)
    asyncio.run(vault.prepare_masks(masks))
    assert [n["text"] for n in snapshot(masks, "first", "second")["nodes"]] == [
        "[secret:login]"
    ] * 2


@pytest.mark.parametrize("protected", [True, False])
def test_missing_named_credential_binding_fails_before_tap(protected):
    async def run():
        device = MemoryDevice()
        device.tap = AsyncMock()
        await DictCredentials({"login": "test"}).bind(device)
        macro = {
            "macro_schema_version": "2.0",
            "actions": [
                {"type": "tap"},
                {
                    "type": "extension_call",
                    "name": "other_vault.type_field",
                    "arguments": {"name": "login"},
                    "protected": protected,
                },
            ],
        }
        with pytest.raises(MacroError):
            await MacroPlayer(device, delay_between_actions=0).replay_macro(macro)
        device.tap.assert_not_awaited()

    asyncio.run(run())


def test_masks_preserve_digit_word_boundaries_and_unrelated_label_text():
    masks = CredentialMasks()
    masks.register("pin", "123")
    masks.register("name", "Ann")
    assert not compare_states(
        snapshot(masks, "Order 1234"), snapshot(masks, "Order 9123"), 1.0
    ).matches
    text = "Ann versus Annie: 123 or 1234"
    assert (
        snapshot(masks, text)["nodes"][0]["text"]
        == "[secret:name] versus Annie: [secret:pin] or 1234"
    )
