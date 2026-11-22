import asyncio
import io
import logging
from contextvars import copy_context
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_core_local.driver import cloud as cloud_module
from mobilerun_sdk import AsyncMobilerun

from mobilerun.core.devices.android import AndroidDevice
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.devices.recording import RecordingDevice


@pytest.fixture(autouse=True)
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


def test_sensitive_extra_fields_and_to_thread_are_redacted():
    from mobilerun.core.devices.components.identifiers import secret_scope

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


def test_secret_record_factory_chains_and_is_idempotent():
    from mobilerun.core.devices.components.identifiers import _secret_input

    original = logging.getLogRecordFactory()
    calls = []

    def previous(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    logging.setLogRecordFactory(previous)
    try:
        with _secret_input("factory-canary"):
            installed = logging.getLogRecordFactory()
            with _secret_input("nested"):
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
