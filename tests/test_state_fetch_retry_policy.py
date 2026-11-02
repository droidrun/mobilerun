"""Retry policy of ``fetch_state_with_retry`` for timeouts vs other errors."""

import asyncio
import logging

import httpx
import pytest

import mobilerun.tools.ui.provider as provider_module
from mobilerun.tools.ui.provider import fetch_state_with_retry

STATE = {"a11y_tree": {}, "phone_state": {}, "device_context": {}}
REQUEST = httpx.Request("GET", "http://127.0.0.1:8080/state_full")


def _read_timeout():
    return httpx.ReadTimeout("", request=REQUEST)


def _fetch(*outcomes, events):
    queue = list(outcomes)

    async def fetch():
        events.append("fetch")
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return fetch


def _recovery(events):
    async def recovery():
        events.append("recover")

    return recovery


def _run(fetch, events, **kwargs):
    return asyncio.run(
        fetch_state_with_retry(
            fetch=fetch, recovery=_recovery(events), retry_delays=[0], **kwargs
        )
    )


@pytest.mark.parametrize(
    "timeout",
    [_read_timeout, lambda: TimeoutError("content provider timed out")],
    ids=["httpx-read-timeout", "timeout-error"],
)
def test_timeouts_are_retried_once_after_recovery(timeout):
    events = []
    fetch = _fetch(*(timeout() for _ in range(7)), events=events)

    with pytest.raises(Exception, match="after 2 attempts") as info:
        _run(fetch, events)

    assert events == ["fetch", "recover", "fetch"]
    assert isinstance(info.value.__cause__, (TimeoutError, httpx.ReadTimeout))


def test_timeout_then_success_returns_state():
    events = []
    assert _run(_fetch(_read_timeout(), STATE, events=events), events) is STATE
    assert events == ["fetch", "recover", "fetch"]


def test_timeouts_count_across_other_errors():
    events = []
    other = {"error": "Parse Error"}
    fetch = _fetch(_read_timeout(), other, _read_timeout(), STATE, events=events)

    with pytest.raises(Exception, match="after 3 attempts"):
        _run(fetch, events)

    assert events == ["fetch", "recover", "fetch", "fetch"]


def test_other_errors_keep_full_schedule_and_late_recovery():
    events = []
    other = {"error": "Parse Error"}

    with pytest.raises(Exception, match="after 7 attempts"):
        _run(_fetch(*(other for _ in range(7)), events=events), events)

    assert events == ["fetch"] * 5 + ["recover"] + ["fetch"] * 2


def test_connect_timeout_is_not_a_busy_device():
    # The request never reached the device, so nothing queues behind it.
    events = []
    connect = httpx.ConnectTimeout("", request=REQUEST)
    fetch = _fetch(connect, connect, connect, STATE, events=events)

    assert _run(fetch, events) is STATE
    assert events == ["fetch"] * 4


def test_worst_case_timeout_waits_one_backoff(monkeypatch):
    slept = []

    async def sleep(delay):
        slept.append(delay)

    monkeypatch.setattr(provider_module.asyncio, "sleep", sleep)
    events = []
    with pytest.raises(Exception, match="after 2 attempts"):
        asyncio.run(
            fetch_state_with_retry(
                fetch=_fetch(*(_read_timeout() for _ in range(7)), events=events)
            )
        )
    assert slept == [1.0]


def _slow_fetch(events, delay, result=STATE):
    async def fetch():
        events.append("fetch")
        await asyncio.sleep(delay)
        if isinstance(result, BaseException):
            raise result
        return result

    return fetch


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _waiting_lines(records):
    return [
        r
        for r in records.records
        if r.levelno == logging.INFO and "Waiting for the UI state" in r.getMessage()
    ]


@pytest.fixture
def mobilerun_info_logs():
    # The mobilerun logger may not propagate (CLI configure_logging), so
    # attach directly instead of relying on caplog.
    logger = logging.getLogger("mobilerun")
    handler = _Records()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def test_slow_fetch_logs_waiting_once(monkeypatch, mobilerun_info_logs):
    monkeypatch.setattr(provider_module, "_SLOW_FETCH_NOTICE_S", 0.01)
    events = []
    result = asyncio.run(fetch_state_with_retry(fetch=_slow_fetch(events, 0.05)))

    assert result is STATE
    assert len(_waiting_lines(mobilerun_info_logs)) == 1


def test_fast_fetch_does_not_log_waiting(monkeypatch, mobilerun_info_logs):
    monkeypatch.setattr(provider_module, "_SLOW_FETCH_NOTICE_S", 1.0)
    events = []
    asyncio.run(fetch_state_with_retry(fetch=_slow_fetch(events, 0)))

    assert _waiting_lines(mobilerun_info_logs) == []


def test_waiting_is_not_repeated_on_retry(monkeypatch, mobilerun_info_logs):
    monkeypatch.setattr(provider_module, "_SLOW_FETCH_NOTICE_S", 0.01)
    events = []
    with pytest.raises(Exception, match="after 2 attempts"):
        asyncio.run(
            fetch_state_with_retry(
                fetch=_slow_fetch(events, 0.05, TimeoutError()), retry_delays=[0]
            )
        )

    assert events == ["fetch", "fetch"]
    assert len(_waiting_lines(mobilerun_info_logs)) == 1


def test_cancelling_the_caller_cancels_the_fetch(monkeypatch):
    monkeypatch.setattr(provider_module, "_SLOW_FETCH_NOTICE_S", 10.0)
    cancelled = []

    async def fetch():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return STATE

    async def run():
        task = asyncio.create_task(fetch_state_with_retry(fetch=fetch))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)

    asyncio.run(run())
    assert cancelled == [True]


def test_device_ui_shows_info_progress_on_stderr(capsys):
    from mobilerun.cli import device_commands

    logger = logging.getLogger("mobilerun")
    saved = (logger.handlers[:], logger.level, logger.propagate)
    logger.handlers = []
    try:
        device_commands._log_progress_to_stderr()
        logger.info("Waiting for the UI state from the device...")
        out, err = capsys.readouterr()
        assert out == ""
        assert err == "Waiting for the UI state from the device...\n"

        # An already-configured logger (SDK / agent CLI) is left alone.
        handlers = logger.handlers[:]
        device_commands._log_progress_to_stderr()
        assert logger.handlers == handlers
    finally:
        logger.handlers, logger.level, logger.propagate = saved
