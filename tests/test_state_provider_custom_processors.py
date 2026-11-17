import asyncio
import threading
from types import SimpleNamespace

import pytest

from mobilerun.tools.filters import ConciseFilter, DetailedFilter, TreeFilter
from mobilerun.tools.formatters import IndexedFormatter, TreeFormatter
from mobilerun.tools.ui.provider import AndroidStateProvider
from mobilerun.tools.ui.state import UIState

RAW = {
    "a11y_tree": {
        "text": "Wi-Fi",
        "boundsInScreen": {"left": 0, "top": 0, "right": 100, "bottom": 100},
    },
    "device_context": {"screen_bounds": {"width": 1080, "height": 2400}},
    "phone_state": {"packageName": "com.android.settings"},
}


class CustomFilter(TreeFilter):
    def get_name(self):
        return "custom"

    def filter(self, a11y_tree, device_context):
        return None


class LockFilter(CustomFilter):
    def __init__(self):
        self.lock = threading.Lock()

    def filter(self, a11y_tree, device_context):
        with self.lock:
            return None


class CountingFilter(CustomFilter):
    def __init__(self):
        self.calls = 0

    def filter(self, a11y_tree, device_context):
        self.calls += 1
        return None


class LoopFilter(CustomFilter):
    def filter(self, a11y_tree, device_context):
        self.loop = asyncio.get_running_loop()
        return None


class CustomFormatter(TreeFormatter):
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0

    def format(self, filtered_tree, phone_state):
        with self.lock:
            self.calls += 1
            self.loop = asyncio.get_running_loop()
        return "custom text", "", [], phone_state


def _provider(tree_filter, tree_formatter):
    async def get_ui_tree():
        return RAW

    return AndroidStateProvider(
        SimpleNamespace(get_ui_tree=get_ui_tree), tree_filter, tree_formatter
    )


def _observe(tree_filter, tree_formatter, method="get_state"):
    provider = _provider(tree_filter, tree_formatter)
    return asyncio.run(getattr(provider, method)())


@pytest.mark.parametrize("method", ["get_state", "get_final_state"])
def test_filter_holding_a_lock_returns_a_state(method):
    state = _observe(LockFilter(), IndexedFormatter(), method)

    assert isinstance(state, UIState)
    assert state.screen_width == 1080


def test_filter_counter_increases_on_the_callers_object():
    tree_filter = CountingFilter()
    provider = _provider(tree_filter, IndexedFormatter())

    async def observe_twice():
        await provider.get_state()
        await provider.get_state()

    asyncio.run(observe_twice())

    assert tree_filter.calls == 2


def test_filter_runs_on_the_event_loop():
    tree_filter = LoopFilter()
    provider = _provider(tree_filter, IndexedFormatter())

    async def observe():
        await provider.get_state()
        return asyncio.get_running_loop()

    loop = asyncio.run(observe())

    assert tree_filter.loop is loop


def test_formatter_is_called_in_place_on_the_event_loop():
    formatter = CustomFormatter()
    provider = _provider(ConciseFilter(), formatter)

    async def observe():
        state = await provider.get_state()
        return state, asyncio.get_running_loop()

    state, loop = asyncio.run(observe())

    assert state.formatted_text == "custom text"
    assert formatter.calls == 1
    assert formatter.loop is loop
    assert formatter.screen_width == 1080


@pytest.mark.parametrize("base", [ConciseFilter, DetailedFilter])
def test_builtin_filter_subclass_is_called_in_place_on_the_event_loop(base):
    class AuditFilter(base):
        def __init__(self):
            super().__init__()
            self.lock = threading.Lock()
            self.calls = 0

        def filter(self, a11y_tree, device_context):
            with self.lock:
                self.calls += 1
                self.loop = asyncio.get_running_loop()
            return super().filter(a11y_tree, device_context)

    tree_filter = AuditFilter()
    provider = _provider(tree_filter, IndexedFormatter())

    async def observe():
        state = await provider.get_state()
        return state, asyncio.get_running_loop()

    state, loop = asyncio.run(observe())

    assert [element["text"] for element in state.elements] == ["Wi-Fi"]
    assert tree_filter.calls == 1
    assert tree_filter.loop is loop


def test_builtin_formatter_subclass_is_called_in_place_on_the_event_loop():
    class AuditFormatter(IndexedFormatter):
        def __init__(self):
            super().__init__()
            self.lock = threading.Lock()
            self.audit = []

        def format(self, filtered_tree, phone_state):
            with self.lock:
                self.audit.append(asyncio.get_running_loop())
            return super().format(filtered_tree, phone_state)

    formatter = AuditFormatter()
    provider = _provider(DetailedFilter(), formatter)

    async def observe():
        state = await provider.get_state()
        return state, asyncio.get_running_loop()

    state, loop = asyncio.run(observe())

    assert [element["text"] for element in state.elements] == ["Wi-Fi"]
    assert formatter.audit == [loop]


@pytest.mark.parametrize("filter_class", [ConciseFilter, DetailedFilter])
def test_builtin_pair_is_processed_off_the_loop_on_copies(monkeypatch, filter_class):
    tree_filter, formatter = filter_class(), IndexedFormatter()
    provider = _provider(tree_filter, formatter)
    seen = []
    original = AndroidStateProvider._state_from_data

    def process(self, raw):
        seen.append(
            (threading.get_ident(), self, self.tree_filter, self.tree_formatter)
        )
        return original(self, raw)

    monkeypatch.setattr(AndroidStateProvider, "_state_from_data", process)
    state = asyncio.run(provider.get_state())

    [(thread, processor, used_filter, used_formatter)] = seen
    assert thread != threading.get_ident()
    assert processor is not provider
    assert used_filter is not tree_filter
    assert used_formatter is not formatter
    assert formatter.screen_width == state.screen_width == 1080
