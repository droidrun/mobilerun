"""Device state providers, observation coordinates and tap safety."""

import asyncio
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

import mobilerun.core.state.android as android_device
import mobilerun.core.state.ios as ios_device
import mobilerun.core.state.screenshot as visual_remote_device
from mobilerun.clients.portal.state_fetch import fetch_state_with_retry
from mobilerun.config.settings import get_settings
from mobilerun.core.devices import (
    AndroidDevice,
    CloudDevice,
    IOSDevice,
    RecordingDevice,
    RemoteDevice,
    StealthDevice,
)
from mobilerun.core.state.android import AndroidStateProvider
from mobilerun.core.state.base import (
    StateProvider,
    _resize_model_screenshot_with_grid,
    _should_resize_model_screenshot,
)
from mobilerun.core.state.filters import DetailedFilter, get_filter
from mobilerun.core.state.formatters import IndexedFormatter
from mobilerun.core.state.ios import IOSStateProvider
from mobilerun.core.state.options import StateProviderOptions
from mobilerun.core.state.screenshot import (
    ScreenshotOnlyStateProvider,
)
from mobilerun.core.state.stealth import StealthUIState
from mobilerun.errors import StateFetchFailed, UnsupportedStateCapability
from mobilerun.utils.geometry import find_uncovered_point
from mobilerun.utils.images import (
    image_dimensions,
    resize_image_to_max_side_with_grid,
)


def test_platform_defaults_are_lazy_and_use_expected_provider(monkeypatch):
    made = []

    def android(driver, **kwargs):
        made.append(("android", kwargs))
        return AndroidStateProvider.__new__(AndroidStateProvider)

    def ios(driver, **kwargs):
        made.append(("ios", kwargs))
        return IOSStateProvider.__new__(IOSStateProvider)

    def screenshot(driver, **kwargs):
        made.append(("screenshot", kwargs))
        return ScreenshotOnlyStateProvider.__new__(ScreenshotOnlyStateProvider)

    monkeypatch.setattr(android_device, "AndroidStateProvider", android)
    monkeypatch.setattr(ios_device, "IOSStateProvider", ios)
    monkeypatch.setattr(visual_remote_device, "ScreenshotOnlyStateProvider", screenshot)
    devices = [
        AndroidDevice("a"),
        CloudDevice("c", api_key="fake"),
        IOSDevice("http://i"),
        RemoteDevice(url="http://v", device_id="v"),
    ]
    assert made == []
    from mobilerun_core_local.driver.ios import IOSPortalDriver

    devices[2]._driver = IOSPortalDriver("http://i")
    [device.state._resolved_provider for device in devices]
    assert [kind for kind, _ in made] == [
        "android",
        "android",
        "ios",
        "screenshot",
    ]
    assert made[0][1]["tree_filter"].__class__ is DetailedFilter
    assert made[0][1]["tree_formatter"].__class__ is IndexedFormatter


def test_provider_override_and_reset_to_default(monkeypatch):
    device = AndroidDevice("a")
    override = object()
    device._set_state_provider(override)
    assert device.state._resolved_provider is override
    device._set_state_provider(None)
    assert isinstance(device.state._resolved_provider, AndroidStateProvider)


def test_use_default_provider_passes_options(monkeypatch):
    args = []
    monkeypatch.setattr(
        android_device,
        "AndroidStateProvider",
        lambda d, **kw: args.append(kw) or object(),
    )
    device = AndroidDevice("a")
    device._reset_state_provider(StateProviderOptions(vision=False, normalized=True))
    _ = device.state._resolved_provider
    assert args[0]["vision_enabled"] is False
    assert args[0]["use_normalized"] is True


def test_wrappers_build_default_provider_against_outer_device():
    for wrapper_type in (RecordingDevice, StealthDevice):
        inner = AndroidDevice("a")
        provider = inner.state._resolved_provider
        wrapped = wrapper_type(inner)
        assert wrapped.state._resolved_provider is not provider
        assert wrapped.state._resolved_provider._device is wrapped


class Snapshot:
    elements = ["first", "second"]
    formatted_text = "indexed UI"

    def get_element_coords(self, index):
        return (index * 10, index * 20)

    def convert_point(self, x, y):
        return x + 1, y + 1


class FakeProvider:
    capabilities = {"element_index"}

    def __init__(self, device):
        self.device = device
        self.reads = 0

    async def get_state(self):
        self.reads += 1
        return Snapshot()

    async def get_final_state(self):
        return Snapshot()


class FullCapabilityProvider(FakeProvider):
    capabilities = {"element_index", "convert_point"}


DEVICE_CASES = [
    (AndroidDevice, ("serial-1",), {"use_tcp": True}),
    (
        CloudDevice,
        ("device-1",),
        {
            "display_id": 2,
            "api_key": "fake",
            "base_url": get_settings().MOBILERUN_CLOUD_BASE_URL,
        },
    ),
    (RemoteDevice, (), {"url": "http://visual-device", "device_id": "v1"}),
]


def patch_driver_init(monkeypatch, device_class):

    driver_class = device_class._backend
    calls = []

    def fake_init(self, *args, **kwargs):
        calls.append((args, kwargs.copy()))
        self._driver_args = args
        self._driver_kwargs = kwargs.copy()
        self.tapped = None
        self._client = None
        self._write_client = None
        self._serial = None

    async def fake_tap(self, x, y):
        self.tapped = (x, y)

    async def fake_ensure_connected(self):
        pass

    monkeypatch.setattr(driver_class, "__init__", fake_init)
    monkeypatch.setattr(driver_class, "tap", fake_tap, raising=False)
    monkeypatch.setattr(driver_class, "ensure_connected", fake_ensure_connected)
    return calls


@pytest.mark.parametrize("device_class,args,kwargs", DEVICE_CASES)
@pytest.mark.parametrize("provider_kind", ["class", "factory", "instance"])
def test_each_device_accepts_provider_without_changing_driver_args(
    monkeypatch, device_class, args, kwargs, provider_kind
):
    calls = patch_driver_init(monkeypatch, device_class)
    expected_args = (args, kwargs)
    if provider_kind == "class":
        provider = FakeProvider
    elif provider_kind == "factory":

        def provider(device):
            return FakeProvider(device)

    else:
        provider = FakeProvider(None)

    device = device_class(*args, **kwargs)
    if provider_kind == "instance":
        device._set_state_provider(provider)
    else:
        device._set_state_provider_factory(provider)

    assert calls == [expected_args]
    assert device._driver._driver_args == args
    assert device._driver._driver_kwargs == kwargs
    assert isinstance(device.state._resolved_provider, FakeProvider)
    if provider_kind == "instance":
        assert device.state._resolved_provider is provider
    else:
        assert device.state._resolved_provider.device is device


def test_refresh_caches_returned_snapshot_and_passthrough():
    async def check():
        device = AndroidDevice("serial")
        device._set_state_provider_factory(FakeProvider)
        snapshot = await device.state.refresh()
        assert device.state.current is snapshot
        assert device.state.current.formatted_text == "indexed UI"
        assert device.state.current.elements == ["first", "second"]
        assert device.state.current.get_element_coords(2) == (20, 40)
        assert device.state._resolved_provider.reads == 1

    asyncio.run(check())


def test_unsupported_capability_names_provider_and_capability():
    device = AndroidDevice("serial")
    device._set_state_provider_factory(FakeProvider)
    with pytest.raises(UnsupportedStateCapability, match="FakeProvider.*convert_point"):
        device.state._require("convert_point")


def test_device_without_default_has_clear_error():
    device = AndroidDevice("serial")

    def unavailable():
        from mobilerun.errors import DeviceNotReady

        raise DeviceNotReady("no state provider attached and no default")

    device._ui_state_view._default_provider = unavailable
    with pytest.raises(RuntimeError, match="no state provider attached and no default"):
        _ = device.state._resolved_provider


def test_attach_replaces_provider_and_clears_cached_state():
    async def check():
        device = AndroidDevice("serial")
        device._set_state_provider_factory(FakeProvider)
        old_state = await device.state.refresh()
        replacement = FullCapabilityProvider(device)
        device._set_state_provider(replacement)
        assert device.state._resolved_provider is replacement
        with pytest.raises(RuntimeError, match="not been refreshed"):
            _ = device.state.current
        assert await device.state.refresh() is not old_state

    asyncio.run(check())


def test_tap_element_uses_indexed_coordinates(monkeypatch):
    async def check():
        patch_driver_init(monkeypatch, AndroidDevice)
        device = AndroidDevice("serial")
        device._set_state_provider_factory(FakeProvider)
        await device.state.refresh()
        await device.tap_element(3)
        assert device._driver.tapped == (30, 60)

    asyncio.run(check())


def test_explicit_view_methods_reject_unsupported_capabilities():
    async def check():
        device = AndroidDevice("serial")
        device._set_state_provider_factory(FakeProvider)
        await device.state.refresh()
        names = dir(device.state)
        assert "formatted_text" not in names
        assert "elements" not in names
        assert "get_element_coords" not in names
        assert device.state.current.formatted_text == "indexed UI"
        with pytest.raises(UnsupportedStateCapability):
            device.state._require("convert_point")

    asyncio.run(check())


def test_state_retry_settings_are_read_from_environment(monkeypatch):
    monkeypatch.setenv("MOBILERUN_STATE_RETRY_DELAYS", "[0,0]")
    monkeypatch.setenv("MOBILERUN_STATE_MAX_RETRIES", "2")
    calls = 0

    async def fetch():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary")
        return {"a11y_tree": {}, "phone_state": {}, "device_context": {}}

    result = asyncio.run(fetch_state_with_retry(fetch))

    assert calls == 2
    assert result["a11y_tree"] == {}


def test_retry_exhaustion_raises_state_fetch_failed(monkeypatch):
    monkeypatch.setenv("MOBILERUN_STATE_MAX_RETRIES", "1")
    monkeypatch.setenv("MOBILERUN_STATE_RETRY_DELAYS", "[0]")

    async def fetch():
        raise RuntimeError("offline")

    try:
        asyncio.run(fetch_state_with_retry(fetch))
    except StateFetchFailed as exc:
        assert str(exc) == "Failed to get state after 1 attempts: offline"
    else:
        raise AssertionError("Expected StateFetchFailed")


NATIVE_W, NATIVE_H = 1080, 2400


def test_screenshot_fallback_uses_configured_max_side(monkeypatch):
    monkeypatch.setenv("MOBILERUN_MODEL_SCREENSHOT_MAX_SIDE", "500")
    image = BytesIO()
    Image.new("RGB", (1000, 2000)).save(image, format="PNG")
    geometry = SimpleNamespace(
        model_screenshot_width=None,
        model_screenshot_height=None,
        model_screenshot_grid=False,
    )
    assert image_dimensions(
        _resize_model_screenshot_with_grid(geometry, image.getvalue())
    ) == (
        250,
        500,
    )


def _combined_data(width=NATIVE_W, height=NATIVE_H):
    return {
        "a11y_tree": {
            "boundsInScreen": {"left": 0, "top": 0, "right": width, "bottom": height},
            "className": "android.widget.FrameLayout",
            "text": "root",
            "children": [
                {
                    "boundsInScreen": {
                        "left": 221,
                        "top": 409,
                        "right": 600,
                        "bottom": 460,
                    },
                    "className": "android.widget.TextView",
                    "text": "Services & preferences",
                    "children": [],
                }
            ],
        },
        "device_context": {"screen_bounds": {"width": width, "height": height}},
        "phone_state": {
            "currentApp": "Settings",
            "packageName": "com.android.settings",
        },
    }


def _provider(width=NATIVE_W, height=NATIVE_H, **kwargs):
    driver = SimpleNamespace(
        _state_recovery=None,
        _get_ui_tree=AsyncMock(return_value=_combined_data(width, height)),
    )
    return AndroidStateProvider(
        driver,
        tree_filter=SimpleNamespace(filter=lambda tree, ctx: tree),
        tree_formatter=IndexedFormatter(),
        **kwargs,
    )


def _get_state(provider):
    return asyncio.run(provider.get_state())


def test_resize_flags_pair_providers_with_agent_gating():
    # The providers declare the flag…
    assert StateProvider.resize_model_screenshot is False
    assert ScreenshotOnlyStateProvider.resize_model_screenshot is True
    assert _provider(vision_enabled=True).resize_model_screenshot is True
    assert _provider().resize_model_screenshot is False


def test_legacy_injected_screenshot_providers_keep_getting_resized():
    """Injected providers that only implement the pre-existing
    ``requires_coordinate_tools`` contract must still get resized screenshots:
    the tool registry, prompts, and validation all still treat them as
    screenshot-coordinate mode."""

    class LegacyScreenshotProvider(StateProvider):
        requires_coordinate_tools = True

    assert _should_resize_model_screenshot(LegacyScreenshotProvider(SimpleNamespace()))

    class DuckTypedLegacyProvider(StateProvider):
        requires_coordinate_tools = True

    assert _should_resize_model_screenshot(DuckTypedLegacyProvider(SimpleNamespace()))

    # Non-coordinate providers are unaffected
    assert not _should_resize_model_screenshot(StateProvider(SimpleNamespace()))
    assert not _should_resize_model_screenshot(_provider())
    assert not _should_resize_model_screenshot(StateProvider(SimpleNamespace()))

    # And the new contract still activates it
    assert _should_resize_model_screenshot(_provider(vision_enabled=True))
    assert _should_resize_model_screenshot(
        ScreenshotOnlyStateProvider(SimpleNamespace())
    )


def test_provider_scale_matches_what_agents_actually_send():
    """The invariant that PR #355 lacked: the provider's scale must describe
    the image the agents really attach."""
    buf = BytesIO()
    Image.new("RGB", (NATIVE_W, NATIVE_H)).save(buf, format="PNG")
    sent = resize_image_to_max_side_with_grid(buf.getvalue(), 2048)
    sent_w, sent_h = image_dimensions(sent)

    state = _get_state(_provider(vision_enabled=True))
    assert sent_w == round(NATIVE_W / state.coordinate_scale_x)
    assert sent_h == round(NATIVE_H / state.coordinate_scale_y)


POINTS_W, POINTS_H = 440, 956
PIXELS_W, PIXELS_H = 1320, 2868  # 3x physical screenshot

RAW_A11Y = """Attributes: Application, pid: 123, label: 'Settings'
Element subtree:
 ->Application, pid: 123, label: 'Settings'
    Window (Main), {{0.0, 0.0}, {440.0, 956.0}}
      Cell, {{20.0, 200.0}, {400.0, 44.0}}, label: 'General'
      StaticText, {{20.0, 119.7}, {133.0, 40.7}}, label: 'Settings'
"""


def _png(width, height):
    buf = BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


class FakeIOSDriver:
    def __init__(
        self, screenshot_bytes=None, screenshot_error=None, screenshot_plan=None
    ):
        # screenshot_plan: per-call behaviors (bytes or Exception); the last
        # entry repeats. Overrides screenshot_bytes/screenshot_error.
        self._plan = screenshot_plan or (
            [screenshot_error] if screenshot_error else [screenshot_bytes]
        )
        self.screenshot_calls = 0
        self.ui_tree_error = None

    async def _get_ui_tree(self):
        if self.ui_tree_error:
            raise self.ui_tree_error
        return {
            "phone_state": {
                "currentApp": "Settings",
                "packageName": "com.apple.Preferences",
            },
            "device_context": {
                "screen_bounds": {"width": POINTS_W, "height": POINTS_H}
            },
            "a11y_tree": RAW_A11Y,
        }

    async def screenshot(self):
        step = self._plan[min(self.screenshot_calls, len(self._plan) - 1)]
        self.screenshot_calls += 1
        if isinstance(step, Exception):
            raise step
        return step


def _general_element(state):
    return next(e for e in state.elements if e.get("text") == "General")


def test_resize_flag_pairs_with_agent_gating():
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    assert _should_resize_model_screenshot(
        IOSStateProvider(driver, vision_enabled=True)
    )
    assert not _should_resize_model_screenshot(IOSStateProvider(driver))
    assert not _should_resize_model_screenshot(
        IOSStateProvider(driver, use_normalized=True, vision_enabled=True)
    )


def test_failed_probe_recovers_and_snapshots_keep_their_contract():
    driver = FakeIOSDriver(
        screenshot_plan=[RuntimeError("probe failed"), _png(PIXELS_W, PIXELS_H)]
    )
    provider = IOSStateProvider(driver, vision_enabled=True)

    fallback = _get_state(provider)
    assert provider.resize_model_screenshot is False
    assert fallback.coordinate_contract_active is False

    good = _get_state(provider)
    assert provider.resize_model_screenshot is True
    assert good.coordinate_contract_active is True
    assert good.convert_point(471, 1024) == (220, 478)
    assert fallback.coordinate_contract_active is False

    provider.resize_model_screenshot = False
    assert good.coordinate_contract_active is True
    assert good.convert_point(471, 1024) == (220, 478)
    provider.resize_model_screenshot = True
    assert fallback.coordinate_contract_active is False


def test_tree_failure_returns_inactive_contract():
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, vision_enabled=True)
    driver.ui_tree_error = RuntimeError("tree fetch failed")

    state = _get_state(provider)
    assert state.coordinate_contract_active is False


def test_non_vision_ios_does_not_require_contract():
    provider = IOSStateProvider(FakeIOSDriver(screenshot_bytes=_png(8, 8)))
    state = _get_state(provider)

    assert provider.requires_active_contract_for_coords is False
    assert state.convert_point(100, 200) == (100, 200)


def test_normalized_mode_shows_normalized_bounds_and_taps_points():
    driver = FakeIOSDriver(screenshot_bytes=_png(8, 8))
    provider = IOSStateProvider(driver, use_normalized=True)
    state = _get_state(provider)

    element = _general_element(state)
    assert element["bounds"] == "20,200,420,244"
    assert element["displayBounds"] == "45,209,954,255"
    assert "(normalized [0-1000])" in state.formatted_text
    assert "(45,209,954,255)" in state.formatted_text
    assert "(20,200,420,244)" not in state.formatted_text
    # element taps keep points
    assert state.get_element_coords(element["index"]) == (220, 222)
    assert driver.screenshot_calls == 0


def _bounds(left, top, right, bottom):
    return {"left": left, "top": top, "right": right, "bottom": bottom}


@pytest.mark.parametrize(
    "name,options,expected_indices,partial_bounds",
    [
        ("concise", {}, [1, 3, 4, 6], (-90, 10, 10, 20)),
        ("detailed", {}, [1, 2, 3, 6], (-90, 10, 10, 20)),
        ("detailed", {"clip_bounds": True}, [1, 2, 3, 6], (0, 10, 10, 20)),
        ("detailed", {"filter_keyboard": False}, [1, 2, 3, 4, 6], (-90, 10, 10, 20)),
    ],
    ids=["concise", "detailed", "clipped", "keyboard-retained"],
)
def test_filter_mixed_observation_preserves_indices_bounds_and_input(
    name, options, expected_indices, partial_bounds
):
    def node(index, bounds, **extra):
        return {"index": index, "boundsInScreen": _bounds(*bounds), **extra}

    tree = node(
        0,
        (0, 0, 100, 100),
        children=[
            node(1, (10, 10, 20, 20)),
            node(2, (20, 20, 25, 30)),  # Width exactly at concise minimum.
            node(3, (-90, 10, 10, 20)),  # Exactly 10% visible.
            node(
                4,
                (10, 40, 60, 90),
                resourceId="com.google.android.inputmethod.latin:id/keyboard",
                children=[node(5, (12, 44, 20, 52))],
            ),
            node(6, (70, 70, 80, 80)),
            node(7, (100, 10, 120, 20)),  # Touches the edge, but entirely off-screen.
        ],
    )
    original = deepcopy(tree)
    context = {
        "screen_bounds": {"width": 100, "height": 100},
        "filtering_params": {"min_element_size": 5},
    }
    filtered = get_filter(name, **options).filter(tree, context)
    assert filtered["index"] == 0
    assert [child["index"] for child in filtered["children"]] == expected_indices
    for child in filtered["children"]:
        index = child["index"]
        source = next(item for item in original["children"] if item["index"] == index)
        expected = _bounds(*partial_bounds) if index == 3 else source["boundsInScreen"]
        assert child["boundsInScreen"] == expected
        if index == 4:
            assert [key["index"] for key in child["children"]] == [5]
    assert tree == original


def test_detailed_filter_keeps_exact_visibility_threshold_but_not_just_below():
    root = {
        "boundsInScreen": _bounds(0, 0, 100, 100),
        "children": [
            {"index": 1, "boundsInScreen": _bounds(-10, 0, 10, 20)},
            {"index": 2, "boundsInScreen": _bounds(-11, 0, 9, 20)},
        ],
    }
    filtered = DetailedFilter(visibility_threshold=0.5).filter(
        root, {"screen_bounds": {"width": 100, "height": 100}}
    )
    assert [child["index"] for child in filtered["children"]] == [1]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Clipping rewrites children on a parent without bounds",
)
def test_clipping_a_boundless_parent_preserves_the_original_child_bounds():
    tree = {"children": [{"index": 1, "boundsInScreen": _bounds(-5, 0, 10, 10)}]}
    result = DetailedFilter(clip_bounds=True).filter(
        tree, {"screen_bounds": {"width": 20, "height": 20}}
    )
    assert result["children"][0]["boundsInScreen"] == _bounds(0, 0, 10, 10)
    assert tree["children"][0]["boundsInScreen"] == _bounds(-5, 0, 10, 10)


def _contains(bounds, point):
    left, top, right, bottom = bounds
    x, y = point
    return left <= x < right and top <= y < bottom


@pytest.mark.parametrize(
    "blockers,fully_covered",
    [
        ([(0, 0, 7, 7), (3, 3, 10, 10)], False),
        ([(0, 0, 4, 10), (5, 0, 10, 10)], False),
        ([(0, 0, 6, 10), (4, 0, 10, 10)], True),
    ],
    ids=["overlap", "one-pixel-strip", "total-occlusion"],
)
def test_uncovered_point_avoids_every_blocker(blockers, fully_covered):
    target = (0, 0, 10, 10)
    point = find_uncovered_point(target, blockers)
    if fully_covered:
        assert point is None
    else:
        assert point is not None and _contains(target, point)
        assert all(not _contains(blocker, point) for blocker in blockers)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Stealth jitter does not recheck sibling bounds",
)
def test_stealth_clear_point_stays_outside_sibling_after_jitter(monkeypatch):
    target = (0, 0, 100, 100)
    sibling = (55, 0, 100, 100)
    state = StealthUIState(
        elements=[
            {"index": i, "bounds": ",".join(map(str, bounds))}
            for i, bounds in enumerate([target, sibling])
        ],
        formatted_text="",
        focused_text="",
        phone_state={},
        screen_width=100,
        screen_height=100,
    )
    monkeypatch.setattr(
        "mobilerun.core.state.stealth.random.randint", lambda low, high: high
    )
    point = state.get_clear_point(0)
    assert _contains(target, point)
    assert not _contains(sibling, point)
