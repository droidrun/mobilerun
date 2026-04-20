import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mobilerun.agent.droid.droid_agent import _effective_disabled_tools
from mobilerun.agent.utils.actions import click_area, click_at, long_press_at, swipe
from mobilerun.agent.utils.signatures import build_tool_registry
from mobilerun.config_manager.config_manager import MobileConfig
from mobilerun.tools.driver.visual_remote import VisualRemoteDriver
from mobilerun.tools.ui.screenshot_provider import ScreenshotOnlyStateProvider
from mobilerun.tools.ui.state import UIState


def _png(width: int = 1320, height: int = 2868) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\r"
        b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
    )


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        json_data=None,
        content: bytes = b"",
    ):
        self.status_code = status_code
        self._json_data = json_data
        self.content = content

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeAsyncClient:
    def __init__(self, *, devices=None, screenshot: bytes | None = None, **kwargs):
        self.devices = devices or []
        self.screenshot_bytes = screenshot or _png()
        self.requests = []
        self.closed = False

    async def get(self, path: str, **kwargs):
        self.requests.append(("GET", path, kwargs))
        if path == "/devices":
            return FakeResponse(json_data={"devices": self.devices})
        if path.endswith("/screenshot"):
            return FakeResponse(content=self.screenshot_bytes)
        return FakeResponse(status_code=404)

    async def post(self, path: str, json=None, **kwargs):
        self.requests.append(("POST", path, json or {}, kwargs))
        return FakeResponse(json_data={"ok": True})

    async def aclose(self):
        self.closed = True


def _device(
    device_id: str = "device-1",
    platform: str = "ios",
    ready: bool = True,
    capabilities: dict | None = None,
) -> dict:
    return {
        "id": device_id,
        "name": "Phone",
        "platform": platform,
        "ready": ready,
        "capabilities": capabilities
        or {
            "screenshot": True,
            "tap": True,
            "swipe": True,
            "type_text": True,
            "press_button": ["enter"],
            "open_app": False,
            "accessibility_tree": False,
        },
    }


class VisualRemoteDriverTest(unittest.TestCase):
    def _connect(self, devices):
        client = FakeAsyncClient(devices=devices)
        with patch(
            "mobilerun.tools.driver.visual_remote.httpx.AsyncClient",
            return_value=client,
        ):
            driver = VisualRemoteDriver("http://localhost:8090")
            asyncio.run(driver.connect())
        return driver, client

    def test_selects_single_ready_device_and_fetches_screenshot(self):
        driver, client = self._connect([_device(platform="android")])

        self.assertEqual(driver.device_id, "device-1")
        self.assertEqual(driver.platform, "Android")
        self.assertIn("tap", driver.supported)
        self.assertIn("input_text", driver.supported)

        screenshot = asyncio.run(driver.screenshot())
        self.assertEqual(screenshot, _png())
        self.assertIn(("GET", "/devices/device-1/screenshot", {}), client.requests)

    def test_multiple_devices_require_device_id(self):
        client = FakeAsyncClient(devices=[_device("a"), _device("b")])
        with patch(
            "mobilerun.tools.driver.visual_remote.httpx.AsyncClient",
            return_value=client,
        ):
            driver = VisualRemoteDriver("http://localhost:8090")
            with self.assertRaisesRegex(ConnectionError, "Multiple visual remote"):
                asyncio.run(driver.connect())

    def test_not_ready_device_fails_clearly(self):
        client = FakeAsyncClient(devices=[_device(ready=False)])
        with patch(
            "mobilerun.tools.driver.visual_remote.httpx.AsyncClient",
            return_value=client,
        ):
            driver = VisualRemoteDriver("http://localhost:8090")
            with self.assertRaisesRegex(ConnectionError, "not ready"):
                asyncio.run(driver.connect())

    def test_actions_send_screenshot_pixel_coordinates_and_screen_size(self):
        driver, client = self._connect([_device()])

        asyncio.run(driver.screenshot())
        asyncio.run(driver.tap(420, 730))
        asyncio.run(driver.swipe(10, 20, 30, 40, duration_ms=500))
        asyncio.run(driver.input_text("hello", clear=True))
        asyncio.run(driver.press_button("enter"))

        post_payloads = [
            request[2] for request in client.requests if request[0] == "POST"
        ]
        self.assertEqual(
            post_payloads[0],
            {
                "action": "tap",
                "x": 420,
                "y": 730,
                "coordinate_space": "screenshot_pixels",
                "screen": {"width": 1320, "height": 2868},
            },
        )
        self.assertEqual(post_payloads[1]["action"], "swipe")
        self.assertEqual(post_payloads[1]["screen"], {"width": 1320, "height": 2868})
        self.assertEqual(
            post_payloads[2],
            {"action": "type_text", "text": "hello", "clear": True},
        )
        self.assertEqual(
            post_payloads[3],
            {"action": "press_button", "button": "enter"},
        )

    def test_open_app_unsupported_returns_failure_message(self):
        driver, client = self._connect([_device()])

        result = asyncio.run(driver.start_app("com.example.app"))

        self.assertIn("does not support app launch", result)
        self.assertFalse([request for request in client.requests if request[0] == "POST"])


class ScreenshotOnlyStateProviderTest(unittest.TestCase):
    def test_state_has_no_elements_and_uses_screenshot_pixel_coordinates(self):
        class FakeDriver:
            async def screenshot(self):
                return _png(1000, 2000)

        provider = ScreenshotOnlyStateProvider(FakeDriver())
        state = asyncio.run(provider.get_state())

        self.assertEqual(state.elements, [])
        self.assertEqual(state.screen_width, 1000)
        self.assertEqual(state.screen_height, 2000)
        self.assertEqual(state.convert_point(500, 500), (500, 500))
        self.assertFalse(state.phone_state["accessibilityTree"])
        self.assertIn("screenshot pixel coordinates", state.formatted_text)
        self.assertIn("1000x2000", state.formatted_text)
        self.assertIn("(0,0) is top-left", state.formatted_text)
        self.assertIn("(999,1999) is bottom-right", state.formatted_text)
        self.assertIn("direct_text_input", provider.supported)
        self.assertNotIn("element_index", provider.supported)

    def test_large_state_uses_model_screenshot_coordinate_space(self):
        class FakeDriver:
            async def screenshot(self):
                return _png(1080, 2316)

        provider = ScreenshotOnlyStateProvider(FakeDriver())
        state = asyncio.run(provider.get_state())

        self.assertEqual(state.screen_width, 955)
        self.assertEqual(state.screen_height, 2048)
        self.assertEqual(state.convert_point(274, 1228), (310, 1389))
        self.assertIn("955x2048", state.formatted_text)
        self.assertIn("(954,2047) is bottom-right", state.formatted_text)

    def test_tool_filter_keeps_coordinate_tools_and_direct_text(self):
        class FakeDriver:
            async def screenshot(self):
                return _png()

        provider = ScreenshotOnlyStateProvider(FakeDriver())

        async def run():
            registry, _ = await build_tool_registry(
                supported_buttons={"enter"},
                platform="ios",
            )
            capabilities = {
                "tap",
                "swipe",
                "input_text",
                "press_button",
                "screenshot",
            } | provider.supported
            registry.disable_unsupported(capabilities)
            registry.disable(
                _effective_disabled_tools(
                    ["click_at", "click_area", "long_press_at"],
                    provider,
                )
            )
            return registry

        registry = asyncio.run(run())

        self.assertIn("click_at", registry.tools)
        self.assertIn("click_area", registry.tools)
        self.assertIn("long_press_at", registry.tools)
        self.assertIn("swipe", registry.tools)
        self.assertIn("type_text", registry.tools)
        self.assertIn("system_button", registry.tools)
        self.assertNotIn("click", registry.tools)
        self.assertNotIn("type", registry.tools)

    def test_coordinate_tools_describe_screenshot_pixel_coordinates(self):
        async def run():
            registry, _ = await build_tool_registry(
                supported_buttons={"enter"},
                platform="ios",
            )
            return registry

        registry = asyncio.run(run())

        self.assertIn("screenshot pixel coordinates", registry.tools["click_at"].description)
        self.assertIn("(0,0) is top-left", registry.tools["click_area"].description)
        self.assertIn(
            "screenshot pixel coordinates",
            registry.tools["long_press_at"].description,
        )
        self.assertIn("bottom-right coordinate is shown", registry.tools["swipe"].description)
        self.assertNotIn("0..1000", registry.tools["click_at"].description)
        self.assertNotIn("0..1000", registry.tools["click_area"].description)
        self.assertNotIn("0..1000", registry.tools["long_press_at"].description)
        self.assertNotIn("0..1000", registry.tools["swipe"].description)


class ScreenshotOnlyCoordinateValidationTest(unittest.TestCase):
    def _ctx(self):
        class FakeDriver:
            def __init__(self):
                self.taps = []
                self.swipes = []

            async def tap(self, x, y):
                self.taps.append((x, y))

            async def swipe(self, x1, y1, x2, y2, duration_ms=1000):
                self.swipes.append((x1, y1, x2, y2, duration_ms))

        driver = FakeDriver()
        ui = UIState(
            elements=[],
            formatted_text="",
            focused_text="",
            phone_state={},
            screen_width=1000,
            screen_height=2000,
            use_normalized=False,
        )
        provider = SimpleNamespace(requires_coordinate_tools=True)
        return SimpleNamespace(driver=driver, ui=ui, state_provider=provider), driver

    def test_click_at_valid_coordinate_converts_and_taps(self):
        ctx, driver = self._ctx()

        result = asyncio.run(click_at(500, 500, ctx=ctx))

        self.assertTrue(result.success)
        self.assertEqual(driver.taps, [(500, 500)])

    def test_click_at_rejects_out_of_range_before_tapping(self):
        ctx, driver = self._ctx()

        result = asyncio.run(click_at(1000, 2000, ctx=ctx))

        self.assertFalse(result.success)
        self.assertIn("Coordinates must be inside the screenshot size 1000x2000", result.summary)
        self.assertEqual(driver.taps, [])

    def test_swipe_rejects_out_of_range_before_swiping(self):
        ctx, driver = self._ctx()

        result = asyncio.run(swipe([500, 2000], [500, 500], ctx=ctx))

        self.assertFalse(result.success)
        self.assertIn("Coordinates must be inside the screenshot size 1000x2000", result.summary)
        self.assertEqual(driver.swipes, [])

    def test_click_area_rejects_out_of_range_corner_before_tapping(self):
        ctx, driver = self._ctx()

        result = asyncio.run(click_area(100, 100, 1000, 300, ctx=ctx))

        self.assertFalse(result.success)
        self.assertIn("Coordinates must be inside the screenshot size 1000x2000", result.summary)
        self.assertEqual(driver.taps, [])

    def test_long_press_at_rejects_out_of_range_before_swiping(self):
        ctx, driver = self._ctx()

        result = asyncio.run(long_press_at(-1, 500, ctx=ctx))

        self.assertFalse(result.success)
        self.assertIn("Coordinates must be inside the screenshot size 1000x2000", result.summary)
        self.assertEqual(driver.swipes, [])


class VisualRemoteConfigTest(unittest.TestCase):
    def test_config_accepts_public_connection_fields(self):
        config = MobileConfig.from_dict(
            {
                "agent": {"vision_only": True},
                "device": {
                    "connection": "visual-remote",
                    "device_id": "phone-1",
                    "serial": "http://localhost:8090",
                },
            }
        )

        self.assertTrue(config.agent.vision_only)
        self.assertEqual(config.device.connection, "visual-remote")
        self.assertEqual(config.device.device_id, "phone-1")


if __name__ == "__main__":
    unittest.main()
