import asyncio
from types import SimpleNamespace

import pytest

from mobilerun.agent.fast_agent.xml_parser import parse_tool_calls_detailed
from mobilerun.agent.utils import actions
from mobilerun.agent.utils.actions import list_deep_links, open_deep_link
from mobilerun.agent.utils.deep_links import format_deep_links
from mobilerun.agent.utils.signatures import build_tool_registry
from mobilerun.macro.replay import MacroPlayer

MAPS = "com.google.android.apps.maps"


def _filter(data, *, callable=True):
    return {"activity": "x.Link", "callable": callable, "data": data}


class _Driver:
    def __init__(self, links=None, error=None) -> None:
        self.links = links
        self.error = error
        self.calls = []

    async def get_deep_links(self, package):
        self.calls.append(("get_deep_links", package))
        if self.error:
            raise self.error
        return self.links

    async def get_ui_tree(self, filter=True):
        return {"phone_state": {"packageName": MAPS}}

    async def open_deep_link(self, deep_link, package=None, action=None):
        self.calls.append(("open_deep_link", deep_link, package))
        if self.error:
            raise self.error


class _Recorder:
    def __init__(self) -> None:
        self.actions = []

    def record_action(self, action, **kwargs) -> None:
        self.actions.append(action)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def instant(_seconds):
        return None

    monkeypatch.setattr(actions.asyncio, "sleep", instant)


def test_format_combines_data_elements_per_filter():
    summary = format_deep_links(
        {
            "package": MAPS,
            "declarationsComplete": True,
            "filters": [
                _filter(
                    [
                        {"scheme": "https"},
                        {"host": "maps.google.com"},
                        {"pathPrefix": "/maps"},
                        {"path": "/dir"},
                    ]
                ),
                _filter([{"scheme": "geo"}]),
                _filter([{"scheme": "google.navigation", "ssp": "q=home"}]),
                _filter([{"scheme": "app", "host": "h", "port": "8080"}]),
                _filter([{"mimeType": "image/*"}]),
                _filter([{"scheme": "hidden"}], callable=False),
            ],
        }
    )

    assert f"Deep links declared by {MAPS}" in summary
    assert "- https://maps.google.com/maps*, https://maps.google.com/dir" in summary
    assert "- geo:*" in summary
    assert "- google.navigation:q=home" in summary
    assert "- app://h:8080 (any path)" in summary
    assert "image/*" not in summary
    assert "hidden" not in summary


def test_format_follows_android_matching_for_ports_types_and_extra_rules():
    summary = format_deep_links(
        {
            "package": "p",
            "filters": [
                _filter([{"scheme": "https", "port": "8443"}]),
                _filter([{"scheme": "https"}, {"host": "h"}, {"port": "8080"}]),
                _filter(
                    [{"scheme": "http"}, {"scheme": "https"}, {"mimeType": "video/*"}]
                ),
                _filter([{"scheme": "content"}, {"mimeType": "application/pdf"}]),
                {
                    **_filter([{"scheme": "https", "host": "shop.example"}]),
                    "additionalRules": [{"tag": "uri-relative-filter-group"}],
                },
            ],
        }
    )

    assert "- https:*" in summary
    # Android pairs a port only with the host in the same <data> element.
    assert "- https://h (any path)" in summary
    assert "video/*" not in summary
    assert "- content:* [type application/pdf]" in summary
    assert "- https://shop.example (any path) [narrowed by extra rules" in summary


def test_format_groups_large_filters_caps_output_and_flags_partial_results():
    hosts = [{"host": f"h{i}.example"} for i in range(12)]
    filters = [_filter([{"scheme": "https"}, {"scheme": "http"}, *hosts])]
    filters += [_filter([{"scheme": f"s{i}"}]) for i in range(100)]

    summary = format_deep_links(
        {
            "package": "p",
            "filters": filters,
            "declarationsComplete": False,
            "warnings": ["output_limit_exceeded"],
        },
        max_chars=300,
    )

    assert "- scheme https, http; host h0.example" in summary
    assert "(+4 more)" in summary
    assert "... +94 more" in summary
    assert "(Declarations may be incomplete: output_limit_exceeded.)" in summary


@pytest.mark.parametrize(
    "phone_state, package",
    [
        ({"packageName": "com.example.app"}, "com.example.app"),
        ({"observationMode": "screenshot_only"}, MAPS),
    ],
)
def test_list_deep_links_defaults_to_the_foreground_app(phone_state, package):
    driver = _Driver(links={"package": package, "filters": []})
    ui = SimpleNamespace(phone_state=phone_state)

    result = asyncio.run(list_deep_links(ctx=SimpleNamespace(driver=driver, ui=ui)))

    assert result.success
    assert driver.calls == [("get_deep_links", package)]
    assert f"{package} declares no deep links" in result.summary


def test_list_deep_links_reports_driver_errors():
    driver = _Driver(
        error=RuntimeError("Package is not installed or not visible to Portal")
    )

    result = asyncio.run(
        list_deep_links(" com.missing\n", ctx=SimpleNamespace(driver=driver))
    )

    assert result.success is False
    assert driver.calls == [("get_deep_links", "com.missing")]
    assert "com.missing" in result.summary
    assert "not installed" in result.summary


def test_list_deep_links_asks_for_a_package_when_none_is_known():
    class _NoTree(_Driver):
        async def get_ui_tree(self, filter=True):
            raise RuntimeError("no tree")

    driver = _NoTree()
    result = asyncio.run(list_deep_links(ctx=SimpleNamespace(driver=driver, ui=None)))

    assert result.success is False
    assert "pass the app's package name" in result.summary
    assert driver.calls == []


def test_open_deep_link_strips_input_and_records_a_macro_action():
    driver = _Driver()
    recorder = _Recorder()
    ctx = SimpleNamespace(driver=driver, macro_recorder=recorder)

    result = asyncio.run(open_deep_link(" geo:0,0?q=Paris\n", MAPS, ctx=ctx))
    unpinned = asyncio.run(open_deep_link("https://example.com", "", ctx=ctx))

    assert result.success and unpinned.success
    assert driver.calls == [
        ("open_deep_link", "geo:0,0?q=Paris", MAPS),
        ("open_deep_link", "https://example.com", None),
    ]
    assert recorder.actions[0] == {
        "action_type": "open_deep_link",
        "deep_link": "geo:0,0?q=Paris",
        "package": MAPS,
    }


def test_open_deep_link_failure_is_reported_and_not_recorded():
    driver = _Driver(error=RuntimeError("No activity found to handle deep link"))
    recorder = _Recorder()

    result = asyncio.run(
        open_deep_link(
            "app://x", ctx=SimpleNamespace(driver=driver, macro_recorder=recorder)
        )
    )

    assert result.success is False
    assert "No activity found" in result.summary
    assert recorder.actions == []


@pytest.mark.parametrize(
    "capabilities, visible",
    [
        (set(), set()),
        ({"open_deep_link"}, {"open_deep_link"}),
        ({"open_deep_link", "get_deep_links"}, {"open_deep_link", "list_deep_links"}),
    ],
)
def test_tools_follow_driver_capabilities(capabilities, visible):
    async def run():
        registry, standard = await build_tool_registry()
        registry.disable_unsupported(capabilities | {"start_app", "get_apps"})
        return registry, standard

    registry, standard = asyncio.run(run())

    assert {"list_deep_links", "open_deep_link"} & set(registry.tools) == visible
    assert not {"list_deep_links", "open_deep_link"} & standard


@pytest.mark.parametrize("separator", ["&", "&amp;"])
def test_xml_tool_call_opens_a_deep_link(separator):
    async def run():
        registry, _ = await build_tool_registry()
        parsed = parse_tool_calls_detailed(
            "<function_calls>\n"
            '<invoke name="open_deep_link">\n'
            f'<parameter name="deep_link">geo:0,0?q=Paris{separator}z=10</parameter>\n'
            f'<parameter name="package">{MAPS}</parameter>\n'
            "</invoke>\n"
            "</function_calls>",
            registry.get_param_types(),
        )
        call = parsed.calls[0]
        driver = _Driver()
        result = await registry.execute(
            call.name, call.parameters, SimpleNamespace(driver=driver)
        )
        return result, driver

    result, driver = asyncio.run(run())

    assert result.success
    assert driver.calls == [("open_deep_link", "geo:0,0?q=Paris&z=10", MAPS)]


def test_macro_replay_opens_recorded_deep_link():
    driver = _Driver()
    player = MacroPlayer.__new__(MacroPlayer)

    async def initialize():
        return driver

    player._initialize_driver = initialize

    ok = asyncio.run(
        player.replay_action(
            {
                "action_type": "open_deep_link",
                "deep_link": "geo:0,0?q=Paris",
                "package": None,
            }
        )
    )

    assert ok is True
    assert driver.calls == [("open_deep_link", "geo:0,0?q=Paris", None)]
