import asyncio
from types import SimpleNamespace

from mobilerun.agent.utils import actions

_DEFAULT_LLM = object()


class FakeDriver:
    def __init__(self):
        self.get_apps_calls = 0
        self.started = []

    async def get_apps(self, include_system=False):
        self.get_apps_calls += 1
        assert include_system
        return [
            {"label": "Google Chrome", "package_name": "com.android.chrome"},
            {"label": "Settings", "package": "com.android.settings"},
            {"label": "Files", "package": "com.example.files"},
            {"label": "Files", "package": "com.vendor.files"},
        ]

    async def start_app(self, package):
        self.started.append(package)
        return f"Started {package}"


def _context(driver, llm=_DEFAULT_LLM):
    return SimpleNamespace(
        driver=driver,
        installed_apps_cache=None,
        app_opener_llm=llm,
        streaming=False,
        macro_recorder=None,
    )


def test_exact_package_launch_skips_app_starter(monkeypatch):
    driver = FakeDriver()
    ctx = _context(driver, llm=None)

    async def no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(actions.asyncio, "sleep", no_sleep)

    result = asyncio.run(actions.open_app("com.android.chrome", ctx=ctx))

    assert result.success
    assert driver.started == ["com.android.chrome"]
    assert driver.get_apps_calls == 1


def test_normalized_exact_label_launches_deterministically(monkeypatch):
    driver = FakeDriver()
    ctx = _context(driver)

    async def no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(actions.asyncio, "sleep", no_sleep)

    result = asyncio.run(actions.open_app("  GOOGLE--chrome ", ctx=ctx))

    assert result.success
    assert driver.started == ["com.android.chrome"]


def test_ambiguous_label_uses_llm_fallback_and_cached_inventory(monkeypatch):
    driver = FakeDriver()
    ctx = _context(driver)
    runs = []

    class FakeAppStarter:
        def __init__(self, **kwargs):
            pass

        async def run(self, **kwargs):
            runs.append(kwargs)
            return "Could not open app: no installed app matches 'Files'"

    async def no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(actions, "AppStarter", FakeAppStarter)
    monkeypatch.setattr(actions.asyncio, "sleep", no_sleep)

    first = asyncio.run(actions.open_app("Files", ctx=ctx))
    second = asyncio.run(actions.open_app("unknown app", ctx=ctx))

    assert not first.success
    assert not second.success
    assert driver.get_apps_calls == 1
    assert len(runs) == 2
    assert runs[0]["installed_apps"] is ctx.installed_apps_cache


def test_unique_label_supports_package_key():
    apps = [{"label": "Settings", "package": "com.android.settings"}]

    assert actions._resolve_installed_app("settings", apps) == "com.android.settings"
    assert actions._resolve_installed_app("Set", apps) is None
