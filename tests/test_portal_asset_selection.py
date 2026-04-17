import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_portal_module():
    console_module = types.ModuleType("rich.console")

    class Console:
        def print(self, *args, **kwargs):
            pass

    console_module.Console = Console

    mobilerun_module = types.ModuleType("mobilerun")
    mobilerun_module.__version__ = "0.6.0"

    async_adbutils_module = types.ModuleType("async_adbutils")
    async_adbutils_module.AdbDevice = object
    async_adbutils_module.adb = object()

    requests_module = types.ModuleType("requests")
    requests_module.RequestException = type("RequestException", (Exception,), {})
    requests_module.ConnectionError = type(
        "ConnectionError", (requests_module.RequestException,), {}
    )

    def get(*args, **kwargs):
        return None

    requests_module.get = get

    stubs = {
        "async_adbutils": async_adbutils_module,
        "mobilerun": mobilerun_module,
        "requests": requests_module,
        "rich": types.ModuleType("rich"),
        "rich.console": console_module,
    }
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in stubs}

    try:
        sys.modules.update(stubs)
        path = Path(__file__).resolve().parents[1] / "mobilerun" / "portal.py"
        spec = importlib.util.spec_from_file_location("portal_asset_module", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


portal = _load_portal_module()
_get_release_assets_by_tag = portal._get_release_assets_by_tag
_normalize_download_base = portal._normalize_download_base
_parse_portal_asset_version = portal._parse_portal_asset_version
_resolve_versioned_portal_apk_asset = portal._resolve_versioned_portal_apk_asset
_select_portal_apk_asset = portal._select_portal_apk_asset
get_latest_release_assets = portal.get_latest_release_assets


def _github_asset(name: str) -> dict:
    return {
        "name": name,
        "browser_download_url": (
            f"https://example.com/releases/download/v0.0.0/{name}"
        ),
    }


class FakeResponse:
    def __init__(self, status_code: int = 200, json_data: dict | None = None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class PortalAssetSelectionTest(unittest.TestCase):
    def test_latest_release_selects_debug_over_unsigned_release(self):
        asset_url, asset_name, asset_version = _select_portal_apk_asset(
            [
                _github_asset("com.mobilerun.portal-0.7.1-release-unsigned.apk"),
                _github_asset("com.mobilerun.portal-0.7.1-debug.apk"),
            ]
        )

        self.assertEqual(asset_name, "com.mobilerun.portal-0.7.1-debug.apk")
        self.assertTrue(asset_url.endswith("/com.mobilerun.portal-0.7.1-debug.apk"))
        self.assertEqual(asset_version, "0.7.1")

    def test_mobilerun_portal_named_asset_matches(self):
        asset_url, asset_name, asset_version = _select_portal_apk_asset(
            [_github_asset("mobilerun-portal-v0.7.1-release.apk")]
        )

        self.assertEqual(asset_name, "mobilerun-portal-v0.7.1-release.apk")
        self.assertTrue(asset_url.endswith("/mobilerun-portal-v0.7.1-release.apk"))
        self.assertEqual(asset_version, "0.7.1")

    def test_internal_repo_named_asset_matches(self):
        asset_url, asset_name, asset_version = _select_portal_apk_asset(
            [_github_asset("mobilerun-portal-internal-v0.7.1-debug.apk")]
        )

        self.assertEqual(asset_name, "mobilerun-portal-internal-v0.7.1-debug.apk")
        self.assertTrue(
            asset_url.endswith("/mobilerun-portal-internal-v0.7.1-debug.apk")
        )
        self.assertEqual(asset_version, "0.7.1")

    def test_legacy_droidrun_portal_asset_is_not_active_match(self):
        with self.assertRaisesRegex(Exception, "Portal APK asset not found"):
            _select_portal_apk_asset([_github_asset("droidrun-portal-v0.4.6.apk")])

    def test_ungh_download_url_asset_matches_without_name(self):
        download_url = (
            "https://example.com/releases/download/v0.7.1/"
            "mobilerun-portal-v0.7.1.apk"
        )

        asset_url, asset_name, asset_version = _select_portal_apk_asset(
            [{"downloadUrl": download_url}]
        )

        self.assertEqual(asset_url, download_url)
        self.assertEqual(asset_name, "mobilerun-portal-v0.7.1.apk")
        self.assertEqual(asset_version, "0.7.1")

    def test_no_matching_apk_reports_seen_assets(self):
        with self.assertRaises(Exception) as ctx:
            _select_portal_apk_asset(
                [
                    _github_asset("release-notes.txt"),
                    _github_asset("other-app-1.0.0.apk"),
                ]
            )

        message = str(ctx.exception)
        self.assertIn("Portal APK asset not found", message)
        self.assertIn("release-notes.txt", message)
        self.assertIn("other-app-1.0.0.apk", message)

    def test_tag_release_lookup_retries_next_host_on_request_error(self):
        original_get = portal.requests.get
        calls = []

        def get(url):
            calls.append(url)
            if "api.github.com" in url:
                raise portal.requests.ConnectionError("dns failure")
            return FakeResponse(
                json_data={
                    "release": {
                        "assets": [
                            _github_asset("com.mobilerun.portal-0.7.1-debug.apk")
                        ]
                    }
                }
            )

        portal.requests.get = get

        try:
            assets = _get_release_assets_by_tag("v0.7.1")
        finally:
            portal.requests.get = original_get

        self.assertEqual(len(calls), 2)
        self.assertIn("api.github.com", calls[0])
        self.assertIn("ungh.cc", calls[1])
        self.assertEqual(assets[0]["name"], "com.mobilerun.portal-0.7.1-debug.apk")

    def test_latest_release_lookup_retries_next_host_on_request_error(self):
        original_get = portal.requests.get
        calls = []

        def get(url):
            calls.append(url)
            if "api.github.com" in url:
                raise portal.requests.ConnectionError("tls failure")
            return FakeResponse(
                json_data={
                    "assets": [_github_asset("com.mobilerun.portal-0.7.1-debug.apk")]
                }
            )

        portal.requests.get = get

        try:
            assets = get_latest_release_assets()
        finally:
            portal.requests.get = original_get

        self.assertEqual(len(calls), 2)
        self.assertIn("api.github.com", calls[0])
        self.assertIn("ungh.cc", calls[1])
        self.assertEqual(assets[0]["name"], "com.mobilerun.portal-0.7.1-debug.apk")

    def test_release_lookup_raises_clear_error_when_all_hosts_fail(self):
        original_get = portal.requests.get

        def get(url):
            raise portal.requests.ConnectionError("offline")

        portal.requests.get = get

        try:
            with self.assertRaisesRegex(Exception, "all configured hosts"):
                get_latest_release_assets()
        finally:
            portal.requests.get = original_get

    def test_versioned_resolution_uses_release_tag_assets(self):
        original = portal._get_release_assets_by_tag

        def release_assets(version, debug=False):
            return [
                _github_asset("com.mobilerun.portal-0.7.1-release-unsigned.apk"),
                _github_asset("com.mobilerun.portal-0.7.1-debug.apk"),
            ]

        portal._get_release_assets_by_tag = release_assets

        try:
            asset_url, asset_version, asset_name = _resolve_versioned_portal_apk_asset(
                "0.7.1",
                "https://github.com/droidrun/mobilerun-portal/releases/download",
            )
        finally:
            portal._get_release_assets_by_tag = original

        self.assertEqual(asset_name, "com.mobilerun.portal-0.7.1-debug.apk")
        self.assertTrue(asset_url.endswith("/com.mobilerun.portal-0.7.1-debug.apk"))
        self.assertEqual(asset_version, "0.7.1")

    def test_versioned_resolution_normalizes_stale_download_base_fallback(self):
        original = portal._get_release_assets_by_tag

        def raise_release_error(version, debug=False):
            raise RuntimeError("offline")

        portal._get_release_assets_by_tag = raise_release_error

        try:
            asset_url, asset_version, asset_name = _resolve_versioned_portal_apk_asset(
                "v0.7.1",
                "https://github.com/droidrun/droidrun-portal/releases/download",
            )
        finally:
            portal._get_release_assets_by_tag = original

        self.assertEqual(asset_name, "com.mobilerun.portal-0.7.1-debug.apk")
        self.assertEqual(
            asset_url,
            "https://github.com/droidrun/mobilerun-portal/releases/download/"
            "v0.7.1/com.mobilerun.portal-0.7.1-debug.apk",
        )
        self.assertEqual(asset_version, "0.7.1")
        self.assertEqual(
            _normalize_download_base(
                "https://github.com/droidrun/droidrun-portal/releases/download"
            ),
            "https://github.com/droidrun/mobilerun-portal/releases/download",
        )

    def test_asset_version_parsing_ignores_build_suffixes(self):
        self.assertEqual(
            _parse_portal_asset_version("com.mobilerun.portal-0.7.1-debug.apk"),
            "0.7.1",
        )
        self.assertEqual(
            _parse_portal_asset_version(
                "com.mobilerun.portal-0.7.1-release-unsigned.apk"
            ),
            "0.7.1",
        )
        self.assertEqual(
            _parse_portal_asset_version("mobilerun-portal-v0.7.1-release.apk"),
            "0.7.1",
        )
        self.assertEqual(
            _parse_portal_asset_version("mobilerun-portal-internal-v0.7.1-debug.apk"),
            "0.7.1",
        )


if __name__ == "__main__":
    unittest.main()
