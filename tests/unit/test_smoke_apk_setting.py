from pathlib import Path

import platformdirs

from mobilerun.config.settings import get_settings


def test_smoke_apk_defaults_to_user_cache_and_respects_override(monkeypatch, tmp_path):
    monkeypatch.delenv("MOBILERUN_SMOKE_APK", raising=False)
    expected = Path(platformdirs.user_cache_dir("droidrun")) / "smoke" / "counter.apk"
    assert expected.is_relative_to(tmp_path), "Expected the isolated pytest cache"
    assert get_settings().MOBILERUN_SMOKE_APK == str(expected)

    override = str(tmp_path / "custom.apk")
    monkeypatch.setenv("MOBILERUN_SMOKE_APK", override)
    assert get_settings().MOBILERUN_SMOKE_APK == override
