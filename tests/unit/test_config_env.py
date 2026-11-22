def test_base_secrets_cover_cloud_environment(monkeypatch):
    from mobilerun.config.env import get_secrets

    monkeypatch.setenv("MOBILERUN_API_KEY", "secret-placeholder-one")
    monkeypatch.setenv("MOBILERUN_DEVICE_TOKEN", "two")
    snapshot = get_secrets()
    assert snapshot.MOBILERUN_API_KEY.get_secret_value() == "secret-placeholder-one"
    assert snapshot.MOBILERUN_DEVICE_TOKEN.get_secret_value() == "two"
    assert "secret-placeholder-one" not in repr(snapshot)
    monkeypatch.setenv("MOBILERUN_API_KEY", "secret-placeholder-changed")
    assert (
        get_secrets().MOBILERUN_API_KEY.get_secret_value()
        == "secret-placeholder-changed"
    )


def test_sdk_paths_read_at_call_time(monkeypatch):
    from mobilerun.config.settings import get_settings

    monkeypatch.setenv("ANDROID_HOME", "/custom/sdk")
    assert str(get_settings().ANDROID_HOME) == "/custom/sdk"
