from mobilerun.config.env import get_secrets, secret_value


def test_secret_names_include_dynamic_cloud_and_trace_keys(monkeypatch):
    monkeypatch.setenv("MOBILERUN_API_KEY", "placeholder")
    assert secret_value(get_secrets().MOBILERUN_API_KEY) == "placeholder"
