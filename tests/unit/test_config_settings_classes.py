"""Environment snapshots preserve the existing runtime configuration rules."""

import pytest

from mobilerun.config import env


@pytest.mark.parametrize("name", env.Secrets.model_fields)
def test_secrets_are_aliased_and_masked(monkeypatch, name):
    alias = name
    monkeypatch.setenv(alias, "sensitive-placeholder")
    snapshot = env.get_secrets()
    assert getattr(snapshot, name).get_secret_value() == "sensitive-placeholder"
    assert "sensitive-placeholder" not in str(getattr(snapshot, name))
    assert "sensitive-placeholder" not in repr(getattr(snapshot, name))
    assert "sensitive-placeholder" not in repr(snapshot)
    assert "sensitive-placeholder" not in str(snapshot)
