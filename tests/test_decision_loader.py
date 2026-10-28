from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from mobilerun import MobileConfig
from mobilerun.agent.selectors import Decision
from mobilerun.agent.utils import decision_loader
from mobilerun.config_manager import DecisionProfile


@pytest.fixture(autouse=True)
def providers(monkeypatch):
    monkeypatch.setattr(decision_loader, "_providers", {})


def test_loader_preserves_native_model_and_constructor_options():
    model = SimpleNamespace(decide=AsyncMock(return_value=Decision(0)))
    factory = Mock(return_value=model)
    decision_loader.register_decision_provider("Example", factory)

    loaded = decision_loader.load_decision_model(
        " EXAMPLE ", model="choice-model", api_key="secret", endpoint="https://example.test"
    )

    assert loaded is model
    factory.assert_called_once_with(
        model="choice-model", api_key="secret", endpoint="https://example.test"
    )
    model.decide.assert_not_called()


def test_loader_leaves_provider_defaults_unchanged():
    factory = Mock(return_value=SimpleNamespace(decide=AsyncMock()))
    decision_loader.register_decision_provider("example", factory)
    decision_loader.load_decision_model("example")
    factory.assert_called_once_with()


@pytest.mark.parametrize("name", ["", " ", "unknown"])
def test_loader_rejects_unknown_providers(name):
    with pytest.raises(ValueError, match="Unsupported decision provider"):
        decision_loader.load_decision_model(name)


def test_registration_rejects_empty_names():
    with pytest.raises(ValueError, match="cannot be empty"):
        decision_loader.register_decision_provider(" ", Mock())


def test_registration_rejects_non_callable_factory():
    with pytest.raises(TypeError, match="must be callable"):
        decision_loader.register_decision_provider("example", None)


def test_registration_rejects_duplicate_providers():
    decision_loader.register_decision_provider("example", Mock())
    with pytest.raises(ValueError, match="already registered"):
        decision_loader.register_decision_provider("EXAMPLE", Mock())


@pytest.mark.parametrize("value", [None, object(), SimpleNamespace(decide=None)])
def test_loader_rejects_objects_without_decide(value):
    decision_loader.register_decision_provider("example", Mock(return_value=value))
    with pytest.raises(TypeError, match="implementing decide"):
        decision_loader.load_decision_model("example")


def test_provider_errors_are_not_hidden():
    error = RuntimeError("Provider unavailable")
    decision_loader.register_decision_provider("example", Mock(side_effect=error))
    with pytest.raises(RuntimeError) as raised:
        decision_loader.load_decision_model("example")
    assert raised.value is error


def test_profile_overrides_preserve_provider_options_and_do_not_mutate_config():
    model = SimpleNamespace(decide=AsyncMock())
    factory = Mock(return_value=model)
    decision_loader.register_decision_provider("example", factory)
    profile = DecisionProfile(
        provider="Example", model="configured-model", base_url="https://original.test",
        kwargs={"timeout": 30, "headers": {"X-Custom": "keep"}},
    )
    profiles = {"active": profile, "unused": DecisionProfile(provider="unregistered")}
    overrides = {"timeout": 5, "model": "overridden-model"}

    loaded = decision_loader.load_decision_models_from_profiles(
        profiles, ["active"], active=overrides,
    )

    assert loaded == {"active": model}
    factory.assert_called_once_with(
        model="overridden-model", base_url="https://original.test",
        timeout=5, headers={"X-Custom": "keep"},
    )
    assert profile.model == "configured-model"
    assert profile.kwargs == {"timeout": 30, "headers": {"X-Custom": "keep"}}
    assert overrides == {"timeout": 5, "model": "overridden-model"}


def test_missing_profiles_are_validated_before_constructing_any_models():
    factory = Mock(return_value=SimpleNamespace(decide=AsyncMock()))
    decision_loader.register_decision_provider("example", factory)

    with pytest.raises(KeyError, match="missing"):
        decision_loader.load_decision_models_from_profiles(
            {"valid": DecisionProfile(provider="Example")}, ["valid", "missing"],
        )

    factory.assert_not_called()


@pytest.mark.parametrize("reasoning", [False, True])
def test_agent_loads_only_the_active_decision_role(reasoning):
    config = MobileConfig()
    config.agent.reasoning = reasoning
    active_role = "executor" if reasoning else "fast_agent"
    inactive_role = "fast_agent" if reasoning else "executor"
    config.decision_profiles = {
        active_role: DecisionProfile(provider="Example"),
        inactive_role: DecisionProfile(provider="unregistered"),
    }
    model = SimpleNamespace(decide=AsyncMock())
    factory = Mock(return_value=model)
    decision_loader.register_decision_provider("example", factory)

    assert decision_loader.load_agent_decision_models(config) == {active_role: model}
    factory.assert_called_once_with()


def test_missing_active_decision_profile_leaves_llm_path_untouched():
    config = MobileConfig()
    config.decision_profiles = {"executor": DecisionProfile(provider="unregistered")}

    assert decision_loader.load_agent_decision_models(config) == {}
