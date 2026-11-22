import ast
import importlib
from pathlib import Path

import pytest

from mobilerun.errors.base import MobilerunError


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("mobilerun.errors.credentials", "CredentialNotFoundError"),
        ("mobilerun.errors.device", "DeviceAuthError"),
        ("mobilerun.errors.device", "DeviceNotFound"),
        (
            "mobilerun.errors.device",
            "DeviceOperationNotSupported",
        ),
        ("mobilerun.errors.operators", "ExpectationFailed"),
        ("mobilerun.errors.cloud", "InvalidAPIKeyError"),
        (
            "mobilerun.errors.credentials",
            "LocalCredentialsNotAllowed",
        ),
        ("mobilerun.errors.operators", "OperatorNotAvailable"),
        ("mobilerun.errors.operators", "OperatorNotSupported"),
        ("mobilerun.errors.device", "PortalApkNotFound"),
        ("mobilerun.errors.device", "PortalInstallFailed"),
        ("mobilerun.errors.device", "UnsupportedStateCapability"),
        ("mobilerun.errors.device", "StateFetchFailed"),
        ("mobilerun.macro.errors", "MacroMismatch"),
        ("mobilerun.agents.errors.keyboard_cleanup_failed", "KeyboardCleanupFailed"),
        ("mobilerun.config.loader", "OutdatedConfigError"),
        ("mobilerun.agents.clients.oauth.profiles", "AuthProfileFormatError"),
        ("mobilerun.agents.clients.oauth.grok", "GrokOAuthError"),
        ("mobilerun.agents.agent.manager.prompts", "ManagerResponseValidationError"),
        (
            "mobilerun.agents.agent.utils.oauth.anthropic_oauth_llm",
            "AnthropicOAuthStreamError",
        ),
    ],
)
def test_errors_share_root(module, name):
    if module.startswith("mobilerun.agents."):
        pytest.importorskip("mobilerun.agents", reason="agent error belongs to add-on")
    assert issubclass(getattr(importlib.import_module(module), name), MobilerunError)


def test_new_error_types_keep_builtin_compatibility():
    from mobilerun.errors import CredentialFormatError, DeviceNotReady, PlaceholderError

    assert issubclass(CredentialFormatError, ValueError)
    assert issubclass(DeviceNotReady, RuntimeError)
    assert issubclass(PlaceholderError, ValueError)


def test_agent_timeout_keeps_builtin_compatibility():
    pytest.importorskip("mobilerun.agents", reason="agent error belongs to add-on")
    from mobilerun.agents.errors import AgentTimeout

    assert issubclass(AgentTimeout, (MobilerunError, TimeoutError))
    assert issubclass(AgentTimeout, TimeoutError)


def test_invalid_values_raise_typed_errors():
    from mobilerun.core.credentials.base import normalize
    from mobilerun.core.state.view import UIStateView
    from mobilerun.errors import CredentialFormatError, DeviceNotReady

    with pytest.raises(CredentialFormatError):
        normalize({"app": 123})
    with pytest.raises(DeviceNotReady):
        _ = UIStateView(None).current


def test_error_classes_discovered_by_ast_share_root():
    root = Path(__file__).resolve().parents[2] / "mobilerun"
    named_errors = {
        "DeviceNotFound",
        "DeviceNotReady",
        "DeviceOperationNotSupported",
        "ExpectationFailed",
        "ExpectationInconclusive",
        "LocalCredentialsNotAllowed",
        "OperatorNotAvailable",
        "OperatorNotSupported",
        "PortalApkNotFound",
        "PortalInstallFailed",
        "StateFetchFailed",
        "UnsupportedStateCapability",
        "MacroMismatch",
        "SandboxNotImplemented",
        "KeyboardCleanupFailed",
        "AgentTimeout",
    }
    for path in root.rglob("*.py"):
        if path.name == "__init__.py" or "clients/cloud" in str(path):
            continue
        module = ".".join(path.relative_to(root.parent).with_suffix("").parts)
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name.startswith("Cloud") or node.name == "MobilerunError":
                continue
            if node.name.endswith("Error") or node.name in named_errors:
                cls = getattr(importlib.import_module(module), node.name)
                if issubclass(cls, BaseException):
                    assert issubclass(cls, MobilerunError), f"{module}.{node.name}"
