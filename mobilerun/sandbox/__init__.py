"""Public sandbox script APIs, with optional runtime exports loaded on demand."""

from importlib import import_module
from typing import Any

__all__ = (
    "ALLOWED_FORMATS",
    "DATASET_MAX_BYTES",
    "ERROR_MESSAGE_MAX_CHARS",
    "RESERVED_ERROR_CODES",
    "RESULT_MAX_BYTES",
    "STATE_MAX_BYTES",
    "Context",
    "Dataset",
    "DatasetItemInvalidError",
    "DatasetTooLargeError",
    "EntryPointError",
    "Log",
    "NotAvailableError",
    "RunInfo",
    "ScriptError",
    "State",
    "key_field",
    "script",
    "cloud",
    "deploy",
    "Remote",
    "Run",
    "routine",
    "DeploymentPreview",
    "DeploymentReceipt",
    "Endpoint",
    "OwnershipReceipt",
    "SandboxError",
    "SandboxConfirmationRequired",
    "SandboxDeclarationError",
    "SandboxDeploymentConflict",
    "SandboxDeploymentError",
    "SandboxPackagingError",
    "SandboxRunError",
    "SandboxRunTimeout",
)


def __getattr__(name: str) -> Any:
    if name in ("cloud", "deploy", "Remote", "Run"):
        return getattr(import_module(".app", __name__), name)
    if name in (
        "routine",
        "DeploymentPreview",
        "DeploymentReceipt",
        "Endpoint",
        "OwnershipReceipt",
    ):
        return getattr(import_module(".models", __name__), name)
    if name in (
        "SandboxError",
        "SandboxConfirmationRequired",
        "SandboxDeclarationError",
        "SandboxDeploymentConflict",
        "SandboxDeploymentError",
        "SandboxPackagingError",
        "SandboxRunError",
        "SandboxRunTimeout",
    ):
        return getattr(import_module(".errors", __name__), name)
    if name in __all__:
        return getattr(import_module("._runtime", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
