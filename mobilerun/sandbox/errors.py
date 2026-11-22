"""Raised while the public sandbox API is not released."""

from mobilerun.errors.base import SandboxError


class SandboxNotImplemented(SandboxError, NotImplementedError):
    """A sandbox operation is intentionally unavailable until backend release."""
