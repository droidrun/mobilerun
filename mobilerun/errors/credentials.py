"""Credential lookup, storage, and policy errors."""

from mobilerun.errors.base import CredentialError


class CredentialNotFoundError(CredentialError, KeyError):
    """Raised when a credential key is not found."""

    value_safe = True


class CredentialFieldNotFound(CredentialNotFoundError):
    """Raised when a requested credential field cannot be found."""


class CredentialReadError(CredentialError):
    """A credential file cannot be read."""


class CredentialFormatError(CredentialError, ValueError):
    """A credential file is malformed."""


class LocalCredentialsNotAllowed(CredentialError, ValueError):
    """A public cloud task was given a local credential provider."""
