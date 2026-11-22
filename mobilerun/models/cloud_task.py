"""Reference-only contributions to server-side cloud tasks; never secret values."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CloudCredentialReference:
    package: str
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.package, str) or not isinstance(self.name, str):
            raise TypeError(
                "Cloud credential references require package and name strings"
            )


@dataclass(frozen=True, slots=True)
class CloudTaskInputs:
    credential_references: tuple[CloudCredentialReference, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.credential_references, tuple) or not all(
            isinstance(getattr(reference, "package", None), str)
            and isinstance(getattr(reference, "name", None), str)
            for reference in self.credential_references
        ):
            raise TypeError(
                "Cloud task inputs require a tuple of credential references"
            )
