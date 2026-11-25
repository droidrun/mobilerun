"""Host-owned credential policies. Macro data cannot relax these rules."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CredentialPolicy:
    """Policy for a named field (optionally limited to one package)."""

    name: str
    field: str = "password"
    package: str | None = None
    requires_approval: bool = False
    mask_from_start: bool = False
    mask_card_numbers: bool = False

    def matches(self, package: str | None, field: str, name: str | None) -> bool:
        # A partial selector must not bypass a protected field it could match.
        return (
            (name is None or self.name == name)
            and self.field == field
            and (package is None or self.package is None or self.package == package)
        )
