"""In-memory credentials using the shared mapping normalizer."""

from collections.abc import Mapping

from mobilerun.core.credentials.base import CredentialProvider, normalize
from mobilerun.models.credentials import Credential


class DictCredentials(CredentialProvider):
    def __init__(
        self, mapping: Mapping[str, object], *, name: str | None = None
    ) -> None:
        super().__init__(name=name)
        self._credentials = normalize(mapping)

    async def _list_credentials(self, package: str | None = None) -> list[Credential]:
        return [
            item
            for item in self._credentials
            if package is None or item.package == package
        ]

    async def _find(
        self,
        package: str | None = None,
        field: str = "password",
        name: str | None = None,
    ) -> Credential | None:
        return next(
            (
                item
                for item in await self._list_credentials(package)
                if field in item.fields and (name is None or item.name == name)
            ),
            None,
        )
