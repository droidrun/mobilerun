"""Repository in-memory Device; never connects to a phone or service."""

from mobilerun.core.devices.base import Device


class MemoryDevice(Device):
    def __init__(
        self,
        *,
        identifier: str = "memory",
        cloud_id: str | None = None,
        platform: str = "android",
        supported_operations: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__()
        self._identifier = identifier
        self._cloud_id = cloud_id
        self._platform = platform
        self._supported_operations = supported_operations
        self.inputs: list[str] = []

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def cloud_id(self) -> str | None:
        return self._cloud_id

    @property
    def platform(self) -> str:
        return self._platform

    @property
    def supported_operations(self) -> set[str]:
        return set(self._supported_operations)

    async def connect(self) -> None:
        return None

    async def _ensure_connected(self) -> None:
        return None

    async def _input_text_raw(
        self, text: str, clear: bool = False, stealth: bool = False, wpm: int = 0
    ) -> bool:
        self.inputs.append(text)
        return True
