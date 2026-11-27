"""Phone number resource interface."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Sequence
from datetime import datetime

from mobilerun.core.extensions import Extension


class PhoneNumber(Extension, ABC):
    role = "phone_number"
    default_name = "sms"
    exclusive = "phone_number"

    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def phone_number(self) -> str | None: ...

    @property
    @abstractmethod
    def device_id(self) -> str | None: ...

    @abstractmethod
    def messages(
        self, *, after: datetime | None = None
    ) -> Awaitable[Sequence[object]]: ...

    @abstractmethod
    def send(self, to: str, text: str) -> Awaitable[object]: ...

    @abstractmethod
    def wait_for_code(
        self,
        *,
        sender: str | None = None,
        timeout: float = 60,
        after: datetime | None = None,
    ) -> Awaitable[str]: ...
