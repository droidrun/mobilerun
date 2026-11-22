"""Mailbox resource interface."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable
from datetime import datetime

from mobilerun.core.extensions import Extension


class Mailbox(Extension, ABC):
    role = "mailbox"
    default_name = "email"

    @property
    @abstractmethod
    def address(self) -> str | None: ...

    @abstractmethod
    def wait_for_code(
        self,
        *,
        sender: str | None = None,
        timeout: float = 60,
        after: datetime | None = None,
    ) -> Awaitable[str]: ...
