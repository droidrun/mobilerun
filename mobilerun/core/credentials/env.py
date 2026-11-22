"""Flat environment secrets, normalized like dict and file credentials."""

import os  # The vault deliberately enumerates an arbitrary caller-specified prefix.

from mobilerun.core.credentials.dict import DictCredentials


class EnvCredentials(DictCredentials):
    def __init__(
        self, prefix: str = "MOBILERUN_SECRET_", *, name: str | None = None
    ) -> None:
        super().__init__(
            {
                key[len(prefix) :]: value
                for key, value in os.environ.items()
                if key.startswith(prefix)
            },
            name=name,
        )
