"""Apps group sandbox tasks into a routines-compatible script."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .models import Task


class App:
    def __init__(self, name: str) -> None:
        """Create a sandbox application with a stable deployment name."""
        self.name = name
        self.tasks: list[Task] = []
        self.entrypoint: Callable[..., Any] | None = None

    def task(
        self,
        *,
        device: str | None = None,
        timeout: float | None = None,
        hosts: Iterable[str] = (),
        credentials: Iterable[str] = (),
        scopes: Iterable[str] = (),
        name: str | None = None,
    ) -> Callable[[Callable[..., Any]], Task]:
        """Register a task and return its typed wrapper."""
        options = {
            "device": device,
            "timeout": timeout,
            "hosts": tuple(hosts),
            "credentials": tuple(credentials),
            "scopes": tuple(scopes),
            "name": name,
        }

        def decorate(func: Callable[..., Any]) -> Task:
            result = Task(func, app=self, options=options)
            self.tasks.append(result)
            return result

        return decorate

    def local_entrypoint(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Mark the function used as the application's local CLI entry point."""

        def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
            self.entrypoint = func
            return func

        return decorate
