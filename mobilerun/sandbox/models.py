"""Task and routine declarations for sandbox scripts."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypedDict, TypeVar, Unpack

from mobilerun_routines import script

from .client import RunResult
from .errors import SandboxNotImplemented

if TYPE_CHECKING:
    from .app import App


class TaskCallOptions(TypedDict, total=False):
    device: str | None


_NOT_IMPLEMENTED = (
    "mobilerun sandbox: not implemented yet (waiting for the sandbox-api release)"
)


class Task:
    def __init__(
        self, func: Callable[..., Any], *, app: App, options: dict[str, Any]
    ) -> None:
        self.func, self.app, self.options = func, app, options
        # Callable objects need not expose a function name.
        self.__name__ = getattr(func, "__name__", "task")
        self.input_model = func.__annotations__.get("input")
        self.output_model = func.__annotations__.get("return")
        self.source_file = Path(__import__("inspect").getsourcefile(func) or "")
        # Register metadata only: script decoration and execution await backend release.

    def __call__(self, input: object, **kwargs: Unpack[TaskCallOptions]) -> RunResult:
        """Run this task locally with a typed input."""
        return self.local(input, **kwargs)

    def local(self, input: object, *, device: str | None = None) -> RunResult:
        """Run locally using the routines-compatible runner."""
        del input, device
        raise SandboxNotImplemented(_NOT_IMPLEMENTED)

    def deploy(self) -> None:
        """Deploy this task's defining script for inspection."""
        raise SandboxNotImplemented(_NOT_IMPLEMENTED)

    def remote(self, input: object) -> RunResult:
        """Start a remote run after deploying the script."""
        del input
        raise SandboxNotImplemented(_NOT_IMPLEMENTED)

    def map(self, inputs: Iterable[object], *, remote: bool = True) -> list[RunResult]:
        """Apply the task to multiple inputs."""
        del inputs, remote
        raise SandboxNotImplemented(_NOT_IMPLEMENTED)


P = ParamSpec("P")
R = TypeVar("R")


def routine(func: Callable[P, R]) -> Callable[P, R]:
    """Register a synchronous or asynchronous routine with upstream's runner.

    Use ``ctx.device()`` for the upstream synchronous device when needed.
    """
    return script(func)
