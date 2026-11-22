"""Actionable errors for agent-facing compatibility paths."""

from contextlib import contextmanager
from importlib.util import find_spec


def is_missing_agents(error: BaseException) -> bool:
    return isinstance(error, ModuleNotFoundError) and (
        error.name == "mobilerun.agents"
        or (error.name or "").startswith("mobilerun.agents.")
    )


def agents_locations() -> tuple[str, ...]:
    try:
        spec = find_spec("mobilerun.agents")
    except ModuleNotFoundError as error:
        if is_missing_agents(error):
            return ()
        raise
    return tuple(spec.submodule_search_locations or ()) if spec else ()


def agents_available() -> bool:
    return bool(agents_locations())


@contextmanager
def require_agents():
    try:
        yield
    except ModuleNotFoundError as error:
        if is_missing_agents(error):
            raise ModuleNotFoundError(
                "Agent commands need the mobilerun-agents add-on: "
                "pip install mobilerun-agents",
                name=error.name,
            ) from error
        raise
