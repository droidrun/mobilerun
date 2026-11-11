"""Mobilerun command-line interface."""

__all__ = ["cli"]


def __getattr__(name):
    if name == "cli":
        from mobilerun.cli.main import cli
        return cli
    raise AttributeError(name)
