"""Interrupt-safe CLI import boundary."""

try:
    from mobilerun.cli.tree import cli as cli
    from mobilerun.cli.tree import run_command as run_command
except KeyboardInterrupt:
    raise SystemExit(130) from None
