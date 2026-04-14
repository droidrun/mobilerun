"""Droidrun Terminal User Interface."""

from droidrun.cli.tui.app import DroidTUI


def run_tui():
    """Run the Droidrun TUI application."""
    app = DroidTUI()
    app.run()


__all__ = ["DroidTUI", "run_tui"]
