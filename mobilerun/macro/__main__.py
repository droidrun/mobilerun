"""
Entry point for running Droidrun macro CLI as a module.

Usage: python -m droidrun.macro <command>
"""

from droidrun.macro.cli import macro_cli

if __name__ == "__main__":
    macro_cli()
