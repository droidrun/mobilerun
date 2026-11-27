"""Terminal checks shared by every interactive prompt."""

import sys
from typing import TextIO


def can_prompt(output: TextIO | None) -> bool:
    """Return whether a prompt drawn on ``output`` can be answered from stdin."""
    stdin = sys.stdin
    return (
        stdin is not None and output is not None and stdin.isatty() and output.isatty()
    )
