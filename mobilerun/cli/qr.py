"""Shared terminal QR rendering."""

import sys
from io import StringIO

import segno


def print_qr(data: str) -> bool:
    """Print a compact QR with its quiet zone to terminal stderr."""
    try:
        if not sys.stderr.isatty():
            return False
        output = StringIO()
        segno.make_qr(data).terminal(out=output, compact=True)
        for line in output.getvalue().splitlines():
            # Compact output uses full blocks for light modules and spaces
            # for dark ones. Pin both colours independently of the theme.
            sys.stderr.write(f"\x1b[97;40m{line}\x1b[0m\n")
    except Exception:
        return False
    return True
