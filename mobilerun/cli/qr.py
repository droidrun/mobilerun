"""Shared terminal QR rendering."""

from __future__ import annotations

import sys
from io import StringIO


def print_qr(data: str) -> bool:
    """Print a compact QR with its quiet zone to terminal stderr."""
    import segno

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


from mobilerun.cli.options import _lazy_imports  # noqa: E402

__getattr__ = _lazy_imports(globals(), {
    'segno': ('segno', None),
})

from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    import segno as segno
