"""Shared terminal QR rendering."""

import sys

import segno


def print_qr(data: str) -> bool:
    """Print a compact QR with its quiet zone to terminal stderr."""
    if not sys.stderr.isatty():
        return False
    segno.make_qr(data).terminal(out=sys.stderr, compact=True)
    return True
