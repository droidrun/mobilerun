"""Stable repository paths independent of test module depth."""

from pathlib import Path

__all__ = ["REPO_ROOT"]

REPO_ROOT = Path(__file__).resolve().parents[2]
