"""Typed standalone extensions: bind explicitly; discover actions and context."""

from .actions import Action, ActionSet, action, device_action
from .base import Extension

__all__ = ["Action", "ActionSet", "Extension", "action", "device_action"]
