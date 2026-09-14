"""ActionContext — composed bag of dependencies for action functions.

Replaces the ``tools=tools_instance`` parameter that action functions
previously received.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from mobilerun_core_local.driver.base import DeviceDriver

    from mobilerun.agent.droid.state import MobileAgentState
    from mobilerun.credential_manager import CredentialManager
    from mobilerun.macro.recorder import MacroRecorder
    from mobilerun.tools.ui.provider import StateProvider
    from mobilerun.tools.ui.state import UIState


class ActionContext:
    """Everything an action function needs to interact with the device."""

    def __init__(
        self,
        driver: "DeviceDriver",
        ui: "Optional[UIState]",
        shared_state: "MobileAgentState",
        state_provider: "StateProvider",
        app_opener_llm=None,
        credential_manager: "Optional[CredentialManager]" = None,
        streaming: bool = False,
        macro_recorder: "Optional[MacroRecorder]" = None,
    ) -> None:
        self.driver = driver
        self.ui = ui  # refreshed each step before tool execution
        self.shared_state = shared_state
        self.state_provider = state_provider
        self.app_opener_llm = app_opener_llm
        self.credential_manager = credential_manager
        self.streaming = streaming
        self.macro_recorder = macro_recorder
        # ActionContext is created for each workflow run. Keep the inventory
        # stable within that run; a subsequent run receives a fresh context.
        self.installed_apps_cache: list[dict[str, Any]] | None = None
