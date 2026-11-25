"""
Macro Replay Module - Replay recorded UI automation sequences.

This module provides functionality to load and replay macro JSON files
that were generated during MobileAgent trajectory recording.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from mobilerun.core.devices import Device
from mobilerun.core.state.base import StateProvider
from mobilerun.errors.base import DeviceError, MobilerunError
from mobilerun.macro.errors import MacroError, MacroMismatch
from mobilerun.macro.models import (
    MACRO_SCHEMA_VERSION,
    UNSUPPORTED_SCHEMA_MESSAGE,
    normalize_ui_state,
)
from mobilerun.macro.settings import get_macro_settings
from mobilerun.macro.utils import StateMatchResult, compare_states

logger = logging.getLogger("mobilerun-macro")

# Reverse map for legacy key_press macro entries
_KEYCODE_TO_BUTTON = {4: "back", 3: "home", 66: "enter"}


class MacroPlayer:
    """
    A class for loading and replaying Mobilerun macro sequences.

    This player can execute recorded UI actions (taps, swipes, text input, key presses)
    on any supplied framework Device. Device lifetime belongs to the caller.
    """

    def __init__(
        self,
        device: Device | None = None,
        delay_between_actions: float | None = None,
        state_timeout: float | None = None,
        state_threshold: float | None = None,
        state_poll_interval: float | None = None,
    ) -> None:
        """
        Initialize the MacroPlayer.

        Args:
            device: Caller-owned target device; replay never constructs a backend.
            delay_between_actions: Delay in seconds between each action (default: 1.0s)
        """
        self.driver: Device | None = device
        settings = get_macro_settings()
        self.delay_between_actions = (
            settings.MOBILERUN_MACRO_DELAY
            if delay_between_actions is None
            else delay_between_actions
        )
        self.state_timeout = (
            settings.MOBILERUN_MACRO_STATE_TIMEOUT
            if state_timeout is None
            else state_timeout
        )
        self.state_threshold = (
            settings.MOBILERUN_MACRO_STATE_THRESHOLD
            if state_threshold is None
            else state_threshold
        )
        self.state_poll_interval = (
            settings.MOBILERUN_MACRO_POLL_INTERVAL
            if state_poll_interval is None
            else state_poll_interval
        )
        self.last_divergence: Optional[Dict[str, Any]] = None
        self.state_provider: StateProvider | None = None
        self.executed_steps = 0

    async def _initialize_driver(self) -> Device:
        """Return the caller-owned device without constructing a backend."""
        if self.driver is None:
            raise MacroError("MacroPlayer requires a Device")
        return self.driver

    def load_macro_from_file(self, macro_file_path: str | Path) -> Dict[str, Any]:
        """
        Load macro data from a JSON file.

        Args:
            macro_file_path: Path to the macro JSON file

        Returns:
            Dictionary containing the macro data
        """
        path = Path(macro_file_path)
        if path.is_dir():
            path = path / "macro.json"
        macro_data = json.loads(path.read_text())
        if macro_data.get("macro_schema_version") != MACRO_SCHEMA_VERSION:
            raise MacroError(UNSUPPORTED_SCHEMA_MESSAGE)
        return macro_data

    def load_macro_from_folder(self, trajectory_folder: str) -> Dict[str, Any]:
        """
        Load macro data from a trajectory folder.

        Args:
            trajectory_folder: Path to the trajectory folder containing macro.json

        Returns:
            Dictionary containing the macro data
        """
        return self.load_macro_from_file(trajectory_folder)

    async def get_current_state_snapshot(self) -> Dict[str, Any]:
        driver = await self._initialize_driver()
        if self.state_provider is not None:
            try:
                return normalize_ui_state(await self.state_provider.get_state())
            except (DeviceError, OSError, NotImplementedError) as e:
                logger.debug("Falling back to raw driver state: %s", e)
        view = driver.state
        if view is not None:
            try:
                return normalize_ui_state(await view.refresh())
            except (DeviceError, OSError, NotImplementedError) as e:
                logger.debug("Falling back to raw device state: %s", e)
        raw_state = await driver._get_ui_tree()
        return normalize_ui_state(raw_state)

    async def wait_for_pre_state(
        self, pre_state: Optional[Dict[str, Any]], step_number: int
    ) -> tuple[bool, Optional[Dict[str, Any]], Optional[StateMatchResult]]:
        if not pre_state:
            return True, None, None

        deadline = time.monotonic() + self.state_timeout
        last_state = None
        last_result = None

        while True:
            current_state = await self.get_current_state_snapshot()
            result = compare_states(pre_state, current_state, self.state_threshold)
            if result.matches:
                return True, current_state, result

            last_state = current_state
            last_result = result
            if time.monotonic() >= deadline:
                logger.error(
                    "State mismatch before step %s: %s", step_number, result.reason
                )
                return False, last_state, last_result

            await asyncio.sleep(self.state_poll_interval)

    async def replay_action(
        self, action: Dict[str, Any], current_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Replay a single action.

        Args:
            action: Action dictionary containing type and parameters

        Returns:
            True if action was executed successfully, False otherwise
        """
        driver = await self._initialize_driver()
        action_type = action.get("action_type", action.get("type", "unknown"))

        try:
            if action_type == "start_app":
                package = action.get("package")
                if not isinstance(package, str) or not package:
                    raise MacroError("start_app action requires a package")
                activity = action.get("activity", None)
                await driver.start_app(package, activity)
                return True

            elif action_type == "open_url":
                await driver.open_url(action["url"])
                return True

            elif action_type == "stop_app":
                await driver.stop_app(action["package"])
                return True

            elif action_type == "press_key_code":
                await driver.press_key_code(action["key_code"])
                return True

            elif action_type == "extension_call":
                name = action.get("name")
                arguments = action.get("arguments")
                if (
                    not isinstance(name, str)
                    or "." not in name
                    or not isinstance(arguments, dict)
                ):
                    raise MacroError("Invalid extension_call action")
                extension = name.split(".", 1)[0]
                if extension not in driver.extensions:
                    # A missing binding is an environment error, not a UI mismatch:
                    # agent-assisted replay must not silently improvise around it.
                    raise MacroError(
                        f"Extension '{extension}' is not bound for macro replay"
                    )
                entry = driver.actions.get(name)
                if entry.sensitive_parameters & arguments.keys():
                    raise MacroError(
                        f"Cannot replay redacted sensitive arguments for {name}"
                    )
                await driver.actions.call(name, arguments, audience="operator")
                return True

            elif action_type == "tap":
                x = action.get("x", 0)
                y = action.get("y", 0)
                logger.info(f" Tapping at ({x}, {y})")
                await driver.tap(x, y)
                return True

            elif action_type == "swipe":
                start_x = action.get("start_x", 0)
                start_y = action.get("start_y", 0)
                end_x = action.get("end_x", 0)
                end_y = action.get("end_y", 0)
                duration_ms = action.get("duration_ms", 300)

                logger.info(
                    f" Swiping from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration_ms}ms"
                )
                await driver.swipe(start_x, start_y, end_x, end_y, duration_ms)
                # Additional wait after swipe for UI to settle
                await asyncio.sleep(2)
                return True

            elif action_type == "drag":
                start_x = action.get("start_x", 0)
                start_y = action.get("start_y", 0)
                end_x = action.get("end_x", 0)
                end_y = action.get("end_y", 0)
                duration = action.get(
                    "duration", action.get("duration_ms", 300) / 1000.0
                )

                logger.info(
                    f" Dragging from ({start_x}, {start_y}) to ({end_x}, {end_y})"
                )
                await driver.drag(start_x, start_y, end_x, end_y, duration)
                return True

            elif action_type == "input_text":
                text = action.get("text", "")
                clear = action.get("clear", False)
                logger.info(f"Inputting text: '{text}'")
                await driver.input_text(text, clear)
                return True

            elif action_type == "type_secret":
                from mobilerun.core.credentials import CredentialProvider

                package = action.get("package")
                name = action.get("name", action.get("secret", action.get("secret_id")))
                if not any(isinstance(v, str) and v.strip() for v in (package, name)):
                    raise MacroError("type_secret action requires a package or name")
                vault = driver.extensions.get("vault")
                if not isinstance(vault, CredentialProvider):
                    candidates = [
                        extension
                        for extension in driver.extensions.values()
                        if isinstance(extension, CredentialProvider)
                    ]
                    if len(candidates) != 1:
                        raise MacroError("Bind one vault to replay a secret action")
                    vault = candidates[0]
                await vault.type_field(
                    package,
                    action.get("field", "password"),
                    name,
                )
                return True

            elif action_type == "key_press":
                keycode = action.get("keycode", 0)
                button = _KEYCODE_TO_BUTTON.get(keycode)
                if button:
                    logger.info(f" Pressing button: {button}")
                    await driver.press_button(button)
                else:
                    raise MacroError(f"Unknown keycode {keycode}")
                return True

            elif action_type == "button_press":
                button = action.get("button", "")
                logger.info(f" Pressing button: {button}")
                await driver.press_button(button)
                return True

            elif action_type == "back":
                logger.info("⬅  Pressing back button")
                await driver.press_button("back")
                return True

            elif action_type == "wait":
                duration = action.get("duration", 1.0)
                logger.info(f"Waiting for {duration} seconds")
                await driver.wait(duration)
                return True

            else:
                raise MacroError(f"Unknown action type: {action_type}")

        except MobilerunError:
            raise
        except Exception as exc:
            raise MacroError(f"Error executing action {action_type}") from exc

    async def replay_macro(
        self,
        macro_data: Dict[str, Any],
        start_from_step: int = 0,
        max_steps: Optional[int] = None,
    ) -> bool:
        """
        Replay a complete macro sequence.

        Args:
            macro_data: Macro data dictionary loaded from JSON
            start_from_step: Step number to start from (0-based, default: 0)
            max_steps: Maximum number of steps to execute (default: all)

        Returns:
            True if all actions were executed successfully, False otherwise
        """
        if not macro_data or "actions" not in macro_data:
            raise MacroError("Invalid macro data - no actions found")
        if macro_data.get("macro_schema_version") != MACRO_SCHEMA_VERSION:
            raise MacroError(UNSUPPORTED_SCHEMA_MESSAGE)

        self.executed_steps = 0
        self.last_divergence = None
        if start_from_step < 0 or (max_steps is not None and max_steps < 0):
            raise MacroError("Step limits must not be negative")

        actions = macro_data["actions"]
        description = macro_data.get("description", "Unknown macro")
        total_actions = len(actions)

        # Apply start_from_step and max_steps filters
        if start_from_step > 0:
            actions = actions[start_from_step:]
            logger.info(f" Starting from step {start_from_step + 1}")

        if max_steps is not None:
            actions = actions[:max_steps]
            logger.info(f" Limiting to {max_steps} steps")

        logger.info(f" Starting macro replay: '{description}'")
        logger.info(f" Total actions to execute: {len(actions)} / {total_actions}")

        success_count = 0
        failed_count = 0

        for i, action in enumerate(actions, start=start_from_step + 1):
            action_type = action.get("action_type", action.get("type", "unknown"))
            description_text = action.get("description", "")

            logger.info(f"\n Step {i}/{total_actions}: {action_type}")
            if description_text:
                logger.info(f"   Description: {description_text}")

            state_matched, current_state, match_result = await self.wait_for_pre_state(
                action.get("pre_state"), i
            )
            if not state_matched:
                self.last_divergence = {
                    "step": i,
                    "action_type": action_type,
                    "reason": match_result.reason if match_result else "state mismatch",
                    "score": match_result.score if match_result else None,
                }
                raise MacroMismatch(
                    i - 1,
                    macro_data["actions"][i - 1 :],
                    current_state or {},
                    self.last_divergence["reason"],
                )

            # Execute the action
            success = await self.replay_action(action, current_state=current_state)

            if success:
                success_count += 1
                self.executed_steps += 1
                logger.info("    Action completed successfully")
            else:
                failed_count += 1
                raise MacroError("Action failed")

            # Wait between actions (except for the last one)
            if i < start_from_step + len(actions):
                logger.debug(f"   Waiting {self.delay_between_actions}s...")
                await asyncio.sleep(self.delay_between_actions)

        # Summary
        total_executed = success_count + failed_count
        success_rate = (
            (success_count / total_executed * 100) if total_executed > 0 else 0
        )

        logger.info("\n Macro replay completed!")
        logger.info(f" Success: {success_count}/{total_executed} ({success_rate:.1f}%)")

        if failed_count > 0:
            logger.warning(f"  Failed actions: {failed_count}")

        return failed_count == 0


# Utility functions for convenience


async def replay_macro_file(
    macro_file_path: str,
    device: Device | None = None,
    delay_between_actions: float | None = None,
    start_from_step: int = 0,
    max_steps: Optional[int] = None,
    state_timeout: float | None = None,
    state_threshold: float | None = None,
) -> bool:
    """
    Convenience function to replay a macro from a file.

    Args:
        macro_file_path: Path to the macro JSON file
        device: Caller-owned target device
        delay_between_actions: Delay between actions in seconds
        start_from_step: Step to start from (0-based)
        max_steps: Maximum steps to execute

    Returns:
        True if replay was successful, False otherwise
    """
    player = MacroPlayer(
        device=device,
        delay_between_actions=delay_between_actions,
        state_timeout=state_timeout,
        state_threshold=state_threshold,
    )

    try:
        macro_data = player.load_macro_from_file(macro_file_path)
        return await player.replay_macro(
            macro_data, start_from_step=start_from_step, max_steps=max_steps
        )
    except MobilerunError:
        raise
    except (OSError, ValueError) as exc:
        raise MacroError("Could not load macro") from exc


async def replay_macro_folder(
    trajectory_folder: str,
    device: Device | None = None,
    delay_between_actions: float | None = None,
    start_from_step: int = 0,
    max_steps: Optional[int] = None,
    state_timeout: float | None = None,
    state_threshold: float | None = None,
) -> bool:
    """
    Convenience function to replay a macro from a trajectory folder.

    Args:
        trajectory_folder: Path to the trajectory folder containing macro.json
        device: Caller-owned target device
        delay_between_actions: Delay between actions in seconds
        start_from_step: Step to start from (0-based)
        max_steps: Maximum steps to execute

    Returns:
        True if replay was successful, False otherwise
    """
    player = MacroPlayer(
        device=device,
        delay_between_actions=delay_between_actions,
        state_timeout=state_timeout,
        state_threshold=state_threshold,
    )

    try:
        macro_data = player.load_macro_from_folder(trajectory_folder)
        return await player.replay_macro(
            macro_data, start_from_step=start_from_step, max_steps=max_steps
        )
    except MobilerunError:
        raise
    except (OSError, ValueError) as exc:
        raise MacroError("Could not load macro") from exc
