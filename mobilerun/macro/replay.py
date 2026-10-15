"""
Macro Replay Module - Replay recorded UI automation sequences.

This module provides functionality to load and replay macro JSON files
that were generated during MobileAgent trajectory recording.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from mobilerun_core_local.driver.android import AndroidDriver
from mobilerun_core_local.driver.base import DeviceDriver
from mobilerun_core_local.driver.ios import (
    IOSPortalDriver,
    IOSPortalHttpDriver,
    create_ios_driver,
    discover_ios_device,
    validate_ios_portal_url,
)

from mobilerun.agent.utils.trajectory import Trajectory
from mobilerun.macro.handoff import run_agent_handoff
from mobilerun.macro.matcher import StateMatchResult, compare_states
from mobilerun.macro.state import (
    MACRO_SCHEMA_VERSION,
    UNSUPPORTED_SCHEMA_MESSAGE,
    normalize_ui_state,
)
from mobilerun.tools.filters import DetailedFilter
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.ui.ios_provider import IOSStateProvider
from mobilerun.tools.ui.provider import AndroidStateProvider, StateProvider

logger = logging.getLogger("mobilerun-macro")

# Reverse map for legacy key_press macro entries
_KEYCODE_TO_BUTTON = {4: "back", 3: "home", 66: "enter"}


async def resolve_ios_portal_url(url: Optional[str] = None) -> str:
    """Validate an iOS portal base URL, or discover a running portal."""
    if url:
        return validate_ios_portal_url(url)
    logger.info("🔍 Searching for iOS portal...")
    return await discover_ios_device()


class MacroPlayer:
    """
    A class for loading and replaying Mobilerun macro sequences.

    This player can execute recorded UI actions (taps, swipes, text input, key presses)
    on Android devices, or on iOS devices through the iOS portal.
    """

    def __init__(
        self,
        device_serial: str = None,
        delay_between_actions: float = 1.0,
        on_mismatch: str = "stop",
        state_timeout: float = 5.0,
        state_threshold: float = 0.85,
        state_poll_interval: float = 0.5,
        handoff_runner=None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        config_path: Optional[str] = None,
        agent: Optional[str] = None,
        ios: bool = False,
        driver: DeviceDriver | None = None,
        state_provider: StateProvider | None = None,
    ):
        """
        Initialize the MacroPlayer.

        Args:
            device_serial: Serial number of the target device, or the iOS portal URL.
                If None, uses ANDROID_SERIAL or the only connected device (iOS:
                portal discovery).
            delay_between_actions: Delay in seconds between each action (default: 1.0s)
            agent: Agent backend for divergence handoff (e.g. "jev"); None keeps
                the configured one.
            ios: Replay on an iOS device through the iOS portal.
            driver: Pre-connected device driver to use instead of creating one.
            state_provider: State provider for pre-action state checks.
        """
        self.device_serial = device_serial
        self.delay_between_actions = delay_between_actions
        self.on_mismatch = on_mismatch
        self.state_timeout = state_timeout
        self.state_threshold = state_threshold
        self.state_poll_interval = state_poll_interval
        self.handoff_runner = handoff_runner or run_agent_handoff
        self.provider = provider
        self.model = model
        self.config_path = config_path
        self.agent = agent
        self.ios = ios
        self.last_divergence: Optional[Dict[str, Any]] = None
        self.state_provider = state_provider
        self.driver: DeviceDriver | None = driver
        self.credential_manager = None
        self._credential_load_error: Optional[str] = None

    async def _initialize_driver(self) -> DeviceDriver:
        """Initialize the driver for the target device."""
        if self.driver is None:
            if self.ios:
                self.driver = await self._create_ios_driver()
            else:
                self.driver = AndroidDriver(serial=self.device_serial)
                await self.driver.connect()
                logger.info(f"🤖 Initialized driver for device: {self.device_serial}")
        return self.driver

    async def _create_ios_driver(self) -> DeviceDriver:
        url = await resolve_ios_portal_url(self.device_serial)
        # The handoff reuses the resolved URL as the agent's device.
        self.device_serial = url
        driver = await create_ios_driver(url, token=self._resolve_device_token())
        await driver.connect()
        logger.info(f"🍎 Initialized iOS driver for portal: {url}")
        return driver

    def _resolve_device_token(self) -> Optional[str]:
        from mobilerun.config_manager.config_manager import DeviceConfig
        from mobilerun.config_manager.loader import ConfigLoader

        device_config = DeviceConfig()
        try:
            if (
                self.config_path is not None
                or ConfigLoader.get_user_config_path().exists()
            ):
                device_config = ConfigLoader.load(self.config_path).device
        except Exception as e:
            logger.debug("Failed to load device config for macro replay: %s", e)
        return device_config.resolve_auth_token()

    def _is_ios(self) -> bool:
        platform = str(getattr(self.driver, "platform", "") or "")
        return self.ios or platform.lower() == "ios"

    def load_macro_from_file(self, macro_file_path: str) -> Dict[str, Any]:
        """
        Load macro data from a JSON file.

        Args:
            macro_file_path: Path to the macro JSON file

        Returns:
            Dictionary containing the macro data
        """
        return Trajectory.load_macro_sequence(macro_file_path)

    def load_macro_from_folder(self, trajectory_folder: str) -> Dict[str, Any]:
        """
        Load macro data from a trajectory folder.

        Args:
            trajectory_folder: Path to the trajectory folder containing macro.json

        Returns:
            Dictionary containing the macro data
        """
        return Trajectory.load_macro_sequence(trajectory_folder)

    def _load_credential_manager(self):
        if self.credential_manager is not None:
            return self.credential_manager

        from mobilerun.config_manager.loader import ConfigLoader
        from mobilerun.credential_manager import FileCredentialManager

        config_path = self.config_path
        if config_path is None:
            user_config_path = ConfigLoader.get_user_config_path()
            if not user_config_path.exists():
                self._credential_load_error = "no Mobilerun config file was found"
                return None
            config_path = str(user_config_path)

        try:
            config = ConfigLoader.load(config_path)
            credential_manager = FileCredentialManager(config.credentials)
        except Exception as e:
            self._credential_load_error = str(e)
            logger.debug("Failed to load credentials for macro replay: %s", e)
            return None

        if not credential_manager.secrets:
            self._credential_load_error = "credentials are disabled or empty"
            return None

        self.credential_manager = credential_manager
        self._credential_load_error = None
        return self.credential_manager

    async def get_current_state_snapshot(self) -> Dict[str, Any]:
        driver = await self._initialize_driver()
        if self.state_provider is None:
            self.state_provider = _default_state_provider(driver)
        if self.state_provider is not None:
            try:
                return normalize_ui_state(await self.state_provider.get_state())
            except Exception as e:
                logger.debug("Falling back to raw driver state: %s", e)
        raw_state = await driver.get_ui_tree()
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
                activity = action.get("activity", None)
                await driver.start_app(package, activity)
                return True

            elif action_type == "tap":
                x = action.get("x", 0)
                y = action.get("y", 0)
                logger.info(f"🫰 Tapping at ({x}, {y})")
                await driver.tap(x, y)
                return True

            elif action_type == "swipe":
                start_x = action.get("start_x", 0)
                start_y = action.get("start_y", 0)
                end_x = action.get("end_x", 0)
                end_y = action.get("end_y", 0)
                duration_ms = action.get("duration_ms", 300)

                logger.info(
                    f"👆 Swiping from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration_ms}ms"
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
                    f"👆 Dragging from ({start_x}, {start_y}) to ({end_x}, {end_y})"
                )
                await driver.drag(start_x, start_y, end_x, end_y, duration)
                return True

            elif action_type == "input_text":
                text = action.get("text", "")
                clear = action.get("clear", False)
                logger.info(f"⌨️  Inputting text: '{text}'")
                await driver.input_text(text, clear)
                return True

            elif action_type == "type_secret":
                secret_id = action.get("secret_id")
                clear = action.get("clear", False)
                if not secret_id:
                    logger.error("❌ Secret macro action is missing secret_id")
                    return False

                credential_manager = self._load_credential_manager()
                if credential_manager is None:
                    detail = (
                        f": {self._credential_load_error}"
                        if self._credential_load_error
                        else ""
                    )
                    logger.error(
                        "❌ Cannot replay secret '%s'; no credentials are available%s",
                        secret_id,
                        detail,
                    )
                    return False

                try:
                    secret_value = await credential_manager.resolve_key(secret_id)
                except Exception as e:
                    logger.error("❌ Secret '%s' is not available: %s", secret_id, e)
                    return False

                logger.info("⌨️  Inputting secret: '%s'", secret_id)
                await driver.input_text(secret_value, clear)
                return True

            elif action_type == "key_press":
                keycode = action.get("keycode", 0)
                button = _KEYCODE_TO_BUTTON.get(keycode)
                if button:
                    logger.info(f"🔘 Pressing button: {button}")
                    await driver.press_button(button)
                else:
                    logger.warning(f"⚠️  Unknown keycode {keycode}, skipping")
                return True

            elif action_type == "button_press":
                button = action.get("button", "")
                logger.info(f"🔘 Pressing button: {button}")
                await driver.press_button(button)
                return True

            elif action_type == "back":
                logger.info("⬅️  Pressing back button")
                await driver.press_button("back")
                return True

            elif action_type == "wait":
                duration = action.get("duration", 1.0)
                logger.info(f"⏳ Waiting for {duration} seconds")
                await asyncio.sleep(duration)
                return True

            else:
                logger.warning(f"⚠️  Unknown action type: {action_type}")
                return False

        except Exception as e:
            logger.error(f"❌ Error executing action {action_type}: {e}")
            return False

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
            logger.error("❌ Invalid macro data - no actions found")
            return False
        if macro_data.get("macro_schema_version") != MACRO_SCHEMA_VERSION:
            raise ValueError(UNSUPPORTED_SCHEMA_MESSAGE)

        actions = macro_data["actions"]
        description = macro_data.get("description", "Unknown macro")
        total_actions = len(actions)

        # Apply start_from_step and max_steps filters
        if start_from_step > 0:
            actions = actions[start_from_step:]
            logger.info(f"📍 Starting from step {start_from_step + 1}")

        if max_steps is not None:
            actions = actions[:max_steps]
            logger.info(f"🎯 Limiting to {max_steps} steps")

        logger.info(f"🎬 Starting macro replay: '{description}'")
        logger.info(f"📊 Total actions to execute: {len(actions)} / {total_actions}")

        success_count = 0
        failed_count = 0

        for i, action in enumerate(actions, start=start_from_step + 1):
            action_type = action.get("action_type", action.get("type", "unknown"))
            description_text = action.get("description", "")

            logger.info(f"\n📍 Step {i}/{total_actions}: {action_type}")
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
                if self.on_mismatch == "agent":
                    logger.warning("🤖 Handing off to agent after macro divergence")
                    handoff_kwargs: Dict[str, Any] = {}
                    # Only set when used, so custom runners keep their signature.
                    if self.agent is not None:
                        handoff_kwargs["agent"] = self.agent
                    if self._is_ios():
                        handoff_kwargs["ios"] = True
                    return await self.handoff_runner(
                        goal=description,
                        device_serial=self.device_serial,
                        provider=self.provider,
                        model=self.model,
                        config_path=self.config_path,
                        divergence=self.last_divergence,
                        current_state=current_state or {},
                        remaining_actions=macro_data["actions"][i - 1 :],
                        **handoff_kwargs,
                    )
                return False

            # Execute the action
            success = await self.replay_action(action, current_state=current_state)

            if success:
                success_count += 1
                logger.info("   ✅ Action completed successfully")
            else:
                failed_count += 1
                logger.error("   ❌ Action failed")

            # Wait between actions (except for the last one)
            if i < start_from_step + len(actions):
                logger.debug(f"   ⏳ Waiting {self.delay_between_actions}s...")
                await asyncio.sleep(self.delay_between_actions)

        # Summary
        total_executed = success_count + failed_count
        success_rate = (
            (success_count / total_executed * 100) if total_executed > 0 else 0
        )

        logger.info("\n🎉 Macro replay completed!")
        logger.info(
            f"📊 Success: {success_count}/{total_executed} ({success_rate:.1f}%)"
        )

        if failed_count > 0:
            logger.warning(f"⚠️  Failed actions: {failed_count}")

        return failed_count == 0


def _default_state_provider(driver: DeviceDriver) -> StateProvider | None:
    """Pick the provider MobileAgent pairs with ``driver`` (no vision)."""
    # The --local iOS portal serves the Android-shaped tree in points.
    if isinstance(driver, (AndroidDriver, IOSPortalHttpDriver)):
        return AndroidStateProvider(
            driver,
            tree_filter=DetailedFilter(),
            tree_formatter=IndexedFormatter(),
            use_normalized=False,
        )
    if isinstance(driver, IOSPortalDriver):
        return IOSStateProvider(driver)
    return None


# Utility functions for convenience


async def replay_macro_file(
    macro_file_path: str,
    device_serial: str = None,
    delay_between_actions: float = 1.0,
    start_from_step: int = 0,
    max_steps: Optional[int] = None,
    on_mismatch: str = "stop",
    state_timeout: float = 5.0,
    state_threshold: float = 0.85,
    config_path: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    agent: Optional[str] = None,
    ios: bool = False,
) -> bool:
    """
    Convenience function to replay a macro from a file.

    Args:
        macro_file_path: Path to the macro JSON file
        device_serial: Target device serial (optional)
        delay_between_actions: Delay between actions in seconds
        start_from_step: Step to start from (0-based)
        max_steps: Maximum steps to execute
        config_path: Config file for credentials and agent handoff
        provider: LLM provider override for agent handoff
        model: Model override for agent handoff
        agent: Agent backend for agent handoff (e.g. "jev")
        ios: Replay on an iOS device (device_serial is the portal URL)

    Returns:
        True if replay was successful, False otherwise
    """
    player = MacroPlayer(
        device_serial=device_serial,
        delay_between_actions=delay_between_actions,
        on_mismatch=on_mismatch,
        state_timeout=state_timeout,
        state_threshold=state_threshold,
        config_path=config_path,
        provider=provider,
        model=model,
        agent=agent,
        ios=ios,
    )

    try:
        macro_data = player.load_macro_from_file(macro_file_path)
        return await player.replay_macro(
            macro_data, start_from_step=start_from_step, max_steps=max_steps
        )
    except Exception as e:
        logger.error(f"❌ Error replaying macro file {macro_file_path}: {e}")
        return False


async def replay_macro_folder(
    trajectory_folder: str,
    device_serial: str = None,
    delay_between_actions: float = 1.0,
    start_from_step: int = 0,
    max_steps: Optional[int] = None,
    on_mismatch: str = "stop",
    state_timeout: float = 5.0,
    state_threshold: float = 0.85,
    config_path: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    agent: Optional[str] = None,
    ios: bool = False,
) -> bool:
    """
    Convenience function to replay a macro from a trajectory folder.

    Args:
        trajectory_folder: Path to the trajectory folder containing macro.json
        device_serial: Target device serial (optional)
        delay_between_actions: Delay between actions in seconds
        start_from_step: Step to start from (0-based)
        max_steps: Maximum steps to execute
        config_path: Config file for credentials and agent handoff
        provider: LLM provider override for agent handoff
        model: Model override for agent handoff
        agent: Agent backend for agent handoff (e.g. "jev")
        ios: Replay on an iOS device (device_serial is the portal URL)

    Returns:
        True if replay was successful, False otherwise
    """
    player = MacroPlayer(
        device_serial=device_serial,
        delay_between_actions=delay_between_actions,
        on_mismatch=on_mismatch,
        state_timeout=state_timeout,
        state_threshold=state_threshold,
        config_path=config_path,
        provider=provider,
        model=model,
        agent=agent,
        ios=ios,
    )

    try:
        macro_data = player.load_macro_from_folder(trajectory_folder)
        return await player.replay_macro(
            macro_data, start_from_step=start_from_step, max_steps=max_steps
        )
    except Exception as e:
        logger.error(f"❌ Error replaying macro folder {trajectory_folder}: {e}")
        return False
