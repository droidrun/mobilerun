"""
Droidrun - A framework for controlling Android devices through LLM agents.
"""

import logging
from importlib.metadata import version

__version__ = version("droidrun")

# Attach a default CLILogHandler so that every consumer (CLI, TUI, SDK,
# tools-only) gets visible output without explicit setup.  CLI and TUI
# replace this with their own handler via ``configure_logging()``.
from droidrun.log_handlers import CLILogHandler

_logger = logging.getLogger("droidrun")
_logger.addHandler(CLILogHandler())
_logger.setLevel(logging.INFO)
_logger.propagate = False

# Import main classes for easier access
from droidrun.agent import ResultEvent
from droidrun.agent.droid import DroidAgent
from droidrun.agent.utils.llm_picker import load_llm

# Import configuration classes
from droidrun.config_manager import (
    # Agent configs
    AgentConfig,
    AppCardConfig,
    FastAgentConfig,
    CredentialsConfig,
    # Feature configs
    DeviceConfig,
    DroidConfig,
    ExecutorConfig,
    LLMProfile,
    LoggingConfig,
    ManagerConfig,
    TelemetryConfig,
    ToolsConfig,
    TracingConfig,
)

# Import macro functionality
from droidrun.macro import MacroPlayer, replay_macro_file, replay_macro_folder
from droidrun.tools import AndroidDriver, DeviceDriver, RecordingDriver

# Make main components available at package level
__all__ = [
    # Agent
    "DroidAgent",
    "load_llm",
    "ResultEvent",
    # Tools / Drivers
    "DeviceDriver",
    "AndroidDriver",
    "RecordingDriver",
    # Macro
    "MacroPlayer",
    "replay_macro_file",
    "replay_macro_folder",
    # Configuration
    "DroidConfig",
    "AgentConfig",
    "FastAgentConfig",
    "ManagerConfig",
    "ExecutorConfig",
    "AppCardConfig",
    "DeviceConfig",
    "LoggingConfig",
    "TracingConfig",
    "TelemetryConfig",
    "ToolsConfig",
    "CredentialsConfig",
    "LLMProfile",
]
