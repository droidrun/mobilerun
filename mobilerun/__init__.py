"""
Mobilerun - A framework for controlling Android devices through LLM agents.
"""

import logging
from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING

from mobilerun.agent.utils.errors import HtmlErrorPageFilter
from mobilerun.log_handlers import CLILogHandler

__version__ = version("mobilerun")

# Attach a default CLILogHandler so that every consumer (CLI, SDK,
# tools-only) gets visible output without explicit setup.  CLI flows can
# replace this with their own handler via ``configure_logging()``.

_logger = logging.getLogger("mobilerun")
_logger.addHandler(CLILogHandler())
_logger.setLevel(logging.INFO)
_logger.propagate = False


# llama-index's retry warnings include provider error bodies verbatim.
for _retry_logger in (
    "llama_index.llms.openai.utils",
    "llama_index.llms.google_genai.utils",
):
    logging.getLogger(_retry_logger).addFilter(HtmlErrorPageFilter())

if TYPE_CHECKING:
    from mobilerun.agent import ResultEvent
    from mobilerun.agent.droid import MobileAgent, MobileAgentState
    from mobilerun.agent.utils.llm_picker import load_llm
    from mobilerun.config_manager import (
        AgentConfig,
        AppCardConfig,
        CredentialsConfig,
        DeviceConfig,
        ExecutorConfig,
        FastAgentConfig,
        LLMProfile,
        LoggingConfig,
        ManagerConfig,
        MobileConfig,
        TelemetryConfig,
        ToolsConfig,
        TracingConfig,
    )
    from mobilerun.macro import MacroPlayer, replay_macro_file, replay_macro_folder
    from mobilerun.tools import AndroidDriver, DeviceDriver, RecordingDriver

_EXPORTS = {
    "ResultEvent": ("mobilerun.agent", "ResultEvent"),
    "MobileAgent": ("mobilerun.agent.droid", "MobileAgent"),
    "MobileAgentState": ("mobilerun.agent.droid", "MobileAgentState"),
    "load_llm": ("mobilerun.agent.utils.llm_picker", "load_llm"),
    "AgentConfig": ("mobilerun.config_manager", "AgentConfig"),
    "AppCardConfig": ("mobilerun.config_manager", "AppCardConfig"),
    "CredentialsConfig": ("mobilerun.config_manager", "CredentialsConfig"),
    "DeviceConfig": ("mobilerun.config_manager", "DeviceConfig"),
    "ExecutorConfig": ("mobilerun.config_manager", "ExecutorConfig"),
    "FastAgentConfig": ("mobilerun.config_manager", "FastAgentConfig"),
    "LLMProfile": ("mobilerun.config_manager", "LLMProfile"),
    "LoggingConfig": ("mobilerun.config_manager", "LoggingConfig"),
    "ManagerConfig": ("mobilerun.config_manager", "ManagerConfig"),
    "MobileConfig": ("mobilerun.config_manager", "MobileConfig"),
    "TelemetryConfig": ("mobilerun.config_manager", "TelemetryConfig"),
    "ToolsConfig": ("mobilerun.config_manager", "ToolsConfig"),
    "TracingConfig": ("mobilerun.config_manager", "TracingConfig"),
    "MacroPlayer": ("mobilerun.macro", "MacroPlayer"),
    "replay_macro_file": ("mobilerun.macro", "replay_macro_file"),
    "replay_macro_folder": ("mobilerun.macro", "replay_macro_folder"),
    "AndroidDriver": ("mobilerun.tools", "AndroidDriver"),
    "DeviceDriver": ("mobilerun.tools", "DeviceDriver"),
    "RecordingDriver": ("mobilerun.tools", "RecordingDriver"),
}

# Make main components available at package level
__all__ = [
    # Agent
    "MobileAgent",
    "MobileAgentState",
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
    "MobileConfig",
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

# Legacy aliases — deprecated, will be removed in v0.8.0
_LEGACY_ALIASES = {
    "DroidAgent": "MobileAgent",
    "DroidAgentState": "MobileAgentState",
    "DroidConfig": "MobileConfig",
}


def __getattr__(name):
    if name in _LEGACY_ALIASES:
        import warnings

        new_name = _LEGACY_ALIASES[name]
        warnings.warn(
            f"{name} has been renamed to {new_name}. "
            f"Update your code to use {new_name}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(import_module(__name__), new_name)
    if name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        value = getattr(import_module(module), attribute)
        globals()[name] = value
        return value
    if name in {"agent", "tools", "macro", "config_manager"}:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
