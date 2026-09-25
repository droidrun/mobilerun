"""
Mobilerun - A framework for controlling Android devices through LLM agents.
"""

import logging
from importlib.metadata import version

__version__ = version("mobilerun")

# Attach a default CLILogHandler so that every consumer (CLI, SDK,
# tools-only) gets visible output without explicit setup.  CLI flows can
# replace this with their own handler via ``configure_logging()``.
from mobilerun.log_handlers import CLILogHandler

_logger = logging.getLogger("mobilerun")
_logger.addHandler(CLILogHandler())
_logger.setLevel(logging.INFO)
_logger.propagate = False

# Import main classes for easier access
from mobilerun.agent import ResultEvent  # noqa: E402
from mobilerun.agent.droid import MobileAgent, MobileAgentState  # noqa: E402
from mobilerun.agent.utils.errors import HtmlErrorPageFilter  # noqa: E402
from mobilerun.agent.utils.llm_picker import load_llm  # noqa: E402

# llama-index's retry warnings include provider error bodies verbatim.
for _retry_logger in (
    "llama_index.llms.openai.utils",
    "llama_index.llms.google_genai.utils",
):
    logging.getLogger(_retry_logger).addFilter(HtmlErrorPageFilter())

# Import configuration classes
from mobilerun.config_manager import (  # noqa: E402
    # Agent configs
    AgentConfig,
    AppCardConfig,
    CredentialsConfig,
    # Feature configs
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

# Import macro functionality
from mobilerun.macro import (  # noqa: E402
    MacroPlayer,
    replay_macro_file,
    replay_macro_folder,
)
from mobilerun.tools import AndroidDriver, DeviceDriver, RecordingDriver  # noqa: E402

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
        return globals()[new_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
