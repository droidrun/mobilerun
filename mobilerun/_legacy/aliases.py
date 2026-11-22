"""Lazy SDK compatibility exports; importing the base never loads agents."""

import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings
from importlib import import_module
from typing import Any

from mobilerun.addons.agents import require_agents

_LEGACY_EXPORTS = {
    name: ("mobilerun.agents", name)
    for name in (
        "MobileAgent",
        "MobileAgentState",
        "load_llm",
        "ResultEvent",
        "AgentConfig",
        "FastAgentConfig",
        "ManagerConfig",
        "ExecutorConfig",
        "AppCardConfig",
        "CredentialsConfig",
        "ToolsConfig",
        "LLMProfile",
    )
}
_LEGACY_EXPORTS.update(
    {
        "HtmlErrorPageFilter": (
            "mobilerun.agents.agent.utils.errors",
            "HtmlErrorPageFilter",
        ),
        "DroidAgent": ("mobilerun.agents", "MobileAgent"),
        "DroidAgentState": ("mobilerun.agents", "MobileAgentState"),
        "DroidConfig": ("mobilerun.agents.config.user_config", "MobileConfig"),
    }
)
_LEGACY_EXPORTS.update(
    {
        name: ("mobilerun_core_local.driver", name)
        for name in ("AndroidDriver", "DeviceDriver", "RecordingDriver")
    }
)
_LEGACY_EXPORTS.update(
    {
        name: ("mobilerun.macro", name)
        for name in ("MacroPlayer", "replay_macro_file", "replay_macro_folder")
    }
)


def resolve_export(name: str) -> Any:
    if name == "CloudDevice":
        from mobilerun.core.devices.cloud import CloudDevice

        return CloudDevice
    if name in _LEGACY_EXPORTS:
        module, attribute = _LEGACY_EXPORTS[name]
        warnings.warn(
            f"mobilerun.{name} is deprecated; import {attribute} from {module} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        with require_agents():
            return getattr(import_module(module), attribute)
    raise AttributeError(f"module 'mobilerun' has no attribute {name!r}")


# Only namespaces present in the 0.7 SDK belong here. Longest prefix wins.
LEGACY_MODULES = {
    "agent": "mobilerun.agents.agent",
    "config_manager": "mobilerun._legacy.config_manager",
    "config_manager.config_manager": "mobilerun._legacy.config",
    "config_manager.loader": "mobilerun.agents.config.loader",
    "config_manager.path_resolver": "mobilerun.agents.utils.paths",
    "config_manager.prompt_loader": "mobilerun.config.prompts",
    "config_manager.env_keys": "mobilerun.agents.clients.oauth.profiles",
    "config_manager.auth_profile_store": "mobilerun.agents.clients.oauth.profiles",
    "config_manager.migrations": "mobilerun.config.migrations",
    "tools": "mobilerun._legacy.tools",
    "tools.driver": "mobilerun_core_local.driver",
    "tools.filters": "mobilerun.core.state.filters",
    "tools.formatters": "mobilerun.core.state.formatters",
    "tools.ui": "mobilerun.core.state",
    "tools.ui.provider": "mobilerun.core.state.android",
    "tools.ui.ios_provider": "mobilerun.core.state.ios",
    "tools.ui.screenshot_provider": "mobilerun.core.state.screenshot",
    "tools.ui.state": "mobilerun.core.state.ui_state",
    "tools.ui.stealth_state": "mobilerun.core.state.stealth",
    "tools.ios": "mobilerun_core_local.driver.ios",
    "tools.helpers": "mobilerun._legacy.helpers",
    "tools.helpers.coordinate": "mobilerun.utils.coordinates",
    "tools.helpers.geometry": "mobilerun.utils.geometry",
    "tools.helpers.images": "mobilerun.utils.images",
    "log_handlers": "mobilerun.logs.handlers",
    "macro.replay": "mobilerun.macro.player",
}


def legacy_module_name(relative: str) -> str | None:
    """Resolve old package trees without importing optional agent code."""
    for old in sorted(LEGACY_MODULES, key=len, reverse=True):
        if relative == old or relative.startswith(old + "."):
            return LEGACY_MODULES[old] + relative[len(old) :]
    return None


class LegacyAliasLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, real_name: str):
        self.fullname = fullname
        self.real_name = real_name

    def create_module(self, spec):
        with require_agents():
            module = import_module(self.real_name)
        self.real_spec = module.__spec__
        warnings.warn(
            f"{self.fullname} is deprecated; import {self.real_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return module

    def exec_module(self, module):
        # Returning the existing module preserves class identity. Do not execute
        # source under the old name or replace the real module's __spec__.
        module.__spec__ = self.real_spec


class LegacyModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("mobilerun."):
            return None
        real_name = legacy_module_name(fullname[len("mobilerun.") :])
        if real_name is None:
            return None
        with require_agents():
            real_spec = importlib.util.find_spec(real_name)
        if real_spec is None:
            return None
        return importlib.machinery.ModuleSpec(
            fullname,
            LegacyAliasLoader(fullname, real_name),
            origin=real_spec.origin,
            is_package=real_spec.submodule_search_locations is not None,
        )


def install_legacy_finder() -> None:
    if not any(isinstance(finder, LegacyModuleFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, LegacyModuleFinder())
