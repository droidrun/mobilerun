"""ToolRegistry — single source of truth for available tools.

Central registry for agent-callable tools.
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set

from droidrun.agent.action_result import ActionResult

if TYPE_CHECKING:
    from llama_index.core.workflow import Context as WorkflowContext

    from droidrun.agent.action_context import ActionContext

logger = logging.getLogger("droidrun")


@dataclass
class ToolEntry:
    fn: Callable
    params: Dict[str, Any]
    description: str
    deps: Optional[Set[str]] = None


class ToolRegistry:
    """Central registry of all agent-callable tools."""

    def __init__(self) -> None:
        self.tools: Dict[str, ToolEntry] = {}

    # -- registration --------------------------------------------------------

    def register(
        self,
        name: str,
        fn: Callable,
        params: Dict[str, Any],
        description: str,
        deps: Optional[Set[str]] = None,
    ) -> None:
        self.tools[name] = ToolEntry(
            fn=fn, params=params, description=description, deps=deps
        )

    def register_from_dict(self, tools_dict: Dict[str, Any]) -> None:
        """Register tools from the existing ``{name: {parameters, description, function}}`` format."""
        for name, spec in tools_dict.items():
            deps = spec.get("deps")
            if deps is not None:
                deps = set(deps)
            self.register(
                name=name,
                fn=spec["function"],
                params=spec.get("parameters", {}),
                description=spec.get("description", f"Tool: {name}"),
                deps=deps,
            )

    def disable(self, tool_names: list[str]) -> None:
        """Remove tools by name (silently ignores unknown names)."""
        for name in tool_names:
            self.tools.pop(name, None)

    def disable_unsupported(self, capabilities: Set[str]) -> None:
        """Remove tools whose ``deps`` set is not satisfied by *capabilities*.

        Tools with ``deps=None`` (custom tools, MCP tools, etc.) are kept.
        """
        to_remove = [
            name
            for name, entry in self.tools.items()
            if entry.deps is not None and not entry.deps <= capabilities
        ]
        if to_remove:
            logger.debug(f"Disabling unsupported tools: {to_remove}")
        self.disable(to_remove)

    # -- query ---------------------------------------------------------------

    def get_signatures(self, exclude: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Return ``{name: {parameters, description}}`` for prompt building.

        Args:
            exclude: Optional set of tool names to omit.  Used by
                     ExecutorAgent to hide flow-control tools (remember,
                     complete) that only FastAgent should see.
        """
        exclude = exclude or set()
        return {
            name: {"parameters": entry.params, "description": entry.description}
            for name, entry in self.tools.items()
            if name not in exclude
        }

    # -- execution -----------------------------------------------------------

    async def execute(
        self,
        name: str,
        args: Dict[str, Any],
        ctx: "ActionContext",
        workflow_ctx: "Optional[WorkflowContext]" = None,
    ) -> ActionResult:
        """Dispatch action by name.

        All actions receive ``ctx=ctx`` as a keyword argument.

        Args:
            workflow_ctx: Optional llama-index workflow Context. When provided,
                a ``ToolExecutionEvent`` is streamed after execution.

        Handles:
        - Unknown tool names
        - Sync vs async functions
        - TypeError (bad arguments)
        - General exceptions
        """
        if name not in self.tools:
            result = ActionResult(
                success=False,
                summary=(
                    f"Unknown tool: {name}. " f"Available: {list(self.tools.keys())}"
                ),
            )
            self._emit_event(workflow_ctx, name, args, result)
            return result

        entry = self.tools[name]
        try:
            if inspect.iscoroutinefunction(entry.fn):
                result = await entry.fn(**args, ctx=ctx)
            else:
                result = entry.fn(**args, ctx=ctx)
        except TypeError as e:
            result = ActionResult(
                success=False,
                summary=f"Invalid arguments for {name}: {e}",
            )
            self._emit_event(workflow_ctx, name, args, result)
            return result
        except Exception as e:
            result = ActionResult(
                success=False,
                summary=f"Failed to execute {name}: {e}",
            )
            self._emit_event(workflow_ctx, name, args, result)
            return result

        # Normalise the return value into ActionResult
        if isinstance(result, ActionResult):
            action_result = result
        elif isinstance(result, tuple):
            action_result = ActionResult(success=result[0], summary=str(result[1]))
        elif isinstance(result, str):
            success = not result.startswith("Failed")
            action_result = ActionResult(success=success, summary=result)
        else:
            action_result = ActionResult(
                success=True, summary=str(result) if result else "Done"
            )

        self._emit_event(workflow_ctx, name, args, action_result)
        return action_result

    @staticmethod
    def _emit_event(
        workflow_ctx: "Optional[WorkflowContext]",
        name: str,
        args: Dict[str, Any],
        result: ActionResult,
    ) -> None:
        if workflow_ctx is None:
            return
        from droidrun.agent.common.events import ToolExecutionEvent

        workflow_ctx.write_event_to_stream(
            ToolExecutionEvent(
                tool_name=name,
                tool_args=args,
                success=result.success,
                summary=result.summary,
            )
        )

    # -- prompt helpers ------------------------------------------------------

    def get_tool_descriptions_xml(self, exclude: Optional[Set[str]] = None) -> str:
        """Build XML ``<functions>`` block for FastAgent system prompt."""
        exclude = exclude or set()
        lines = ["<functions>"]
        for name, entry in self.tools.items():
            if name in exclude:
                continue
            lines.append(f"<function>{self._spec_to_json(name, entry)}</function>")
        lines.append("</functions>")
        return "\n".join(lines)

    def get_tool_descriptions_text(self, exclude: Optional[Set[str]] = None) -> str:
        """Build text tool descriptions for executor prompts."""
        exclude = exclude or set()
        descriptions = []
        for name, entry in self.tools.items():
            if name in exclude:
                continue
            params = self._format_params(entry.params)
            descriptions.append(f"- {name}({params}): {entry.description}")
        return "\n".join(descriptions)

    def get_param_types(self, exclude: Optional[Set[str]] = None) -> Dict[str, str]:
        """Build a flat ``{param_name: type_string}`` map for XML coercion.

        Note: parameter names are global (not per-tool).
        """
        exclude = exclude or set()
        param_types: Dict[str, str] = {}
        for name, entry in self.tools.items():
            if name in exclude:
                continue
            for param_name, param_info in entry.params.items():
                param_types[param_name] = param_info.get("type", "string")
        return param_types

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _format_params(parameters: Dict[str, Any]) -> str:
        parts = []
        for name, info in parameters.items():
            type_str = info.get("type", "string")
            if info.get("required", True):
                parts.append(f"{name}: {type_str}")
            else:
                default = info.get("default", "None")
                parts.append(f"{name}: {type_str} = {default}")
        return ", ".join(parts)

    @staticmethod
    def _spec_to_json(name: str, entry: ToolEntry) -> str:
        parameters = entry.params
        properties = {}
        required = []
        for param_name, param_info in parameters.items():
            properties[param_name] = {"type": param_info.get("type", "string")}
            if param_info.get("description"):
                properties[param_name]["description"] = param_info["description"]
            if param_info.get("required", True):
                required.append(param_name)
            if "default" in param_info:
                properties[param_name]["default"] = param_info["default"]

        tool_dict = {
            "name": name,
            "description": entry.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
        return json.dumps(tool_dict, separators=(",", ":"))
