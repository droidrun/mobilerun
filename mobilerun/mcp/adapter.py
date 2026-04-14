"""Adapter to convert MCP tools to Droidrun custom tool format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from droidrun.mcp.client import MCPClientManager


def schema_to_parameters(input_schema: dict) -> dict[str, dict[str, Any]]:
    """Convert JSON Schema properties to Droidrun parameters format.

    Args:
        input_schema: JSON Schema with "properties" and "required" fields.

    Returns:
        Dict mapping param name to {"type": ..., "required": ..., "default": ...}.
    """
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    parameters: dict[str, dict[str, Any]] = {}
    for prop_name, prop_info in properties.items():
        param: dict[str, Any] = {
            "type": prop_info.get("type", "string"),
            "required": prop_name in required,
        }
        if "default" in prop_info:
            param["default"] = prop_info["default"]
        if "description" in prop_info:
            param["description"] = prop_info["description"]
        parameters[prop_name] = param

    return parameters


def mcp_to_droidrun_tools(mcp_manager: "MCPClientManager") -> dict[str, dict[str, Any]]:
    """Convert discovered MCP tools to Droidrun custom tool format."""
    custom_tools: dict[str, dict[str, Any]] = {}

    for tool_name, tool_info in mcp_manager.tools.items():
        wrapper = _create_tool_wrapper(tool_name, mcp_manager)
        custom_tools[tool_name] = {
            "parameters": schema_to_parameters(tool_info.input_schema),
            "description": tool_info.description,
            "function": wrapper,
        }

    return custom_tools


def _create_tool_wrapper(tool_name: str, manager: "MCPClientManager"):
    """Create async wrapper function for an MCP tool."""

    async def mcp_tool_wrapper(*, ctx=None, **kwargs) -> str:
        result = await manager.call_tool(tool_name, kwargs)

        if hasattr(result, "content") and result.content:
            text_parts = []
            for block in result.content:
                if hasattr(block, "text") and block.text:
                    text_parts.append(block.text)
            if text_parts:
                return "\n".join(text_parts)

        return str(result)

    mcp_tool_wrapper.__name__ = f"mcp_{tool_name}"
    return mcp_tool_wrapper
