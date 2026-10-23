"""MCP Client Manager - handles connections to MCP servers."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mobilerun.mcp.config import MCPConfig, MCPServerConfig

logger = logging.getLogger("mobilerun")


@dataclass
class MCPToolInfo:
    """Cached metadata about an MCP tool."""

    server_name: str
    original_name: str
    description: str
    input_schema: dict


class MCPClientManager:
    """Manages MCP server connections with lazy initialization."""

    def __init__(self, config: "MCPConfig"):
        self.config = config
        self._tools: dict[str, MCPToolInfo] = {}
        self._server_tools: dict[str, list[str]] = {}
        self._sessions: dict = {}  # type: ignore
        self._shutdown_events: dict[str, asyncio.Event] = {}
        self._connection_tasks: dict[str, asyncio.Task] = {}

    async def discover_tools(self) -> dict[str, MCPToolInfo]:
        """Discover tools from all configured servers."""
        if not self.config.enabled:
            return {}

        for server_name, server_config in self.config.servers.items():
            if not server_config.enabled:
                continue
            try:
                await self._discover_server_tools(server_name, server_config)
            except Exception as e:
                logger.warning(f"MCP '{server_name}': discovery failed: {e}")

        if self._tools:
            logger.info(
                f"MCP: discovered {len(self._tools)} tools from {len(self._server_tools)} servers"
            )
        return self._tools

    async def _discover_server_tools(
        self, server_name: str, config: "MCPServerConfig"
    ) -> None:
        """Connect temporarily to fetch tool schemas."""
        # Lazy import to avoid circular dependency with mobilerun.mcp package
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env if config.env else None,
        )

        server_tool_names: list[str] = []

        async with AsyncExitStack() as stack:
            transport = await stack.enter_async_context(stdio_client(server_params))
            read, write = transport
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            response = await session.list_tools()
            prefix = config.prefix if config.prefix is not None else f"{server_name}_"

            for tool in response.tools:
                if not self._should_include_tool(tool.name, config):
                    continue

                tool_name = f"{prefix}{tool.name}" if prefix else tool.name
                self._tools[tool_name] = MCPToolInfo(
                    server_name=server_name,
                    original_name=tool.name,
                    description=tool.description or f"MCP tool: {tool.name}",
                    input_schema=tool.inputSchema,
                )
                server_tool_names.append(tool_name)

        self._server_tools[server_name] = server_tool_names
        logger.debug(f"MCP '{server_name}': discovered {len(server_tool_names)} tools")

    def _should_include_tool(self, tool_name: str, config: "MCPServerConfig") -> bool:
        """Check if tool passes include/exclude filters."""
        if tool_name in config.exclude_tools:
            return False
        if config.include_tools is not None:
            return tool_name in config.include_tools
        return True

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call an MCP tool, connecting lazily if needed."""
        if tool_name not in self._tools:
            raise ValueError(f"Unknown MCP tool: {tool_name}")

        tool_info = self._tools[tool_name]
        server_name = tool_info.server_name

        if server_name not in self._sessions:
            await self._connect_server(server_name)

        session = self._sessions[server_name]
        return await session.call_tool(tool_info.original_name, arguments)

    async def _connect_server(self, server_name: str) -> None:
        """Establish persistent connection to a server."""
        # Lazy import to avoid circular dependency with mobilerun.mcp package
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        config = self.config.servers[server_name]
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env if config.env else None,
        )

        logger.debug(f"MCP '{server_name}': connecting...")

        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        shutdown = asyncio.Event()

        # The transport must be closed by the task that opened it, so one task
        # owns the connection until disconnect_all() signals shutdown.
        async def own_connection() -> None:
            try:
                async with AsyncExitStack() as stack:
                    read, write = await stack.enter_async_context(
                        stdio_client(server_params)
                    )
                    session = await stack.enter_async_context(
                        ClientSession(read, write)
                    )
                    await session.initialize()
                    ready.set_result(session)
                    await shutdown.wait()
            except BaseException as e:
                if ready.done():
                    raise
                if isinstance(e, asyncio.CancelledError):
                    ready.cancel()
                    raise
                ready.set_exception(e)

        task = asyncio.create_task(own_connection(), name=f"mcp:{server_name}")
        try:
            session = await ready
        except BaseException:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise

        self._sessions[server_name] = session
        self._shutdown_events[server_name] = shutdown
        self._connection_tasks[server_name] = task
        logger.info(f"MCP '{server_name}': connected")

    async def disconnect_all(self) -> None:
        """Clean up all server connections."""
        for shutdown in self._shutdown_events.values():
            shutdown.set()

        for server_name, task in list(self._connection_tasks.items()):
            try:
                await task
                logger.debug(f"MCP '{server_name}': disconnected")
            except Exception as e:
                logger.warning(f"MCP '{server_name}': disconnect error: {e}")

        self._sessions.clear()
        self._shutdown_events.clear()
        self._connection_tasks.clear()

    @property
    def tools(self) -> dict[str, MCPToolInfo]:
        return self._tools

    @property
    def connected_servers(self) -> list[str]:
        return list(self._sessions.keys())
