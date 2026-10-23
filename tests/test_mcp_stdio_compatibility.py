import asyncio
import os
import sys
import time

import mobilerun.mcp.client as mcp_client
from mobilerun.mcp.adapter import mcp_to_mobilerun_tools
from mobilerun.mcp.client import MCPClientManager
from mobilerun.mcp.config import MCPConfig, MCPServerConfig


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_real_stdio_server_discovery_filtering_invocation_and_cleanup(tmp_path):
    server_script = tmp_path / "compatibility_server.py"
    server_script.write_text("""
import os

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

server = FastMCP("mobilerun-compatibility-test")


@server.tool()
def describe(value: str) -> list[TextContent]:
    return [
        TextContent(type="text", text=f"value={value}"),
        TextContent(type="text", text=f"transport={os.environ['MCP_TEST_TRANSPORT']}"),
        TextContent(type="text", text=f"pid={os.getpid()}"),
    ]


@server.tool()
def excluded() -> str:
    return "excluded"


@server.tool()
def not_included() -> str:
    return "not included"


if __name__ == "__main__":
    server.run(transport="stdio")
""".lstrip())

    manager = MCPClientManager(
        MCPConfig(
            enabled=True,
            servers={
                "fixture": MCPServerConfig(
                    command=sys.executable,
                    args=[str(server_script)],
                    env={"MCP_TEST_TRANSPORT": "stdio"},
                    prefix="compat_",
                    include_tools=["describe", "excluded"],
                    exclude_tools=["excluded"],
                )
            },
        )
    )

    async def exercise_manager() -> tuple[str, int]:
        pid = 0
        try:
            discovered = await manager.discover_tools()

            assert list(discovered) == ["compat_describe"]
            assert manager.connected_servers == []
            assert discovered["compat_describe"].original_name == "describe"
            assert discovered["compat_describe"].input_schema["required"] == ["value"]

            tools = mcp_to_mobilerun_tools(manager)
            assert list(tools) == ["compat_describe"]
            assert tools["compat_describe"]["parameters"]["value"] == {
                "type": "string",
                "required": True,
            }

            result = await tools["compat_describe"]["function"](value="hello")
            lines = result.splitlines()
            assert lines[:2] == ["value=hello", "transport=stdio"]
            pid = int(lines[2].removeprefix("pid="))
            assert manager.connected_servers == ["fixture"]
            assert _process_exists(pid)
            return result, pid
        finally:
            await manager.disconnect_all()
            assert manager.connected_servers == []

    result, server_pid = asyncio.run(exercise_manager())

    assert result.startswith("value=hello\ntransport=stdio\npid=")
    deadline = time.monotonic() + 2
    while _process_exists(server_pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not _process_exists(server_pid)


def test_persistent_connection_closes_from_a_different_task(tmp_path, monkeypatch):
    server_script = tmp_path / "pid_server.py"
    server_script.write_text("""
import os

from mcp.server.fastmcp import FastMCP

server = FastMCP("mobilerun-task-test")


@server.tool()
def pid() -> str:
    return str(os.getpid())


if __name__ == "__main__":
    server.run(transport="stdio")
""".lstrip())

    manager = MCPClientManager(
        MCPConfig(
            enabled=True,
            servers={
                "fixture": MCPServerConfig(
                    command=sys.executable, args=[str(server_script)]
                )
            },
        )
    )

    async def exercise_manager() -> int:
        await manager.discover_tools()
        # Agent steps call tools and finalization disconnects in separate tasks.
        result = await asyncio.create_task(manager.call_tool("fixture_pid", {}))
        await asyncio.create_task(manager.disconnect_all())
        assert manager.connected_servers == []
        return int(result.content[0].text)

    warnings = []
    monkeypatch.setattr(
        mcp_client.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    server_pid = asyncio.run(exercise_manager())

    assert warnings == []
    deadline = time.monotonic() + 2
    while _process_exists(server_pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not _process_exists(server_pid)
