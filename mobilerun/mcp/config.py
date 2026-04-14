"""MCP configuration models."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""

    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    prefix: Optional[str] = None
    enabled: bool = True
    include_tools: Optional[List[str]] = None
    exclude_tools: List[str] = field(default_factory=list)


@dataclass
class MCPConfig:
    """MCP client configuration."""

    enabled: bool = False
    servers: Dict[str, MCPServerConfig] = field(default_factory=dict)
