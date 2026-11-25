"""Install, remove and inspect the Mobilerun MCP server."""

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from mobilerun.config.settings import get_settings
from mobilerun.mcp.settings import DEFAULT_MCP_FORMATS, MCPFormat, get_mcp_settings
from mobilerun.mcp.utils import check_path, edit, parse_config, read_source, write_file


def config_path(agent: str, scope: str) -> Path:
    paths = get_settings().MOBILERUN_AGENTS.get(agent)
    if paths is None or paths.mcp is None:
        raise ValueError(f"Unknown MCP agent: {agent}")
    if scope not in ("user", "project"):
        raise ValueError(f"Unknown scope: {scope}")
    relative = paths.mcp[scope == "project"]
    if relative is None:
        raise ValueError(f"{agent} does not support {scope} scope")
    path = (Path.home() if scope == "user" else Path.cwd()) / relative
    try:
        fmt = _agent_format(agent)
    except ValueError:
        fmt = None
    if fmt and fmt.jsonc_alternative:
        alternative = path.with_suffix(".jsonc")
        if alternative.exists() or alternative.is_symlink():
            return alternative
    return path


def _agent_format(agent: str) -> MCPFormat:
    settings = get_settings()
    paths = settings.MOBILERUN_AGENTS.get(agent)
    if paths is not None:
        if paths.mcp_format is not None:
            return MCPFormat.model_validate(paths.mcp_format)
        if "MOBILERUN_AGENTS" not in settings.model_fields_set:
            fmt = DEFAULT_MCP_FORMATS.get(agent)
            if fmt is not None:
                return fmt
    raise ValueError(f"No MCP format configured for agent: {agent}")


def _server_entries(
    data: MutableMapping[str, Any], fmt: MCPFormat, create: bool
) -> MutableMapping[str, Any]:
    key = fmt.servers_key
    if key not in data:
        if not create:
            return {}
        data[key] = {}
    servers = data[key]
    if not isinstance(servers, MutableMapping):
        raise ValueError(f"Invalid config: {key} must be an object")
    legacy = any(
        isinstance(value, MutableMapping)
        and any(field in value for field in ("type", "url", "command"))
        for value in servers.values()
    )
    if fmt.nested_servers and ("servers" in servers or not legacy):
        if "servers" not in servers:
            if not create:
                return {}
            servers["servers"] = {}
        servers = servers["servers"]
        if not isinstance(servers, MutableMapping):
            raise ValueError(f"Invalid config: {key}.servers must be an object")
    return servers


def _server_entry(fmt: MCPFormat, scope: str, api_key: bool) -> dict[str, Any]:
    if scope == "user" and api_key and not fmt.user_api_key:
        raise ValueError(
            "This agent's user scope cannot safely reference an API key; use --project or OAuth."
        )
    entry: dict[str, Any] = {"url": get_mcp_settings().MOBILERUN_MCP_URL}
    if fmt.entry_type:
        entry["type"] = fmt.entry_type
    if api_key:
        if fmt.bearer_token_env_var:
            entry["bearer_token_env_var"] = "MOBILERUN_API_KEY"
        elif fmt.api_key_ref:
            entry["headers"] = {"Authorization": f"Bearer {fmt.api_key_ref}"}
        else:
            raise ValueError(
                "This agent does not support API key references; use OAuth."
            )
        if fmt.disable_oauth_with_api_key:
            entry["oauth"] = False
    return entry


def status(agent: str, scope: str) -> str:
    name = get_mcp_settings().MOBILERUN_MCP_SERVER_NAME
    path = config_path(agent, scope)
    return (
        "installed"
        if name
        in _server_entries(
            parse_config(path, read_source(path)), _agent_format(agent), False
        )
        else "not installed"
    )


def install(
    agent: str, scope: str, *, api_key: bool = False, force: bool = False
) -> str:
    fmt = _agent_format(agent)
    return _update(
        config_path(agent, scope), fmt, _server_entry(fmt, scope, api_key), force=force
    )


def uninstall(agent: str, scope: str) -> str:
    return _update(config_path(agent, scope), _agent_format(agent), None)


def _update(
    path: Path, fmt: MCPFormat, entry: dict[str, Any] | None, *, force: bool = False
) -> str:
    """Apply one install or removal, re-evaluating conflicts after an editor save."""
    name = get_mcp_settings().MOBILERUN_MCP_SERVER_NAME
    for _ in range(3):
        snapshot = read_source(path)
        data = parse_config(path, snapshot)
        servers = _server_entries(data, fmt, entry is not None)
        keys = [fmt.servers_key]
        if fmt.nested_servers and data.get(fmt.servers_key) is not servers:
            keys.append("servers")
        keys.append(name)
        if path.suffix == ".jsonc":
            check_path(snapshot or "{}", keys)
        if entry is None:
            if name not in servers:
                return "not installed"
            del servers[name]
        else:
            if name in servers:
                if servers[name] == entry:
                    return "exists"
                if not force:
                    raise ValueError(
                        f"{name} exists with different settings; use --force to replace it"
                    )
            servers[name] = entry
        content = None
        if path.suffix == ".jsonc":
            content = edit(snapshot or "{}", keys, entry, remove=entry is None)
        if write_file(path, data, content, snapshot):
            return "removed" if entry is None else "installed"
    raise ValueError(
        f"Config {path} changed repeatedly during update; retry the command"
    )
