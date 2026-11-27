# MCP

Install the public Mobilerun MCP server into a coding agent:

```sh
mobilerun mcp install --agent claude-code --project
mobilerun mcp install --agent codex --api-key
mobilerun mcp list
mobilerun mcp uninstall --agent codex
```

OAuth is the default. After installing, follow the printed agent-specific sign-in step; for example, Codex uses `codex mcp login mobilerun`. For API-key authentication, set `MOBILERUN_API_KEY` in the agent's environment; only its reference is saved. Claude Code user scope supports OAuth only. Use `--force` to replace an existing different Mobilerun entry. Codex project configuration also requires project trust.

`MOBILERUN_MCP_URL` overrides the server URL (an empty value uses the default).
`MOBILERUN_MCP_SERVER_NAME` overrides the server key and sign-in hint name.
Settings are read from the environment on each operation; use the same server
name when checking or removing an existing installation.

| File | Purpose |
| --- | --- |
| `installer.py` | Resolve agent paths and install, remove and inspect the server. |
| `settings.py` | Environment-backed MCP URL and server name. |
| `utils.py` | Read and atomically write files, preserving JSONC comments and formatting. |
| `cli/` | MCP commands. |

Agent MCP format defaults and OAuth instructions live in `settings.py`. The shared `MOBILERUN_AGENTS` JSON registry can replace these using each agent’s `mcp_format` object; the replacement registry must supply a format for every MCP-enabled agent.
