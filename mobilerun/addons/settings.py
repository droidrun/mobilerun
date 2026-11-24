"""Entry-point groups supported by the base package."""

CLI_ENTRY_POINT_GROUP = "mobilerun.cli"
OPERATOR_ENTRY_POINT_GROUP = "mobilerun.operators"

# Descriptions avoid importing plugin implementations while rendering root help.
CLI_PLUGIN_HELP = {
    "macro": "Record and replay device macros.",
    "mcp": "Manage coding-agent MCP connections.",
    "sandbox": "Deploy and run cloud functions.",
    "skills": "Manage reusable skills.",
}
