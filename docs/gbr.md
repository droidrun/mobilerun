# Build Remote Agent (companion, not a replacement)

**Mobilerun drives phones** (LLM agents inspect UI, tap, swipe, type). **Build Remote Agent does the inverse:** a phone spectates a *desktop* coding agent (Grok Build, Claude Code, Codex, …).

Use this when you want a spectator/veto phone on the **coding-agent host**, not when you want an agent to control a device. Keep using Mobilerun / Portal for device control. Pair `gbr-agent` on the same machine that runs the desktop coding agent.

## Install + pair

```bash
# macOS / Linux
curl -fsSL https://grokbuildremote.com/install.sh | bash
gbr-agent version          # must print v0.6.0 or newer
gbr-agent pair             # QR in browser + printed 8-char code
gbr-agent run              # leave running
```

```powershell
# Windows
irm https://grokbuildremote.com/install.ps1 | iex
gbr-agent version
gbr-agent pair
gbr-agent run
```

Phone: open [Build Remote Agent](https://grokbuildremote.com/) → **Scan QR from computer** (or type the 8-char code). Sessions appear in the app. **Unpair** in Settings before changing PCs. Force-close is not enough.

## Attach

After `gbr-agent run`:

| How | Where |
|-----|--------|
| Bot API | `http://127.0.0.1:8788` |
| MCP stdio | `node …/GrokBuildRemote-Agents/mcp/gbr-mcp/bin/gbr-mcp.js` |

```bash
curl -sS http://127.0.0.1:8788/health
curl -sS http://127.0.0.1:8788/v1/sessions
```

Phone is spectator + veto, not orchestrator. Do not commit mailbox keys. Phone **Settings → Bot API** is the only place a relay key is copied.

MCP clone:

```bash
git clone https://github.com/LinespottingOrg/GrokBuildRemote-Agents.git
cd GrokBuildRemote-Agents/mcp/gbr-mcp && npm install
node bin/gbr-mcp.js --diagnose
```

Protocol `gbr/1`. Independent product by Linespotting AB. Not affiliated with xAI or SpaceX.

Docs: https://github.com/LinespottingOrg/GrokBuildRemote-Agents/blob/main/docs/BOT-API.md
Website: https://grokbuildremote.com/

Do not merge this with Portal accessibility, reverse WebSocket, or Mobilerun Cloud. Different product, different protocol (`gbr/1`).
