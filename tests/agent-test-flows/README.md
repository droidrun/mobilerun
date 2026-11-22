# agent-test-flows

Briefs that tell an AI agent (Claude / Codex / Gemini) how to regression-test mobilerun before a
release or PR. The agent reads this, decides the concrete tasks itself, runs them on real devices,
and reports the results. Keep these short — the agent works out the details.

## How to use
Point an agent at this folder and ask for a **smoke** pass (quick, ~10–15 min) or a **full** pass
(thorough, can take hours). The agent plans the tasks, picks models, runs them via `mobilerun run …`
(and `mobilerun device` / `mobilerun macro` / `mobilerun agent llm configure` as needed), and writes a
comparison table with screenshots. Smoke = one cheap model + a few core checks; full = sweep the
areas below.

## What to cover
- **Modes:** vision and non-vision; reasoning and non-reasoning.
- **Platforms:** Android (emulator/device) and iOS (simulator/physical via ios-portal).
- **UI actions:** open apps, tap, type, scroll, swipe, multi-step navigation, back/home.
- **Models & providers:** a representative set across Anthropic, OpenAI, Gemini, and the others in
  the registry — api-key and OAuth.
- **OAuth:** login and token refresh per provider.
- **Macros:** record a sequence (`mobilerun agent run … --save-trajectory action`) and replay it.
- **Cloud:** the cloud device backend.
- Any other mobilerun functionality worth a check.

The agent uses its judgment on the exact tasks and depth; there is no fixed script.

## Guidance
- **Verify independently** — don't trust the agent's own "done"; re-read the screen or check the
  value (e.g. `mobilerun device ui`/screenshot).
- **Capture evidence** — screenshots (and trajectories), summarized in a results table under
  `tests/agent-test-flows/reports/`.
- **Set up the device first** — install the Portal if it's missing; on physical devices keep tasks
  read-only and never destructive/paid.
- **Treat known limits as expected**, not failures (e.g. Gemini OAuth short-window rate limits).
- Optionally compare the run against a known-good ("ground-truth") release.

## Cost and cleanup

These are manual, agent-directed flows, not an automated test suite. Cloud phones, proxies, cloud tasks and LLM calls can incur charges. Agree on a budget and resource scope before running. Use test accounts and avoid destructive actions. Record created resource IDs, release any temporary phones, numbers, mailboxes and proxies afterward, and check the cloud dashboard after failures or interrupted runs. Do not delete existing shared resources.
