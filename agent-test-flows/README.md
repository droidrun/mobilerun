# agent-test-flows

Briefs that tell an AI agent (Claude / Codex / Gemini) how to regression-test mobilerun before a
release or PR. The agent reads this, decides the concrete tasks itself, runs them on real devices,
and reports the results. Keep these short — the agent works out the details.

## How to use
Point an agent at this folder and ask for a **smoke** pass (quick, ~10–15 min) or a **full** pass
(thorough, can take hours). The agent plans the tasks, picks models, runs them via `mobilerun run …`
(and `mobilerun device` / `mobilerun macro` as needed), and writes a
comparison table with screenshots. Smoke = one cheap model + a few core checks; full = sweep the
areas below.

## What to cover
- **Modes:** vision and non-vision; reasoning and non-reasoning.
- **Platforms:** Android (emulator/device) and iOS (simulator/physical via ios-portal).
- **UI actions:** open apps, tap, type, scroll, swipe, multi-step navigation, back/home.
- **Models & providers:** a representative set across Anthropic, OpenAI, Gemini, and the others in
  the registry — configured by API key via `mobilerun configure`.
- **Macros:** record a sequence (`mobilerun run … --save-trajectory`) and replay it.
- **Cloud:** the cloud device backend.
- Any other mobilerun functionality worth a check.

The agent uses its judgment on the exact tasks and depth; there is no fixed script.

## Guidance
- **Verify independently** — don't trust the agent's own "done"; re-read the screen or check the
  value (e.g. `mobilerun device ui`/screenshot).
- **Capture evidence** — screenshots (and trajectories), summarized in a results table under
  `reports/`.
- **Set up the device first** — install the Portal if it's missing; on physical devices keep tasks
  read-only and never destructive/paid.
- **Treat known limits as expected**, not failures (e.g. provider free-tier short-window rate limits).
- Optionally compare the run against a known-good ("ground-truth") release.
