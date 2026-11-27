# state/

**Purpose:** UI state models and platform-specific state providers.

**What belongs here / what doesn't:** Put state fetching, element parsing, filtering, and formatting here, not feature logic.

**Contents:**

- `__init__.py` — Public models and provider exports.
- `android.py` — Android state provider.
- `base.py` — Provider interface and screenshot resizing helpers.
- `elements.py` — UI element model.
- `filters/` — UI tree filters.
- `formatters/` — UI tree formatters.
- `ios.py` — iOS state provider.
- `options.py` — Typed options, vision policy and provider factories.
- `screenshot.py` — Screenshot-only provider.
- `stealth.py` — Randomized coordinate resolution.
- `ui_state.py` — Parsed UI state and element resolution.
- `view.py` — Device-facing cached state view.

**Examples:**

Run in an async function with an existing `device`.

```python
state = await device.state.refresh()
print(state.formatted_text)
assert device.state.current is state
```

`StateProvider` declares all consumed capability flags, including
`use_normalized` and `requires_active_contract_for_coords`. The public state
controller exposes only `refresh()`, `current`, and `capabilities`. Read snapshot
fields and coordinate methods from the returned `UIState` or
`device.state.current`. No snapshot fields are forwarded by the controller.
Configure devices with `state_options=`, a `state_provider=` instance, or a
`state_provider_factory=(device, options) -> StateProvider`. An instance and
factory are mutually exclusive. Provider selection and coordinate plumbing are
private framework hooks.

**Depends on / used by:** Uses shared packages and platform drivers; used by devices and features.

See [devices](../devices/README.md), [filters](filters/README.md), and [formatters](formatters/README.md).
