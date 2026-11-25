# core/

**Purpose:** Shared device, state, and operator abstractions.

**What belongs here / what doesn't:** Put platform-neutral contracts and device orchestration here, not feature or CLI logic.

**Contents:**

- `__init__.py` — Public device and Workspace exports.
- `workspace.py` — Account-wide file and app classmethods; see [Workspace](WORKSPACE.md).
- `devices/` — Device interfaces, implementations, and wrappers.
- `credentials/` — Credential providers that bind themselves to devices.
- [`webhooks/`](webhooks/README.md) — Organization webhook endpoint management.
- `phone_number/` — Phone number interface and server-bound cloud numbers.
- `mailbox/` — Mailbox interface and locally associated cloud mailboxes.
- `proxy/` — Connect/SOCKS5 parameters; selection and switching live on phones.
- `state/` — UI state, providers, filters, and formatters.
- `operator/` — Device operator contract.

**Examples:**

Use an existing `device`.

```python
state = await device.state.refresh()
print(state.formatted_text)
```

**Depends on / used by:** Uses clients and other shared packages; used by devices and features.

See [devices](devices/README.md), [state](state/README.md), and [Workspace](WORKSPACE.md).
