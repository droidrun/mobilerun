"""Save, use and delete a demo cloud vault credential on a local Android device.

Vault access uses your cloud account; the local device is free. The demo opens
Settings search and types a non-secret username. Run: python -m tests.demos.vault_demo [--serial emulator-5554].
"""

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mobilerun.core.credentials import CloudCredentials  # noqa: E402
from mobilerun.core.devices import AndroidDevice  # noqa: E402

PACKAGE = "com.example.demo"


async def main(serial: str | None = None) -> None:
    vault = CloudCredentials()
    device = None
    bound = False
    name = f"mobilerun-demo-{uuid4().hex}"
    attempted_save = False
    settings_open = False
    try:
        device = AndroidDevice(serial)
        await vault.bind(device)
        bound = True
        for credential in await vault.list(PACKAGE):
            print(
                f"{credential['package']}: {credential['name']} ({', '.join(credential['fields'])})"
            )
        # A unique name avoids updating or deleting anyone's existing credential.
        attempted_save = True
        await vault.save(
            PACKAGE, name, {"username": "demo-user", "password": "demo-password"}
        )
        # Settings may reopen on its previous search page or its home page.
        await device.start_app("com.android.settings")
        settings_open = True
        await device.wait(1)
        state = await device.state.refresh()
        edit = next(
            (item for item in state.elements if item.get("className") == "EditText"),
            None,
        )
        if edit is None:
            search = next(
                (
                    item
                    for item in state.elements
                    if "search"
                    in " ".join(
                        str(item.get(key, ""))
                        for key in ("text", "contentDescription", "resourceId")
                    ).lower()
                    and "index" in item
                ),
                None,
            )
            if search is None:
                raise RuntimeError("Settings search entry not found")
            await device.tap_element(search["index"])
            await device.wait(1)
            state = await device.state.refresh()
            edit = next(
                (
                    item
                    for item in state.elements
                    if item.get("className") == "EditText"
                ),
                None,
            )
        if edit is None:
            raise RuntimeError("Settings search input not found")
        clear = next(
            (
                item
                for item in state.elements
                if "clear" in str(item.get("resourceId", "")).lower()
                and "index" in item
            ),
            None,
        )
        if clear is not None:
            await device.tap_element(clear["index"])
            state = await device.state.refresh()
            edit = next(
                (
                    item
                    for item in state.elements
                    if item.get("className") == "EditText"
                ),
                None,
            )
        if edit is None or "index" not in edit:
            raise RuntimeError("Settings search input not found")
        await device.tap_element(edit["index"])
        await vault.type_field(PACKAGE, "username", name=name)
        state = await device.state.refresh()
        if state.focused_text != "demo-user":
            raise RuntimeError("Demo username was not visible in the focused field")
        print(f"Field now contains: {state.focused_text}")
    finally:
        try:
            if settings_open and device is not None:
                await device.press_button("home")
        finally:
            try:
                if attempted_save:
                    await vault.delete(PACKAGE, name)
            finally:
                try:
                    if bound:
                        await vault.unbind()
                finally:
                    try:
                        if device is not None:
                            await device.close()
                    finally:
                        await vault.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default=None, help="Local Android device serial")
    asyncio.run(main(parser.parse_args().serial))
