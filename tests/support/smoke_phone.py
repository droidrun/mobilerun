"""Conservative cleanup for minute phones created by the smoke suite."""

import asyncio
import sys

from mobilerun.clients.cloud import devices
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.models import Billing, Plan


def _candidate(listed, name):
    matches = [item for item in listed if item.name == name]
    if len(matches) > 1:
        raise AssertionError(
            f"Ambiguous phone name {name!r}: IDs {[item.id for item in matches]}"
        )
    if not matches:
        return None
    item = matches[0]
    if item.billing_strategy != Plan.MINUTE:
        raise AssertionError(
            f"Cannot confirm minute billing for {name!r}, ID {item.id}"
        )
    return item.id


async def _recover_phone(name):
    terminated = None
    async with cloud_client(billing=Billing.INCLUDED) as client:
        for _ in range(30):
            listed = await devices.list_devices(billing=Billing.INCLUDED, client=client)
            candidate = _candidate(listed, name)
            if terminated is None and candidate is not None:
                await devices.terminate(
                    candidate, billing=Billing.INCLUDED, client=client
                )
                terminated = candidate
            elif terminated is not None:
                if candidate not in (None, terminated):
                    raise AssertionError(
                        f"Phone {name!r} changed IDs: {terminated} to {candidate}"
                    )
                if not any(item.id == terminated for item in listed):
                    return
            await asyncio.sleep(1)
    message = f"unresolved: phone named {name!r} may exist"
    if terminated is not None:
        message += f"; could not verify removal of ID {terminated}"
    original = sys.exception()
    if original is None:
        raise AssertionError(message)
    original.add_note(message)


async def close_smoke_phone(device, name):
    """Close first; recover by name only when create never returned a receipt."""
    try:
        try:
            await device.close()
        finally:
            # name is the cached *server response*, not the constructor argument.
            # A returned name with a cleared ID means readiness rollback succeeded.
            if device.cloud_id is None and device.name is None:
                await _recover_phone(name)
    except Exception as exc:
        exc.add_note(f"Smoke phone cleanup failed: name={name!r}, id={device.cloud_id}")
        raise
