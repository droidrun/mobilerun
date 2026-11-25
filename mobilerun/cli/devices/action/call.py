"""Inspect and invoke actions on a selected device's extensions."""

import json
from collections.abc import Mapping
from typing import Any

import click

from mobilerun.cli.base.menu import is_interactive
from mobilerun.cli.base.prompts import SelectChoice
from mobilerun.cli.base.prompts import text_prompt_async as text_prompt
from mobilerun.cli.base.run import coro
from mobilerun.cli.base.safety import confirm_action_async
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.cli.devices.base.pickers import choose_value_async as choose_value
from mobilerun.config.settings import get_settings
from mobilerun.core.extensions import Action
from mobilerun.errors.base import MobilerunError
from mobilerun.models.billing import Billing
from mobilerun.models.context import JSONValue


def _convert(value: str, schema: Mapping[str, Any], name: str) -> JSONValue:
    types = [schema.get("type")]
    types.extend(item.get("type") for item in schema.get("anyOf", []))
    if "string" in types or types == [None]:
        return value
    try:
        parsed: JSONValue = json.loads(value)
    except json.JSONDecodeError:
        raise click.ClickException(f"Invalid value for {name}") from None
    return parsed


async def _arguments(action: Action, supplied: tuple[str, ...]) -> dict[str, JSONValue]:
    properties: Mapping[str, Any] = action.input_schema.get("properties", {})
    values: dict[str, str] = {}
    for item in supplied:
        key, separator, value = item.partition("=")
        if not separator or not key or key not in properties or key in values:
            raise click.ClickException("Arguments must be unique known key=value pairs")
        values[key] = value
    for name in action.input_schema.get("required", []):
        if name not in values:
            if not is_interactive():
                raise click.ClickException(f"Missing required argument: {name}")
            values[name] = await text_prompt(
                name, secret=name in action.sensitive_parameters
            )
    return {
        name: _convert(value, properties[name], name) for name, value in values.items()
    }


@click.command("call")
@click.argument("action", required=False)
@click.option(
    "--arg",
    "arguments",
    multiple=True,
    help="Action argument as key=value (repeatable).",
)
@click.option("--json", "json_output", is_flag=True, help="Output JSON.")
@click.option("--yes", "-y", is_flag=True, help="Confirm a chargeable action.")
@device_command
@coro
async def call(
    action: str | None,
    arguments: tuple[str, ...],
    json_output: bool,
    yes: bool,
    target: DeviceTarget,
) -> None:
    """Call an action on the selected device."""
    try:
        async with target.open() as device:
            selected = await choose_value(
                action,
                "action",
                (
                    [
                        SelectChoice(entry.name, entry.name, entry.description)
                        for entry in device.actions.list(audience="cli")
                    ]
                    if action is None and is_interactive()
                    else []
                ),
            )
            entry = device.actions.get(selected)
            if not entry.public:
                raise click.ClickException(f"Action {selected} is restricted")
            if entry.name == "sms.send" or (
                entry.name.endswith(".send")
                and device.extensions[entry.owner].role == "phone_number"
            ):
                billing = get_settings().MOBILERUN_BILLING
                if billing is not Billing.PAID:
                    raise click.ClickException(
                        "Sending SMS may incur charges; set MOBILERUN_BILLING=paid."
                    )
                await confirm_action_async(
                    "send SMS",
                    billing.value,
                    "Sending SMS may incur charges.",
                    yes=yes,
                    err=True,
                )
            result = await device.actions.call(
                selected, await _arguments(entry, arguments), audience="cli"
            )
            serialized = entry.serialize(result)
            click.echo(
                json.dumps(serialized, ensure_ascii=False)
                if json_output or not isinstance(serialized, str)
                else serialized
            )
    except click.ClickException:
        raise
    except MobilerunError as exc:
        raise click.ClickException(" ".join(str(exc).splitlines())) from exc
