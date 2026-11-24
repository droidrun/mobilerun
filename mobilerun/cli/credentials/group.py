"""Manage vault and private local credential stores."""

import asyncio
import json
import sys
from contextlib import redirect_stdout
from itertools import groupby

import click

from mobilerun.cli.base import render
from mobilerun.cli.base.menu import MenuGroup, is_interactive
from mobilerun.cli.base.prompts import (
    SelectChoice,
    confirm_prompt,
    multi_select_prompt,
    select_prompt,
    text_prompt,
)
from mobilerun.config.settings import get_settings
from mobilerun.core.credentials.cloud import FIELD_ORDER, FIELD_TYPES, CloudCredentials
from mobilerun.core.credentials.file import FileCredentials
from mobilerun.models.billing import Billing
from mobilerun.utils.paths import local_store_path


def _run(operation, *, read=True, local=False):
    async def invoke():
        vault = (
            FileCredentials(get_settings().local_credentials_path)
            if local
            else (
                CloudCredentials(billing=Billing.NO_COMMERCE)
                if read
                else CloudCredentials()
            )
        )
        try:
            return await operation(vault)
        finally:
            await vault.close()

    try:
        return asyncio.run(invoke())
    except PermissionError as exc:
        raise click.ClickException(str(exc)) from None


def _cancel():
    click.echo("Cancelled.")
    raise click.exceptions.Exit(0)


def _pick(label, values):
    values = sorted(set(value for value in values if value is not None))
    if not values:
        raise click.ClickException(f"No {label.lower()} found.")
    result = select_prompt(label, [SelectChoice(value, value) for value in values])
    if result is None:
        _cancel()
    return result


def _resolve(package, name, field=None, *, need_field=False, local=False):
    if package and name and (field or not need_field):
        return package, name, field
    if not is_interactive():
        raise click.UsageError("Missing required package, name, or field.")
    entries = _run(lambda vault: vault.list(), local=local)
    if not package:
        choices = [
            SelectChoice(item, item)
            for item in sorted({e["package"] for e in entries if e["package"]})
        ]
        if local and any(e["package"] is None for e in entries):
            choices.append(SelectChoice("__general__", "No app (general)"))
        if not choices:
            raise click.ClickException("No package found.")
        package = select_prompt("Package", choices)
        if package is None:
            _cancel()
        if package == "__general__":
            package = None
    matching = [item for item in entries if item["package"] == package]
    name = name or _pick("Name", [item["name"] for item in matching])
    if need_field and not field:
        field = _pick(
            "Field",
            [f for item in matching if item["name"] == name for f in item["fields"]],
        )
    return package, name, field


def _installed_packages():
    async def fetch():
        from mobilerun.cli.devices.base.selection import connect_target
        from mobilerun.config.loader import ConfigLoader

        config = ConfigLoader.load(None)
        serial = config.device.serial
        if not serial:
            return []
        device = await connect_target(serial, config=config, allow_auto_setup=False)
        try:
            apps = await device.list_apps()
            return [
                package
                for item in apps
                if (
                    package := item.get("package")
                    or item.get("package_name")
                    or item.get("packageName")
                )
            ]
        finally:
            await device.close()

    try:
        return asyncio.run(asyncio.wait_for(fetch(), timeout=1.5))
    except Exception:
        return []


def _save_package(local):
    entries = _run(lambda vault: vault.list(), local=local)
    packages = {item["package"] for item in entries if item["package"]}
    packages.update(_installed_packages())
    choices = [SelectChoice(package, package) for package in sorted(packages)]
    if local:
        choices.append(SelectChoice("__general__", "No app (general)"))
    choices.append(
        SelectChoice("__manual__", "Enter manually…", "e.g. com.google.android.gm")
    )
    selected = select_prompt("Package", choices)
    if selected is None:
        _cancel()
    if selected == "__general__":
        return None
    if selected == "__manual__":
        selected = text_prompt("Package (e.g. com.google.android.gm)").strip()
        if not selected:
            _cancel()
    return selected


@click.group(cls=MenuGroup, help="Manage credentials for agent tasks.")
def credentials():
    pass


@credentials.command("path")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def path(as_json):
    """Show the local credentials file path."""
    settings = get_settings()
    info = local_store_path(
        settings.local_credentials_path,
        overridden=settings.MOBILERUN_CREDENTIALS_FILE is not None,
    )
    click.echo(info.format(as_json))


@credentials.command("list")
@click.option("--package")
@click.option("--json", "as_json", is_flag=True)
@click.option("--local", is_flag=True)
@click.option("--vault", is_flag=True)
def list_credentials(package, as_json, local, vault):
    """List names and field names without values."""
    if local and vault:
        raise click.UsageError("Choose --local or --vault, not both.")
    safe = []
    for source in (["local"] if local else ["vault"] if vault else ["vault", "local"]):
        entries = _run(lambda store: store.list(package), local=source == "local")
        safe.extend(
            {
                "source": item.get("source", source) if source == "vault" else source,
                "package": item["package"],
                "name": item["name"],
                "fields": list(item["fields"]),
            }
            for item in entries
        )
    if as_json:
        click.echo(json.dumps(safe, ensure_ascii=False))
    else:
        if not safe:
            render.empty("credentials", "Add one: mobilerun credentials save")
            return
        safe.sort(
            key=lambda item: ((item["package"] or "").lower(), item["name"].lower())
        )
        render.heading(len(safe), "credential")
        for package, group in groupby(safe, key=lambda item: item["package"]):
            render.message(package or "General")
            rows = []
            for item in group:
                fields = sorted(
                    item["fields"],
                    key=lambda field: (
                        FIELD_ORDER.index(field)
                        if field in FIELD_ORDER
                        else len(FIELD_ORDER)
                    ),
                )
                rows.append(
                    [
                        f"  {item['name']}",
                        render.secondary(", ".join(fields)),
                        render.secondary(item["source"]),
                    ]
                )
            render.table(rows)


@credentials.command("save")
@click.argument("package", required=False)
@click.argument("name", required=False)
@click.option(
    "--field", "fields", multiple=True, type=click.Choice(sorted(FIELD_TYPES))
)
@click.option("--from-json", "json_file", type=click.File("r"))
@click.option("--local", is_flag=True)
def save(package, name, fields, json_file, local):
    """Save fields from hidden prompts or a JSON object in FILE (or -)."""
    interactive = is_interactive()
    if interactive and not package:
        if not local:
            store = select_prompt(
                "Store",
                [
                    SelectChoice("vault", "Vault (cloud tasks)"),
                    SelectChoice("local", "Local file (local devices only)"),
                ],
            )
            if store is None:
                _cancel()
            local = store == "local"
        package = _save_package(local)
    if interactive and not name:
        name = text_prompt("Name", default="default")
        if not name:
            _cancel()
    if not name or (not package and not (interactive and local)):
        raise click.UsageError("Missing required PACKAGE or NAME.")
    if not fields and not json_file and interactive:
        fields = multi_select_prompt(
            "Fields", [SelectChoice(field, field) for field in sorted(FIELD_TYPES)]
        )
        if not fields:
            _cancel()
    if bool(fields) == bool(json_file):
        raise click.UsageError("Specify either --field or --from-json.")
    if json_file:
        try:
            values = json.load(json_file)
        except (ValueError, UnicodeError):
            raise click.ClickException("Invalid JSON input.") from None
        if (
            not isinstance(values, dict)
            or not values
            or not all(
                isinstance(k, str) and k in FIELD_TYPES and isinstance(v, str)
                for k, v in values.items()
            )
        ):
            raise click.ClickException(
                "JSON must be an object of supported field names and string values."
            )
    else:
        values = {field: click.prompt(field, hide_input=True) for field in fields}
    existing = _run(lambda store: store.list(package), local=local)
    if any(item["name"] == name and item["package"] == package for item in existing):
        with redirect_stdout(sys.stderr):
            if not interactive or not confirm_prompt(
                f"Update existing credential {name}?", default=False
            ):
                raise click.Abort()
    _run(lambda store: store.save(package, name, values), read=False, local=local)
    render.success("Credential saved.")


@credentials.command("update")
@click.argument("package", required=False)
@click.argument("name", required=False)
@click.argument("field", required=False)
@click.option("--stdin", "from_stdin", is_flag=True, help="Read the value from stdin.")
@click.option("--local", is_flag=True)
def update(package, name, field, from_stdin, local):
    """Update one field from a hidden prompt or stdin."""
    package, name, field = _resolve(package, name, field, need_field=True, local=local)
    if field not in FIELD_TYPES:
        raise click.UsageError("Unsupported credential field.")
    value = (
        click.get_text_stream("stdin").read().removesuffix("\n")
        if from_stdin
        else click.prompt(field, hide_input=True)
    )
    _run(
        lambda store: store.update(package, name, field, value), read=False, local=local
    )
    render.success("Credential updated.")


@credentials.command("delete")
@click.argument("package", required=False)
@click.argument("name", required=False)
@click.argument("field", required=False)
@click.option("--yes", "-y", is_flag=True)
@click.option("--local", is_flag=True)
def delete(package, name, field, yes, local):
    """Delete a credential or one field."""
    package, name, field = _resolve(package, name, field, local=local)
    if field is None and is_interactive():
        entries = _run(lambda store: store.list(package), local=local)
        fields = [
            f
            for item in entries
            if item["package"] == package and item["name"] == name
            for f in item["fields"]
        ]
        choices = [SelectChoice("__all__", "Entire credential")] + [
            SelectChoice(value, value) for value in sorted(set(fields))
        ]
        selected = select_prompt("Field", choices)
        if selected is None:
            _cancel()
        field = None if selected == "__all__" else selected
    if not yes:
        interactive = is_interactive()
        with redirect_stdout(sys.stderr):
            if not interactive or not confirm_prompt(
                f"Delete {name}{' field ' + field if field else ''}?", default=False
            ):
                raise click.Abort()
    _run(lambda store: store.delete(package, name, field), read=False, local=local)
    render.success("Credential deleted.")
