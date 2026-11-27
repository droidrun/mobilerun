"""Workspace CLI command groups."""

import asyncio
import copy
import json
from dataclasses import asdict

import click

from mobilerun.cli.base import render
from mobilerun.cli.base.cleanup import cleanup
from mobilerun.cli.base.errors import safe_error_message
from mobilerun.cli.base.menu import MenuGroup, PromptingArgument
from mobilerun.cli.base.prompts import SelectChoice
from mobilerun.cli.base.run import coro
from mobilerun.core.workspace import Workspace


def file_choices(ctx: click.Context) -> list[SelectChoice]:
    return [
        SelectChoice(item.id, item.name, item.id)
        for item in asyncio.run(Workspace.list_files())
    ]


def app_choices(ctx: click.Context) -> list[SelectChoice]:
    return [
        SelectChoice(item.id, item.name or item.package, item.id)
        for item in asyncio.run(Workspace.list_apps())
        if item.id is not None
    ]


@click.group(cls=MenuGroup)
def workspace() -> None:
    """Manage workspace files and apps."""


@workspace.group(cls=MenuGroup)
def files() -> None:
    """Manage workspace files."""


@files.command("upload")
@click.option("--json", "as_json", is_flag=True)
@click.argument(
    "paths", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False)
)
@coro
async def upload_files(paths: tuple[str, ...], as_json: bool) -> None:
    """Upload local files to the workspace."""
    entries = []
    for path in paths:
        click.echo(f"Uploading {path}…", err=True)
        try:
            item = await Workspace.upload_file(path)
        except (click.Abort, click.exceptions.Exit):
            raise
        except Exception as exc:
            if not as_json:
                raise
            entries.append(
                {"path": path, "status": "failed", "error": safe_error_message(exc)}
            )
            continue
        entries.append({"id": item.id, "status": "uploaded", "name": item.name})
        if not as_json:
            render.success(f"Uploaded {item.name}")
            render.kv([("File ID", render.secondary(item.id))])
    if as_json:
        click.echo(json.dumps(entries))
        if any(entry["status"] == "failed" for entry in entries):
            click.get_current_context().exit(1)


@files.command("list")
@click.option("--json", "as_json", is_flag=True)
@coro
async def list_files(as_json: bool) -> None:
    """List workspace files."""
    items = await Workspace.list_files()
    if as_json:
        click.echo(json.dumps([asdict(item) for item in items], default=str))
    else:
        if not items:
            render.empty("files", "Upload one: mobilerun workspace files upload")
            return
        render.heading(len(items), "file")
        render.table(
            [
                [item.name, str(item.size_bytes), render.secondary(item.id)]
                for item in items
            ]
        )


@files.command("download")
@click.option("--json", "as_json", is_flag=True)
@click.argument("file_id", cls=PromptingArgument, choices=file_choices)
@click.argument("dest", required=False, default=".")
@coro
async def download_file(file_id: str, dest: str, as_json: bool) -> None:
    """Download a workspace file."""
    click.echo(f"Downloading {file_id}…", err=True)
    path = await Workspace.download_file(file_id, dest)
    click.echo(
        json.dumps({"id": file_id, "status": "downloaded", "path": str(path)})
        if as_json
        else path
    )


@files.command("rm")
@click.option("--json", "as_json", is_flag=True)
@click.argument(
    "file_id", cls=PromptingArgument, choices=file_choices, nargs=-1, required=True
)
@click.option("--yes", "-y", is_flag=True)
@coro
async def delete_file(file_id: tuple[str, ...], yes: bool, as_json: bool) -> None:
    """Delete workspace files."""

    async def remove(item_id: str) -> str:
        await Workspace.delete_file(item_id)
        return "deleted"

    await cleanup(
        file_id,
        action="Delete",
        noun="workspace file",
        billing="no_commerce",
        yes=yes,
        as_json=as_json,
        operation=remove,
    )


_remove_file = copy.copy(delete_file)
_remove_file.name = "remove"
_remove_file.hidden = True
files.add_command(_remove_file)


@workspace.group(
    cls=MenuGroup,
    epilog="Tip: add more apps, including from the Mobilerun App Store, in the dashboard: https://cloud.mobilerun.ai/apps",
)
def apps() -> None:
    """Manage the workspace app library."""


@apps.command("list")
@click.option("--json", "as_json", is_flag=True)
@coro
async def list_apps(as_json: bool) -> None:
    """List workspace apps."""
    items = await Workspace.list_apps()
    if as_json:
        click.echo(json.dumps([asdict(item) for item in items], default=str))
    else:
        if not items:
            render.empty(
                "apps",
                "Upload one: mobilerun workspace apps upload, or add from the App Store: https://cloud.mobilerun.ai/apps",
            )
            return
        render.heading(len(items), "app")
        render.table(
            [
                [
                    item.name or item.package,
                    render.secondary(item.package if item.name else ""),
                    render.secondary(item.id),
                ]
                for item in items
            ]
        )


@apps.command("upload")
@click.option("--json", "as_json", is_flag=True)
@click.argument(
    "apk", cls=PromptingArgument, type=click.Path(exists=True, dir_okay=False)
)
@coro
async def upload_app(apk: str, as_json: bool) -> None:
    """Upload an APK to the workspace."""
    click.echo(f"Uploading {apk}…", err=True)
    item = await Workspace.upload_app(apk)
    if as_json:
        click.echo(
            json.dumps(
                {
                    "id": item.id,
                    "status": "uploaded",
                    "name": item.name,
                    "package": item.package,
                }
            )
        )
    else:
        render.success(f"Uploaded {item.name or item.package}")
        render.kv([("Package", item.package), ("App ID", render.secondary(item.id))])


@apps.command("remove")
@click.option("--json", "as_json", is_flag=True)
@click.argument(
    "app_id", cls=PromptingArgument, choices=app_choices, nargs=-1, required=True
)
@click.option("--yes", "-y", is_flag=True)
@coro
async def delete_app(app_id: tuple[str, ...], yes: bool, as_json: bool) -> None:
    """Delete apps from the workspace."""

    async def remove(item_id: str) -> str:
        await Workspace.delete_app(item_id)
        return "deleted"

    await cleanup(
        app_id,
        action="Delete",
        noun="workspace app",
        billing="no_commerce",
        yes=yes,
        as_json=as_json,
        operation=remove,
    )
