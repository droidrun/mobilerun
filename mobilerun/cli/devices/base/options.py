from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, overload

import click

from mobilerun.cli.base.menu import PromptingParameter, is_interactive
from mobilerun.cli.base.prompts import SelectChoice, confirm_prompt, select_prompt
from mobilerun.cli.devices.base.default import save_default_device
from mobilerun.cli.devices.base.discovery import DiscoveredDevice, discover_devices
from mobilerun.cli.devices.base.selection import detect_device_kind
from mobilerun.config.loader import ConfigLoader
from mobilerun.config.user_config import MobileConfig

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device


@dataclass(frozen=True)
class DeviceTarget:
    """A resolved CLI target; opening it always releases the connection."""

    identifier: str | None
    kind: str | None
    config: MobileConfig
    tcp: bool | None = None
    base_url: str | None = None
    allow_auto_setup: bool = False

    def __str__(self) -> str:
        return self.identifier or "no device"

    @asynccontextmanager
    async def open(self) -> AsyncIterator[Device | None]:
        from mobilerun.cli.devices.base.selection import connect_target

        if self.identifier is None:
            yield None
            return
        device = await connect_target(
            self.identifier,
            kind=self.kind,
            config=self.config,
            tcp=self.tcp,
            base_url=self.base_url,
            allow_auto_setup=self.allow_auto_setup,
        )
        try:
            yield device
        finally:
            await device.close()


def device_command(
    f: Callable[..., Any] | None = None,
    *,
    optional: bool = False,
    base_url_name: str = "base_url",
    cloud_only: bool = False,
    allow_auto_setup: bool = False,
) -> Callable[..., Any]:
    """Add device controls and inject exactly one DeviceTarget parameter."""
    if f is None:
        return lambda command: device_command(
            command,
            optional=optional,
            base_url_name=base_url_name,
            cloud_only=cloud_only,
            allow_auto_setup=allow_auto_setup,
        )

    @wraps(f)
    def invoke(*args: Any, **kwargs: Any) -> Any:
        identifier = kwargs.pop("target")
        android = kwargs.pop("android")
        ios = kwargs.pop("ios")
        cloud = kwargs.pop("cloud")
        remote = kwargs.pop("visual_remote")
        tcp = kwargs.pop("tcp")
        base_url = kwargs.pop(base_url_name)
        kind = (
            selected_device_kind(identifier, android, ios, cloud, remote, tcp, base_url)
            if identifier is not None
            else None
        )
        if (
            identifier is None
            and not kwargs.get("dry_run")
            and (android or ios or cloud or remote or tcp is not None or base_url)
        ):
            raise click.UsageError("Device type and transport options require DEVICE.")
        if cloud_only and kind != "cloud":
            raise click.UsageError("This command requires a cloud device ID.")
        config = click.get_current_context().meta["device_config"]
        kwargs["target"] = DeviceTarget(
            identifier,
            kind,
            config,
            (
                tcp
                if tcp is not None
                else (config.device.use_tcp if kind == "android" else None)
            ),
            base_url,
            allow_auto_setup,
        )
        return f(*args, **kwargs)

    return device_options(
        invoke, optional=optional, base_url_name=base_url_name, cloud_only=cloud_only
    )


def choose_device(
    devices: list[DiscoveredDevice], *, save: Callable[[str], None] | None = None
) -> str:
    if not devices:
        raise click.ClickException(
            "No devices found. Plug in a phone, run `mobilerun device setup`, "
            "or `mobilerun account login` for cloud devices."
        )
    if len(devices) == 1:
        device = devices[0]
        click.echo(
            f"Using {device.id}" + (f" ({device.name})" if device.name else ""),
            err=True,
        )
        return device.id
    if not is_interactive():
        candidates = ", ".join(f"{device.id} ({device.source})" for device in devices)
        raise click.ClickException(
            f"Devices found: {candidates}. Pass a device explicitly "
            "or run `mobilerun device connect <id>`."
        )
    choices = []
    for device in devices:
        label = device.id
        if device.name:
            label += f" ({device.name})"
        if device.state:
            label += f" [{device.state}]"
        label += f" {device.source}"
        if not device.ready:
            label += " (not ready)"
        choices.append(SelectChoice(device.id, label))
    try:
        target = select_prompt("Select a device", choices)
        if target is None:
            click.get_current_context().exit(0)
        if save and confirm_prompt(f"Save {target} as default device?", default=False):
            save(target)
    except (KeyboardInterrupt, click.Abort, EOFError):
        click.get_current_context().exit(130)
    return target


@overload
def device_options(
    f: Callable[..., Any],
    *,
    optional: bool = False,
    base_url_name: str = "base_url",
    cloud_only: bool = False,
) -> Callable[..., Any]: ...


@overload
def device_options(
    f: None = None,
    *,
    optional: bool = False,
    base_url_name: str = "base_url",
    cloud_only: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


def device_options(
    f: Callable[..., Any] | None = None,
    *,
    optional: bool = False,
    base_url_name: str = "base_url",
    cloud_only: bool = False,
) -> Callable[..., Any]:
    """Add target options: DEVICE, explicit config, env, saved config, discovery."""
    if f is None:
        return lambda command: device_options(
            command,
            optional=optional,
            base_url_name=base_url_name,
            cloud_only=cloud_only,
        )
    original = f
    is_device_command = f.__module__.startswith("mobilerun.cli.devices.")

    def resolve(
        ctx: click.Context, target: str | None, kwargs: dict[str, Any]
    ) -> str | None:
        path = kwargs.get("config_path", kwargs.get("config"))
        config = ConfigLoader.load(path)
        ctx.meta["device_config"] = config
        explicit = target or kwargs.get("device_target")
        if target and kwargs.get("device_target") and target != kwargs["device_target"]:
            raise click.UsageError(
                "DEVICE and --device refer to different devices; choose one."
            )
        if explicit:
            return explicit
        if kwargs.get("dry_run"):
            return None
        if cloud_only:
            from mobilerun.cli.base.choices import cloud_devices
            from mobilerun.cli.devices.base.pickers import choose_value

            saved = config.device.serial
            if (
                saved
                and not saved.lower().startswith(("http://", "https://"))
                and detect_device_kind(saved) == "cloud"
            ):
                return saved
            choices = []
            if is_interactive():
                try:
                    choices = list(cloud_devices(ctx))
                except Exception:
                    click.echo(
                        "Could not load cloud devices; enter an ID manually.", err=True
                    )
            return choose_value(None, "cloud device", choices)
        saved = config.device.serial
        requested = [
            kind
            for kind in ("android", "ios", "cloud", "visual_remote")
            if kwargs.get(kind)
        ]
        if len(requested) > 1:
            raise click.UsageError("Choose at most one device type flag.")
        kind = requested[0] if requested else None
        if saved and kind and _infer_device_kind(saved, config) != kind:
            saved = None
        if saved and saved.lower().startswith(("http://", "https://")):
            return saved
        if (
            not saved
            and (kind or getattr(config.device, "platform", "android")) == "ios"
        ):
            from mobilerun_core_local.driver.ios import discover_ios_device

            return asyncio.run(discover_ios_device())
        if kind == "visual_remote":
            raise click.UsageError("Pass a remote device URL with --remote.")
        devices = asyncio.run(
            discover_devices(
                local=kind != "cloud",
                cloud=kind != "android",
                base_url=kwargs.get(base_url_name),
            )
        )
        if saved:
            if not any(device.id == saved for device in devices):
                raise click.ClickException(
                    f"Default device {saved} is not available. Run `mobilerun device connect` "
                    "to pick another or `mobilerun device disconnect` to clear it."
                )
            return saved
        if optional and not devices:
            return None
        save = (
            partial(save_default_device, config_path=path)
            if path
            else save_default_device
        )
        return choose_device(devices, save=save)

    def target_callback(
        ctx: click.Context, param: click.Parameter, value: str | None
    ) -> str | None:
        # Only pick early when a required command argument would prompt first.
        # Explicit kind options are eager too, so Click processes them before DEVICE.
        if (
            not is_device_command
            or ctx.resilient_parsing
            or not is_interactive()
            or not any(
                isinstance(p, PromptingParameter) and p.required
                for p in ctx.command.params
            )
        ):
            return value
        return resolve(ctx, value, ctx.params)

    @wraps(original)
    def with_config(*args: Any, **kwargs: Any) -> Any:
        ctx = click.get_current_context()
        target = kwargs.get("target")
        if target and kwargs.get("device_target") and target != kwargs["device_target"]:
            raise click.UsageError(
                "DEVICE and --device refer to different devices; choose one."
            )
        if "device_config" not in ctx.meta:
            target = resolve(ctx, target, kwargs)
        kwargs["target"] = target
        kwargs.pop("device_target", None)
        if is_device_command:
            kwargs.pop("config_path", None)
        return original(*args, **kwargs)

    decorated: Callable[..., Any] = with_config
    decorated = click.argument(
        "target",
        required=False,
        default=None,
        metavar="[DEVICE]",
        is_eager=is_device_command,
        callback=target_callback,
    )(decorated)
    decorated = click.option(
        "--device",
        "-d",
        "device_target",
        default=None,
        is_eager=True,
        help="Device to use",
    )(decorated)
    if is_device_command:
        decorated = click.option(
            "--config",
            "config_path",
            is_eager=True,
            type=click.Path(exists=True, dir_okay=False),
            help="Use a specific configuration file",
        )(decorated)
    for option in (
        click.option(
            "--base-url",
            base_url_name,
            is_eager=True,
            default=None,
            help="Cloud API base URL (cloud only)",
        ),
        click.option(
            "--tcp/--no-tcp", default=None, help="Use TCP communication (Android only)"
        ),
        click.option(
            "-v",
            "--remote",
            "visual_remote",
            is_eager=True,
            is_flag=True,
            help="Interpret DEVICE as a remote device URL",
        ),
        click.option(
            "-c",
            "--cloud",
            is_eager=True,
            is_flag=True,
            help="Interpret DEVICE as a cloud device id",
        ),
        click.option(
            "-i",
            "--ios",
            is_eager=True,
            is_flag=True,
            help="Interpret DEVICE as an iOS URL",
        ),
        click.option(
            "-a",
            "--android",
            is_eager=True,
            is_flag=True,
            help="Interpret DEVICE as Android (including Portal HTTP URLs)",
        ),
    ):
        decorated = option(decorated)
    return decorated


def _infer_device_kind(target: str, config: MobileConfig | None) -> str:
    """Use configured transport for a known URL; otherwise infer from the ID."""
    if (
        config
        and target.lower().startswith(("http://", "https://"))
        and (config.device.serial == target or config.device.platform == "ios")
    ):
        return (
            "visual_remote"
            if (config.device.control_backend or "").lower() == "visual-remote"
            else config.device.platform
        )
    return detect_device_kind(target)


def selected_device_kind(
    target: str | None,
    android: bool,
    ios: bool,
    cloud: bool,
    visual_remote: bool,
    tcp: bool | None,
    base_url: str | None,
) -> str:
    if target is None:
        raise click.UsageError("Missing argument 'DEVICE'")
    flags = [
        ("android", android),
        ("ios", ios),
        ("cloud", cloud),
        ("visual_remote", visual_remote),
    ]
    selected = [kind for kind, enabled in flags if enabled]
    if len(selected) > 1:
        raise click.UsageError("Choose at most one device type flag.")
    ctx = click.get_current_context(silent=True)
    config = ctx.meta.get("device_config") if ctx else None
    try:
        kind = selected[0] if selected else _infer_device_kind(target, config)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    if tcp is not None and kind != "android":
        raise click.UsageError("--tcp/--no-tcp may only be used with Android.")
    if base_url is not None and kind != "cloud":
        raise click.UsageError("--base-url may only be used with cloud.")
    return kind
