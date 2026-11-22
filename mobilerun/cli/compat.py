"""Deprecated 0.7 command names, resolved only when requested."""

from copy import copy
from functools import wraps
from importlib import import_module

import click


def legacy_command(name: str, module: str, attr: str, target: str) -> click.Command:
    original = getattr(import_module(module), attr)

    @wraps(original.callback)
    def invoke(*args, **kwargs):
        click.echo(f"Deprecated: use 'mobilerun {target}' instead.", err=True)
        legacy_device = kwargs.pop("legacy_device", None)
        if legacy_device is not None:
            kwargs["target" if name == "run" else "device_target"] = legacy_device
        if name == "run":
            backend = kwargs.pop("legacy_control_backend", None)
            device_id = kwargs.pop("legacy_device_id", None)
            if backend:
                from mobilerun_core_local.driver.visual_remote import (
                    VISUAL_REMOTE_DEFAULT_URL,
                )

                from mobilerun.config.loader import ConfigLoader

                if not (kwargs.get("target") or kwargs.get("device_target")):
                    config = ConfigLoader.load(kwargs.get("config"))
                    kwargs["target"] = config.device.serial or VISUAL_REMOTE_DEFAULT_URL
                # Previously iOS selected the platform independently of the backend.
                kwargs["ios"] = False
                kwargs["visual_remote"] = True
            if device_id is not None:
                click.get_current_context().meta["legacy_remote_device_id"] = device_id
        return original.callback(*args, **kwargs)

    if name == "configure":
        command = click.Group(
            name=name,
            params=original.params,
            callback=invoke,
            invoke_without_command=True,
        )
        # The original configure group had provider-specific OAuth subcommands.
        login = import_module(module.rsplit(".", 1)[0] + ".login.group").login
        for subcommand in login.commands.values():
            command.add_command(subcommand)
        original_invoke = command.callback

        @click.pass_context
        def configure(ctx, **kwargs):
            if ctx.invoked_subcommand:
                click.echo(
                    "Deprecated: use 'mobilerun agent llm login' instead.", err=True
                )
            else:
                return original_invoke(**kwargs)

        command.callback = configure
        return command
    command = copy(original)
    if name in {"run", "setup", "ping"}:
        command.params = [copy(param) for param in original.params]
        for param in command.params:
            if isinstance(param, click.Option):
                if name == "run" and param.name in {"cloud", "android"}:
                    param.opts = [opt for opt in param.opts if opt not in {"-c", "-a"}]
                elif name == "run" and param.name in {"config", "agent"}:
                    param.opts = [*param.opts, "-c" if param.name == "config" else "-a"]
        if name == "run":
            command.params.append(
                click.Option(
                    ["--control-backend", "legacy_control_backend"],
                    type=click.Choice(["visual_remote"]),
                    hidden=True,
                )
            )
            command.params.append(
                click.Option(["--device-id", "legacy_device_id"], hidden=True)
            )
    command.callback = invoke
    return command
