"""Mobilerun CLI entry point."""

try:
    import os
    import warnings
    from importlib.metadata import version

    import click

    from mobilerun.cli.options import Group, examples, verbose_option
    from mobilerun.cli.run import run, run_command

    __all__ = ["cli", "run_command"]


    @click.group(
        cls=Group,
        command_order=["run", "device", "configure", "auth", "doctor", "env", "macro", "completion"],
        epilog=examples(
            'mobilerun device list',
            'mobilerun configure -p openai',
            'mobilerun run "Open Settings"',
            docs='https://docs.mobilerun.ai/framework/guides/cli',
        ),
    )
    @click.version_option(version("mobilerun"), "-V", "--version")
    @verbose_option
    def cli(verbose):
        """Mobilerun — control mobile devices and run LLM tasks."""
        warnings.filterwarnings("ignore")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"


    from mobilerun.cli.auth import auth_cli  # noqa: E402
    from mobilerun.cli.completion import completion  # noqa: E402
    from mobilerun.cli.configure import configure  # noqa: E402
    from mobilerun.cli.device_lifecycle import device_cli  # noqa: E402
    from mobilerun.cli.doctor_command import doctor  # noqa: E402
    from mobilerun.cli.env import env  # noqa: E402
    from mobilerun.cli.script import script_cli  # noqa: E402
    from mobilerun.macro.cli import macro_cli  # noqa: E402

    cli.add_command(run)
    cli.add_command(device_cli, "device")
    cli.add_command(auth_cli, "auth")
    cli.add_command(configure)
    cli.add_command(macro_cli, "macro")
    cli.add_command(doctor)
    cli.add_command(env)
    cli.add_command(script_cli, "script")

    cli.add_command(completion)

    _EXAMPLES = {
        "device screenshot": "mobilerun device screenshot -d emulator-5554",
        "device tap": "mobilerun device tap 100 200",
        "device swipe": "mobilerun device swipe 100 500 100 100",
        "device long-press": "mobilerun device long-press 100 200",
        "device type": 'mobilerun device type "Hello"',
        "device press": "mobilerun device press home",
        "device apps": "mobilerun device apps -j",
        "device start": "mobilerun device start com.android.settings",
        "device connect": "mobilerun device connect 192.0.2.1:5555",
        "device disconnect": "mobilerun device disconnect 192.0.2.1:5555",
        "device setup": "mobilerun device setup -d emulator-5554",
        "device ping": "mobilerun device ping -d emulator-5554",
        "auth": "mobilerun auth status",
        "auth status": "mobilerun auth status -j",
        "auth logout": "mobilerun auth logout",
        "doctor": "mobilerun doctor -j",
        "env": "mobilerun env -j",
        "macro": "mobilerun macro list ./macros",
        "macro list": "mobilerun macro list ./macros",
        "completion": "mobilerun completion bash",
    }
    for _path, _example in _EXAMPLES.items():
        _command = cli
        for _name in _path.split():
            _command = _command.commands[_name]
        if not _command.epilog:
            _command.epilog = examples(_example)

except KeyboardInterrupt:
    raise SystemExit(130) from None
