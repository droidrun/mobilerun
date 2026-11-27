import click

from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.core.state.filters import ConciseFilter
from mobilerun.core.state.options import StateProviderOptions
from mobilerun.logs.progress import log_progress_to_stderr


@click.command()
@click.option("--raw", is_flag=True, help="Print the unmodified tree.")
@device_command
@coro
async def ui(
    raw: bool,
    target: DeviceTarget,
) -> None:
    """Print the UI accessibility tree with element bounds for targeting."""
    from mobilerun.core.state.ios import IOSStateProvider

    log_progress_to_stderr()
    async with target.open() as device:
        provider = device._create_state_provider(
            StateProviderOptions(tree_filter=ConciseFilter())
        )
        state = (
            await provider.get_state(strict=True)
            if isinstance(provider, IOSStateProvider)
            else await provider.get_state()
        )
        tree = state.formatted_text
        click.echo(tree if raw else "\n".join(f"  {line}" for line in tree.split("\n")))
        if state.phone_state:
            click.echo(f"\nPhone state: {state.phone_state}")
