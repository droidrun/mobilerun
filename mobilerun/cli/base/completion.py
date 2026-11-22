"""Install or print Click shell-completion scripts."""

import os  # SHELL identifies the user's current shell for completion installation.
import re
import shutil
import subprocess
import sys
from pathlib import Path

import click
from click.shell_completion import ShellComplete, get_completion_class

from mobilerun.cli.base import menu
from mobilerun.cli.base.prompts import SelectChoice, confirm_prompt, select_prompt

_SHELLS = ("zsh", "bash", "fish")


def _stdout_is_tty() -> bool:
    return sys.stdout.isatty()


def _bash_version() -> tuple[int, int]:
    configured = os.environ.get("SHELL", "")
    bash = configured if Path(configured).name == "bash" else shutil.which("bash")
    if not bash:
        raise click.ClickException(
            "Cannot detect bash version; install bash 4.4 or newer."
        )
    try:
        result = subprocess.run(
            [bash, "--norc", "-c", 'printf "%s" "$BASH_VERSION"'],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise click.ClickException(
            "Cannot detect bash version; install bash 4.4 or newer."
        ) from exc
    match = re.match(r"^(\d+)\.(\d+)", result.stdout)
    if match is None:
        raise click.ClickException(
            "Cannot detect bash version; install bash 4.4 or newer."
        )
    return int(match.group(1)), int(match.group(2))


def _completion_script(shell: str) -> str:
    from mobilerun.cli.app import cli

    if shell == "bash":
        version = _bash_version()
        if version < (4, 4):
            raise click.ClickException(
                f"bash {version[0]}.{version[1]} is too old for completion (needs 4.4+). "
                "Use zsh (macOS default) or `brew install bash`."
            )
    cls = get_completion_class(shell)
    if cls is None:
        raise click.ClickException(f"Unsupported shell: {shell}")
    # BashComplete.source() probes PATH and prints a warning; the selected shell
    # was checked above, so use the common generator without that extra probe.
    return ShellComplete.source(cls(cli, {}, "mobilerun", "_MOBILERUN_COMPLETE"))


@click.command()
@click.argument("shell", required=False, type=click.Choice(_SHELLS))
@click.option(
    "--print",
    "print_script",
    is_flag=True,
    help="Print the script instead of installing.",
)
@click.option(
    "--yes", "-y", is_flag=True, help="Add shell startup configuration without asking."
)
def completion(shell: str | None, print_script: bool, yes: bool) -> None:
    """Install completion for SHELL, or print it when output is redirected."""
    tty = _stdout_is_tty()
    if shell is None:
        detected = Path(os.environ.get("SHELL", "")).name
        if detected in _SHELLS:
            shell = detected
        elif tty:
            try:
                shell = select_prompt(
                    "Select your shell", [SelectChoice(s, s) for s in _SHELLS]
                )
            except (KeyboardInterrupt, EOFError, click.Abort):
                raise click.exceptions.Exit(130) from None
            if shell is None:
                return
        else:
            raise click.ClickException(
                "Cannot detect shell; specify bash, zsh, or fish."
            )
    script = _completion_script(shell)
    if print_script or not tty:
        click.echo(script)
        return

    home = Path.home()
    script_path = home / ".mobilerun" / f"completion.{shell}"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script + "\n", encoding="utf-8")
    if shell == "fish":
        fish_path = home / ".config" / "fish" / "completions" / "mobilerun.fish"
        fish_path.parent.mkdir(parents=True, exist_ok=True)
        fish_path.write_text(script + "\n", encoding="utf-8")
        click.echo("Installed fish completion. Restart your shell.")
        return

    rc_name = (
        ".zshrc"
        if shell == "zsh"
        else (".bash_profile" if sys.platform == "darwin" else ".bashrc")
    )
    rc = home / rc_name
    line = f'source "$HOME/.mobilerun/completion.{shell}"'
    existing = rc.read_text(encoding="utf-8") if rc.exists() else ""
    if line not in (item.strip() for item in existing.splitlines()):
        if not yes and (
            not menu.is_interactive()
            or not confirm_prompt(f"Add completion to ~/{rc_name}?", default=False)
        ):
            click.echo(
                f"Saved completion to ~/.mobilerun/completion.{shell}; ~/{rc_name} unchanged."
            )
            return
        with rc.open("a", encoding="utf-8") as stream:
            if existing and not existing.endswith("\n"):
                stream.write("\n")
            stream.write(line + "\n")
    click.echo(
        f"Installed {shell} completion. Restart your shell or run: source ~/{rc_name}"
    )
