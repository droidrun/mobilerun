"""Write, inspect, and run local sandbox scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import click

from mobilerun.cli.options import Group, emit, examples, json_option, verbose_option

python_option = click.option("--python", "python_path", help="Python interpreter for the script.")


def _interpreter(python_path):
    if python_path:
        interpreter = python_path
    elif os.environ.get("VIRTUAL_ENV"):
        executable = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
        interpreter = str(Path(os.environ["VIRTUAL_ENV"]).joinpath(*executable))
    else:
        interpreter = sys.executable
    result = subprocess.run(
        [interpreter, "-c", "import mobilerun_sandbox"], capture_output=True, check=False,
    )
    if result.returncode:
        click.echo(
            f"mobilerun-sandbox is not installed in {interpreter}. "
            "Install it with: pip install mobilerun-sandbox", err=True,
        )
        raise click.exceptions.Exit(1)
    return interpreter


@click.group(cls=Group, epilog=examples("mobilerun script init script.py"))
@verbose_option
def script_cli(verbose):
    """Write and test sandbox scripts locally."""


@script_cli.command("init", epilog=examples("mobilerun script init script.py"))
@click.argument("path", type=click.Path(dir_okay=False))
@python_option
@verbose_option
def init_script(path, python_path, verbose):
    """Create a starter script."""
    interpreter = _interpreter(python_path)
    result = subprocess.run(
        [interpreter, "-m", "mobilerun_sandbox", "new", path],
        capture_output=True, text=True, check=False,
    )
    if result.stdout:
        emit(result.stdout.rstrip("\n"))
    if result.stderr:
        click.echo(result.stderr, err=True, nl=False)
    raise click.exceptions.Exit(result.returncode)


@script_cli.command("check", epilog=examples("mobilerun script check script.py -j"))
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@python_option
@json_option
@verbose_option
def check_script(path, python_path, json_output, verbose):
    """Inspect a script's entry point and schemas."""
    interpreter = _interpreter(python_path)
    result = subprocess.run(
        [interpreter, "-m", "mobilerun_sandbox", "inspect", path],
        capture_output=True, text=True, check=False,
    )
    if result.stderr:
        click.echo(result.stderr, err=True, nl=False)
    if result.returncode:
        raise click.exceptions.Exit(result.returncode)
    if json_output:
        emit(result.stdout.rstrip("\n"))
    else:
        document = json.loads(result.stdout)
        emit(f"Entry point: {document['entrypoint']}")
        emit(f"Output kind: {document['outputKind']}")
        emit(f"Dataset key: {document['datasetKey']}")
        emit("Input schema:\n" + json.dumps(document["inputSchema"], indent=2))
        emit("Output schema:\n" + json.dumps(document["outputSchema"], indent=2))


@script_cli.command("run", epilog=examples('mobilerun script run script.py --local --input \'{}\''))
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--local", is_flag=True, required=True, help="Run in your local Python environment.")
@click.option("--input", "input_json", help="Script input as JSON.")
@click.option("--input-file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Read script input from a JSON file.")
@click.option("--state-file", type=click.Path(dir_okay=False), help="Read and save script state.")
@python_option
@json_option
@verbose_option
def run_script(path, local, input_json, input_file, state_file, python_path, json_output, verbose):
    """Run a script and print its result."""
    if (input_json is None) == (input_file is None):
        raise click.UsageError("Provide exactly one of --input or --input-file.")
    try:
        data = json.loads(input_json if input_json is not None else input_file.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        option = "--input" if input_json is not None else "--input-file"
        raise click.BadParameter(str(exc), param_hint=option) from None
    interpreter = _interpreter(python_path)
    with tempfile.TemporaryDirectory(prefix="mobilerun-script-") as directory:
        input_path = Path(directory) / "input.json"
        result_path = Path(directory) / "result.json"
        input_path.write_text(json.dumps(data), encoding="utf-8")
        command = [interpreter, "-m", "mobilerun_sandbox", "run", path,
                   "--input-file", str(input_path), "--result-file", str(result_path)]
        if state_file is not None:
            command.extend(["--state-file", state_file])
        # Inherit stderr so script logs appear immediately while it runs.
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=False)
        if result_path.exists():
            raw = result_path.read_text(encoding="utf-8")
            emit(raw.rstrip("\n") if json_output else json.dumps(json.loads(raw), indent=2))
        elif not result.returncode:
            raise click.ClickException("The script did not write a result file.")
        raise click.exceptions.Exit(result.returncode)
