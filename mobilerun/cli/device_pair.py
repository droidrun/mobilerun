"""Pair and connect a local Android device through adb wireless debugging."""

import secrets
import shutil
import string
import subprocess
import sys
import time
from typing import NamedTuple

import click

from mobilerun.cli.options import CONTEXT_SETTINGS, Command, emit, verbose_option
from mobilerun.cli.qr import print_qr


class MdnsService(NamedTuple):
    name: str
    service_type: str
    address: str

    @property
    def ip(self) -> str:
        return self.address.rsplit(":", 1)[0]


def parse_mdns_services(output: str) -> list[MdnsService]:
    """Read adb's tabular services, ignoring its header and unrelated lines."""
    services = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        name, service_type, address = fields
        service_type = service_type.rstrip(".")
        if service_type not in {"_adb-tls-pairing._tcp", "_adb-tls-connect._tcp"}:
            continue
        host, separator, port = address.rpartition(":")
        if host and separator and port.isdigit() and 0 < int(port) < 65536:
            services.append(MdnsService(name, service_type, address))
    return services


def _run_adb(
    *args: str, input: str | None = None, timeout: float = 10
) -> subprocess.CompletedProcess[str]:
    """Keep adb execution in one replaceable boundary; never log its input."""
    return subprocess.run(
        ["adb", *args], input=input, capture_output=True, text=True,
        timeout=timeout, check=False,
    )


def _adb(*args: str, deadline: float, password: str = "", input: str | None = None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise click.ClickException("Timed out waiting for wireless debugging.")
    try:
        result = _run_adb(*args, input=input, timeout=min(10, remaining))
    except subprocess.TimeoutExpired:
        raise click.ClickException("Timed out waiting for adb wireless debugging.") from None
    # adb diagnostics must not echo the pairing password, even on failure.
    if password:
        result.stdout = result.stdout.replace(password, "[redacted]")
        result.stderr = result.stderr.replace(password, "[redacted]")
    if result.returncode:
        raise click.ClickException(
            result.stderr.strip() or result.stdout.strip() or "adb command failed."
        )
    return result


def _wait_for_service(
    service_type: str, deadline: float, password: str, *, name: str | None = None,
    ip: str | None = None,
) -> MdnsService:
    while True:
        result = _adb("mdns", "services", deadline=deadline, password=password)
        for service in parse_mdns_services(result.stdout):
            if (service.service_type == service_type
                    and (name is None or service.name == name)
                    and (ip is None or service.ip == ip)):
                return service
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise click.ClickException("Timed out waiting for wireless debugging.")
        time.sleep(min(1, remaining))


@click.command(cls=Command, context_settings=CONTEXT_SETTINGS)
@verbose_option
@click.option(
    "--timeout", type=click.IntRange(min=1), default=120, show_default=True,
    help="Maximum time in seconds to pair and connect.",
)
def pair(timeout: int, verbose: bool | None):
    """Pair a local Android device by scanning a wireless debugging QR code."""
    if not sys.stderr.isatty():
        raise click.UsageError("This command needs a terminal.")
    if not shutil.which("adb"):
        raise click.ClickException("adb is missing; install Android SDK Platform-Tools.")
    deadline = time.monotonic() + timeout
    check = _adb("mdns", "check", deadline=deadline)
    if "mdns daemon version" not in check.stdout.lower():
        raise click.ClickException("adb mDNS is unavailable; enable adb mDNS discovery.")

    name = "mobilerun-" + "".join(
        secrets.choice(string.ascii_lowercase + string.digits) for _ in range(6)
    )
    password = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))
    click.echo(
        'On the device, open Developer options → Wireless debugging → '
        '"Pair device with QR code"; keep both devices on the same Wi-Fi and scan below.',
        err=True,
    )
    if not print_qr(f"WIFI:T:ADB;S:{name};P:{password};;"):
        raise click.UsageError("This command needs a terminal.")
    pairing = _wait_for_service(
        "_adb-tls-pairing._tcp", deadline, password, name=name
    )
    result = _adb("pair", pairing.address, input=password + "\n",
                  deadline=deadline, password=password)
    if "successfully paired to" not in result.stdout.lower():
        raise click.ClickException(
            result.stderr.strip() or result.stdout.strip() or "adb pairing failed."
        )
    connection = _wait_for_service(
        "_adb-tls-connect._tcp", deadline, password, ip=pairing.ip
    )
    result = _adb("connect", connection.address, deadline=deadline, password=password)
    if not any(line.lower().startswith(("connected to ", "already connected to "))
               for line in result.stdout.splitlines()):
        raise click.ClickException(
            result.stderr.strip() or result.stdout.strip() or "adb connection failed."
        )
    emit(connection.address)
    click.echo(
        f"Paired and connected. Run `mobilerun device setup -d {connection.address}`.",
        err=True,
    )
