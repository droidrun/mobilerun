"""Pair and connect a local Android device through adb wireless debugging."""

import re
import secrets
import shutil
import string
import subprocess
import sys
import time
from ipaddress import ip_address
from typing import NamedTuple

import click

from mobilerun.cli.device_commands import device_cli
from mobilerun.cli.options import emit, examples, verbose_option
from mobilerun.cli.qr import print_qr


class MdnsService(NamedTuple):
    name: str
    service_type: str
    address: str


def parse_mdns_services(output: str) -> list[MdnsService]:
    """Read adb's tabular services, ignoring its header and unrelated lines."""
    services = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        name, service_type, address = fields
        service_type = service_type.rstrip(".")
        host, _, port = address.rpartition(":")
        if not port.isdigit() or not 0 < int(port) < 65536:
            continue
        if ":" in host and not (host.startswith("[") and host.endswith("]")):
            continue
        try:
            ip = ip_address(host.removeprefix("[").removesuffix("]"))
        except ValueError:
            continue
        if not ip.is_link_local:
            services.append(MdnsService(name, service_type, address))
    return sorted(services, key=lambda service: service.address.startswith("["))


def parse_adb_devices(output: str) -> dict[str, str]:
    """Read the serial and state columns, ignoring adb's header."""
    devices = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2:
            devices[fields[0]] = fields[1]
    return devices


def _run_adb(
    *args: str, input: str | None = None, timeout: float = 10
) -> subprocess.CompletedProcess[str]:
    """Keep adb execution in one replaceable boundary; never log its input."""
    return subprocess.run(
        ["adb", *args], input=input, capture_output=True, text=True,
        timeout=timeout, check=False,
    )


def _adb_failure(result: subprocess.CompletedProcess[str], fallback: str) -> click.ClickException:
    return click.ClickException(result.stderr.strip() or result.stdout.strip() or fallback)


def adb_connect_succeeded(output: str) -> bool:
    """Recognize adb's connect replies consistently for both commands."""
    return any(line.lower().startswith(("connected to ", "already connected to "))
               for line in output.splitlines())


def _adb(*args: str, deadline: float, input: str | None = None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise click.ClickException("Timed out waiting for wireless debugging.")
    try:
        result = _run_adb(*args, input=input, timeout=min(10, remaining))
    except subprocess.TimeoutExpired:
        raise click.ClickException("Timed out waiting for adb wireless debugging.") from None
    # adb diagnostics must not echo the pairing password, even on failure.
    if input:
        secret = input.rstrip("\r\n")
        if secret:
            result.stdout = result.stdout.replace(secret, "[redacted]")
            result.stderr = result.stderr.replace(secret, "[redacted]")
    if result.returncode:
        raise _adb_failure(result, "adb command failed.")
    return result


def _wait_for_service(service_type: str, deadline: float, *, name: str) -> MdnsService:
    while True:
        result = _adb("mdns", "services", deadline=deadline)
        for service in parse_mdns_services(result.stdout):
            if service.service_type == service_type and service.name == name:
                return service
        time.sleep(1)


def _guid_transport(devices: dict[str, str], guid: str) -> str | None:
    return next((serial for serial, state in devices.items()
                 if serial.startswith(guid) and state == "device"), None)


def _connect_addresses(guid: str, deadline: float) -> list[str]:
    result = _adb("mdns", "services", deadline=deadline)
    return [service.address for service in parse_mdns_services(result.stdout)
            if service.name == guid and service.service_type == "_adb-tls-connect._tcp"]


def _wait_for_transport(guid: str, deadline: float) -> str:
    fallback_at = time.monotonic() + 15
    fallback = None
    while True:
        devices = parse_adb_devices(_adb("devices", deadline=deadline).stdout)
        serial = _guid_transport(devices, guid)
        if serial:
            addresses = _connect_addresses(guid, deadline)
            # Recheck at completion so a late auto-connect wins over TCP.
            devices = parse_adb_devices(_adb("devices", deadline=deadline).stdout)
            serial = _guid_transport(devices, guid)
            if serial:
                for address in dict.fromkeys([*addresses, fallback]):
                    if address and address in devices:
                        _adb("disconnect", address, deadline=deadline)
                        click.echo(f"Disconnected duplicate transport {address}; keeping {serial}.",
                                   err=True)
                return serial
        elif (fallback and devices.get(fallback) == "device"
              and not any(serial.startswith(guid) for serial in devices)):
            return fallback
        elif fallback is None and time.monotonic() >= fallback_at:
            address = next((address for address in _connect_addresses(guid, deadline)
                            if not address.startswith("[")), None)
            if address:
                # The transport may have appeared during mDNS discovery.
                devices = parse_adb_devices(_adb("devices", deadline=deadline).stdout)
                if not any(serial.startswith(guid) for serial in devices):
                    result = _adb("connect", address, deadline=deadline)
                    if not adb_connect_succeeded(result.stdout):
                        raise _adb_failure(result, "adb connection failed.")
                    fallback = address
        time.sleep(1)


@device_cli.command(
    epilog=examples(
        'mobilerun device pair',
        'mobilerun device pair --timeout 180',
    ),
)
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
        raise click.ClickException("Could not display the pairing QR code.")
    pairing = _wait_for_service(
        "_adb-tls-pairing._tcp", deadline, name=name
    )
    result = _adb("pair", pairing.address, input=password + "\n",
                  deadline=deadline)
    if "successfully paired to" not in result.stdout.lower():
        raise _adb_failure(result, "adb pairing failed.")
    identity = re.search(r"\[guid=([^\]\s]+)\]", result.stdout)
    if not identity:
        raise _adb_failure(result, "adb pairing did not return a device GUID.")
    serial = _wait_for_transport(identity.group(1), deadline)
    emit(serial)
    click.echo(
        f"Paired and connected. Run `mobilerun device setup -d {serial}`.",
        err=True,
    )
