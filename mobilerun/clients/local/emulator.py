"""Android emulator process and ADB helpers."""

import asyncio
from pathlib import Path

from mobilerun.config.settings import get_settings

_processes: dict[str, asyncio.subprocess.Process] = {}


def emulator_binary() -> str:
    sdk = get_settings().ANDROID_HOME or get_settings().ANDROID_SDK_ROOT
    return str(Path(sdk or Path.home() / "Library/Android/sdk") / "emulator/emulator")


async def _command(*args: str) -> str:
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await process.communicate()
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(stderr.decode(errors="replace").strip())
    return stdout.decode(errors="replace")


async def list_avds() -> list[str]:
    return (await _command(emulator_binary(), "-list-avds")).splitlines()


async def find_running(name: str) -> str | None:
    output = await _command("adb", "devices")
    for line in output.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2 or not parts[0].startswith("emulator-"):
            continue
        try:
            result = await _command("adb", "-s", parts[0], "emu", "avd", "name")
        except RuntimeError:
            result = ""
        if not result.strip():
            try:
                result = await _command(
                    "adb", "-s", parts[0], "shell", "getprop", "ro.boot.qemu.avd_name"
                )
            except RuntimeError:
                continue
        if result.splitlines() and result.splitlines()[0].strip() == name:
            return parts[0]
    return None


async def start_avd(name: str, headless: bool = False) -> str:
    """Let the emulator allocate its console port and report it over a socket."""
    loop = asyncio.get_running_loop()
    reported = loop.create_future()

    async def receive(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            value = (await reader.read(32)).decode().strip()
            if not reported.done():
                reported.set_result(value)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(receive, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    process = None
    try:
        args = [emulator_binary(), "-avd", name, "-report-console", f"tcp:{port}"]
        if headless:
            args.append("-no-window")
        process = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        value = await asyncio.wait_for(reported, timeout=30)
        if not value.isdigit() or not 0 < int(value) < 65536:
            raise RuntimeError("Emulator reported an invalid console port")
        serial = f"emulator-{value}"
        _processes[serial] = process
        return serial
    except BaseException:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        raise
    finally:
        server.close()
        await server.wait_closed()


async def wait_boot(serial: str, timeout: float = 300) -> None:
    async def wait() -> None:
        await _command("adb", "-s", serial, "wait-for-device")
        while (
            await _command(
                "adb", "-s", serial, "shell", "getprop", "sys.boot_completed"
            )
        ).strip() != "1":
            await asyncio.sleep(1)

    try:
        await asyncio.wait_for(wait(), timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Emulator {serial} did not boot within {timeout} seconds"
        ) from exc


async def stop(serial: str) -> None:
    process = _processes.get(serial)
    try:
        await _command("adb", "-s", serial, "emu", "kill")
    except Exception:
        if process is None:
            raise
    if process is not None:
        if process.returncode is None:
            process.kill()
            await process.wait()
        _processes.pop(serial, None)
