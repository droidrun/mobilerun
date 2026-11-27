"""App descriptions and Android package inspection."""

import re
import shutil
import subprocess
from pathlib import Path

from mobilerun.config.settings import get_settings


def inspect_apk(path: str | Path) -> tuple[str, int, str, str]:
    """Read APK metadata using Android's aapt tool."""
    aapt = shutil.which("aapt")
    if not aapt:
        roots = [
            get_settings().ANDROID_HOME,
            get_settings().ANDROID_SDK_ROOT,
            str(Path.home() / "Library/Android/sdk"),
        ]
        for root in filter(None, roots):
            candidates = sorted(
                (Path(root) / "build-tools").glob("*/aapt"), reverse=True
            )
            if candidates:
                aapt = str(candidates[0])
                break
    if not aapt:
        raise RuntimeError(
            "Android build-tools aapt is required to inspect APKs; install build-tools or put aapt on PATH"
        )
    try:
        output = subprocess.run(
            [aapt, "dump", "badging", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"APK inspection timed out: {path}") from exc
    package_line = next(
        (line for line in output.splitlines() if line.startswith("package: ")), ""
    )
    fields = dict(re.findall(r"(\w+)='([^']*)'", package_line))
    if not all(fields.get(key) for key in ("name", "versionCode")):
        raise ValueError(f"APK metadata is missing package or version: {path}")
    label = re.search(r"^application-label:'([^']*)'", output, re.MULTILINE)
    return (
        fields["name"],
        int(fields["versionCode"]),
        fields.get("versionName", ""),
        label.group(1) if label else fields["name"],
    )
