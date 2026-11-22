import re
from typing import TypeAlias

import requests
from mobilerun_core_local.driver.android.portal import GITHUB_API_HOSTS

VersionTuple: TypeAlias = tuple[int, ...]


def parse_version_tuple(version_str: str) -> VersionTuple:
    """Parse a version string like '0.4.23' or 'v0.4.23' into a comparable tuple."""
    cleaned = version_str.lstrip("v").strip()
    parts = []
    for p in cleaned.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def parse_portal_version(version_str: str) -> VersionTuple:
    """Like parse_version_tuple, but '0.7.26-dev' reads as 0.7.26."""
    cleaned = version_str.lstrip("v").strip()
    parts = []
    for p in cleaned.split("."):
        match = re.match(r"\d+", p)
        parts.append(int(match.group()) if match else 0)
    return tuple(parts)


def get_latest_sdk_version() -> str | None:
    for host in GITHUB_API_HOSTS:
        try:
            url = f"{host}/repos/droidrun/mobilerun/releases/latest"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if "release" in data:
                    tag = data["release"].get(
                        "tag", data["release"].get("tag_name", "")
                    )
                else:
                    tag = data.get("tag_name", "")
                return tag.lstrip("v")
        except Exception:
            continue
    return None
