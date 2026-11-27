import requests
from mobilerun_core_local.driver.android.portal import GITHUB_API_HOSTS, REPO


def get_latest_portal_version() -> str | None:
    """Fetch latest portal release version from GitHub."""
    for host in GITHUB_API_HOSTS:
        try:
            url = f"{host}/repos/{REPO}/releases/latest"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if "release" in data:
                    tag = data["release"].get(
                        "tag", data["release"].get("tag_name", "")
                    )
                else:
                    tag = data.get("tag_name", "")
                return tag.lstrip("v").strip() if tag else None
        except Exception:
            continue
    return None
