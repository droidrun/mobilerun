"""Private cloud credential persistence."""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from mobilerun.config.env import get_secrets, secret_value
from mobilerun.config.settings import get_settings
from mobilerun.errors.credentials import CredentialFormatError, CredentialReadError


def credential_path() -> Path:
    return get_settings().mobilerun_cloud_credential_path


def _chmod_private(path: Path, mode: int) -> None:
    if os.name != "nt":
        os.chmod(path, mode)


def save_token(token: str, auth_url: str) -> Path:
    """Persist the session token to the credentials dir."""
    return save_credentials({"access_token": token, "auth_url": auth_url})


def save_credentials(credentials: dict) -> Path:
    """Atomically replace saved cloud credentials with a private file."""
    path = credential_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _chmod_private(path.parent, 0o700)

    payload = json.dumps(credentials)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
        _chmod_private(tmp_path, 0o600)
        os.replace(tmp_path, path)
        _chmod_private(path, 0o600)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


def load_credentials() -> Optional[dict]:
    path = credential_path()
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    except UnicodeError as exc:
        raise CredentialFormatError(
            "Cloud credentials must be UTF-8 JSON; run `mobilerun account login` again."
        ) from exc
    except OSError as exc:
        raise CredentialReadError(
            "Cannot read cloud credentials; check file permissions."
        ) from exc
    try:
        data = json.loads(text)
        if not isinstance(data, dict) or any(
            key in data and not isinstance(data[key], str)
            for key in ("api_key", "access_token", "auth_url", "auth_type", "saved_at")
        ):
            raise ValueError("Expected a credential object containing string fields")
        return data
    except (ValueError, UnicodeError) as exc:
        raise CredentialFormatError(
            "Invalid cloud credentials; run `mobilerun account login` again."
        ) from exc


def clear_credentials() -> bool:
    """Delete the saved login. Returns True if a file was removed."""
    path = credential_path()
    if path.exists():
        path.unlink()
        return True
    return False


# Cloud device ids are UUIDs (RFC 4122); ADB serials never are. Used to
# auto-route `-d <id>` to the cloud driver when the user didn't pass `--cloud`.
_CLOUD_DEVICE_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def looks_like_cloud_device_id(value: Optional[str]) -> bool:
    return bool(value and _CLOUD_DEVICE_ID_RE.match(value))


def resolve_cloud_api_key() -> Optional[str]:
    """Resolve the Mobilerun cloud API key for ``--cloud`` commands.

    Order: the ``MOBILERUN_API_KEY`` env var, then the credential persisted by
    ``mobilerun account login`` (``<config dir>/credentials/mobilerun-cloud.json``).
    """
    env_key = secret_value(get_secrets().MOBILERUN_API_KEY)
    if env_key:
        return env_key
    data = load_credentials()
    return (data.get("api_key") or data.get("access_token")) if data else None
