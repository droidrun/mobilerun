import json
from pathlib import Path

from mobilerun.agent.utils.trajectory import MACRO_SCHEMA_VERSION, Trajectory
from mobilerun.cli.deviceauth import load_credentials
from mobilerun.credential_manager.file_credential_manager import FileCredentialManager


def test_trajectory_load_supports_utf8(tmp_path: Path):
    """Verify trajectory and macro loaders handle UTF-8 characters on all platforms."""
    traj_dir = tmp_path / "sample_traj"
    traj_dir.mkdir()

    sample_trajectory = {
        "user_prompt": "Search for cafe with Wi-Fi in Tokyo (東京) ☕",
        "steps": [
            {"thought": "Opening Google Maps — searching for 'Café'", "action": "tap"}
        ],
    }
    sample_macro = {
        "macro_schema_version": MACRO_SCHEMA_VERSION,
        "description": "Tokyo cafe search: café & ☕",
        "total_actions": 1,
        "actions": [{"action": "tap", "target": "Search 🔍"}],
    }

    (traj_dir / "trajectory.json").write_text(
        json.dumps(sample_trajectory), encoding="utf-8"
    )
    (traj_dir / "macro.json").write_text(
        json.dumps(sample_macro), encoding="utf-8"
    )

    loaded_traj = Trajectory.load_trajectory_folder(str(traj_dir))
    assert loaded_traj["trajectory_data"]["user_prompt"] == sample_trajectory["user_prompt"]
    assert loaded_traj["macro_data"]["description"] == sample_macro["description"]

    loaded_macro = Trajectory.load_macro_sequence(str(traj_dir / "macro.json"))
    assert loaded_macro["description"] == sample_macro["description"]


def test_file_credential_manager_loads_utf8_secrets(tmp_path: Path):
    """Verify secrets file loader preserves non-ASCII UTF-8 strings."""
    secrets_file = tmp_path / "secrets.yaml"
    secrets_content = """secrets:
  API_KEY:
    value: "p@ssw0rd—with—em-dash-and-é-accents"
    enabled: true
  NOTE: "東京カフェ"
"""
    secrets_file.write_text(secrets_content, encoding="utf-8")

    mgr = FileCredentialManager(str(secrets_file))
    secrets = mgr.secrets
    assert secrets["API_KEY"] == "p@ssw0rd—with—em-dash-and-é-accents"
    assert secrets["NOTE"] == "東京カフェ"


def test_deviceauth_load_credentials_utf8(tmp_path: Path, monkeypatch):
    """Verify deviceauth does not fail on non-ASCII token data."""
    cred_file = tmp_path / "credentials.json"
    data = {"access_token": "token-with-special-char-—é", "user": "Wassim"}
    cred_file.write_text(json.dumps(data), encoding="utf-8")

    monkeypatch.setattr("mobilerun.cli.deviceauth.credential_path", lambda: cred_file)
    loaded = load_credentials()
    assert loaded is not None
    assert loaded["access_token"] == data["access_token"]
    assert loaded["user"] == "Wassim"


def test_process_exists_helper_lifecycle():
    """Verify the cross-platform process existence helper correctly identifies alive and dead processes."""
    import subprocess
    import sys
    import time

    from tests.test_mcp_stdio_compatibility import _process_exists

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
    try:
        assert _process_exists(proc.pid) is True
    finally:
        proc.terminate()
        proc.wait()

    deadline = time.monotonic() + 2
    while _process_exists(proc.pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _process_exists(proc.pid) is False
