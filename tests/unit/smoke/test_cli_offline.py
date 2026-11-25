"""Offline smoke contracts for CLI completion, privacy, installers, and app IDs."""

import json
import subprocess
import sys
from pathlib import Path


def test_installed_cli_bash_completion(monkeypatch):
    monkeypatch.setenv("_MOBILERUN_COMPLETE", "bash_complete")
    monkeypatch.setenv("COMP_WORDS", "mobilerun ")
    monkeypatch.setenv("COMP_CWORD", "1")
    result = subprocess.run(
        [str(Path(sys.executable).with_name("mobilerun"))],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert {"plain,account", "plain,device", "plain,macro"} <= set(
        result.stdout.splitlines()
    )


def test_offline_cli_protects_local_skill_and_macro_text(tmp_path):
    from mobilerun.skills.bundle import bundle_folders

    def cli(*args):
        return subprocess.run(
            [sys.executable, "-m", "mobilerun", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
        )

    name = next(iter(bundle_folders()))
    target = tmp_path / "skills"
    options = ["--agent", "codex", "--dir", str(target)]
    installed = cli("skills", "install", "bundle", name, *options)
    assert installed.returncode == 0, installed.stderr
    notes = target / name / "user-notes.md"
    notes.write_text("keep me")
    refused = cli("skills", "uninstall", name, *options)
    assert refused.returncode != 0
    assert notes.read_text() == "keep me"

    macro = tmp_path / "macro.json"
    macro.write_text(
        json.dumps(
            {
                "macro_schema_version": "2.0",
                "actions": [
                    {
                        "type": "tap",
                        "x": 4,
                        "y": 5,
                        "pre_state": {"text": "SYNTHETIC-PRIVATE"},
                    }
                ],
            }
        )
    )
    preview = cli("macro", "replay", str(macro), "--dry-run")
    assert preview.returncode == 0, preview.stderr
    assert "tap x=4 y=5" in preview.stdout
    assert "SYNTHETIC-PRIVATE" not in preview.stdout + preview.stderr


def test_local_workspace_id_fails_without_device_or_credentials(monkeypatch):
    monkeypatch.delenv("MOBILERUN_CONFIG", raising=False)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mobilerun",
            "device",
            "app",
            "install",
            "--android",
            "--device",
            "offline-serial",
            "--app-id",
            "workspace-id",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "Pass a local APK path" in result.stderr


def test_local_config_and_skill_dependencies(tmp_path, monkeypatch):
    from mobilerun.skills.bundle import list_skills

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "active"))

    def cli(*args, check=True):
        return subprocess.run(
            [sys.executable, "-m", "mobilerun", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=check,
        )

    cli("mcp", "install", "--agent", "codex")
    assert (tmp_path / "active/config.toml").is_file()
    cli("mcp", "uninstall", "--agent", "codex")
    assert not (tmp_path / ".codex/config.toml").exists()

    skill = next(s for s in list_skills() if s.parent)
    options = ["--agent", "codex", "--dir", str(tmp_path / "skills")]
    cli("skills", "install", skill.name, *options)
    blocked = cli("skills", "uninstall", skill.parent, *options, check=False)
    assert blocked.returncode != 0 and skill.name in blocked.stdout + blocked.stderr
    cli("skills", "uninstall", skill.parent, skill.name, *options)
